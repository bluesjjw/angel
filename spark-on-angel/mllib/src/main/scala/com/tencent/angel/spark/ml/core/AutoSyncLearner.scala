package com.tencent.angel.spark.ml.core

import com.tencent.angel.ml.core.conf.{MLConf, SharedConf}
import com.tencent.angel.ml.core.network.layers.{AngelGraph, PlaceHolder, STATUS}
import com.tencent.angel.ml.core.optimizer.decayer.{StandardDecay, StepSizeScheduler}
import com.tencent.angel.ml.core.optimizer.loss.{L2Loss, LogLoss, LossFunc}
import com.tencent.angel.ml.core.utils.paramsutils.JsonUtils
import com.tencent.angel.ml.feature.LabeledData
import com.tencent.angel.ml.math2.matrix.{BlasDoubleMatrix, BlasFloatMatrix, Matrix}
import com.tencent.angel.model.{ModelLoadContext, ModelSaveContext}
import com.tencent.angel.spark.automl.tuner.config.{Configuration, ConfigurationSpace}
import com.tencent.angel.spark.automl.tuner.kernel.Matern5Iso
import com.tencent.angel.spark.automl.tuner.model.GPModel
import com.tencent.angel.spark.automl.tuner.parameter.ParamSpace
import com.tencent.angel.spark.automl.tuner.solver.Solver
import com.tencent.angel.spark.automl.utils.{AutoMLException, DataUtils}
import com.tencent.angel.spark.context.{AngelPSContext, PSContext}
import com.tencent.angel.spark.ml.core.metric.{AUC, Precision}
import com.tencent.angel.spark.ml.util.{DataLoader, SparkUtils}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import com.tencent.angel.spark.automl.sync.model.{Pow3, PowerLaw, PowerLawEnsemble}
import org.apache.spark.ml.linalg.{Vector, Vectors}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.Random

class AutoSyncLearner (tuneIter: Int = 20, minimize: Boolean = false) {

  // Shared configuration with Angel-PS
  val conf = SharedConf.get()

  // Some params
  var numEpoch: Int = conf.getInt(MLConf.ML_EPOCH_NUM)
  var fraction: Double = conf.getDouble(MLConf.ML_BATCH_SAMPLE_RATIO)
  var validationRatio: Double = conf.getDouble(MLConf.ML_VALIDATE_RATIO)

  println(s"fraction=$fraction validateRatio=$validationRatio numEpoch=$numEpoch")

  val autoSyncStrategy = "GP" // GP or AIMD
  var curBarrier = 1
  val maxBarrier = (1.0 / fraction).toInt * 5
  var historyBatch = new ArrayBuffer[Int]()
  var historyMetric = new ArrayBuffer[Double]()

  // GP model
  val cs: ConfigurationSpace = new ConfigurationSpace("cs")
  val covFunc = Matern5Iso()
  val initCovParams = BDV(10, 0.1)
  val initNoiseStdDev = 0.1
  val gpModel: GPModel = GPModel(covFunc, initCovParams, initNoiseStdDev)

  // power-law model
  val powerLawModel = new PowerLawEnsemble()

  def addHistory(batch: Int, metric: Double): Unit = {
    historyBatch += batch
    historyMetric += metric
  }

  def trainGP(): Unit = {
    println(s"history X: ${historyBatch.mkString("[",",","]")}, history Y: ${historyMetric.mkString("[",",","]")}")
    val breezeX: BDM[Double] = DataUtils.toBreeze(
      historyBatch.toArray.map{ batch: Int => Vectors.dense(Array(batch.toDouble)) } )
    val breezeY: BDV[Double] = DataUtils.toBreeze(historyMetric.toArray)
    gpModel.fit(breezeX, breezeY)
  }

  def predictGP(batch: Int): (Double, Double) = {
    val breezeX = DataUtils.toBreeze(Vectors.dense(Array(batch.toDouble))).toDenseMatrix
    val pred = gpModel.predict(breezeX)
    (pred(0, 0), pred(0, 1))
  }

  def nextBarrierGP(): Int = {
    trainGP()
    val optimalBatchNum = historyBatch.last + 1
    var batchNum = historyBatch.last + 1
    while (batchNum < historyBatch.last + maxBarrier) {
      val gpPred = predictGP(batchNum)
      val mean = gpPred._1
      val variance = gpPred._2
      val bestMetric = if (minimize) historyMetric.min else historyMetric.max
      println(s"batch num: $batchNum, current optimal metric: $bestMetric, mean of GP: $mean, variance of GP: $variance")
      batchNum += 1
    }
    optimalBatchNum - historyBatch.last
  }

  def nextBarrierAIMD(): Int = {
    val curMetric = historyMetric.last
    val lastMetric = historyMetric(historyMetric.length - 2)
    if (curMetric > lastMetric)
      curBarrier + 1
    else if (curBarrier > 1)
      curBarrier / 2
    else 1
  }

  def nextBarrierPowerLaw(): Int = {
    
  }

  def updateBarrier(batchNum: Int, metric: Double): Unit = {
    addHistory(batchNum, metric)
    curBarrier = autoSyncStrategy match {
      case "AIMD" => nextBarrierAIMD()
      case "GP" => nextBarrierGP()
    }
  }

  def evaluate(data: RDD[LabeledData], model: GraphModel): (Double, Double) = {
    val scores = data.mapPartitions { case iter =>
      val output = model.forward(1, iter.toArray)
      Iterator.single((output, model.graph.placeHolder.getLabel))
    }
    (new AUC().cal(scores), new Precision().cal(scores))
  }

  def train(data: RDD[LabeledData], model: GraphModel): (Double, Double) = {
    // split data into train and validate
    val ratios = Array(1 - validationRatio, validationRatio)
    val splits = data.randomSplit(ratios)
    val train = splits(0)
    val validate = splits(1)

    println(s"numTrain[${train.count}], numValid[${validate.count}]")

    data.cache()
    train.cache()
    validate.cache()

    train.count()
    validate.count()
    data.unpersist()

    val numSplits = (1.0 / fraction).toInt
    val manifold = OfflineLearner.buildManifold(train, numSplits)

    train.unpersist()

    var numUpdate = 1

    val bcNumSplits = data.sparkContext.broadcast(numSplits)
    var barrier = conf.getInt(MLConf.SYNC_BATCH, MLConf.DEFAULT_SYNC_BATCH)

    val isAIMD = false

    var validateAuc: ArrayBuffer[Double] = new ArrayBuffer[Double]
    var validatePrecision: ArrayBuffer[Double] = new ArrayBuffer[Double]
    validateAuc += 0.1
    validatePrecision += 0.1

    val t1 = System.nanoTime

    var totalBatch = 0
    for (epoch <- 0 until numEpoch) {
      val bcBarrier = data.sparkContext.broadcast(barrier)
      //val batchIterator = OfflineLearner.buildManifoldIterator(manifold, numSplits)
      var innerBatch = 0
      while (innerBatch < numSplits) {
        // on executor
        val (sumLoss, batchSize) = manifold.mapPartitionsWithIndex { (partId, iter) =>
          PSContext.instance()
          val batchIter = iter.toList
          var left = bcBarrier.value
          var counter = 0
          var retBatchSize: Int = Int.MaxValue
          var retLoss: Double = 0
          while (left > 0) {
            val takeNum = left min bcNumSplits.value
            val indices = Random.shuffle(List.range(0, bcNumSplits.value)).take(takeNum)
            println(s"partition $partId, batch indices: ${indices.mkString(",")}")
            batchIter.zipWithIndex.foreach { case (batch, idx) =>
              if (indices.contains(idx)) {
                println(s"counter $counter")
                // feed data
                model.feedData(batch)
                // full model
                if (counter == 0)
                  model.pullParams(epoch)
                // forward
                model.predict()
                val loss = model.getLoss()
                // backward
                model.backward()
                retBatchSize = retBatchSize min batch.length
                retLoss += loss
                counter += 1
              }
            }
            left -= takeNum
          }
          println("------push gradient to ps------")
          model.pushGradient()
          Thread.sleep(10)
          Iterator.single((retLoss, retBatchSize))
        }.reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2))

        innerBatch += bcBarrier.value
        totalBatch += bcBarrier.value
        numUpdate += bcBarrier.value

        val (lr, boundary) = model.update(numUpdate, batchSize)

        val loss = sumLoss / model.graph.taskNum / bcBarrier.value
        println(f"epoch=[$epoch] batch[$totalBatch] lr[$lr%.3f] batchSize[$batchSize] trainLoss=$loss")

        var validateMetricLog = ""
        if (validationRatio > 0.0) {
          val metrics = evaluate(validate, model)
          validateAuc += metrics._1
          validatePrecision += metrics._2
          updateBarrier(totalBatch, metrics._1)
          validateMetricLog = f"validateAuc=${validateAuc.last}%.5f validatePrecision=${validatePrecision.last}%.5f"
          val timeCost = (System.nanoTime - t1) / 1e9d
          println(f"time[$timeCost%.2f s] barrier[$barrier] epoch[$epoch] batch[$numUpdate] batchsize[$batchSize] lr[$lr%.5f] $validateMetricLog")
        }
      }
    }
    (validateAuc.max, validatePrecision.max)
  }

  /**
    * Predict the output with a RDD with data and a trained model
    * @param data: examples to be predicted
    * @param model: a trained model
    * @return RDD[(label, predict value)]
    *
    */
  def predict(data: RDD[(LabeledData, String)], model: GraphModel): RDD[(String, Double)] = {
    val scores = data.mapPartitions { iterator =>
      PSContext.instance()
      val samples = iterator.toArray
      val output  = model.forward(1, samples.map(f => f._1))
      val labels = samples.map(f => f._2)

      (output, model.getLossFunc()) match {
        case (mat :BlasDoubleMatrix, _: LogLoss) =>
          // For LogLoss, the output is (value, sigmoid(value), label)
          (0 until mat.getNumRows).map(idx => (labels(idx), mat.get(idx, 1))).iterator
        case (mat :BlasFloatMatrix, _: LogLoss) =>
          // For LogLoss, the output is (value, sigmoid(value), label)
          (0 until mat.getNumRows).map(idx => (labels(idx), mat.get(idx, 1).toDouble)).iterator
        case (mat: BlasDoubleMatrix , _: L2Loss) =>
          // For L2Loss, the output is (value, _, _)
          (0 until mat.getNumRows).map(idx => (labels(idx), mat.get(idx, 0))).iterator
        case (mat: BlasFloatMatrix , _: L2Loss) =>
          // For L2Loss, the output is (value, _, _)
          (0 until mat.getNumRows).map(idx => (labels(idx), mat.get(idx, 0).toDouble)).iterator
      }
    }
    scores
  }

  def train(input: String,
            modelOutput: String,
            modelInput: String,
            dim: Int,
            model: GraphModel): Unit = {
    val conf = SparkContext.getOrCreate().getConf
    val data = SparkContext.getOrCreate().textFile(input)
      .repartition(SparkUtils.getNumCores(conf))
      .map(f => DataLoader.parseIntFloat(f, dim))

    model.init(data.getNumPartitions)

    if (modelInput.length > 0) model.load(modelInput)
    train(data, model)
    if (modelOutput.length > 0) model.save(modelOutput)

  }

  def predict(input: String,
              output: String,
              modelInput: String,
              dim: Int,
              model: GraphModel): Unit = {
    val dataWithLabels = SparkContext.getOrCreate().textFile(input)
      .map(f => (DataLoader.parseIntFloat(f, dim), DataLoader.parseLabel(f)))

    model.init(dataWithLabels.getNumPartitions)
    model.load(modelInput)

    val predicts = predict(dataWithLabels, model)
    if (output.length > 0)
      predicts.map(f => s"${f._1} ${f._2}").saveAsTextFile(output)
  }

}

object AutoSyncLearner {

  /**
    * Build manifold view for a RDD. A manifold RDD is to split a RDD to multiple RDD.
    * First, we shuffle the RDD and split it into several splits inside every partition.
    * Then, we hold the manifold RDD into cache.
    * @param data, RDD to be split
    * @param numSplit, the number of splits
    * @return
    */
  def buildManifold[T: ClassTag](data: RDD[T], numSplit: Int): RDD[Array[T]] = {
    def shuffleAndSplit(iterator: Iterator[T]): Iterator[Array[T]] = {
      val samples = Random.shuffle(iterator).toArray
      val sizes = Array.tabulate(numSplit)(_ => samples.length / numSplit)
      val left = samples.length % numSplit
      //for (i <- 0 until left) sizes(i) += 1

      var idx = 0
      val manifold = new Array[Array[T]](numSplit)
      for (a <- 0 until numSplit) {
        manifold(a) = new Array[T](sizes(a))
        for (b <- 0 until sizes(a)) {
          manifold(a)(b) = samples(idx)
          idx += 1
        }
      }
      manifold.iterator
    }

    val manifold = data.mapPartitions(it => shuffleAndSplit(it))
    manifold.cache()
    manifold.count()
    manifold
  }

  /**
    * Return an iterator for the manifold RDD. Each element returned by the iterator is a RDD
    * which contains a split for the manifold RDD.
    * @param manifold, RDD to be split
    * @param numSplit, number of splits to split the manifold RDD
    * @return
    */
  def buildManifoldIterator[T: ClassTag](manifold: RDD[Array[T]], numSplit: Double): Iterator[RDD[Array[T]]] = {

    def skip[T](partitionId: Int, iterator: Iterator[Array[T]], skipNum: Int): Iterator[Array[T]] = {
      (0 until skipNum).foreach(_ => iterator.next())
      Iterator.single(iterator.next())
    }

    new Iterator[RDD[Array[T]]] with Serializable {
      var index = 0

      override def hasNext(): Boolean = index < numSplit

      override def next(): RDD[Array[T]] = {
        val batch = manifold.mapPartitionsWithIndex((partitionId, it) => skip(partitionId, it, index - 1), true)
        index += 1
        batch
      }
    }
  }
}

