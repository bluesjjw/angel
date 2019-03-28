package com.tencent.angel.spark.ml.core

import com.tencent.angel.ml.core.conf.{MLConf, SharedConf}
import com.tencent.angel.ml.core.network.layers.{AngelGraph, PlaceHolder, STATUS}
import com.tencent.angel.ml.core.optimizer.decayer.{StandardDecay, StepSizeScheduler}
import com.tencent.angel.ml.core.optimizer.loss.{L2Loss, LogLoss, LossFunc}
import com.tencent.angel.ml.core.utils.paramsutils.JsonUtils
import com.tencent.angel.ml.feature.LabeledData
import com.tencent.angel.ml.math2.matrix.{BlasDoubleMatrix, BlasFloatMatrix, Matrix}
import com.tencent.angel.model.{ModelLoadContext, ModelSaveContext}
import com.tencent.angel.spark.automl.tuner.config.Configuration
import com.tencent.angel.spark.automl.tuner.parameter.ParamSpace
import com.tencent.angel.spark.automl.tuner.solver.Solver
import com.tencent.angel.spark.automl.utils.AutoMLException
import com.tencent.angel.spark.context.{AngelPSContext, PSContext}
import com.tencent.angel.spark.ml.core.metric.{AUC, Precision}
import com.tencent.angel.spark.ml.util.{DataLoader, SparkUtils}
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.json4s.JsonAST.JValue

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.Random

class AutoSyncLearner (tuneIter: Int = 20, minimize: Boolean = true) {

  // Shared configuration with Angel-PS
  val conf = SharedConf.get()

  // Some params
  var numEpoch: Int = conf.getInt(MLConf.ML_EPOCH_NUM)
  var fraction: Double = conf.getDouble(MLConf.ML_BATCH_SAMPLE_RATIO)
  var validationRatio: Double = conf.getDouble(MLConf.ML_VALIDATE_RATIO)

  println(s"fraction=$fraction validateRatio=$validationRatio numEpoch=$numEpoch")

  val solver: Solver = Solver(minimize)

  // param name -> param type (continuous or discrete), value type (int, double,...)
  val paramType: mutable.Map[String, (String, String)] = new mutable.HashMap[String, (String, String)]()

  def addParam(param: ParamSpace[AnyVal]): this.type = {
    solver.addParam(param)
    this
  }

  def addParam(pType: String, vType: String, name: String, config: String): this.type = {
    paramType += name -> (pType.toLowerCase, vType.toLowerCase)
    solver.addParam(pType, vType, name, config)
    this
  }

  def setParam(name: String, vType: String, value: Double): Unit = {
    println(s"set param[$name] type[$vType] value[$value]")
    vType match {
      case "int" => conf.setInt(name, value.toInt)
      case "long" => conf.setLong(name, value.toLong)
      case "float" => conf.setFloat(name, value.toFloat)
      case "double" => conf.setDouble(name, value)
      case _ => throw new AutoMLException(s"unsupported value type $vType")
    }
  }

  def resetParam(paramMap: mutable.Map[String, Double]): this.type = {
    paramMap.foreach(println)
    numEpoch = paramMap.getOrElse(MLConf.ML_EPOCH_NUM, numEpoch.toDouble).toInt
    fraction = paramMap.getOrElse(MLConf.ML_BATCH_SAMPLE_RATIO, fraction)
    validationRatio = paramMap.getOrElse(MLConf.ML_VALIDATE_RATIO, validationRatio)
    println(s"fraction=$fraction validateRatio=$validationRatio numEpoch=$numEpoch")
    this
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
          println(s"partition $partId")
          PSContext.instance()
          val indices = Random.shuffle(List.range(0, bcNumSplits.value)).take(bcBarrier.value)
          println(s"batch indices: ${indices.mkString(",")}")
          var counter = 0
          var retBatchSize: Int = Int.MaxValue
          var retLoss: Double = 0
          iter.zipWithIndex.foreach { case (batch, idx) =>
            if (indices.contains(idx)) {
              println(s"use batch $idx")
              model.feedData(batch)
              println(s"counter $counter")
              if (counter == 0)
                model.pullParams(epoch)
              model.predict()
              val loss = model.getLoss()

              model.backward()

              retBatchSize = retBatchSize min batch.length
              retLoss += loss
              counter += 1
            }
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
        if (true) {
          //println(s"calculate metrics")
          var validateMetricLog = ""
          if (validationRatio > 0.0) {
            val metrics = evaluate(validate, model)
            if (isAIMD) {
              if (metrics._1 > validateAuc.last)
                barrier += 1
              else if (barrier > 1)
                barrier /= 2
            }
            validateAuc += metrics._1
            validatePrecision += metrics._2
            validateMetricLog = f"validateAuc=${validateAuc.last}%.5f validatePrecision=${validatePrecision.last}%.5f"
          }
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

