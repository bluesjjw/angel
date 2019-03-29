package com.tencent.angel.spark.examples.cluster

import com.tencent.angel.RunningMode
import com.tencent.angel.conf.AngelConf
import com.tencent.angel.ml.core.conf.{MLConf, SharedConf}
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.{ArgsUtil, AutoSyncLearner, GraphModel, OfflineLearner}
import org.apache.spark.{SparkConf, SparkContext}

object AutoOfflineRunner {

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)

    val input = params.getOrElse("input", "")
    val output = params.getOrElse("output", "")
    val actionType = params.getOrElse("actionType", "train")
    val network = params.getOrElse("network", "LogisticRegression")
    val modelPath = params.getOrElse("model", "")

    // set running mode, use angel_ps mode for spark
    SharedConf.get().set(AngelConf.ANGEL_RUNNING_MODE, RunningMode.ANGEL_PS_WORKER.toString)

    // build SharedConf with params
    SharedConf.addMap(params)

    val dim = SharedConf.indexRange.toInt

    println(s"dim=$dim")

    // load data
    val conf = new SparkConf()

    // we set the load model path for angel-ps to load the meta information of model
    if (modelPath.length > 0)
      conf.set(AngelConf.ANGEL_LOAD_MODEL_PATH, modelPath)

    val sc = new SparkContext(conf)

    // start PS
    PSContext.getOrCreate(sc)

    val className = "com.tencent.angel.spark.ml.classification." + network
    val model = GraphModel(className)
    val learner = new AutoSyncLearner(5,false)

    actionType match {
      case MLConf.ANGEL_ML_TRAIN => learner.train(input, output, modelPath, dim, model)
      case MLConf.ANGEL_ML_PREDICT => learner.predict(input, output, modelPath, dim, model)
    }
  }
}
