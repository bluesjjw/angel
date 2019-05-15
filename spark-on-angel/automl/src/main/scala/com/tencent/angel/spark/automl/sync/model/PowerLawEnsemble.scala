package com.tencent.angel.spark.automl.sync.model

import breeze.optimize.{AdaDeltaGradientDescent, LBFGS}
import com.tencent.angel.spark.automl.sync.data.MetricHistory
import breeze.linalg.{DenseVector => BDV}

import scala.collection.mutable.ArrayBuffer

class PowerLawEnsemble {

  val numCurve = 3

  val models: Array[PowerLaw] = new Array[PowerLaw](numCurve)

  models(0) = new Pow3()
  models(1) = new Pow4().setParams(BDV(1.0, 1.0, 1.0, 2.0))
  models(2) = new Exp4()

  val history = new MetricHistory()

  def setData(variables: Array[Double], metrics: Array[Double]): Unit = {
    history.setHistory(variables, metrics)
  }

  def setData(variables: Array[Int], metrics: Array[Double]): Unit = {
    history.setHistory(variables.map(_.toDouble), metrics)
  }

  def train(): Unit = {
    //models.foreach { model =>
    val model = new Pow4()
      println(s"${model.getClass.getSimpleName}")
      println(s"variables: ${history.getBatches().mkString(",")}")
      println(s"metrics: ${history.getMetrics().mkString(",")}")
      val kernelDiffFunc = new PowerLawDiffFunc(history, model)
      val optimizer = new LBFGS[BDV[Double]](maxIter = 10, m = 7)
      println(s"initial params: ${model.params}")
      val newParams = optimizer.minimize(kernelDiffFunc, model.params)
      //model.setParams(newParams)
      println(s"new params: $newParams")
      println(s"labels: ${history.getMetrics().mkString(",")}")
      println(s"predictions: ${model.predictBatch(history.getBatches()).mkString(",")}")
    //}
  }

  def predict(variable: Int): (Double, Double) = {
    predict(variable.toDouble)
  }

  def predict(variable: Double): (Double, Double) = {
    val preds: ArrayBuffer[Double] = new ArrayBuffer[Double]
    models.foreach { model =>
//      val kernelDiffFunc = new PowerLawDiffFunc(history, model)
//      val optimizer = new LBFGS[BDV[Double]](maxIter = 10, m = 7, tolerance = 1e-10)
//      val newParams = optimizer.minimize(kernelDiffFunc, model.params)
      val pred = model.predict(variable)
      preds += pred
      //println(s"new params: $newParams")
      println(s"prediction: $pred")
    }
    val mean = preds.sum / preds.length
    val variance = preds.map(pred => math.sqrt(pred - mean)).sum / preds.length
    (mean, variance)
  }
}

object PowerLawEnsemble {

  def main(args: Array[String]): Unit = {
    val variables: Array[Double] = Array(1.0, 2.0, 3.0, 4.0, 5.0)
    val metrics: Array[Double] = Array(0.5, 0.8, 0.9, 0.95, 0.97)
    val model = new PowerLawEnsemble
    model.setData(variables, metrics)
    model.train()
    val (mean, variance) = model.predict(6)
    println(s"mean $mean, variance $variance")

//    val history = new MetricHistory()
//    history.addHistory(variables, metrics)
//    val initParams = BDV(1.0, 1.0, 1.0, 2.0)
//    println(s"initial params: $initParams")
//    val model = new Pow4()
//
//    val kernelDiffFunc = new PowerLawDiffFunc(history, model)
//
//    val optimizer = new LBFGS[BDV[Double]](maxIter = 10, m = 7)
//    val newParams = optimizer.minimize(kernelDiffFunc, initParams)
//
//    println(s"new params: $newParams")
//    println(s"labels: ${metrics.mkString(",")}")
//    println(s"predictions: ${model.predictBatch(variables).mkString(",")}")

  }
}
