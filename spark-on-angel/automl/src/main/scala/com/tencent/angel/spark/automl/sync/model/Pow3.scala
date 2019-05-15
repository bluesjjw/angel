package com.tencent.angel.spark.automl.sync.model

import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.LBFGS
import com.tencent.angel.spark.automl.sync.data.MetricHistory

/**
  * y = a - b * power(x, -c)
  * params: a, b, c
  */
class Pow3 extends PowerLaw {

  override val numParams: Int = 3
  override val params: BDV[Double] = BDV(1.0, 1.0, 2.0)

  override def predict(x: Double): Double = {
    params(0) - params(1) * math.pow(x, -params(2))
  }

  override def predictBatch(X: Array[Double]): Array[Double] = {
    X.map(predict)
  }

  override def grad(x: Double): BDV[Double] = {
    val right = -params(1) * math.pow(x, -params(2))
    BDV(Array(1, right / params(1), -right * math.log(params(2))))
  }

  override def gradBatch(X: Array[Double]): BDV[Double] = {
    val grads: Array[BDV[Double]] = X.map(grad)
    (1.0 / grads.length) * grads.reduce((a: BDV[Double], b: BDV[Double]) => a + b)
  }
}

object Pow3 {

  def main(args: Array[String]): Unit = {
    //val batches = Array(1.0, 2.0, 3.0)
    //val metrics = Array(2.0, 3.0, 10.0/3.0)
    val batches = Array(1.0, 2.0, 3.0, 4.0, 5.0)
    val metrics = Array(0.5, 0.8, 0.9, 0.95, 0.97)
    val history = new MetricHistory()
    history.addHistory(batches, metrics)

    val initParams = BDV(1.0, 1.0, 2.0)
    println(s"initial params: $initParams")
    val model = new Pow3().setParams(initParams)

    val kernelDiffFunc = new PowerLawDiffFunc(history, model)

    val optimizer = new LBFGS[BDV[Double]](maxIter = 10, m = 7, tolerance = 1e-10)
    val newParams = optimizer.minimize(kernelDiffFunc, initParams)

    println(s"new params: $newParams")
    println(s"labels: ${metrics.mkString(",")}")
    println(s"predictions: ${model.predictBatch(batches).mkString(",")}")
    println(s"predictions: ${model.predictBatch(Array(6.0)).mkString(",")}")
  }
}