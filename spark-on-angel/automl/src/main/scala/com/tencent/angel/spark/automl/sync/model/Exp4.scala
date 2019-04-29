package com.tencent.angel.spark.automl.sync.model

import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.LBFGS
import com.tencent.angel.spark.automl.sync.data.MetricHistory

/**
  * y = a - exp(-b * power(x, c) + d)
  */
class Exp4 extends PowerLaw {

  override val numParams: Int = 4
  override val params: BDV[Double] = BDV(1.0, 1.0, 1.0, 1.0)

  override def predict(x: Double): Double = {
    params(0) - math.exp(-params(1) * math.pow(x, params(2)) + params(3))
  }

  override def predictBatch(X: Array[Double]): Array[Double] = {
    X.map(predict)
  }

  override def grad(x: Double): BDV[Double] = {
    val tmp = -math.exp(-params(1) * math.pow(x, params(2)) + params(3))
    BDV(Array(1,
      tmp * (-1.0) * math.pow(x, params(2)),
      tmp * (-1.0) * params(1) * math.pow(x, params(2)) * math.log(params(2)),
      tmp
    ))
  }

  override def gradBatch(X: Array[Double]): BDV[Double] = {
    val grads: Array[BDV[Double]] = X.map(grad)
    (1.0 / grads.length) * grads.reduce((a: BDV[Double], b: BDV[Double]) => a + b)
  }
}

object Exp4 {

  def main(args: Array[String]): Unit = {
    val batches = Array(1.0, 2.0, 3.0)
    val metrics = Array(2.0, 3.0, 10.0/3.0)
    val history = new MetricHistory()
    history.addHistory(batches, metrics)

    val initParams = BDV(1.0, 1.0, 1.0, 1.0)
    println(s"initial params: $initParams")
    val model = new Exp4().setParams(initParams)

    val kernelDiffFunc = new PowerLawDiffFunc(history, model)

    val optimizer = new LBFGS[BDV[Double]](maxIter = 20, m = 7, tolerance = 1e-10)
    val newParams = optimizer.minimize(kernelDiffFunc, initParams)

    println(s"new params: $newParams")
    println(s"labels: ${metrics.mkString(",")}")
    println(s"predictions: ${model.predictBatch(batches).mkString(",")}")
  }
}
