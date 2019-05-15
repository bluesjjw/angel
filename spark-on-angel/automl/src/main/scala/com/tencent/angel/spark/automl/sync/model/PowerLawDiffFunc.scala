package com.tencent.angel.spark.automl.sync.model

import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.{DiffFunction, LBFGS}
import com.tencent.angel.spark.automl.sync.data.MetricHistory

class PowerLawDiffFunc(history: MetricHistory, model: PowerLaw) extends DiffFunction[BDV[Double]] {

  var iter: Int = _

  override def calculate(params: BDV[Double]): (Double, BDV[Double]) = {

    model.setParams(params)
    val preds: Array[Double] = model.predictBatch(history.getBatches())
    //println(s"labels in diff func: ${history.getMetrics().mkString(",")}")
    //println(s"preds in diff func: ${preds.mkString(",")}")
    val loss = MSError.loss(history.getMetrics(), preds)

    val grad: BDV[Double] = MSError.grad(history.getMetrics(), history.getBatches(),
      model.predict _, model.grad _)

    iter += 1
    //println(s"loss: $loss, grad: ${grad}")
    (loss, grad)
  }
}

object PowerLawDiffFunc {

  def main(args: Array[String]): Unit = {
    val batches = Array(1.0, 2.0, 3.0)
    val metrics = Array(2.0, 3.0, 10.0/3.0)
    val history = new MetricHistory()
    history.addHistory(batches, metrics)

    val initParams = BDV(1.0, 1.0, 1.0)
    println(s"initial params: $initParams")
    val model = new Pow3().setParams(initParams)

    val kernelDiffFunc = new PowerLawDiffFunc(history, model)

    val optimizer = new LBFGS[BDV[Double]](maxIter = 10, m = 7, tolerance = 1e-10)
    val newParams = optimizer.minimize(kernelDiffFunc, initParams)

    println(s"new params: $newParams")
    println(s"labels: ${metrics.mkString(",")}")
    println(s"predictions: ${model.predictBatch(batches).mkString(",")}")
  }
}
