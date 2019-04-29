package com.tencent.angel.spark.automl.sync.model

import breeze.linalg.{DenseVector => BDV}

trait PowerLaw {

  val numParams = 1
  val params: BDV[Double] = new BDV[Double](numParams)

  def setParams(newParams: BDV[Double]): this.type = {
    require(params.length == newParams.length, s"the size of new params should be equal to ${params.length}.")
    (0 until params.length).foreach { idx =>
      params(idx) = newParams(idx)
    }
    this
  }

  def predictBatch(X: Array[Double]): Array[Double]

  def predict(x: Double): Double

  def gradBatch(X: Array[Double]): BDV[Double]

  def grad(x: Double): BDV[Double]
}
