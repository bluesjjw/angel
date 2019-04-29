package com.tencent.angel.spark.automl.sync.model

import breeze.linalg.{Vector, DenseVector => BDV}

/**
  * loss = (pred - label)^2
  * grad = 2 * (pred - label) * grad(x)
  **/
object MSError {

  def loss(label: Double, pred: Double): Double = {
    math.pow(pred - label, 2)
  }

  def loss(labels: Array[Double], preds: Array[Double]): Double = {
    labels.zip(preds).map( pair => loss(pair._1, pair._2) ).sum
  }

  def grad(label: Double, pred: Double): Double = {
    2 * (pred - label)
  }

  def grad(label: Double, feature: Double, predFunc: Double => Double, predFuncDiff: Double => Double): Double = {
    grad(label, predFunc(feature)) * predFuncDiff(feature)
  }

  def grad(label: Double, feature: Double, predFunc: Double => Double, predFuncDiff: Double => BDV[Double]): BDV[Double] = {
    grad(label, predFunc(feature)) * predFuncDiff(feature)
  }

  def grad(labels: Array[Double], features: Array[Double],
           predFunc: Double => Double, predFuncDiff: Double => BDV[Double]): BDV[Double] = {
    require(labels.length == features.length, s"size of labels and features should be equal.")
    (1.0 / labels.length) * labels.zip(features).map { ins =>
      grad(ins._1, ins._2, predFunc, predFuncDiff)
    }.reduce((a: BDV[Double], b: BDV[Double]) => a + b)
  }
}
