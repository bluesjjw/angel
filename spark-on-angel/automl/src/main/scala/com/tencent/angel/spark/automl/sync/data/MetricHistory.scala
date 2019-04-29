package com.tencent.angel.spark.automl.sync.data

import scala.collection.mutable.ArrayBuffer

class MetricHistory {

  var batches: ArrayBuffer[Double] = new ArrayBuffer[Double]()
  var metrics: ArrayBuffer[Double] = new ArrayBuffer[Double]()

  def getBatches(): Array[Double] = batches.toArray

  def getMetrics(): Array[Double] = metrics.toArray

  def getHistory(): Array[(Double, Double)] = batches.zip(metrics).toArray

  def addHistory(batches: Array[Double], metrics: Array[Double]): Unit = {
    batches.zip(metrics).foreach( pair => addHistory(pair._1, pair._2))
  }

  def addHistory(batch: Double, metric: Double): Unit = {
    batches += batch
    metrics += metric
  }
}
