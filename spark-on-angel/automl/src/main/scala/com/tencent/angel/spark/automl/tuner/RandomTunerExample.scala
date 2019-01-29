/*
 * Tencent is pleased to support the open source community by making Angel available.
 *
 * Copyright (C) 2017-2018 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */


package com.tencent.angel.spark.automl.tuner

import com.tencent.angel.spark.automl.tuner.config.{Configuration, ConfigurationSpace}
import com.tencent.angel.spark.automl.tuner.parameter.{ContinuousSpace, DiscreteSpace, ParamSpace}
import com.tencent.angel.spark.automl.tuner.surrogate.{NormalSurrogate, Surrogate}
import com.tencent.angel.spark.automl.tuner.trail.{TestTrail, Trail}

object RandomTunerExample extends App {

  override def main(args: Array[String]): Unit = {
    val param1: ParamSpace[Double] = new ContinuousSpace("param1", 0, 10, 11)
    val param2: ParamSpace[Double] = new ContinuousSpace("param2", -5, 5, 11)
    val param3: ParamSpace[Double] = new DiscreteSpace[Double]("param3", Array(0.0, 1.0, 3.0, 5.0))
    val param4: ParamSpace[Double] = new DiscreteSpace[Double]("param4", Array(-5.0, -3.0, 0.0, 3.0, 5.0))
    val cs: ConfigurationSpace = new ConfigurationSpace("cs")
    cs.addParam(param1)
    cs.addParam(param2)
    cs.addParam(param3)
    cs.addParam(param4)
    val sur: NormalSurrogate = new NormalSurrogate(cs, true)
    val trail: Trail = new TestTrail()
    (0 until 100).foreach{ iter =>
      println(s"------iteration $iter starts------")
      val configs: Array[Configuration] = cs.sample(TunerParam.sampleSize)
      val results: Array[Double] = trail.evaluate(configs)
      sur.update(configs.map(_.getVector), results.map(-_))
      val result = sur.curBest
      println()
      println(s"Best configuration ${result._1.toArray.mkString(",")}, best performance: ${result._2}")
    }
  }
}
