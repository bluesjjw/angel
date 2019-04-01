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

package com.tencent.angel.spark.automl

import com.tencent.angel.spark.automl.tuner.config.Configuration
import com.tencent.angel.spark.automl.tuner.parameter.{DiscreteSpace, ParamSpace}
import com.tencent.angel.spark.automl.tuner.solver.Solver
import com.tencent.angel.spark.automl.tuner.trail.{TestTrail, Trail}
import org.apache.spark.ml.linalg.Vector
import org.junit.Test

class TunerTest {

  @Test def testRandom(): Unit = {
    val param1 = ParamSpace.fromConfigString("param3", "{2.0,3.0,4.0,5.0,6.0}")
    val param2 = ParamSpace.fromConfigString("param4", "{3:10:1}")
    val solver: Solver = Solver(Array(param1, param2), true, surrogate = "Random")
    val trail: Trail = new TestTrail()
    (0 until 100).foreach { iter =>
      println(s"------iteration $iter starts------")
      val configs: Array[Configuration] = solver.suggest()
      val results: Array[Double] = trail.evaluate(configs)
      solver.feed(configs, results)
    }
    val result: (Vector, Double) = solver.optimal
    solver.stop
    println(s"Best configuration ${result._1.toArray.mkString(",")}, best performance: ${result._2}")
  }

  @Test def testGrid(): Unit = {
    val param1 = ParamSpace.fromConfigString("param1", "[0.1,10]")
    val param2 = ParamSpace.fromConfigString("param2", "[-5:5:10]")
//    val param3 = ParamSpace.fromConfigString("param3", "{0.0,1.0,3.0,5.0}")
//    val param4 = ParamSpace.fromConfigString("param4", "{-5.0,-3.0,0.0,3.0,5.0}")
//    val solver: Solver = Solver(Array(param1, param2, param3, param4), true, surrogate = "Grid")
    val solver: Solver = Solver(Array(param1, param2), true, surrogate = "Grid")
    val trail: Trail = new TestTrail()
    (0 until 3000).foreach { iter =>
      println(s"------iteration $iter starts------")
      val configs: Array[Configuration] = solver.suggest()
      val results: Array[Double] = trail.evaluate(configs)
      solver.feed(configs, results)
    }
    val result: (Vector, Double) = solver.optimal
    solver.stop
    println(s"Best configuration ${result._1.toArray.mkString(",")}, best performance: ${result._2}")
  }

  @Test def testGaussianProcess(): Unit = {
    val param1 = ParamSpace.fromConfigString("param1", "[1,10]")
    val param2 = ParamSpace.fromConfigString("param2", "[-5:5:10]")
    val param3 = ParamSpace.fromConfigString("param3", "{0.0,1.0,3.0,5.0}")
    val param4 = ParamSpace.fromConfigString("param4", "{-5:5:1}")
    val solver: Solver = Solver(Array(param1, param2, param3, param4), true, surrogate = "GaussianProcess")
    val trail: Trail = new TestTrail()
    (0 until 100).foreach { iter =>
      println(s"------iteration $iter starts------")
      val configs: Array[Configuration] = solver.suggest
      val results: Array[Double] = trail.evaluate(configs)
      solver.feed(configs, results)
    }
    val result: (Vector, Double) = solver.optimal
    solver.stop
    println(s"Best configuration ${result._1.toArray.mkString(",")}, best performance: ${result._2}")
  }

  @Test def testRandomForest(): Unit = {
    val param1 = ParamSpace.fromConfigString("param1", "[1,10]")
    val param2 = ParamSpace.fromConfigString("param2", "[-5:5:10]")
    val param3 = ParamSpace.fromConfigString("param3", "{0.0,1.0,3.0,5.0}")
    val param4 = ParamSpace.fromConfigString("param4", "{-5:5:1}")
    val solver: Solver = Solver(Array(param1, param2, param3, param4), true, "RandomForest")
    val trail: Trail = new TestTrail()
    (0 until 25).foreach { iter =>
      println(s"------iteration $iter starts------")
      val configs: Array[Configuration] = solver.suggest
      val results: Array[Double] = trail.evaluate(configs)
      solver.feed(configs, results)
    }
    val result: (Vector, Double) = solver.optimal
    solver.stop
    println(s"Best configuration ${result._1.toArray.mkString(",")}, best performance: ${result._2}")
  }

}
