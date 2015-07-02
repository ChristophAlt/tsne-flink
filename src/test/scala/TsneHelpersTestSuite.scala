/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package de.tu_berlin.dima.impro3

import org.apache.flink.api.scala._
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.DenseVector
import org.apache.flink.ml.metrics.distances.SquaredEuclideanDistanceMetric
import org.scalatest._
import Inspectors._

import de.tu_berlin.dima.impro3.TsneHelpers._


class TsneHelpersTestSuite extends FlatSpec with Matchers {
  "kNearestNeighbors" should "return the k nearest neighbors for each SparseVector" in  {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val input = env.fromCollection(TsneHelpersTestSuite.input)
    val neighbors = 2
    val metric = SquaredEuclideanDistanceMetric()

    val result = kNearestNeighbors(input, neighbors, metric).collect()
    val expectedResult = TsneHelpersTestSuite.knnResult

    result.size should equal (expectedResult.size)
    forAll(result) {expectedResult should contain (_)}
  }
}

object TsneHelpersTestSuite {
  val input: Seq[LabeledVector] = List(
    LabeledVector(0.0, DenseVector(0.0, 0.0, 0.0, 0.0)),
    LabeledVector(1.0, DenseVector(1.0, 1.0, 1.0, 1.0)),
    LabeledVector(2.0, DenseVector(2.0, 2.0, 2.0, 2.0)),
    LabeledVector(3.0, DenseVector(3.0, 3.0, 3.0, 3.0)),
    LabeledVector(4.0, DenseVector(4.0, 4.0, 4.0, 4.0)),
    LabeledVector(5.0, DenseVector(5.0, 5.0, 5.0, 5.0)),
    LabeledVector(6.0, DenseVector(6.0, 6.0, 6.0, 6.0)),
    LabeledVector(7.0, DenseVector(7.0, 7.0, 7.0, 7.0)),
    LabeledVector(8.0, DenseVector(8.0, 8.0, 8.0, 8.0))
  ).toSeq

  val knnResult: Seq[(Long, Long, Double)] = List(
    (0L, 1L, 4.0), (0L, 2L, 16.0), (1L, 2L, 4.0), (1L, 0L, 4.0), (2L, 3L, 4.0), (2L, 1L, 4.0),
    (3L, 4L, 4.0), (3L, 2L, 4.0), (4L, 5L, 4.0), (4L, 3L, 4.0), (5L, 6L, 4.0), (5L, 4L, 4.0),
    (6L, 7L, 4.0), (6L, 5L, 4.0), (7L, 8L, 4.0), (7L, 6L, 4.0), (8L, 7L, 4.0), (8L, 6L, 16.0)
  ).toSeq

  val pairwiseAffinitiesResult: Seq[LabeledVector] = List(
    LabeledVector(0.0, DenseVector(0.0, 0.733896425817, 0.230388739954, 0.0334070949154, 0.00223752119073, 6.92223896587e-05, 9.89183817531e-07, 6.52917640895e-09, 1.99063111076e-11)),
    LabeledVector(1.0, DenseVector(0.499999999991, 0.0, 0.499999999991, 1.8875672721e-11, 8.01905445259e-29, 3.83382403679e-53, 2.06266870227e-84, 1.24886378343e-122, 8.5092043508e-168)),
    LabeledVector(2.0, DenseVector(1.88756727207e-11, 0.499999999981, 0.0, 0.499999999981, 1.88756727207e-11, 8.01905445244e-29, 3.83382403672e-53, 2.06266870223e-84, 1.24886378341e-122)),
    LabeledVector(3.0, DenseVector(8.01905445244e-29, 1.88756727207e-11, 0.499999999981, 0.0, 0.499999999981, 1.88756727207e-11, 8.01905445244e-29, 3.83382403672e-53, 2.06266870223e-84)),
    LabeledVector(4.0, DenseVector(3.83382403672e-53, 8.01905445244e-29, 1.88756727207e-11, 0.499999999981, 0.0, 0.499999999981, 1.88756727207e-11, 8.01905445244e-29, 3.83382403672e-53)),
    LabeledVector(5.0, DenseVector(2.06266870223e-84, 3.83382403672e-53, 8.01905445244e-29, 1.88756727207e-11, 0.499999999981, 0.0, 0.499999999981, 1.88756727207e-11, 8.01905445244e-29)),
    LabeledVector(6.0, DenseVector(1.24886378341e-122, 2.06266870223e-84, 3.83382403672e-53, 8.01905445244e-29, 1.88756727207e-11, 0.499999999981, 0.0, 0.499999999981, 1.88756727207e-11)),
    LabeledVector(7.0, DenseVector(8.5092043508e-168, 1.24886378343e-122, 2.06266870227e-84, 3.83382403679e-53, 8.01905445259e-29, 1.8875672721e-11, 0.499999999991, 0.0, 0.499999999991)),
    LabeledVector(8.0, DenseVector(1.99063111076e-11, 6.52917640895e-09, 9.89183817531e-07, 6.92223896587e-05, 0.00223752119073, 0.0334070949154, 0.230388739954, 0.733896425817, 0.0))
  ).toSeq

  val jointProbabilitiesResult: Seq[LabeledVector] = List(
    LabeledVector(0.0, DenseVector(0.0, 0.0685498014338, 0.0127993744429, 0.00185594971752, 0.000124306732818, 3.84568831437e-06, 5.49546565295e-08, 3.6273202272e-10, 2.21181234528e-12)),
    LabeledVector(1.0, DenseVector(0.0685498014338, 0.0, 0.055555555554, 2.09729696898e-12, 8.9100605028e-30, 4.25980448528e-54, 2.29185411361e-85, 1.38762642604e-123, 3.6273202272e-10)),
    LabeledVector(2.0, DenseVector(0.0127993744429, 0.055555555554, 0.0, 0.0555555555535, 2.09729696896e-12, 8.91006050271e-30, 4.25980448524e-54, 2.29185411361e-85, 5.49546565295e-08)),
    LabeledVector(3.0, DenseVector(0.00185594971752, 2.09729696898e-12, 0.0555555555535, 0.0, 0.0555555555535, 2.09729696896e-12, 8.91006050271e-30, 4.25980448528e-54, 3.84568831437e-06)),
    LabeledVector(4.0, DenseVector(0.000124306732818, 8.9100605028e-30, 2.09729696896e-12, 0.0555555555535, 0.0, 0.0555555555535, 2.09729696896e-12, 8.9100605028e-30, 0.000124306732818)),
    LabeledVector(5.0, DenseVector(3.84568831437e-06, 4.25980448528e-54, 8.91006050271e-30, 2.09729696896e-12, 0.0555555555535, 0.0, 0.0555555555535, 2.09729696898e-12, 0.00185594971752)),
    LabeledVector(6.0, DenseVector(5.49546565295e-08, 2.29185411361e-85, 4.25980448524e-54, 8.91006050271e-30, 2.09729696896e-12, 0.0555555555535, 0.0, 0.055555555554, 0.0127993744429)),
    LabeledVector(7.0, DenseVector(3.6273202272e-10, 1.38762642604e-123, 2.29185411361e-85, 4.25980448528e-54, 8.9100605028e-30, 2.09729696898e-12, 0.055555555554, 0.0, 0.0685498014338)),
    LabeledVector(8.0, DenseVector(2.21181234528e-12, 3.6273202272e-10, 5.49546565295e-08, 3.84568831437e-06, 0.000124306732818, 0.00185594971752, 0.0127993744429, 0.0685498014338, 0.0))
  ).toSeq
}
