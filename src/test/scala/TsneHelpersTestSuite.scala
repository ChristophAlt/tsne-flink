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

import de.tu_berlin.dima.impro3.TsneHelpers._


class TsneHelpersTestSuite extends FlatSpec with Matchers with Inspectors {
  "kNearestNeighbors" should "return the k nearest neighbors for each SparseVector" in  {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val neighbors = 2
    val metric = SquaredEuclideanDistanceMetric()

    val input = env.fromCollection(TsneHelpersTestSuite.input)

    val results = kNearestNeighbors(input, neighbors, metric).collect()
    val expectedResults = TsneHelpersTestSuite.knnResults

    results.size should equal (expectedResults.size)
    forAll(results) {expectedResults should contain (_)}
  }

  "pairwiseAffinities" should "return the pairwise similarity p_i|j between datapoints" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val neighbors = 10
    val perplexity = 2.0
    val metric = SquaredEuclideanDistanceMetric()

    val input = env.fromCollection(TsneHelpersTestSuite.input)

    val knn = kNearestNeighbors(input, neighbors, metric)
    val results = pairwiseAffinities(knn, perplexity).collect()
    val expectedResults = TsneHelpersTestSuite.pairwiseAffinitiesResults

    results.size should equal (expectedResults.size)
    for (expected <- expectedResults) {
      val result = results.find(x => x._1 == expected._1 && x._2 == expected._2)
      result match {
        case Some(result) => result._3 should equal(expected._3 +- 1e-5)
        case None => fail("expected result not found")
      }
    }
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

  val knnResults: Seq[(Long, Long, Double)] = List(
    (0L, 1L, 4.0), (0L, 2L, 16.0), (1L, 2L, 4.0), (1L, 0L, 4.0), (2L, 3L, 4.0), (2L, 1L, 4.0),
    (3L, 4L, 4.0), (3L, 2L, 4.0), (4L, 5L, 4.0), (4L, 3L, 4.0), (5L, 6L, 4.0), (5L, 4L, 4.0),
    (6L, 7L, 4.0), (6L, 5L, 4.0), (7L, 8L, 4.0), (7L, 6L, 4.0), (8L, 7L, 4.0), (8L, 6L, 16.0)
  ).toSeq

  val pairwiseAffinitiesResults: Seq[(Long, Long, Double)] = List(
    (0L, 1L, 0.733896425817), (0L, 2L, 0.230388739954), (0L, 3L, 0.0334070949154), (0L, 4L, 0.00223752119073), (0L, 5L, 6.92223896587e-05), (0L, 6L, 9.89183817531e-07), (0L, 7L, 6.52917640895e-09), (0L, 8L, 1.99063111076e-11), (1L, 0L, 0.499999999991), (1L, 2L, 0.499999999991), (1L, 3L, 1.8875672721e-11), (1L, 4L, 8.01905445259e-29), (1L, 5L, 3.83382403679e-53), (1L, 6L, 2.06266870227e-84), (1L, 7L, 1.24886378343e-122), (1L, 8L, 8.5092043508e-168), (2L, 0L, 1.88756727207e-11), (2L, 1L, 0.499999999981), (2L, 3L, 0.499999999981), (2L, 4L, 1.88756727207e-11), (2L, 5L, 8.01905445244e-29), (2L, 6L, 3.83382403672e-53), (2L, 7L, 2.06266870223e-84), (2L, 8L, 1.24886378341e-122), (3L, 0L, 8.01905445244e-29), (3L, 1L, 1.88756727207e-11), (3L, 2L, 0.499999999981), (3L, 4L, 0.499999999981), (3L, 5L, 1.88756727207e-11), (3L, 6L, 8.01905445244e-29), (3L, 7L, 3.83382403672e-53), (3L, 8L, 2.06266870223e-84), (4L, 0L, 3.83382403672e-53), (4L, 1L, 8.01905445244e-29), (4L, 2L, 1.88756727207e-11), (4L, 3L, 0.499999999981), (4L, 5L, 0.499999999981), (4L, 6L, 1.88756727207e-11), (4L, 7L, 8.01905445244e-29), (4L, 8L, 3.83382403672e-53), (5L, 0L, 2.06266870223e-84), (5L, 1L, 3.83382403672e-53), (5L, 2L, 8.01905445244e-29), (5L, 3L, 1.88756727207e-11), (5L, 4L, 0.499999999981), (5L, 6L, 0.499999999981), (5L, 7L, 1.88756727207e-11), (5L, 8L, 8.01905445244e-29), (6L, 0L, 1.24886378341e-122), (6L, 1L, 2.06266870223e-84), (6L, 2L, 3.83382403672e-53), (6L, 3L, 8.01905445244e-29), (6L, 4L, 1.88756727207e-11), (6L, 5L, 0.499999999981), (6L, 7L, 0.499999999981), (6L, 8L, 1.88756727207e-11), (7L, 0L, 8.5092043508e-168), (7L, 1L, 1.24886378343e-122), (7L, 2L, 2.06266870227e-84), (7L, 3L, 3.83382403679e-53), (7L, 4L, 8.01905445259e-29), (7L, 5L, 1.8875672721e-11), (7L, 6L, 0.499999999991), (7L, 8L, 0.499999999991), (8L, 0L, 1.99063111076e-11), (8L, 1L, 6.52917640895e-09), (8L, 2L, 9.89183817531e-07), (8L, 3L, 6.92223896587e-05), (8L, 4L, 0.00223752119073), (8L, 5L, 0.0334070949154), (8L, 6L, 0.230388739954), (8L, 7L, 0.733896425817)
  ).toSeq

  val jointProbabilitiesResults: Seq[LabeledVector] = List(
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
