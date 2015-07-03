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

import de.tu_berlin.dima.impro3.TsneHelpers._
import org.apache.flink.api.scala._
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.SparseVector
import org.apache.flink.ml.metrics.distances.SquaredEuclideanDistanceMetric
import org.scalatest._


class TsneHelpersTestSuite extends FlatSpec with Matchers with Inspectors {
  "kNearestNeighbors" should "return the k nearest neighbors for each SparseVector" in  {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val neighbors = 2
    val metric = SquaredEuclideanDistanceMetric()

    val input = env.fromCollection(TsneHelpersTestSuite.inputDense)

    val results = kNearestNeighbors(input, neighbors, metric).collect()
    val expectedResults = TsneHelpersTestSuite.knnResults

    results.size should equal (expectedResults.size)
    forAll(results) {expectedResults should contain (_)}
  }

  "pairwiseAffinities" should "return the pairwise similarity p_i|j between datapoints" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val perplexity = 2.0
    val neighbors = 10
    val metric = SquaredEuclideanDistanceMetric()

    val input = env.fromCollection(TsneHelpersTestSuite.inputDense)

    val knn = kNearestNeighbors(input, neighbors, metric)
    val results = pairwiseAffinities(knn, perplexity).collect()
    val expectedResults = TsneHelpersTestSuite.pairwiseAffinitiesResults

    results.size should equal (expectedResults.size)
    for (expected <- expectedResults) {
      val result = results.find(x => x._1 == expected._1 && x._2 == expected._2)
      result match {
        case Some(result) => result._3 should equal (expected._3 +- 1e-5)
        case None => fail("expected result not found")
      }
    }
  }

  "jointProbabilities" should "return the symmetrized probability distribution p_ij over the datapoints" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val pairwiseAffinities = env.fromCollection(TsneHelpersTestSuite.pairwiseAffinitiesResults)
    val results = jointDistribution(pairwiseAffinities).collect()
    val expectedResults = TsneHelpersTestSuite.jointProbabilitiesResults

    results.size should equal (expectedResults.size)
    for (expected <- expectedResults) {
      val result = results.find(x => x._1 == expected._1 && x._2 == expected._2)
      result match {
        case Some(result) => result._3 should equal (expected._3 +- 1e-12)
        case _ => fail("expected result not found")
      }
    }
    results.map(_._3).sum should equal (1.0 +- 1e-12)
  }

  //TODO: add testcase for jointProbabilities with sparse input
}

object TsneHelpersTestSuite {
  val inputDense: Seq[LabeledVector] = List(
    LabeledVector(0.0, SparseVector.fromCOO(4, List((0, 0.0),(1, 0.0),(2, 0.0),(3, 0.0)))),
    LabeledVector(1.0, SparseVector.fromCOO(4, List((0, 1.0),(1, 1.0),(2, 1.0),(3, 1.0)))),
    LabeledVector(2.0, SparseVector.fromCOO(4, List((0, 2.0),(1, 2.0),(2, 2.0),(3, 2.0)))),
    LabeledVector(3.0, SparseVector.fromCOO(4, List((0, 3.0),(1, 3.0),(2, 3.0),(3, 3.0)))),
    LabeledVector(4.0, SparseVector.fromCOO(4, List((0, 4.0),(1, 4.0),(2, 4.0),(3, 4.0)))),
    LabeledVector(5.0, SparseVector.fromCOO(4, List((0, 5.0),(1, 5.0),(2, 5.0),(3, 5.0)))),
    LabeledVector(6.0, SparseVector.fromCOO(4, List((0, 6.0),(1, 6.0),(2, 6.0),(3, 6.0)))),
    LabeledVector(7.0, SparseVector.fromCOO(4, List((0, 7.0),(1, 7.0),(2, 7.0),(3, 7.0)))),
    LabeledVector(8.0, SparseVector.fromCOO(4, List((0, 8.0),(1, 8.0),(2, 8.0),(3, 8.0))))
  ).toSeq

  // result for k = 2
  val knnResults: Seq[(Long, Long, Double)] = List(
    (0L, 1L, 4.0), (0L, 2L, 16.0), (1L, 2L, 4.0), (1L, 0L, 4.0), (2L, 3L, 4.0), (2L, 1L, 4.0),
    (3L, 4L, 4.0), (3L, 2L, 4.0), (4L, 5L, 4.0), (4L, 3L, 4.0), (5L, 6L, 4.0), (5L, 4L, 4.0),
    (6L, 7L, 4.0), (6L, 5L, 4.0), (7L, 8L, 4.0), (7L, 6L, 4.0), (8L, 7L, 4.0), (8L, 6L, 16.0)
  ).toSeq

  val pairwiseAffinitiesResults: Seq[(Long, Long, Double)] = List(
    (0L, 1L, 0.733896425817), (0L, 2L, 0.230388739954), (0L, 3L, 0.0334070949154),
    (0L, 4L, 0.00223752119073), (0L, 5L, 6.92223896587e-05), (0L, 6L, 9.89183817531e-07),
    (0L, 7L, 6.52917640895e-09), (0L, 8L, 1.99063111076e-11), (1L, 0L, 0.499999999991),
    (1L, 2L, 0.499999999991), (1L, 3L, 1.8875672721e-11), (1L, 4L, 8.01905445259e-29),
    (1L, 5L, 3.83382403679e-53), (1L, 6L, 2.06266870227e-84), (1L, 7L, 1.24886378343e-122),
    (1L, 8L, 8.5092043508e-168), (2L, 0L, 1.88756727207e-11), (2L, 1L, 0.499999999981),
    (2L, 3L, 0.499999999981), (2L, 4L, 1.88756727207e-11), (2L, 5L, 8.01905445244e-29),
    (2L, 6L, 3.83382403672e-53), (2L, 7L, 2.06266870223e-84), (2L, 8L, 1.24886378341e-122),
    (3L, 0L, 8.01905445244e-29), (3L, 1L, 1.88756727207e-11), (3L, 2L, 0.499999999981),
    (3L, 4L, 0.499999999981), (3L, 5L, 1.88756727207e-11), (3L, 6L, 8.01905445244e-29),
    (3L, 7L, 3.83382403672e-53), (3L, 8L, 2.06266870223e-84), (4L, 0L, 3.83382403672e-53),
    (4L, 1L, 8.01905445244e-29), (4L, 2L, 1.88756727207e-11), (4L, 3L, 0.499999999981),
    (4L, 5L, 0.499999999981), (4L, 6L, 1.88756727207e-11), (4L, 7L, 8.01905445244e-29),
    (4L, 8L, 3.83382403672e-53), (5L, 0L, 2.06266870223e-84), (5L, 1L, 3.83382403672e-53),
    (5L, 2L, 8.01905445244e-29), (5L, 3L, 1.88756727207e-11), (5L, 4L, 0.499999999981),
    (5L, 6L, 0.499999999981), (5L, 7L, 1.88756727207e-11), (5L, 8L, 8.01905445244e-29),
    (6L, 0L, 1.24886378341e-122), (6L, 1L, 2.06266870223e-84), (6L, 2L, 3.83382403672e-53),
    (6L, 3L, 8.01905445244e-29), (6L, 4L, 1.88756727207e-11), (6L, 5L, 0.499999999981),
    (6L, 7L, 0.499999999981), (6L, 8L, 1.88756727207e-11), (7L, 0L, 8.5092043508e-168),
    (7L, 1L, 1.24886378343e-122), (7L, 2L, 2.06266870227e-84), (7L, 3L, 3.83382403679e-53),
    (7L, 4L, 8.01905445259e-29), (7L, 5L, 1.8875672721e-11), (7L, 6L, 0.499999999991),
    (7L, 8L, 0.499999999991), (8L, 0L, 1.99063111076e-11), (8L, 1L, 6.52917640895e-09),
    (8L, 2L, 9.89183817531e-07), (8L, 3L, 6.92223896587e-05), (8L, 4L, 0.00223752119073),
    (8L, 5L, 0.0334070949154), (8L, 6L, 0.230388739954), (8L, 7L, 0.733896425817)
  ).toSeq

  val jointProbabilitiesResults: Seq[(Long, Long, Double)] = List(
    (0L, 1L, 0.0685498014338), (0L, 2L, 0.0127993744429), (0L, 3L, 0.00185594971752),
    (0L, 4L, 0.000124306732818), (0L, 5L, 3.84568831437e-06), (0L, 6L, 5.49546565295e-08),
    (0L, 7L, 3.6273202272e-10), (0L, 8L, 2.21181234528e-12), (1L, 0L, 0.0685498014338),
    (1L, 2L, 0.055555555554), (1L, 3L, 2.09729696898e-12), (1L, 4L, 8.9100605028e-30),
    (1L, 5L, 4.25980448528e-54), (1L, 6L, 2.29185411361e-85), (1L, 7L, 1.38762642604e-123),
    (1L, 8L, 3.6273202272e-10), (2L, 0L, 0.0127993744429), (2L, 1L, 0.055555555554),
    (2L, 3L, 0.0555555555535), (2L, 4L, 2.09729696896e-12), (2L, 5L, 8.91006050271e-30),
    (2L, 6L, 4.25980448524e-54), (2L, 7L, 2.29185411361e-85), (2L, 8L, 5.49546565295e-08),
    (3L, 0L, 0.00185594971752), (3L, 1L, 2.09729696898e-12), (3L, 2L, 0.0555555555535),
    (3L, 4L, 0.0555555555535), (3L, 5L, 2.09729696896e-12), (3L, 6L, 8.91006050271e-30),
    (3L, 7L, 4.25980448528e-54), (3L, 8L, 3.84568831437e-06), (4L, 0L, 0.000124306732818),
    (4L, 1L, 8.9100605028e-30), (4L, 2L, 2.09729696896e-12), (4L, 3L, 0.0555555555535),
    (4L, 5L, 0.0555555555535), (4L, 6L, 2.09729696896e-12), (4L, 7L, 8.9100605028e-30),
    (4L, 8L, 0.000124306732818), (5L, 0L, 3.84568831437e-06), (5L, 1L, 4.25980448528e-54),
    (5L, 2L, 8.91006050271e-30), (5L, 3L, 2.09729696896e-12), (5L, 4L, 0.0555555555535),
    (5L, 6L, 0.0555555555535), (5L, 7L, 2.09729696898e-12), (5L, 8L, 0.00185594971752),
    (6L, 0L, 5.49546565295e-08), (6L, 1L, 2.29185411361e-85), (6L, 2L, 4.25980448524e-54),
    (6L, 3L, 8.91006050271e-30), (6L, 4L, 2.09729696896e-12), (6L, 5L, 0.0555555555535),
    (6L, 7L, 0.055555555554), (6L, 8L, 0.0127993744429), (7L, 0L, 3.6273202272e-10),
    (7L, 1L, 1.38762642604e-123), (7L, 2L, 2.29185411361e-85), (7L, 3L, 4.25980448528e-54),
    (7L, 4L, 8.9100605028e-30), (7L, 5L, 2.09729696898e-12), (7L, 6L, 0.055555555554),
    (7L, 8L, 0.0685498014338), (8L, 0L, 2.21181234528e-12), (8L, 1L, 3.6273202272e-10),
    (8L, 2L, 5.49546565295e-08), (8L, 3L, 3.84568831437e-06), (8L, 4L, 0.000124306732818),
    (8L, 5L, 0.00185594971752), (8L, 6L, 0.0127993744429), (8L, 7L, 0.0685498014338)
  ).toSeq
}
