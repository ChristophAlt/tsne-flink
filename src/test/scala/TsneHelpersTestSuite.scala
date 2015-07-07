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
import org.apache.flink.ml.math.{DenseVector, SparseVector}
import org.apache.flink.ml.metrics.distances.SquaredEuclideanDistanceMetric
import org.scalatest._
import org.apache.flink.ml._


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

  "jointProbabilities" should "return the sparse symmetrized probability distribution p_ij over the datapoints" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val pairwiseAffinities = env.fromCollection(TsneHelpersTestSuite.pairwiseAffinitiesResultsSparse)
    val results = jointDistribution(pairwiseAffinities).collect()
    val expectedResults = TsneHelpersTestSuite.jointProbabilitiesResultsSparse
    
    results.size should equal (expectedResults.size)
    for (expected <- expectedResults) {
      val result = results.find(x => x._1 == expected._1 && x._2 == expected._2)
      result match {
        case Some(result) => result._3 should equal (expected._3 +- 1e-6)
        case _ => fail("expected result not found")
      }
    }
    results.map(_._3).sum should equal (1.0 +- 1e-12)
  }

  "SquaredEuclideanDistance" should "return the squared euclidean distance for all the datapoints" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val y = env.fromCollection(TsneHelpersTestSuite.yInit)
    val results = computeDistances(y, new SquaredEuclideanDistanceMetric).map(t => (t._1, t._2, t._3)).collect()
    val expectedResults = TsneHelpersTestSuite.distancesDenseWithoutVectorDiff
    
    print(results)
        
    results.size should equal (expectedResults.size)
    for (expected <- expectedResults) {
      val result = results.find(x => x._1 == expected._1 && x._2 == expected._2)
      result match {
        case Some(result) => result._3 should equal (expected._3 +- 1e-6)
        case _ => fail("expected result not found")
      }
    }
  }

  "Q" should "return the Q" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val y = env.fromCollection(TsneHelpersTestSuite.yInit)
    val DD = computeDistances(y, new SquaredEuclideanDistanceMetric)
    
    //TODO: check this function and where the error is ...
    val results = computeLowDimAffinities(DD).q.collect()
    
    val expectedResults = TsneHelpersTestSuite.QDense

    print(results)

    results.size should equal (expectedResults.size)
    for (expected <- expectedResults) {
      val result = results.find(x => x._1 == expected._1 && x._2 == expected._2)
      result match {
        case Some(result) => result._3 should equal (expected._3 +- 1e-6)
        case _ => fail("expected result not found")
      }
    }
  }
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

  // calculated by Van der Maaten Python implementation
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

  // calculated by Van der Maaten Python implementation
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

  //calculated by c++ program
  val pairwiseAffinitiesResultsSparse: Seq[(Long, Long, Double)] = List(
    (0L, 5L, 0.999490),(0L, 4L, 0.000000),(1L, 5L, 0.999490),(1L, 6L, 0.000000),
    (2L, 3L, 0.999490),(2L, 4L, 0.000000),(3L, 4L, 0.499252),(3L, 2L, 0.499252),
    (4L, 3L, 0.499252),(4L, 5L, 0.499252),(5L, 6L, 0.499252),(5L, 4L, 0.499252),
    (6L, 5L, 0.499252),(6L, 7L, 0.499252),(7L, 9L, 0.999490),(7L, 6L, 0.000000),
    (8L, 9L, 0.999490),(8L, 7L, 0.000000),(9L, 8L, 0.999490),(9L, 7L, 0.000000),
    (10L, 11L, 0.999490),(10L, 8L, 0.000000),(11L, 10L, 0.999490),(11L, 8L, 0.000000)
  ).toSeq

  //calculated by c++ program
  val jointProbabilitiesResultsSparse: Seq[(Long, Long, Double)] = List(
    (0L, 5L, 0.041680),(0L, 4L, 0.000000),(1L, 5L, 0.041680),(1L, 6L, 0.000000),
    (2L, 3L, 0.062500),(2L, 4L, 0.000000),(3L, 2L, 0.062500),(3L, 4L, 0.041639),
    (4L, 0L, 0.000000),(4L, 2L, 0.000000),(4L, 3L, 0.041639),(4L, 5L, 0.041639),
    (5L, 0L, 0.041680),(5L, 1L, 0.041680),(5L, 4L, 0.041639),(5L, 6L, 0.041639),
    (6L, 1L, 0.000000),(6L, 5L, 0.041639),(6L, 7L, 0.020820),(7L, 6L, 0.020820),
    (7L, 9L, 0.041680),(7L, 8L, 0.000000),(8L, 9L, 0.083361),(8L, 7L, 0.000000),
    (8L, 10L, 0.000000),(8L, 11L, 0.000000),(9L, 7L, 0.041680),(9L, 8L, 0.083361),
    (10L, 11L, 0.083361),(10L, 8L, 0.000000),(11L, 10L, 0.083361),(11L, 8L, 0.000000)
  )

  //calculated by c++ program
  val yInit: Seq[LabeledVector] = List(
    LabeledVector(0, new DenseVector(Array(0.00011122716070838681,0.00006080563805333057))), 
    LabeledVector(1, new DenseVector(Array(-0.00007120823209735604,-0.00017189452155069314))), 
    LabeledVector(2, new DenseVector(Array(-0.00004000535669646060,-0.00022717203149772645))), 
    LabeledVector(3, new DenseVector(Array(0.00008663309627528085,-0.00010325765463991235))), 
    LabeledVector(4, new DenseVector(Array(-0.00003582032920213515,-0.00011138115570423478))), 
    LabeledVector(5, new DenseVector(Array(0.00000474475662365405,0.00008448473882524468))), 
    LabeledVector(6, new DenseVector(Array(0.00022241394002237897,-0.00000064174153521060))), 
    LabeledVector(7, new DenseVector(Array(-0.00007498357621804678,0.00001542291223429679))), 
    LabeledVector(8, new DenseVector(Array(-0.00002279772806102704,-0.00013284059681958318))), 
    LabeledVector(9, new DenseVector(Array(0.00000541931482402651,-0.00010620372323138086))), 
    LabeledVector(10, new DenseVector(Array(0.00007868893550591110,-0.00000695497272360187))), 
    LabeledVector(11, new DenseVector(Array(-0.00017049612819062857,-0.00006410186066559613)))
  )

  //calculated by c++ program
  val distancesDenseWithoutVectorDiff: Seq[(Long, Long, Double)] = List(
    (0L, 1L, 0.00000008743203682792),
    (0L, 2L, 0.00000010580241248067),
    (0L, 3L, 0.00000002752163201469),
    (0L, 4L, 0.00000005127125623350),
    (0L, 5L, 0.00000001189920219303),
    (0L, 6L, 0.00000001613828035252),
    (0L, 7L, 0.00000003673403034945),
    (0L, 8L, 0.00000005546153509011),
    (0L, 9L, 0.00000003908742700742),
    (0L, 10L, 0.00000000565023647219),
    (0L, 11L, 0.00000009496989474430),
    (1L, 0L, 0.00000008743203682792),
    (1L, 2L, 0.00000000402922253923),
    (1L, 3L, 0.00000002962490444177),
    (1L, 4L, 0.00000000491417111739),
    (1L, 5L, 0.00000007149918164657),
    (1L, 6L, 0.00000011554149462334),
    (1L, 7L, 0.00000003510207422302),
    (1L, 8L, 0.00000000386878593795),
    (1L, 9L, 0.00000001018706193102),
    (1L, 10L, 0.00000004967421562277),
    (1L, 11L, 0.00000002147734405132),
    (2L, 0L, 0.00000010580241248067),
    (2L, 1L, 0.00000000402922253923),
    (2L, 3L, 0.00000003139207056314),
    (2L, 4L, 0.00000001342504137215),
    (2L, 5L, 0.00000009913251513031),
    (2L, 6L, 0.00000012017985956091),
    (2L, 7L, 0.00000006007578256524),
    (2L, 8L, 0.00000000919452205169),
    (2L, 9L, 0.00000001669673238757),
    (2L, 10L, 0.00000006258388797655),
    (2L, 11L, 0.00000004361972206036),
    (3L, 0L, 0.00000002752163201469),
    (3L, 1L, 0.00000002962490444177),
    (3L, 2L, 0.00000003139207056314),
    (3L, 4L, 0.00000001506083268070),
    (3L, 5L, 0.00000004195290647493),
    (3L, 6L, 0.00000002896646315099),
    (3L, 7L, 0.00000004020502578140),
    (3L, 8L, 0.00000001285025578293),
    (3L, 9L, 0.00000000660435761776),
    (3L, 10L, 0.00000000933731623460),
    (3L, 11L, 0.00000006764861427620),
    (4L, 0L, 0.00000005127125623350),
    (4L, 1L, 0.00000000491417111739),
    (4L, 2L, 0.00000001342504137215),
    (4L, 3L, 0.00000001506083268070),
    (4L, 5L, 0.00000004000897482789),
    (4L, 6L, 0.00000007894815565242),
    (4L, 7L, 0.00000001761303156259),
    (4L, 8L, 0.00000000063009575346),
    (4L, 9L, 0.00000000172751404642),
    (4L, 10L, 0.00000002401719939588),
    (4L, 11L, 0.00000002037290257254),
    (5L, 0L, 0.00000001189920219303),
    (5L, 1L, 0.00000007149918164657),
    (5L, 2L, 0.00000009913251513031),
    (5L, 3L, 0.00000004195290647493),
    (5L, 4L, 0.00000004000897482789),
    (5L, 6L, 0.00000005462639106003),
    (5L, 7L, 0.00000001112614294980),
    (5L, 8L, 0.00000004798888997574),
    (5L, 9L, 0.00000003636254459029),
    (5L, 10L, 0.00000001382896243871),
    (5L, 11L, 0.00000005278734525874),
    (6L, 0L, 0.00000001613828035252),
    (6L, 1L, 0.00000011554149462334),
    (6L, 2L, 0.00000012017985956091),
    (6L, 3L, 0.00000002896646315099),
    (6L, 4L, 0.00000007894815565242),
    (6L, 5L, 0.00000005462639106003),
    (6L, 7L, 0.00000008870335576671),
    (6L, 8L, 0.00000007760529950274),
    (6L, 9L, 0.00000005822999934460),
    (6L, 10L, 0.00000002069673381130),
    (6L, 11L, 0.00000015840550842319),
    (7L, 0L, 0.00000003673403034945),
    (7L, 1L, 0.00000003510207422302),
    (7L, 2L, 0.00000006007578256524),
    (7L, 3L, 0.00000004020502578140),
    (7L, 4L, 0.00000001761303156259),
    (7L, 5L, 0.00000001112614294980),
    (7L, 6L, 0.00000008870335576671),
    (7L, 8L, 0.00000002470543086484),
    (7L, 9L, 0.00000002125766334262),
    (7L, 10L, 0.00000002411601059474),
    (7L, 11L, 0.00000001544683708909),
    (8L, 0L, 0.00000005546153509011),
    (8L, 1L, 0.00000000386878593795),
    (8L, 2L, 0.00000000919452205169),
    (8L, 3L, 0.00000001285025578293),
    (8L, 4L, 0.00000000063009575346),
    (8L, 5L, 0.00000004798888997574),
    (8L, 6L, 0.00000007760529950274),
    (8L, 7L, 0.00000002470543086484),
    (8L, 9L, 0.00000000150572454373),
    (8L, 10L, 0.00000002614673323598),
    (8L, 11L, 0.00000002653983124889),
    (9L, 0L, 0.00000003908742700742),
    (9L, 1L, 0.00000001018706193102),
    (9L, 2L, 0.00000001669673238757),
    (9L, 3L, 0.00000000660435761776),
    (9L, 4L, 0.00000000172751404642),
    (9L, 5L, 0.00000003636254459029),
    (9L, 6L, 0.00000005822999934460),
    (9L, 7L, 0.00000002125766334262),
    (9L, 8L, 0.00000000150572454373),
    (9L, 10L, 0.00000001521875179222),
    (9L, 11L, 0.00000003271880992255),
    (10L, 0L, 0.00000000565023647219),
    (10L, 1L, 0.00000004967421562277),
    (10L, 2L, 0.00000006258388797655),
    (10L, 3L, 0.00000000933731623460),
    (10L, 4L, 0.00000002401719939588),
    (10L, 5L, 0.00000001382896243871),
    (10L, 6L, 0.00000002069673381130),
    (10L, 7L, 0.00000002411601059474),
    (10L, 8L, 0.00000002614673323598),
    (10L, 9L, 0.00000001521875179222),
    (10L, 11L, 0.00000006535896277090),
    (11L, 0L, 0.00000009496989474430),
    (11L, 1L, 0.00000002147734405132),
    (11L, 2L, 0.00000004361972206036),
    (11L, 3L, 0.00000006764861427620),
    (11L, 4L, 0.00000002037290257254),
    (11L, 5L, 0.00000005278734525874),
    (11L, 6L, 0.00000015840550842319),
    (11L, 7L, 0.00000001544683708909),
    (11L, 8L, 0.00000002653983124889),
    (11L, 9L, 0.00000003271880992255),
    (11L, 10L, 0.00000006535896277090)
  )

  //calculated by c++ program
  val QDense: Seq[(Long, Long, Double)] = List(
    (0L, 1L, 0.99999991256797093886),
    (0L, 2L, 0.99999989419759882647),
    (0L, 3L, 0.99999997247836869274),
    (0L, 4L, 0.99999994872874653495),
    (0L, 5L, 0.99999998810079782086),
    (0L, 6L, 0.99999998386171995435),
    (0L, 7L, 0.99999996326597107021),
    (0L, 8L, 0.99999994453846796638),
    (0L, 9L, 0.99999996091257448150),
    (0L, 10L, 0.99999999434976349910),
    (0L, 11L, 0.99999990503011415566),
    (1L, 0L, 0.99999991256797093886),
    (1L, 2L, 0.99999999597077748881),
    (1L, 3L, 0.99999997037509635867),
    (1L, 4L, 0.99999999508582892993),
    (1L, 5L, 0.99999992850082342066),
    (1L, 6L, 0.99999988445851872854),
    (1L, 7L, 0.99999996489792708498),
    (1L, 8L, 0.99999999613121404174),
    (1L, 9L, 0.99999998981293825029),
    (1L, 10L, 0.99999995032578681275),
    (1L, 11L, 0.99999997852265631870),
    (2L, 0L, 0.99999989419759882647),
    (2L, 1L, 0.99999999597077748881),
    (2L, 3L, 0.99999996860793050057),
    (2L, 4L, 0.99999998657495892118),
    (2L, 5L, 0.99999990086749479534),
    (2L, 6L, 0.99999987982015481336),
    (2L, 7L, 0.99999993992422109290),
    (2L, 8L, 0.99999999080547807839),
    (2L, 9L, 0.99999998330326789198),
    (2L, 10L, 0.99999993741611581388),
    (2L, 11L, 0.99999995638027983080),
    (3L, 0L, 0.99999997247836869274),
    (3L, 1L, 0.99999997037509635867),
    (3L, 2L, 0.99999996860793050057),
    (3L, 4L, 0.99999998493916764097),
    (3L, 5L, 0.99999995804709529423),
    (3L, 6L, 0.99999997103353766015),
    (3L, 7L, 0.99999995979497591225),
    (3L, 8L, 0.99999998714974436265),
    (3L, 9L, 0.99999999339564227263),
    (3L, 10L, 0.99999999066268385750),
    (3L, 11L, 0.99999993235139028602),
    (4L, 0L, 0.99999994872874653495),
    (4L, 1L, 0.99999999508582892993),
    (4L, 2L, 0.99999998657495892118),
    (4L, 3L, 0.99999998493916764097),
    (4L, 5L, 0.99999995999102675448),
    (4L, 6L, 0.99999992105185064872),
    (4L, 7L, 0.99999998238696885533),
    (4L, 8L, 0.99999999936990424665),
    (4L, 9L, 0.99999999827248586826),
    (4L, 10L, 0.99999997598280121469),
    (4L, 11L, 0.99999997962709796795),
    (5L, 0L, 0.99999998810079782086),
    (5L, 1L, 0.99999992850082342066),
    (5L, 2L, 0.99999990086749479534),
    (5L, 3L, 0.99999995804709529423),
    (5L, 4L, 0.99999995999102675448),
    (5L, 6L, 0.99999994537361203140),
    (5L, 7L, 0.99999998887385721513),
    (5L, 8L, 0.99999995201111235854),
    (5L, 9L, 0.99999996363745680128),
    (5L, 10L, 0.99999998617103780241),
    (5L, 11L, 0.99999994721265761388),
    (6L, 0L, 0.99999998386171995435),
    (6L, 1L, 0.99999988445851872854),
    (6L, 2L, 0.99999987982015481336),
    (6L, 3L, 0.99999997103353766015),
    (6L, 4L, 0.99999992105185064872),
    (6L, 5L, 0.99999994537361203140),
    (6L, 7L, 0.99999991129665211087),
    (6L, 8L, 0.99999992239470647526),
    (6L, 9L, 0.99999994177000417483),
    (6L, 10L, 0.99999997930326656181),
    (6L, 11L, 0.99999984159451660304),
    (7L, 0L, 0.99999996326597107021),
    (7L, 1L, 0.99999996489792708498),
    (7L, 2L, 0.99999993992422109290),
    (7L, 3L, 0.99999995979497591225),
    (7L, 4L, 0.99999998238696885533),
    (7L, 5L, 0.99999998887385721513),
    (7L, 6L, 0.99999991129665211087),
    (7L, 8L, 0.99999997529456974288),
    (7L, 9L, 0.99999997874233703676),
    (7L, 10L, 0.99999997588399003323),
    (7L, 11L, 0.99999998455316307933),
    (8L, 0L, 0.99999994453846796638),
    (8L, 1L, 0.99999999613121404174),
    (8L, 2L, 0.99999999080547807839),
    (8L, 3L, 0.99999998714974436265),
    (8L, 4L, 0.99999999936990424665),
    (8L, 5L, 0.99999995201111235854),
    (8L, 6L, 0.99999992239470647526),
    (8L, 7L, 0.99999997529456974288),
    (8L, 9L, 0.99999999849427556597),
    (8L, 10L, 0.99999997385326744315),
    (8L, 11L, 0.99999997346016944455),
    (9L, 0L, 0.99999996091257448150),
    (9L, 1L, 0.99999998981293825029),
    (9L, 2L, 0.99999998330326789198),
    (9L, 3L, 0.99999999339564227263),
    (9L, 4L, 0.99999999827248586826),
    (9L, 5L, 0.99999996363745680128),
    (9L, 6L, 0.99999994177000417483),
    (9L, 7L, 0.99999997874233703676),
    (9L, 8L, 0.99999999849427556597),
    (9L, 10L, 0.99999998478124840773),
    (9L, 11L, 0.99999996728119122658),
    (10L, 0L, 0.99999999434976349910),
    (10L, 1L, 0.99999995032578681275),
    (10L, 2L, 0.99999993741611581388),
    (10L, 3L, 0.99999999066268385750),
    (10L, 4L, 0.99999997598280121469),
    (10L, 5L, 0.99999998617103780241),
    (10L, 6L, 0.99999997930326656181),
    (10L, 7L, 0.99999997588399003323),
    (10L, 8L, 0.99999997385326744315),
    (10L, 9L, 0.99999998478124840773),
    (10L, 11L, 0.99999993464104153240),
    (11L, 0L, 0.99999990503011415566),
    (11L, 1L, 0.99999997852265631870),
    (11L, 2L, 0.99999995638027983080),
    (11L, 3L, 0.99999993235139028602),
    (11L, 4L, 0.99999997962709796795),
    (11L, 5L, 0.99999994721265761388),
    (11L, 6L, 0.99999984159451660304),
    (11L, 7L, 0.99999998455316307933),
    (11L, 8L, 0.99999997346016944455),
    (11L, 9L, 0.99999996728119122658),
    (11L, 10L, 0.99999993464104153240)
  )
}
