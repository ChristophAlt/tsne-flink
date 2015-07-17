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
import org.apache.flink.ml.math.Breeze._


class TsneHelpersTestSuite extends FlatSpec with Matchers with Inspectors {
  "kNearestNeighbors" should "return the k nearest neighbors for each SparseVector" in  {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val neighbors = 2
    val metric = SquaredEuclideanDistanceMetric()

    val input = env.fromCollection(TsneHelpersTestSuite.knnInput)

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

    val input = TsneHelpersTestSuite
      .readInput(getClass.getResource("/dense_input.csv").getPath, 28*28, env, Array(0,1,2))

    val knn = kNearestNeighbors(input, neighbors, metric)
    val results = pairwiseAffinities(knn, perplexity).collect()
    val expectedResults = TsneHelpersTestSuite.densePairwiseAffinitiesResults

    results.size should equal (expectedResults.size)
    for (expected <- expectedResults) {
      val result = results.find(x => x._1 == expected._1 && x._2 == expected._2)
      result match {
        case Some(result) => result._3 should equal (expected._3 +- 1e-12)
        case None => fail("expected result not found")
      }
    }
  }

  "jointProbabilities" should "return the symmetrized probability distribution p_ij over the datapoints" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val pairwiseAffinities = env.fromCollection(TsneHelpersTestSuite.densePairwiseAffinitiesResults)
    val results = jointDistribution(pairwiseAffinities).collect()
    val expectedResults = TsneHelpersTestSuite.denseJointProbabilitiesResults

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

    val pairwiseAffinities = env.fromCollection(TsneHelpersTestSuite.sparsePairwiseAffinitiesResults)
    val results = jointDistribution(pairwiseAffinities).collect()
    val expectedResults = TsneHelpersTestSuite.sparseJointProbabilitiesResults
    
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
        case Some(result) => result._3 should equal (expected._3 +- 1e-12)
        case _ => fail("expected result not found")
      }
    }
  }

  "sumLowDimAffinities" should "return the sum over Q" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val embedding = env.fromCollection(TsneHelpersTestSuite.initialEmbedding)

    val results = sumLowDimAffinities(embedding, SquaredEuclideanDistanceMetric()).collect()

    val expectedResult = TsneHelpersTestSuite.denseSumQ

    results.size should equal (1)
    results(0) should equal (expectedResult +- 1e-12)
  }

  "centerEmbedding" should "compute the centered embedding as LabeledVectors" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val nComponents = 2

    val embeddingSeq = TsneHelpersTestSuite.centeringInput.map( g => {

      val lastGradient = breeze.linalg.DenseVector.fill(nComponents, 0.0)
      val gains = breeze.linalg.DenseVector.fill(nComponents, 1.0)

      (g.label, g.vector, lastGradient.fromBreeze, gains.fromBreeze)
    })

    val embedding = env.fromCollection(embeddingSeq)

    val results = centerEmbedding(embedding).map(x => LabeledVector(x._1, x._2)).collect()

    val expectedResults = TsneHelpersTestSuite.centeringResults

    results.size should equal (expectedResults.size)
    for (expected <- expectedResults) {
      val result = results.find(x => x.label == expected.label)
      result match {
        case Some(result) => result.vector should equal (expected.vector)
        case _ => fail("expected result not found")
      }
    }
  }

  "Gradient" should "return the gradient" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val jointDistribution = env.fromCollection(TsneHelpersTestSuite.denseJointProbabilitiesResults)
    val embedding = env.fromCollection(TsneHelpersTestSuite.initialEmbedding)
    val sumQ = env.fromCollection(List(TsneHelpersTestSuite.denseSumQ))

    val results = gradient(jointDistribution, embedding, SquaredEuclideanDistanceMetric(), sumQ).collect()

    val expectedResults = TsneHelpersTestSuite.denseGradientResults
    
    results.size should equal (expectedResults.size)
    for (expected <- expectedResults) {
      val result = results.find(x => x.label == expected.label)
      result match {
        case Some(result) => {
          for (i <- 0 until result.vector.size){
            result.vector(i) should equal (expected.vector(i) +- 1e-12)
          }
        }
        case _ => fail("expected result not found")
      }
    }
  }

  "initWorkingSet" should "randomly initialize the embedding and initialize gradient and gains to zero" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val randomState = 0
    val nComponents = 2

    val zeroVector = breeze.linalg.DenseVector.fill(nComponents, 0.0).fromBreeze
    val oneVector = breeze.linalg.DenseVector.fill(nComponents, 1.0).fromBreeze

    val input = TsneHelpersTestSuite
      .readInput(getClass.getResource("/dense_input.csv").getPath, 28*28, env, Array(0,1,2))

    val results = initWorkingSet(input, nComponents, randomState).collect()

    results.size should equal (input.count())
    for (result <- results) {
      result._3 should equal (zeroVector)
      result._4 should equal (oneVector)
    }
  }

  "updateEmbedding" should "return the updated embedding by the gradient with momentum and gain" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val minGain = 0.01
    val momentum = 0.5
    val learningRate = 300
    val nComponents = 2

    val initialEmbeddingSeq = TsneHelpersTestSuite.initialEmbedding
    val gradient = env.fromCollection(TsneHelpersTestSuite.denseGradientResults)

    val workingSetSeq = initialEmbeddingSeq.map( g => {

      val lastGradient = breeze.linalg.DenseVector.fill(nComponents, 0.0)
      val gains = breeze.linalg.DenseVector.fill(nComponents, 1.0)

      (g.label, g.vector, lastGradient.fromBreeze, gains.fromBreeze)
    })

    val workingSet = env.fromCollection(workingSetSeq)

    val results = updateEmbedding(gradient, workingSet, minGain, momentum, learningRate)
      .map(x => LabeledVector(x._1, x._2)).collect()

    val expectedResults = TsneHelpersTestSuite.updatedEmbeddingResults

    results.size should equal (expectedResults.size)
    for (expected <- expectedResults) {
      val result = results.find(x => x.label == expected.label)
      result match {
        case Some(result) => {
          for (i <- 0 until result.vector.size){
            result.vector(i) should equal (expected.vector(i) +- 1e-9)
          }
        }
        case _ => fail("expected result not found")
      }
    }
  }

  "iterationComputation" should "iteratively compute the embedding" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val momentum = 0.5
    val learningRate = 300
    val nComponents = 2
    val iterations = 1
    val metric = SquaredEuclideanDistanceMetric()

    val jointDistribution = env.fromCollection(TsneHelpersTestSuite.denseJointProbabilitiesResults)

    val initialEmbeddingSeq = TsneHelpersTestSuite.initialEmbedding

    val workingSetSeq = initialEmbeddingSeq.map( g => {

      val lastGradient = breeze.linalg.DenseVector.fill(nComponents, 0.0)
      val gains = breeze.linalg.DenseVector.fill(nComponents, 1.0)

      (g.label, g.vector, lastGradient.fromBreeze, gains.fromBreeze)
    })

    val workingSet = env.fromCollection(workingSetSeq)

    val results = iterationComputation(iterations, momentum, workingSet, jointDistribution, metric, learningRate)
      .map(x => LabeledVector(x._1, x._2)).collect()

    val expectedResults = TsneHelpersTestSuite.updatedAndCentredEmbeddingResults

    results.size should equal (expectedResults.size)
    for (expected <- expectedResults) {
      val result = results.find(x => x.label == expected.label)
      result match {
        case Some(result) => {
          for (i <- 0 until result.vector.size){
            result.vector(i) should equal (expected.vector(i) +- 1e-9)
          }
        }
        case _ => fail("expected result not found")
      }
    }
  }
}

object TsneHelpersTestSuite {
  val knnInput: Seq[LabeledVector] = List(
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

  // Python implementation van der Maaten, perplexity: 2.0, nComponents: 2, earlyExaggeration:

  val densePairwiseAffinitiesResults: Seq[(Long, Long, Double)] = List(
    (0L, 1L, 2.370974987703e-02), (0L, 2L, 5.153826240184e-05), (0L, 3L, 1.945495759780e-02), (0L, 4L, 8.216433537309e-04), (0L, 5L, 4.872518553230e-03), (0L, 6L, 7.036533081247e-02), (0L, 7L, 8.338103510412e-01), (0L, 8L, 4.291578136374e-02), (0L, 9L, 3.998129138403e-03), (1L, 0L, 7.724806516559e-01), (1L, 2L, 1.365877675224e-09), (1L, 3L, 3.248251955497e-05), (1L, 4L, 5.336682282239e-04), (1L, 5L, 1.440504524837e-01), (1L, 6L, 1.402694896096e-05), (1L, 7L, 8.229819286616e-02), (1L, 8L, 5.685570334838e-05), (1L, 9L, 5.336682282239e-04), (2L, 0L, 1.898500331402e-03), (2L, 1L, 8.687658519724e-04), (2L, 3L, 5.062254851994e-02), (2L, 4L, 1.449209955883e-02), (2L, 5L, 3.548272991202e-03), (2L, 6L, 2.928791467503e-02), (2L, 7L, 3.166935688445e-02), (2L, 8L, 2.316520085169e-02), (2L, 9L, 8.444473403355e-01), (3L, 0L, 2.390453996554e-04), (3L, 1L, 1.395251795149e-08), (3L, 2L, 1.371014800172e-06), (3L, 4L, 2.369600685486e-03), (3L, 5L, 1.347198827355e-04), (3L, 6L, 3.128918783462e-02), (3L, 7L, 2.411490824021e-05), (3L, 8L, 7.330981240389e-01), (3L, 9L, 2.328438222830e-01), (4L, 0L, 1.019336232626e-04), (4L, 1L, 2.549661856719e-04), (4L, 2L, 8.105432254715e-05), (4L, 3L, 9.980288326134e-03), (4L, 5L, 5.017868132870e-03), (4L, 6L, 1.751495817431e-01), (4L, 7L, 1.008626268887e-03), (4L, 8L, 7.770114760492e-01), (4L, 9L, 3.139420534839e-02), (5L, 0L, 9.336380191993e-06), (5L, 1L, 1.005438326603e-03), (5L, 2L, 1.769727010776e-13), (5L, 3L, 2.563249124685e-03), (5L, 4L, 1.665948534565e-02), (5L, 6L, 1.977751128205e-10), (5L, 7L, 2.760372711471e-01), (5L, 8L, 1.333923297742e-08), (5L, 9L, 7.037252061386e-01), (6L, 0L, 2.657830365339e-02), (6L, 1L, 4.705122579295e-03), (6L, 2L, 7.129194416069e-03), (6L, 3L, 3.052697452006e-02), (6L, 4L, 5.312632631673e-02), (6L, 5L, 4.705122579295e-03), (6L, 7L, 8.775567604535e-03), (6L, 8L, 8.480859960929e-01), (6L, 9L, 1.636739223770e-02), (7L, 0L, 8.439216060511e-01), (7L, 1L, 2.173031429002e-02), (7L, 2L, 4.863220572021e-03), (7L, 3L, 1.840043131472e-02), (7L, 4L, 1.319325614475e-02), (7L, 5L, 6.961968258558e-02), (7L, 6L, 7.370873853279e-03), (7L, 8L, 2.500183873820e-03), (7L, 9L, 1.840043131472e-02), (8L, 0L, 1.718948684906e-02), (8L, 1L, 3.907353474692e-03), (8L, 2L, 4.531278151932e-03), (8L, 3L, 3.605396373725e-02), (8L, 4L, 7.022214844512e-02), (8L, 5L, 4.531278151932e-03), (8L, 6L, 8.397228403177e-01), (8L, 7L, 3.907353474692e-03), (8L, 9L, 1.993429739760e-02), (9L, 0L, 4.341453089465e-07), (9L, 1L, 2.423850003604e-08), (9L, 2L, 9.192527632300e-02), (9L, 3L, 8.003537295921e-01), (9L, 4L, 9.192527632300e-02), (9L, 5L, 5.132223630089e-03), (9L, 6L, 9.710748028923e-05), (9L, 7L, 7.776147410114e-06), (9L, 8L, 1.055815212027e-02)
  ).toSeq

  val denseJointProbabilitiesResults: Seq[(Long, Long, Double)] = List(
    (0L, 1L, 3.980952007665e-02), (0L, 2L, 9.750192969019e-05), (0L, 3L, 9.847001498727e-04), (0L, 4L, 4.617884884968e-05), (0L, 5L, 2.440927466711e-04), (0L, 6L, 4.847181723293e-03), (0L, 7L, 8.388659785461e-02), (0L, 8L, 3.005263410640e-03), (0L, 9L, 1.999281641856e-04), (1L, 0L, 3.980952007665e-02), (1L, 2L, 4.343836089250e-05), (1L, 3L, 1.624823603646e-06), (1L, 4L, 3.943172069479e-05), (1L, 5L, 7.252794540517e-03), (1L, 6L, 2.359574764128e-04), (1L, 7L, 5.201425357809e-03), (1L, 8L, 1.982104589020e-04), (1L, 9L, 2.668462333619e-05), (2L, 0L, 9.750192969019e-05), (2L, 1L, 4.343836089250e-05), (2L, 3L, 2.531195976737e-03), (2L, 4L, 7.286576940686e-04), (2L, 5L, 1.774136495689e-04), (2L, 6L, 1.820855454555e-03), (2L, 7L, 1.826628872824e-03), (2L, 8L, 1.384823950181e-03), (2L, 9L, 4.681863083292e-02), (3L, 0L, 9.847001498727e-04), (3L, 1L, 1.624823603646e-06), (3L, 2L, 2.531195976737e-03), (3L, 4L, 6.174944505810e-04), (3L, 5L, 1.348984503710e-04), (3L, 6L, 3.090808117734e-03), (3L, 7L, 9.212273111480e-04), (3L, 8L, 3.845760438881e-02), (3L, 9L, 5.165987759376e-02), (4L, 0L, 4.617884884968e-05), (4L, 1L, 3.943172069479e-05), (4L, 2L, 7.286576940686e-04), (4L, 3L, 6.174944505810e-04), (4L, 5L, 1.083867673926e-03), (4L, 6L, 1.141379540299e-02), (4L, 7L, 7.100941206818e-04), (4L, 8L, 4.236168122471e-02), (4L, 9L, 6.165974083570e-03), (5L, 0L, 2.440927466711e-04), (5L, 1L, 7.252794540517e-03), (5L, 2L, 1.774136495689e-04), (5L, 3L, 1.348984503710e-04), (5L, 4L, 1.083867673926e-03), (5L, 6L, 2.352561388535e-04), (5L, 7L, 1.728284768663e-02), (5L, 8L, 2.265645745583e-04), (5L, 9L, 3.544287148844e-02), (6L, 0L, 4.847181723293e-03), (6L, 1L, 2.359574764128e-04), (6L, 2L, 1.820855454555e-03), (6L, 3L, 3.090808117734e-03), (6L, 4L, 1.141379540299e-02), (6L, 5L, 2.352561388535e-04), (6L, 7L, 8.073220728907e-04), (6L, 8L, 8.439044182053e-02), (6L, 9L, 8.232249858994e-04), (7L, 0L, 8.388659785461e-02), (7L, 1L, 5.201425357809e-03), (7L, 2L, 1.826628872824e-03), (7L, 3L, 9.212273111480e-04), (7L, 4L, 7.100941206818e-04), (7L, 5L, 1.728284768663e-02), (7L, 6L, 8.073220728907e-04), (7L, 8L, 3.203768674256e-04), (7L, 9L, 9.204103731065e-04), (8L, 0L, 3.005263410640e-03), (8L, 1L, 1.982104589020e-04), (8L, 2L, 1.384823950181e-03), (8L, 3L, 3.845760438881e-02), (8L, 4L, 4.236168122471e-02), (8L, 5L, 2.265645745583e-04), (8L, 6L, 8.439044182053e-02), (8L, 7L, 3.203768674256e-04), (8L, 9L, 1.524622475894e-03), (9L, 0L, 1.999281641856e-04), (9L, 1L, 2.668462333619e-05), (9L, 2L, 4.681863083292e-02), (9L, 3L, 5.165987759376e-02), (9L, 4L, 6.165974083570e-03), (9L, 5L, 3.544287148844e-02), (9L, 6L, 8.232249858994e-04), (9L, 7L, 9.204103731065e-04), (9L, 8L, 1.524622475894e-03)
  ).toSeq

  val denseSumQ = 3.365625463923e+01

  val denseUnnormLowDimAffinitiesResults: Seq[(Long, Long, Double)] = List(
    (0L, 1L, 1.997990966178e-01), (0L, 2L, 3.438741251510e-01), (0L, 3L, 5.084645547121e-01), (0L, 4L, 2.228754579760e-01), (0L, 5L, 2.111669473429e-01), (0L, 6L, 4.799407190805e-01), (0L, 7L, 3.639911864625e-01), (0L, 8L, 6.947858598830e-01), (0L, 9L, 2.137434062812e-01), (1L, 0L, 1.997990966178e-01), (1L, 2L, 8.232738803226e-02), (1L, 3L, 1.487280153275e-01), (1L, 4L, 1.811394269082e-01), (1L, 5L, 4.318749070054e-01), (1L, 6L, 1.805549827630e-01), (1L, 7L, 2.031044587733e-01), (1L, 8L, 1.379549302261e-01), (1L, 9L, 9.072703100166e-02), (2L, 0L, 3.438741251510e-01), (2L, 1L, 8.232738803226e-02), (2L, 3L, 3.962129377163e-01), (2L, 4L, 1.468393804551e-01), (2L, 5L, 1.011844233815e-01), (2L, 6L, 2.913681119953e-01), (2L, 7L, 2.107258420207e-01), (2L, 8L, 5.761512990942e-01), (2L, 9L, 2.914080750559e-01), (3L, 0L, 5.084645547121e-01), (3L, 1L, 1.487280153275e-01), (3L, 2L, 3.962129377163e-01), (3L, 4L, 4.123285538511e-01), (3L, 5L, 2.365319708095e-01), (3L, 6L, 9.006682201122e-01), (3L, 7L, 6.704571955142e-01), (3L, 8L, 7.699293990309e-01), (3L, 9L, 5.264164083150e-01), (4L, 0L, 2.228754579760e-01), (4L, 1L, 1.811394269082e-01), (4L, 2L, 1.468393804551e-01), (4L, 3L, 4.123285538511e-01), (4L, 5L, 4.650305789265e-01), (4L, 6L, 5.463238327253e-01), (4L, 7L, 7.661566330118e-01), (4L, 8L, 2.544194531822e-01), (4L, 9L, 3.606532994232e-01), (5L, 0L, 2.111669473429e-01), (5L, 1L, 4.318749070054e-01), (5L, 2L, 1.011844233815e-01), (5L, 3L, 2.365319708095e-01), (5L, 4L, 4.650305789265e-01), (5L, 6L, 3.168065370999e-01), (5L, 7L, 4.263239066536e-01), (5L, 8L, 1.793300697535e-01), (5L, 9L, 1.573034975607e-01), (6L, 0L, 4.799407190805e-01), (6L, 1L, 1.805549827630e-01), (6L, 2L, 2.913681119953e-01), (6L, 3L, 9.006682201122e-01), (6L, 4L, 5.463238327253e-01), (6L, 5L, 3.168065370999e-01), (6L, 7L, 8.729481827509e-01), (6L, 8L, 6.082097572678e-01), (6L, 9L, 4.645101020686e-01), (7L, 0L, 3.639911864625e-01), (7L, 1L, 2.031044587733e-01), (7L, 2L, 2.107258420207e-01), (7L, 3L, 6.704571955142e-01), (7L, 4L, 7.661566330118e-01), (7L, 5L, 4.263239066536e-01), (7L, 6L, 8.729481827509e-01), (7L, 8L, 4.178341871825e-01), (7L, 9L, 4.118776956674e-01), (8L, 0L, 6.947858598830e-01), (8L, 1L, 1.379549302261e-01), (8L, 2L, 5.761512990942e-01), (8L, 3L, 7.699293990309e-01), (8L, 4L, 2.544194531822e-01), (8L, 5L, 1.793300697535e-01), (8L, 6L, 6.082097572678e-01), (8L, 7L, 4.178341871825e-01), (8L, 9L, 3.551252754453e-01), (9L, 0L, 2.137434062812e-01), (9L, 1L, 9.072703100166e-02), (9L, 2L, 2.914080750559e-01), (9L, 3L, 5.264164083150e-01), (9L, 4L, 3.606532994232e-01), (9L, 5L, 1.573034975607e-01), (9L, 6L, 4.645101020686e-01), (9L, 7L, 4.118776956674e-01), (9L, 8L, 3.551252754453e-01)
  ).toSeq

  val initialEmbedding: Seq[LabeledVector] = List(
    LabeledVector(0.0, DenseVector(1.764052345968e+00, 4.001572083672e-01)),
    LabeledVector(1.0, DenseVector(9.787379841057e-01, 2.240893199201e+00)),
    LabeledVector(2.0, DenseVector(1.867557990150e+00, -9.772778798764e-01)),
    LabeledVector(3.0, DenseVector(9.500884175256e-01, -1.513572082977e-01)),
    LabeledVector(4.0, DenseVector(-1.032188517936e-01, 4.105985019384e-01)),
    LabeledVector(5.0, DenseVector(1.440435711609e-01, 1.454273506963e+00)),
    LabeledVector(6.0, DenseVector(7.610377251470e-01, 1.216750164928e-01)),
    LabeledVector(7.0, DenseVector(4.438632327454e-01, 3.336743273743e-01)),
    LabeledVector(8.0, DenseVector(1.494079073158e+00, -2.051582637658e-01)),
    LabeledVector(9.0, DenseVector(3.130677016509e-01, -8.540957393017e-01))
  ).toSeq

  val denseGradientResults: Seq[LabeledVector] = List(
    LabeledVector(0.0, DenseVector(2.039671351114e-02, -2.841077532513e-02)),
    LabeledVector(1.0, DenseVector(-8.392113494907e-03, 2.231906609600e-03)),
    LabeledVector(2.0, DenseVector(4.925836551601e-03, 1.893423603195e-02)),
    LabeledVector(3.0, DenseVector(-2.069923187117e-03, 3.379074534054e-02)),
    LabeledVector(4.0, DenseVector(9.411374074020e-03, 5.816153185684e-03)),
    LabeledVector(5.0, DenseVector(6.038471365385e-03, -9.556999264301e-04)),
    LabeledVector(6.0, DenseVector(-3.232836693981e-02, 1.179730064230e-02)),
    LabeledVector(7.0, DenseVector(-2.424283588089e-02, -2.266466720419e-02)),
    LabeledVector(8.0, DenseVector(4.632554267724e-02, -1.538485019566e-02)),
    LabeledVector(9.0, DenseVector(-2.006469867665e-02, -5.154349158673e-03))
  ).toSeq

  val gradientWithMomentumAndGainResults: Seq[LabeledVector] = List(
    LabeledVector(0.0, DenseVector(-7.342816864009e+00, 6.818586078031e+00)),
    LabeledVector(1.0, DenseVector(2.014107238778e+00, -8.034863794560e-01)),
    LabeledVector(2.0, DenseVector(-1.773301158576e+00, -6.816324971503e+00)),
    LabeledVector(3.0, DenseVector(4.967815649080e-01, -1.216466832260e+01)),
    LabeledVector(4.0, DenseVector(-3.388094666647e+00, -2.093815146846e+00)),
    LabeledVector(5.0, DenseVector(-2.173849691539e+00, 2.293679823432e-01)),
    LabeledVector(6.0, DenseVector(7.758808065556e+00, -4.247028231228e+00)),
    LabeledVector(7.0, DenseVector(5.818280611414e+00, 5.439520129006e+00)),
    LabeledVector(8.0, DenseVector(-1.667719536380e+01, 3.692364046958e+00)),
    LabeledVector(9.0, DenseVector(4.815527682396e+00, 1.237043798081e+00))
  ).toSeq

  val updatedGainsResults: Seq[LabeledVector] = List(
    LabeledVector(0.0, DenseVector(1.200000000000e+00, 8.000000000000e-01)),
    LabeledVector(1.0, DenseVector(8.000000000000e-01, 1.200000000000e+00)),
    LabeledVector(2.0, DenseVector(1.200000000000e+00, 1.200000000000e+00)),
    LabeledVector(3.0, DenseVector(8.000000000000e-01, 1.200000000000e+00)),
    LabeledVector(4.0, DenseVector(1.200000000000e+00, 1.200000000000e+00)),
    LabeledVector(5.0, DenseVector(1.200000000000e+00, 8.000000000000e-01)),
    LabeledVector(6.0, DenseVector(8.000000000000e-01, 1.200000000000e+00)),
    LabeledVector(7.0, DenseVector(8.000000000000e-01, 8.000000000000e-01)),
    LabeledVector(8.0, DenseVector(1.200000000000e+00, 8.000000000000e-01)),
    LabeledVector(9.0, DenseVector(8.000000000000e-01, 8.000000000000e-01))
  ).toSeq

  val updatedEmbeddingResults: Seq[LabeledVector] = List(
    LabeledVector(0.0, DenseVector(-5.578764518042e+00, 7.218743286398e+00)),
    LabeledVector(1.0, DenseVector(2.992845222883e+00, 1.437406819745e+00)),
    LabeledVector(2.0, DenseVector(9.425683157355e-02, -7.793602851380e+00)),
    LabeledVector(3.0, DenseVector(1.446869982434e+00, -1.231602553089e+01)),
    LabeledVector(4.0, DenseVector(-3.491313518441e+00, -1.683216644908e+00)),
    LabeledVector(5.0, DenseVector(-2.029806120378e+00, 1.683641489306e+00)),
    LabeledVector(6.0, DenseVector(8.519845790703e+00, -4.125353214736e+00)),
    LabeledVector(7.0, DenseVector(6.262143844159e+00, 5.773194456381e+00)),
    LabeledVector(8.0, DenseVector(-1.518311629065e+01, 3.487205783192e+00)),
    LabeledVector(9.0, DenseVector(5.128595384047e+00, 3.829480587797e-01))
  ).toSeq

  val updatedAndCentredEmbeddingResults: Seq[LabeledVector] = List(
    LabeledVector(0.0, DenseVector(-5.394920178871e+00, 7.812249121209e+00)),
    LabeledVector(1.0, DenseVector(3.176689562054e+00, 2.030912654557e+00)),
    LabeledVector(2.0, DenseVector(2.781011707444e-01, -7.200097016568e+00)),
    LabeledVector(3.0, DenseVector(1.630714321604e+00, -1.172251969608e+01)),
    LabeledVector(4.0, DenseVector(-3.307469179270e+00, -1.089710810096e+00)),
    LabeledVector(5.0, DenseVector(-1.845961781207e+00, 2.277147324118e+00)),
    LabeledVector(6.0, DenseVector(8.703690129873e+00, -3.531847379924e+00)),
    LabeledVector(7.0, DenseVector(6.445988183330e+00, 6.366700291192e+00)),
    LabeledVector(8.0, DenseVector(-1.499927195148e+01, 4.080711618003e+00)),
    LabeledVector(9.0, DenseVector(5.312439723218e+00, 9.764538935911e-01))
  ).toSeq

  val centeringInput: Seq[LabeledVector] = List(
    LabeledVector(0.0, SparseVector.fromCOO(2, List((0, -2.0),(1, 4.0)))),
    LabeledVector(1.0, SparseVector.fromCOO(2, List((0, -6.0),(1, 4.0)))),
    LabeledVector(2.0, SparseVector.fromCOO(2, List((0, 2.0),(1, -8.0))))
  ).toSeq

  val centeringResults: Seq[LabeledVector] = List(
    LabeledVector(0.0, SparseVector.fromCOO(2, List((0, 0.0),(1, 4.0)))),
    LabeledVector(1.0, SparseVector.fromCOO(2, List((0, -4.0),(1, 4.0)))),
    LabeledVector(2.0, SparseVector.fromCOO(2, List((0, 4.0),(1, -8.0))))
  ).toSeq

  // C++ implementation

  val sparsePairwiseAffinitiesResults: Seq[(Long, Long, Double)] = List(
    (0L, 5L, 0.999490),(0L, 4L, 0.000000),(1L, 5L, 0.999490),(1L, 6L, 0.000000),
    (2L, 3L, 0.999490),(2L, 4L, 0.000000),(3L, 4L, 0.499252),(3L, 2L, 0.499252),
    (4L, 3L, 0.499252),(4L, 5L, 0.499252),(5L, 6L, 0.499252),(5L, 4L, 0.499252),
    (6L, 5L, 0.499252),(6L, 7L, 0.499252),(7L, 9L, 0.999490),(7L, 6L, 0.000000),
    (8L, 9L, 0.999490),(8L, 7L, 0.000000),(9L, 8L, 0.999490),(9L, 7L, 0.000000),
    (10L, 11L, 0.999490),(10L, 8L, 0.000000),(11L, 10L, 0.999490),(11L, 8L, 0.000000)
  ).toSeq

  val sparseJointProbabilitiesResults: Seq[(Long, Long, Double)] = List(
    (0L, 5L, 0.041680),(0L, 4L, 0.000000),(1L, 5L, 0.041680),(1L, 6L, 0.000000),
    (2L, 3L, 0.062500),(2L, 4L, 0.000000),(3L, 2L, 0.062500),(3L, 4L, 0.041639),
    (4L, 0L, 0.000000),(4L, 2L, 0.000000),(4L, 3L, 0.041639),(4L, 5L, 0.041639),
    (5L, 0L, 0.041680),(5L, 1L, 0.041680),(5L, 4L, 0.041639),(5L, 6L, 0.041639),
    (6L, 1L, 0.000000),(6L, 5L, 0.041639),(6L, 7L, 0.020820),(7L, 6L, 0.020820),
    (7L, 9L, 0.041680),(7L, 8L, 0.000000),(8L, 9L, 0.083361),(8L, 7L, 0.000000),
    (8L, 10L, 0.000000),(8L, 11L, 0.000000),(9L, 7L, 0.041680),(9L, 8L, 0.083361),
    (10L, 11L, 0.083361),(10L, 8L, 0.000000),(11L, 10L, 0.083361),(11L, 8L, 0.000000)
  )

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

  private def readInput(inputPath: String, dimension: Int, env: ExecutionEnvironment,
                        fields: Array[Int]): DataSet[LabeledVector] = {
    env.readCsvFile[(Int, Int, Double)](inputPath, includedFields = fields)
      .groupBy(_._1).reduceGroup(
        elements => {
          val elementsIterable = elements.toIterable
          val entries = elementsIterable.map(x => (x._2, x._3))
          LabeledVector(elementsIterable.head._1.toDouble, SparseVector.fromCOO(dimension, entries))
        })
  }
}
