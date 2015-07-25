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

import breeze.linalg._
import de.tu_berlin.dima.impro3.TsneHelpers._
import org.apache.flink.api.scala._
import org.scalatest._


class TsneHelpersTestSuite extends FlatSpec with Matchers with Inspectors {

  "kNearestNeighbors" should "return the k nearest neighbors for each SparseVector" in  {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val neighbors = 2
    val metric = Tsne.getMetric("sqeucledian")

    val input = env.fromCollection(TsneHelpersTestSuite.knnInput)

    val results = kNearestNeighbors(input, neighbors, metric).collect()
    val expectedResults = TsneHelpersTestSuite.knnResults

    results.size should equal (expectedResults.size)
    forAll(results) {expectedResults should contain (_)}
  }

  "partitionKnn" should "return the k nearest neighbors for each SparseVector" in  {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val neighbors = 2
    val metric = Tsne.getMetric("sqeucledian")

    val input = env.fromCollection(TsneHelpersTestSuite.knnInput)

    val results = partitionKnn(input, neighbors, metric, env.getParallelism).collect()
    val expectedResults = TsneHelpersTestSuite.knnResults

    results.size should equal (expectedResults.size)
    forAll(results) {expectedResults should contain (_)}
  }

  /*"projectKnn" should "return the k nearest neighbors for each SparseVector" in  {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val neighbors = 2
    val dimension = 4
    val iterations = 4
    val metric = SquaredEuclideanDistanceMetric()

    val input = env.fromCollection(TsneHelpersTestSuite.knnInput)

    val results = projectKnn(input, neighbors, metric, dimension, iterations).collect()
    val expectedResults = TsneHelpersTestSuite.knnResults

    results.size should equal (expectedResults.size)
    forAll(results) {expectedResults should contain (_)}
  }*/

  "pairwiseAffinities" should "return the pairwise similarity p_i|j between datapoints" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val perplexity = 2.0
    val neighbors = 10
    val metric = Tsne.getMetric("sqeucledian")

    val input = Tsne
      .readInput(getClass.getResource("/dense_input.csv").getPath, 28*28, env, Array(0,1,2))

    val knn = kNearestNeighbors(input, neighbors, metric)
    val results = pairwiseAffinities(knn, perplexity).collect()
    val expectedResults = TsneHelpersTestSuite.densePairwiseAffinitiesResults

    results.size should equal (expectedResults.size)
    for (expected <- expectedResults) {
      val result = results.find(x => x._1 == expected._1 && x._2 == expected._2)
      result match {
        case Some(res) => res._3 should equal (expected._3 +- 1e-12)
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
        case Some(res) => res._3 should equal (expected._3 +- 1e-12)
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
        case Some(res) => res._3 should equal (expected._3 +- 1e-6)
        case _ => fail("expected result not found")
      }
    }
    results.map(_._3).sum should equal (1.0 +- 1e-12)
  }

  /*"SquaredEuclideanDistance" should "return the squared euclidean distance for all the datapoints" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val y = env.fromCollection(TsneHelpersTestSuite.yInit)
    val results = computeDistances(y, Tsne.getMetric("sqeucledian")).map(t => (t._1, t._2, t._3)).collect()
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
  }*/

  "centerEmbedding" should "compute the centered embedding as LabeledVectors" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val nComponents = 2

    val embeddingSeq = TsneHelpersTestSuite.centeringInput.map( g => {

      val lastGradient = DenseVector.fill(nComponents, 0.0)
      val gains = DenseVector.fill(nComponents, 1.0)

      (g._1, g._2, lastGradient.toVector, gains.toVector)
    })

    val embedding = env.fromCollection(embeddingSeq)

    val results = centerEmbedding(embedding).map(x => (x._1, x._2)).collect()

    val expectedResults = TsneHelpersTestSuite.centeringResults

    results.size should equal (expectedResults.size)
    for (expected <- expectedResults) {
      val result = results.find(x => x._1 == expected._1)
      result match {
        case Some(res) => res._2 should equal (expected._2)
        case _ => fail("expected result not found")
      }
    }
  }

  "Gradient" should "return the gradient" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val jointDistribution = env.fromCollection(TsneHelpersTestSuite.denseJointProbabilitiesResults)
    val embedding = env.fromCollection(TsneHelpersTestSuite.initialEmbedding)
    // the result of setting theta to zero is the same as not using a quad-tree at all
    val theta = 0.0

    val grad = embedding.iterate(1) {
      currentEmbedding =>
        gradient(jointDistribution, currentEmbedding, Tsne.getMetric("sqeucledian"), theta)
    }

    val results = grad.collect()

    val expectedResults = TsneHelpersTestSuite.denseGradientResults
    
    results.size should equal (expectedResults.size)
    for (expected <- expectedResults) {
      val result = results.find(x => x._1 == expected._1)
      result match {
        case Some(res) =>
          for (i <- 0 until res._2.size){
            res._2(i) should equal (expected._2(i) +- 1e-12)
          }
        case _ => fail("expected result not found")
      }
    }
  }

  "initWorkingSet" should "randomly initialize the embedding and initialize gradient and gains to zero" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val randomState = 0
    val nComponents = 2

    val zeroVector = DenseVector.fill(nComponents, 0.0)
    val oneVector = DenseVector.fill(nComponents, 1.0)

    val input = Tsne
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

      (g._1, g._2, lastGradient.toVector, gains.toVector)
    })

    val workingSet = env.fromCollection(workingSetSeq)

    val results = updateEmbedding(gradient, workingSet, minGain, momentum, learningRate)
      .map(x => (x._1, x._2)).collect()

    val expectedResults = TsneHelpersTestSuite.updatedEmbeddingResults

    results.size should equal (expectedResults.size)
    for (expected <- expectedResults) {
      val result = results.find(x => x._1 == expected._1)
      result match {
        case Some(res) =>
          for (i <- 0 until res._2.size){
            res._2(i) should equal (expected._2(i) +- 1e-9)
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
    val metric = Tsne.getMetric("sqeucledian")
    // the result of setting theta to zero is the same as not using a quad-tree at all
    val theta = 0.0

    val jointDistribution = env.fromCollection(TsneHelpersTestSuite.denseJointProbabilitiesResults)

    val initialEmbeddingSeq = TsneHelpersTestSuite.initialEmbedding

    val workingSetSeq = initialEmbeddingSeq.map( g => {

      val lastGradient = breeze.linalg.DenseVector.fill(nComponents, 0.0)
      val gains = breeze.linalg.DenseVector.fill(nComponents, 1.0)

      (g._1, g._2, lastGradient.toVector, gains.toVector)
    })

    val workingSet = env.fromCollection(workingSetSeq)

    val results = iterationComputation(iterations, momentum, workingSet, jointDistribution, metric, learningRate, theta)
      .map(x => (x._1, x._2)).collect()

    val expectedResults = TsneHelpersTestSuite.updatedAndCentredEmbeddingResults

    results.size should equal (expectedResults.size)
    for (expected <- expectedResults) {
      val result = results.find(x => x._1 == expected._1)
      result match {
        case Some(res) =>
          for (i <- 0 until res._2.size){
            res._2(i) should equal (expected._2(i) +- 1e-9)
          }
        case _ => fail("expected result not found")
      }
    }
  }
}

object TsneHelpersTestSuite {
  val knnInput: Seq[(Int, Vector[Double])] = List(
    (0, SparseVector(4)(0 -> 0.0, 1 -> 0.0, 2 -> 0.0, 3 -> 0.0)),
    (1, SparseVector(4)(0 -> 1.0, 1 -> 1.0, 2 -> 1.0, 3 -> 1.0)),
    (2, SparseVector(4)(0 -> 2.0, 1 -> 2.0, 2 -> 2.0, 3 -> 2.0)),
    (3, SparseVector(4)(0 -> 3.0, 1 -> 3.0, 2 -> 3.0, 3 -> 3.0)),
    (4, SparseVector(4)(0 -> 4.0, 1 -> 4.0, 2 -> 4.0, 3 -> 4.0)),
    (5, SparseVector(4)(0 -> 5.0, 1 -> 5.0, 2 -> 5.0, 3 -> 5.0)),
    (6, SparseVector(4)(0 -> 6.0, 1 -> 6.0, 2 -> 6.0, 3 -> 6.0)),
    (7, SparseVector(4)(0 -> 7.0, 1 -> 7.0, 2 -> 7.0, 3 -> 7.0)),
    (8, SparseVector(4)(0 -> 8.0, 1 -> 8.0, 2 -> 8.0, 3 -> 8.0))
  ).toSeq

  // result for k = 2
  val knnResults: Seq[(Int, Int, Double)] = List(
    (0, 1, 4.0), (0, 2, 16.0), (1, 2, 4.0), (1, 0, 4.0), (2, 3, 4.0), (2, 1, 4.0),
    (3, 4, 4.0), (3, 2, 4.0), (4, 5, 4.0), (4, 3, 4.0), (5, 6, 4.0), (5, 4, 4.0),
    (6, 7, 4.0), (6, 5, 4.0), (7, 8, 4.0), (7, 6, 4.0), (8, 7, 4.0), (8, 6, 16.0)
  ).toSeq

  // Python implementation van der Maaten, perplexity: 2.0, nComponents: 2, earlyExaggeration:

  val densePairwiseAffinitiesResults: Seq[(Int, Int, Double)] = List(
    (0, 1, 2.370974987703e-02), (0, 2, 5.153826240184e-05), (0, 3, 1.945495759780e-02), (0, 4, 8.216433537309e-04), (0, 5, 4.872518553230e-03), (0, 6, 7.036533081247e-02), (0, 7, 8.338103510412e-01), (0, 8, 4.291578136374e-02), (0, 9, 3.998129138403e-03), (1, 0, 7.724806516559e-01), (1, 2, 1.365877675224e-09), (1, 3, 3.248251955497e-05), (1, 4, 5.336682282239e-04), (1, 5, 1.440504524837e-01), (1, 6, 1.402694896096e-05), (1, 7, 8.229819286616e-02), (1, 8, 5.685570334838e-05), (1, 9, 5.336682282239e-04), (2, 0, 1.898500331402e-03), (2, 1, 8.687658519724e-04), (2, 3, 5.062254851994e-02), (2, 4, 1.449209955883e-02), (2, 5, 3.548272991202e-03), (2, 6, 2.928791467503e-02), (2, 7, 3.166935688445e-02), (2, 8, 2.316520085169e-02), (2, 9, 8.444473403355e-01), (3, 0, 2.390453996554e-04), (3, 1, 1.395251795149e-08), (3, 2, 1.371014800172e-06), (3, 4, 2.369600685486e-03), (3, 5, 1.347198827355e-04), (3, 6, 3.128918783462e-02), (3, 7, 2.411490824021e-05), (3, 8, 7.330981240389e-01), (3, 9, 2.328438222830e-01), (4, 0, 1.019336232626e-04), (4, 1, 2.549661856719e-04), (4, 2, 8.105432254715e-05), (4, 3, 9.980288326134e-03), (4, 5, 5.017868132870e-03), (4, 6, 1.751495817431e-01), (4, 7, 1.008626268887e-03), (4, 8, 7.770114760492e-01), (4, 9, 3.139420534839e-02), (5, 0, 9.336380191993e-06), (5, 1, 1.005438326603e-03), (5, 2, 1.769727010776e-13), (5, 3, 2.563249124685e-03), (5, 4, 1.665948534565e-02), (5, 6, 1.977751128205e-10), (5, 7, 2.760372711471e-01), (5, 8, 1.333923297742e-08), (5, 9, 7.037252061386e-01), (6, 0, 2.657830365339e-02), (6, 1, 4.705122579295e-03), (6, 2, 7.129194416069e-03), (6, 3, 3.052697452006e-02), (6, 4, 5.312632631673e-02), (6, 5, 4.705122579295e-03), (6, 7, 8.775567604535e-03), (6, 8, 8.480859960929e-01), (6, 9, 1.636739223770e-02), (7, 0, 8.439216060511e-01), (7, 1, 2.173031429002e-02), (7, 2, 4.863220572021e-03), (7, 3, 1.840043131472e-02), (7, 4, 1.319325614475e-02), (7, 5, 6.961968258558e-02), (7, 6, 7.370873853279e-03), (7, 8, 2.500183873820e-03), (7, 9, 1.840043131472e-02), (8, 0, 1.718948684906e-02), (8, 1, 3.907353474692e-03), (8, 2, 4.531278151932e-03), (8, 3, 3.605396373725e-02), (8, 4, 7.022214844512e-02), (8, 5, 4.531278151932e-03), (8, 6, 8.397228403177e-01), (8, 7, 3.907353474692e-03), (8, 9, 1.993429739760e-02), (9, 0, 4.341453089465e-07), (9, 1, 2.423850003604e-08), (9, 2, 9.192527632300e-02), (9, 3, 8.003537295921e-01), (9, 4, 9.192527632300e-02), (9, 5, 5.132223630089e-03), (9, 6, 9.710748028923e-05), (9, 7, 7.776147410114e-06), (9, 8, 1.055815212027e-02)
  ).toSeq

  val denseJointProbabilitiesResults: Seq[(Int, Int, Double)] = List(
    (0, 1, 3.980952007665e-02), (0, 2, 9.750192969019e-05), (0, 3, 9.847001498727e-04), (0, 4, 4.617884884968e-05), (0, 5, 2.440927466711e-04), (0, 6, 4.847181723293e-03), (0, 7, 8.388659785461e-02), (0, 8, 3.005263410640e-03), (0, 9, 1.999281641856e-04), (1, 0, 3.980952007665e-02), (1, 2, 4.343836089250e-05), (1, 3, 1.624823603646e-06), (1, 4, 3.943172069479e-05), (1, 5, 7.252794540517e-03), (1, 6, 2.359574764128e-04), (1, 7, 5.201425357809e-03), (1, 8, 1.982104589020e-04), (1, 9, 2.668462333619e-05), (2, 0, 9.750192969019e-05), (2, 1, 4.343836089250e-05), (2, 3, 2.531195976737e-03), (2, 4, 7.286576940686e-04), (2, 5, 1.774136495689e-04), (2, 6, 1.820855454555e-03), (2, 7, 1.826628872824e-03), (2, 8, 1.384823950181e-03), (2, 9, 4.681863083292e-02), (3, 0, 9.847001498727e-04), (3, 1, 1.624823603646e-06), (3, 2, 2.531195976737e-03), (3, 4, 6.174944505810e-04), (3, 5, 1.348984503710e-04), (3, 6, 3.090808117734e-03), (3, 7, 9.212273111480e-04), (3, 8, 3.845760438881e-02), (3, 9, 5.165987759376e-02), (4, 0, 4.617884884968e-05), (4, 1, 3.943172069479e-05), (4, 2, 7.286576940686e-04), (4, 3, 6.174944505810e-04), (4, 5, 1.083867673926e-03), (4, 6, 1.141379540299e-02), (4, 7, 7.100941206818e-04), (4, 8, 4.236168122471e-02), (4, 9, 6.165974083570e-03), (5, 0, 2.440927466711e-04), (5, 1, 7.252794540517e-03), (5, 2, 1.774136495689e-04), (5, 3, 1.348984503710e-04), (5, 4, 1.083867673926e-03), (5, 6, 2.352561388535e-04), (5, 7, 1.728284768663e-02), (5, 8, 2.265645745583e-04), (5, 9, 3.544287148844e-02), (6, 0, 4.847181723293e-03), (6, 1, 2.359574764128e-04), (6, 2, 1.820855454555e-03), (6, 3, 3.090808117734e-03), (6, 4, 1.141379540299e-02), (6, 5, 2.352561388535e-04), (6, 7, 8.073220728907e-04), (6, 8, 8.439044182053e-02), (6, 9, 8.232249858994e-04), (7, 0, 8.388659785461e-02), (7, 1, 5.201425357809e-03), (7, 2, 1.826628872824e-03), (7, 3, 9.212273111480e-04), (7, 4, 7.100941206818e-04), (7, 5, 1.728284768663e-02), (7, 6, 8.073220728907e-04), (7, 8, 3.203768674256e-04), (7, 9, 9.204103731065e-04), (8, 0, 3.005263410640e-03), (8, 1, 1.982104589020e-04), (8, 2, 1.384823950181e-03), (8, 3, 3.845760438881e-02), (8, 4, 4.236168122471e-02), (8, 5, 2.265645745583e-04), (8, 6, 8.439044182053e-02), (8, 7, 3.203768674256e-04), (8, 9, 1.524622475894e-03), (9, 0, 1.999281641856e-04), (9, 1, 2.668462333619e-05), (9, 2, 4.681863083292e-02), (9, 3, 5.165987759376e-02), (9, 4, 6.165974083570e-03), (9, 5, 3.544287148844e-02), (9, 6, 8.232249858994e-04), (9, 7, 9.204103731065e-04), (9, 8, 1.524622475894e-03)
  ).toSeq

  val denseSumQ = 3.365625463923e+01

  val denseUnnormLowDimAffinitiesResults: Seq[(Int, Int, Double)] = List(
    (0, 1, 1.997990966178e-01), (0, 2, 3.438741251510e-01), (0, 3, 5.084645547121e-01), (0, 4, 2.228754579760e-01), (0, 5, 2.111669473429e-01), (0, 6, 4.799407190805e-01), (0, 7, 3.639911864625e-01), (0, 8, 6.947858598830e-01), (0, 9, 2.137434062812e-01), (1, 0, 1.997990966178e-01), (1, 2, 8.232738803226e-02), (1, 3, 1.487280153275e-01), (1, 4, 1.811394269082e-01), (1, 5, 4.318749070054e-01), (1, 6, 1.805549827630e-01), (1, 7, 2.031044587733e-01), (1, 8, 1.379549302261e-01), (1, 9, 9.072703100166e-02), (2, 0, 3.438741251510e-01), (2, 1, 8.232738803226e-02), (2, 3, 3.962129377163e-01), (2, 4, 1.468393804551e-01), (2, 5, 1.011844233815e-01), (2, 6, 2.913681119953e-01), (2, 7, 2.107258420207e-01), (2, 8, 5.761512990942e-01), (2, 9, 2.914080750559e-01), (3, 0, 5.084645547121e-01), (3, 1, 1.487280153275e-01), (3, 2, 3.962129377163e-01), (3, 4, 4.123285538511e-01), (3, 5, 2.365319708095e-01), (3, 6, 9.006682201122e-01), (3, 7, 6.704571955142e-01), (3, 8, 7.699293990309e-01), (3, 9, 5.264164083150e-01), (4, 0, 2.228754579760e-01), (4, 1, 1.811394269082e-01), (4, 2, 1.468393804551e-01), (4, 3, 4.123285538511e-01), (4, 5, 4.650305789265e-01), (4, 6, 5.463238327253e-01), (4, 7, 7.661566330118e-01), (4, 8, 2.544194531822e-01), (4, 9, 3.606532994232e-01), (5, 0, 2.111669473429e-01), (5, 1, 4.318749070054e-01), (5, 2, 1.011844233815e-01), (5, 3, 2.365319708095e-01), (5, 4, 4.650305789265e-01), (5, 6, 3.168065370999e-01), (5, 7, 4.263239066536e-01), (5, 8, 1.793300697535e-01), (5, 9, 1.573034975607e-01), (6, 0, 4.799407190805e-01), (6, 1, 1.805549827630e-01), (6, 2, 2.913681119953e-01), (6, 3, 9.006682201122e-01), (6, 4, 5.463238327253e-01), (6, 5, 3.168065370999e-01), (6, 7, 8.729481827509e-01), (6, 8, 6.082097572678e-01), (6, 9, 4.645101020686e-01), (7, 0, 3.639911864625e-01), (7, 1, 2.031044587733e-01), (7, 2, 2.107258420207e-01), (7, 3, 6.704571955142e-01), (7, 4, 7.661566330118e-01), (7, 5, 4.263239066536e-01), (7, 6, 8.729481827509e-01), (7, 8, 4.178341871825e-01), (7, 9, 4.118776956674e-01), (8, 0, 6.947858598830e-01), (8, 1, 1.379549302261e-01), (8, 2, 5.761512990942e-01), (8, 3, 7.699293990309e-01), (8, 4, 2.544194531822e-01), (8, 5, 1.793300697535e-01), (8, 6, 6.082097572678e-01), (8, 7, 4.178341871825e-01), (8, 9, 3.551252754453e-01), (9, 0, 2.137434062812e-01), (9, 1, 9.072703100166e-02), (9, 2, 2.914080750559e-01), (9, 3, 5.264164083150e-01), (9, 4, 3.606532994232e-01), (9, 5, 1.573034975607e-01), (9, 6, 4.645101020686e-01), (9, 7, 4.118776956674e-01), (9, 8, 3.551252754453e-01)
  ).toSeq

  val initialEmbedding: Seq[(Int, Vector[Double])] = List(
    (0, DenseVector(1.764052345968e+00, 4.001572083672e-01)),
    (1, DenseVector(9.787379841057e-01, 2.240893199201e+00)),
    (2, DenseVector(1.867557990150e+00, -9.772778798764e-01)),
    (3, DenseVector(9.500884175256e-01, -1.513572082977e-01)),
    (4, DenseVector(-1.032188517936e-01, 4.105985019384e-01)),
    (5, DenseVector(1.440435711609e-01, 1.454273506963e+00)),
    (6, DenseVector(7.610377251470e-01, 1.216750164928e-01)),
    (7, DenseVector(4.438632327454e-01, 3.336743273743e-01)),
    (8, DenseVector(1.494079073158e+00, -2.051582637658e-01)),
    (9, DenseVector(3.130677016509e-01, -8.540957393017e-01))
  ).toSeq

  val denseGradientResults: Seq[(Int, Vector[Double])] = List(
    (0, DenseVector(2.039671351114e-02, -2.841077532513e-02)),
    (1, DenseVector(-8.392113494907e-03, 2.231906609600e-03)),
    (2, DenseVector(4.925836551601e-03, 1.893423603195e-02)),
    (3, DenseVector(-2.069923187117e-03, 3.379074534054e-02)),
    (4, DenseVector(9.411374074020e-03, 5.816153185684e-03)),
    (5, DenseVector(6.038471365385e-03, -9.556999264301e-04)),
    (6, DenseVector(-3.232836693981e-02, 1.179730064230e-02)),
    (7, DenseVector(-2.424283588089e-02, -2.266466720419e-02)),
    (8, DenseVector(4.632554267724e-02, -1.538485019566e-02)),
    (9, DenseVector(-2.006469867665e-02, -5.154349158673e-03))
  ).toSeq

  val gradientWithMomentumAndGainResults: Seq[(Int, Vector[Double])] = List(
    (0, DenseVector(-7.342816864009e+00, 6.818586078031e+00)),
    (1, DenseVector(2.014107238778e+00, -8.034863794560e-01)),
    (2, DenseVector(-1.773301158576e+00, -6.816324971503e+00)),
    (3, DenseVector(4.967815649080e-01, -1.216466832260e+01)),
    (4, DenseVector(-3.388094666647e+00, -2.093815146846e+00)),
    (5, DenseVector(-2.173849691539e+00, 2.293679823432e-01)),
    (6, DenseVector(7.758808065556e+00, -4.247028231228e+00)),
    (7, DenseVector(5.818280611414e+00, 5.439520129006e+00)),
    (8, DenseVector(-1.667719536380e+01, 3.692364046958e+00)),
    (9, DenseVector(4.815527682396e+00, 1.237043798081e+00))
  ).toSeq

  val updatedGainsResults: Seq[(Int, Vector[Double])] = List(
    (0, DenseVector(1.200000000000e+00, 8.000000000000e-01)),
    (1, DenseVector(8.000000000000e-01, 1.200000000000e+00)),
    (2, DenseVector(1.200000000000e+00, 1.200000000000e+00)),
    (3, DenseVector(8.000000000000e-01, 1.200000000000e+00)),
    (4, DenseVector(1.200000000000e+00, 1.200000000000e+00)),
    (5, DenseVector(1.200000000000e+00, 8.000000000000e-01)),
    (6, DenseVector(8.000000000000e-01, 1.200000000000e+00)),
    (7, DenseVector(8.000000000000e-01, 8.000000000000e-01)),
    (8, DenseVector(1.200000000000e+00, 8.000000000000e-01)),
    (9, DenseVector(8.000000000000e-01, 8.000000000000e-01))
  ).toSeq

  val updatedEmbeddingResults: Seq[(Int, Vector[Double])] = List(
    (0, DenseVector(-5.578764518042e+00, 7.218743286398e+00)),
    (1, DenseVector(2.992845222883e+00, 1.437406819745e+00)),
    (2, DenseVector(9.425683157355e-02, -7.793602851380e+00)),
    (3, DenseVector(1.446869982434e+00, -1.231602553089e+01)),
    (4, DenseVector(-3.491313518441e+00, -1.683216644908e+00)),
    (5, DenseVector(-2.029806120378e+00, 1.683641489306e+00)),
    (6, DenseVector(8.519845790703e+00, -4.125353214736e+00)),
    (7, DenseVector(6.262143844159e+00, 5.773194456381e+00)),
    (8, DenseVector(-1.518311629065e+01, 3.487205783192e+00)),
    (9, DenseVector(5.128595384047e+00, 3.829480587797e-01))
  ).toSeq

  val updatedAndCentredEmbeddingResults: Seq[(Int, Vector[Double])] = List(
    (0, DenseVector(-5.394920178871e+00, 7.812249121209e+00)),
    (1, DenseVector(3.176689562054e+00, 2.030912654557e+00)),
    (2, DenseVector(2.781011707444e-01, -7.200097016568e+00)),
    (3, DenseVector(1.630714321604e+00, -1.172251969608e+01)),
    (4, DenseVector(-3.307469179270e+00, -1.089710810096e+00)),
    (5, DenseVector(-1.845961781207e+00, 2.277147324118e+00)),
    (6, DenseVector(8.703690129873e+00, -3.531847379924e+00)),
    (7, DenseVector(6.445988183330e+00, 6.366700291192e+00)),
    (8, DenseVector(-1.499927195148e+01, 4.080711618003e+00)),
    (9, DenseVector(5.312439723218e+00, 9.764538935911e-01))
  ).toSeq

  val centeringInput: Seq[(Int, Vector[Double])] = List(
    (0, SparseVector(2)(0 -> -2.0, 1 -> 4.0)),
    (1, SparseVector(2)(0 -> -6.0, 1 -> 4.0)),
    (2, SparseVector(2)(0 -> 2.0, 1 -> -8.0))
  ).toSeq

  val centeringResults: Seq[(Int, Vector[Double])] = List(
    (0, SparseVector(2)(0 -> 0.0, 1 -> 4.0)),
    (1, SparseVector(2)(0 -> -4.0, 1 -> 4.0)),
    (2, SparseVector(2)(0 -> 4.0, 1 -> -8.0))
  ).toSeq

  // C++ implementation

  val sparsePairwiseAffinitiesResults: Seq[(Int, Int, Double)] = List(
    (0, 5, 0.999490),(0, 4, 0.000000),(1, 5, 0.999490),(1, 6, 0.000000),
    (2, 3, 0.999490),(2, 4, 0.000000),(3, 4, 0.499252),(3, 2, 0.499252),
    (4, 3, 0.499252),(4, 5, 0.499252),(5, 6, 0.499252),(5, 4, 0.499252),
    (6, 5, 0.499252),(6, 7, 0.499252),(7, 9, 0.999490),(7, 6, 0.000000),
    (8, 9, 0.999490),(8, 7, 0.000000),(9, 8, 0.999490),(9, 7, 0.000000),
    (10, 11, 0.999490),(10, 8, 0.000000),(11, 10, 0.999490),(11, 8, 0.000000)
  ).toSeq

  val sparseJointProbabilitiesResults: Seq[(Int, Int, Double)] = List(
    (0, 5, 0.041680),(0, 4, 0.000000),(1, 5, 0.041680),(1, 6, 0.000000),
    (2, 3, 0.062500),(2, 4, 0.000000),(3, 2, 0.062500),(3, 4, 0.041639),
    (4, 0, 0.000000),(4, 2, 0.000000),(4, 3, 0.041639),(4, 5, 0.041639),
    (5, 0, 0.041680),(5, 1, 0.041680),(5, 4, 0.041639),(5, 6, 0.041639),
    (6, 1, 0.000000),(6, 5, 0.041639),(6, 7, 0.020820),(7, 6, 0.020820),
    (7, 9, 0.041680),(7, 8, 0.000000),(8, 9, 0.083361),(8, 7, 0.000000),
    (8, 10, 0.000000),(8, 11, 0.000000),(9, 7, 0.041680),(9, 8, 0.083361),
    (10, 11, 0.083361),(10, 8, 0.000000),(11, 10, 0.083361),(11, 8, 0.000000)
  )
}
