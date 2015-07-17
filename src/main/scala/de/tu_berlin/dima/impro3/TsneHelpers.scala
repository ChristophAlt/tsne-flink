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

import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.api.common.operators.Order
import org.apache.flink.api.scala._
import org.apache.flink.configuration.Configuration
import org.apache.flink.ml._
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.Breeze._
import org.apache.flink.ml.math.{DenseVector, Vector}
import org.apache.flink.ml.metrics.distances.DistanceMetric
import org.apache.flink.util.Collector

import scala.collection.JavaConverters._
import scala.math._
import scala.util.Random


object TsneHelpers {

  //============================= TSNE steps ===============================================//

  def kNearestNeighbors(input: DataSet[LabeledVector], k: Int, metric: DistanceMetric):
  DataSet[(Long, Long, Double)] = {
    // compute k nearest neighbors for all points
    input
      .cross(input) {
      (v1, v2) =>
        //   i         j     /----------------- d ----------------\
        (v1.label.toLong, v2.label.toLong, metric.distance(v1.vector, v2.vector))
    }
      // remove distances == 0
      .filter(x => x._1 != x._2)
      // group by i
      .groupBy(_._1)
      // sort
      .sortGroup(_._3, Order.ASCENDING)
      // either take the n nearest neighbors or take the 3u nearest neighbors by default
      .first(k)
  }

  def pairwiseAffinities(input: DataSet[(Long, Long, Double)], perplexity: Double):
  DataSet[(Long, Long, Double)] = {
    // compute pairwise affinities p_j|i
    input
      // group on i
      .groupBy(_._1)
      // compute pairwise affinities for each point i
      // binary search for sigma_i and the result is p_j|i
      .reduceGroup {
      (knn, affinities: Collector[(Long, Long, Double)]) =>
        val knnSeq = knn.toSeq
        // do a binary search to find sigma_i resulting in given perplexity
        // return pairwise affinities
        val pwAffinities = binarySearch(knnSeq, perplexity)
        for (p <- pwAffinities) {
          affinities.collect(p)
        }
    }
  }

  def jointDistribution(input: DataSet[(Long, Long, Double)]): DataSet[(Long, Long, Double)] = {

    val inputTransposed =
      input.map(x => (x._2, x._1, x._3))

    val jointDistribution =
      input.union(inputTransposed).groupBy(0, 1).reduce((x, y) => (x._1, x._2, x._3 + y._3))

    // collect the sum over the joint distribution for normalization
    val sumP = jointDistribution.sum(2).map(x => max(x._3, Double.MinValue))

    jointDistribution.mapWithBcVariable(sumP) { (p, sumP) => (p._1, p._2, max(p._3 / sumP, Double.MinValue)) }
  }

  def initWorkingSet(input: DataSet[LabeledVector], nComponents: Int, randomState: Int): DataSet[(Double, Vector, Vector, Vector)] = {
    // init Y (embedding) by sampling from N(0, I)
    input
      .map(new RichMapFunction[LabeledVector, (Double, Vector, Vector, Vector)] {
      private var gaussian: Random = null

      override def open(parameters: Configuration) {
        gaussian = new Random(randomState)
      }

      def map(inp: LabeledVector): (Double, Vector, Vector, Vector) = {

        val y = breeze.linalg.DenseVector.rand[Double](nComponents)
        val lastGradient = breeze.linalg.DenseVector.fill(nComponents, 0.0)
        val gains = breeze.linalg.DenseVector.fill(nComponents, 1.0)
        
        (inp.label, y.fromBreeze, lastGradient.fromBreeze, gains.fromBreeze)
      }
    })
  }

  def computeDistances(points: DataSet[LabeledVector], metric: DistanceMetric):
  DataSet[(Long, Long, Double, Vector)] = {
    val distances = points
      .cross(points) {
      (e1, e2) =>
        //   i         j      /----------------- d ---------------\ /---------- difference vector----------\
        (e1.label.toLong, e2.label.toLong, metric.distance(e1.vector, e2.vector), (e1.vector.asBreeze - e2.vector.asBreeze).fromBreeze)
    } // remove distances == 0
      .filter(x => x._1 != x._2)

    distances
  }

  def sumLowDimAffinities(embedding: DataSet[LabeledVector], metric: DistanceMetric): DataSet[Double] = {
    embedding
      .map(new RichMapFunction[LabeledVector, Double] {
      private var embedding: Traversable[LabeledVector] = null

      override def open(parameters: Configuration) {
        embedding = getRuntimeContext
          .getBroadcastVariable[LabeledVector]("embedding").asScala
      }

      def map(vector: LabeledVector): Double = {
        val leftVector = vector.vector
        val index = vector.label
        embedding
          .map(rightVector => {
            if (index != rightVector.label) {
              1 / (1 + metric.distance(leftVector, rightVector.vector))
            } else {
              0.0
            }
        }).sum
      }
    }).withBroadcastSet(embedding, "embedding")
    .reduce((x, y) => x + y)
  }

  def gradient(highDimAffinities: DataSet[(Long, Long, Double)], embedding: DataSet[LabeledVector],
               metric: DistanceMetric, sumOverAllAffinities: DataSet[Double]):
    DataSet[LabeledVector] = {
    // this is not the optimized version
    // compute attracting forces
    val attrForces = highDimAffinities
      .map(new RichMapFunction[(Long, Long, Double), (Long, Vector)] {
      private var embedding: Map[Long, Vector] = null

      override def open(parameters: Configuration) {
        embedding = getRuntimeContext.getBroadcastVariable[LabeledVector]("embedding")
          .asScala.map(x => x.label.toLong -> x.vector).toMap
      }

      def map(pij: (Long, Long, Double)): (Long, Vector) = {
        val i = pij._1
        val j = pij._2
        val p = pij._3

        val partialGradient = (p / (1 + metric.distance(embedding(i), embedding(j))) *
          (embedding(i).asBreeze - embedding(j).asBreeze)).fromBreeze

        (i, partialGradient)
      }
    }).withBroadcastSet(embedding, "embedding")
      .groupBy(_._1).reduce((v1, v2) => (v1._1, (v1._2.asBreeze + v2._2.asBreeze).fromBreeze))

    // compute repulsive forces
    val repForces = embedding
      .map(new RichMapFunction[LabeledVector, (Long, Vector)] {
      private var embedding: Traversable[LabeledVector] = null
      private var sumQ: Double = 0.0

      override def open(parameters: Configuration) {
        embedding = getRuntimeContext.getBroadcastVariable[LabeledVector]("embedding").asScala
        sumQ = getRuntimeContext.getBroadcastVariable[Double]("sumQ").get(0)
      }

      def map(vector: LabeledVector): (Long, Vector) = {
        val leftVector = vector.vector
        val index = vector.label.toLong
        val sumVector = embedding
          .map(rightVector => {
            val Z = 1 / (1 + metric.distance(leftVector, rightVector.vector))
            val q = Z / sumQ
            q * Z * (leftVector.asBreeze - rightVector.vector.asBreeze)
        }).reduce((v1, v2) => v1 + v2)

        (index, sumVector.fromBreeze)
      }
    }).withBroadcastSet(embedding, "embedding")
      .withBroadcastSet(sumOverAllAffinities, "sumQ")
      .groupBy(_._1).reduce((v1, v2) => (v1._1, (v1._2.asBreeze + v2._2.asBreeze).fromBreeze))

    // put everything together
    attrForces.join(repForces).where(0).equalTo(0)((attr, rep) => LabeledVector(attr._1, (attr._2.asBreeze - rep._2.asBreeze).fromBreeze))
  }

  def centerEmbedding(embedding: DataSet[(Double, Vector, Vector, Vector)]): DataSet[(Double, Vector, Vector, Vector)] = {
    // center embedding
    val sumAndCount = embedding.map(x => (x._2, 1)).reduce((x, y) => ((x._1.asBreeze + y._1.asBreeze).fromBreeze, x._2 + y._2))

    embedding.mapWithBcVariable(sumAndCount) {
      (v, sumAndCount) => (v._1, (v._2.asBreeze - (sumAndCount._1.asBreeze :/ sumAndCount._2.toDouble)).fromBreeze, v._3, v._4)
    }
  }

  def centerInput(embedding: DataSet[LabeledVector]): DataSet[LabeledVector] = {
    // center embedding
    val sumAndCount = embedding.map(x => (x.vector.asBreeze, 1)).reduce((x, y) => (x._1 + y._1, x._2 + y._2))


    embedding.mapWithBcVariable(sumAndCount) {
      (lv, sumAndCount) => LabeledVector(lv.label, (lv.vector.asBreeze - (sumAndCount._1 :/ sumAndCount._2.toDouble)).fromBreeze)
    }
  }
  
  def updateEmbedding(gradient: DataSet[LabeledVector], workingSet: DataSet[(Double, Vector, Vector, Vector)], minGain: Double, momentum: Double, learningRate: Double):
  DataSet[(Double, Vector, Vector, Vector)] = {
    gradient.map(t => (t.label, t.vector)).join(workingSet).where(0).equalTo(0) {
      (dY, rest) =>
        val currentEmbedding = rest._2
        val previousGradient = rest._3
        val gain = rest._4
        val currentGradient = dY._2

        val d = currentGradient.size
        val newEmbedding = new Array[Double](d)
        val newGain = new Array[Double](d)
        val newGradient = new Array[Double](d)

        for (i <- 0 until d) {
          if ((currentGradient (i) > 0.0) == (previousGradient(i) > 0.0)) {
            newGain(i) = Math.max(gain(i) * 0.8, minGain)
          } else {
            newGain(i) = Math.max(gain(i) + 0.2, minGain)
          }
          newGradient(i) = momentum * previousGradient(i) - learningRate * newGain(i) * currentGradient(i)
          newEmbedding(i) = newGradient(i) + currentEmbedding(i)
        }
        (dY._1, DenseVector(newEmbedding), DenseVector(newGradient), DenseVector(newGain))
    }
  }
  
  def iterationComputation (iterations: Int, momentum: Double, workingSet: DataSet[(Double, Vector, Vector, Vector)],
                            highdimAffinites: DataSet[(Long, Long, Double)], metric: DistanceMetric,
                            learningRate: Double) = {
    workingSet.iterate(iterations) {
      // (label, embedding, gradient, gains)
      workingSet =>

      val currentEmbedding = workingSet.map(t => LabeledVector(t._1, t._2))
      // compute pairwise differences yi - yj
      val distances = computeDistances(currentEmbedding, metric)
      // Compute Q-matrix and normalization sum
      val sumAffinities = sumLowDimAffinities(currentEmbedding, metric)

      val dY = gradient(highdimAffinites, currentEmbedding, metric, sumAffinities)

      val minGain = 0.01

      val updatedEmbedding = updateEmbedding(dY, workingSet, minGain, momentum, learningRate)

      val centeredEmbedding = centerEmbedding(updatedEmbedding)

      centeredEmbedding
    }
  }

  def optimize(highDimAffinities: DataSet[(Long, Long, Double)], initialWorkingSet: DataSet[(Double, Vector, Vector, Vector)],
               learningRate: Double, iterations: Int, metric: DistanceMetric,
               earlyExaggeration: Double, initialMomentum: Double, finalMomentum: Double):
  DataSet[LabeledVector] = {

    val iterInitMomentumExaggeration = min(iterations, 20)
    val iterExaggeration = min(iterations - iterInitMomentumExaggeration, 101-20)
    val iterWoExaggeration = iterations - iterExaggeration - iterInitMomentumExaggeration

    var embedding: DataSet[(Double, Vector, Vector, Vector)] = null

    // early exaggeration
    val exaggeratedAffinities = highDimAffinities.map(x => (x._1, x._2, x._3 * earlyExaggeration))

    // iterate with initial momentum and exaggerated input
    embedding = iterationComputation(iterInitMomentumExaggeration, initialMomentum,
      initialWorkingSet, exaggeratedAffinities, metric, learningRate)

    if (iterExaggeration > 0) {
      // iterate with final momentum and exaggerated input
      embedding = iterationComputation(iterExaggeration, finalMomentum, embedding, exaggeratedAffinities,
        metric, learningRate)
    }

    // iterate with final momentum and standard input
    if (iterWoExaggeration > 0) {
      embedding = iterationComputation(iterWoExaggeration, finalMomentum, embedding, highDimAffinities,
        metric, learningRate)
    }

    embedding.map(x => LabeledVector(x._1, x._2))
  }

  //============================= binary search ===============================================//

  private def binarySearch(distances: Seq[(Long, Long, Double)], perplexity: Double):
  Seq[(Long, Long, Double)] = {
    // try to approximate beta_i (1/sigma_i**2) so we can compute p_j|i
    // set target to log(perplexity) (entropy) to make computation simpler
    approximateBeta(1.0,
      distances, log(perplexity), Double.NegativeInfinity, Double.NegativeInfinity, Double.PositiveInfinity)
  }

  private def approximateBeta(beta: Double, distances: Seq[(Long, Long, Double)], targetH: Double,
                              previousH: Double, min: Double,
                              max: Double, iterations: Int = 50):
  Seq[(Long, Long, Double)] = {
    // compute the entropy
    val h = computeH(distances, beta)

    // check whether we reached the expected entropy or we're out of iterations
    // in both cases, compute p_j|i
    if (isCloseEnough(h, targetH) || iterations == 0) {
      computeP(distances, beta)
    }
    // otherwise, adapt beta and try again
    else {
      val nextBeta = {
        if (h - targetH > 0) {
          if (max == Double.PositiveInfinity || max == Double.NegativeInfinity) {
            beta * 2
          }
          else {
            (beta + max) / 2
          }
        }
        else {
          if (min == Double.PositiveInfinity || min == Double.NegativeInfinity) {
            beta / 2
          }
          else {
            (beta + min) / 2
          }
        }
      }
      if (h - targetH > 0) {
        // difference is positive -> set lower bound to current guess
        approximateBeta(nextBeta, distances, targetH, h, beta, max, iterations - 1)
      }
      else {
        // difference is negative -> set upper bound to current guess
        approximateBeta(nextBeta, distances, targetH, h, min, beta, iterations - 1)
      }
    }
  }

  private def isCloseEnough(a: Double, b: Double, tol: Double = 1e-5): Boolean = {
    abs(a - b) < tol
  }

  private def computeH(distances: Seq[(Long, Long, Double)], beta: Double = 1.0): Double = {
    //                          d_i      exp(-d_i * beta)
    val p = distances.map(d => (d._3, exp(-d._3 * beta)))
    val sumP = if (p.map(_._2).sum == 0.0) 1e-7 else p.map(_._2).sum
    log(sumP) + beta * p.map(p => p._1 * p._2).sum / sumP
  }

  private def computeP(distances: Seq[(Long, Long, Double)], beta: Double):
  Seq[(Long, Long, Double)] = {
    //                            i     j      exp(-d_i * beta)
    val p = distances.map(d => (d._1, d._2, exp(-d._3 * beta)))
    val sumP = if (p.map(_._3).sum == 0.0) 1e-7 else p.map(_._3).sum
    //            i     j      p_i|j
    p.map(p => (p._1, p._2, p._3 / sumP))
  }
}
