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
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.{DenseVector, Vector}
import org.apache.flink.ml.metrics.distances.DistanceMetric
import org.apache.flink.util.Collector
import org.apache.flink.ml._
import org.apache.flink.ml.math.Breeze._

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
      .filter(_._3 != 0)
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

  def initWorkingSet(input: DataSet[LabeledVector], dimension: Int, randomState: Int): DataSet[(Double, Vector, Vector, Vector)] = {
    // init Y (embedding) by sampling from N(0, 10e-4*I)
    input
      .map(new RichMapFunction[LabeledVector, (Double, Vector, Vector, Vector)] {
      private var gaussian: Random = null
      private val sigma = 10e-2

      override def open(parameters: Configuration) {
        gaussian = new Random(randomState)
      }

      def map(inp: LabeledVector): (Double, Vector, Vector, Vector) = {
        var yValues = Array.fill(dimension){gaussian.nextDouble * sigma}
        var lastGradientValues = Array.fill(dimension){(0.0).toDouble}
        var gainValues = Array.fill(dimension){(1.0).toDouble}
        
        val y = DenseVector(yValues)
        val lastGradient = DenseVector(lastGradientValues)
        val gains = DenseVector(gainValues)
        
        (inp.label, y, lastGradient, gains)
        
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
      .filter(_._3 != 0) //TODO: check whether this is really right for dense matrices??

    distances
  }

  // Compute Q-matrix and normalization sum
  def computeLowDimAffinities(distances: DataSet[(Long, Long, Double, Vector)]):
    {val q: DataSet[(Long, Long, Double)]; val sumQ: DataSet[Double]} = {
    // unnormalized q_ij
    val unnormAffinities = distances
      .map { d =>
      // i     j   1 / (1 + dij)
      (d._1, d._2, 1 / (1 + d._3))
    }

    // sum over q_i
    /*val sumAffinities = unnormAffinities
      .groupBy(_._1)
      //                     i       j     sum(q_i)
      .reduce((a1, a2) => (a1._1, a1._2, a1._3 + a2._3))*/

    val sumOverAllAffinities = unnormAffinities.sum(2).map(x => x._3)

    /*// make affinities a probability distribution by q_ij / sum(q)
    val lowDimAffinities = unnormAffinities.mapWithBcVariable(sumOverAllAffinities) {
      (q, sumQ) => (q._1, q._2, max(q._3 / sumQ, Double.MinValue))
    }*/

    new {
      val q = unnormAffinities
      val sumQ = sumOverAllAffinities
    }
  }
  
  def calcPQ_unittest(lowDimAffinities: DataSet[(Long, Long, Double)],
               highDimAffinities: DataSet[(Long, Long, Double)], sumOverAllAffinities: DataSet[Double]) = {
        highDimAffinities
        .join(lowDimAffinities).where(0, 1).equalTo(0, 1).mapWithBcVariable(sumOverAllAffinities) {
      //                i           j       (p - (num / sum(num)) * num
      //                                    (p -  q)              * num    
      (pQ, sumQ) => (pQ._1._1, pQ._1._2, pQ._1._3 - (pQ._2._3 / sumQ))
    }
  }

  def gradient(lowDimAffinities: DataSet[(Long, Long, Double)],
               highDimAffinities: DataSet[(Long, Long, Double)], sumOverAllAffinities: DataSet[Double],
               distances: DataSet[(Long, Long, Double, Vector)]): DataSet[LabeledVector] = {
    // this is not the optimized version
    highDimAffinities
      .join(lowDimAffinities).where(0, 1).equalTo(0, 1).mapWithBcVariable(sumOverAllAffinities) {
      //                i           j       (p - (q / sum(q)) * q
      (pQ, sumQ) => (pQ._1._1, pQ._1._2, (pQ._1._3 - (pQ._2._3 / sumQ)) * pQ._2._3)

    }.join(distances).where(0, 1).equalTo(0, 1) {
    //                             ((p -  q)* num) * (yi -yj)      
      (mul, d) => (mul._1, mul._2, (mul._3         * d._4.asBreeze).fromBreeze)
    }.groupBy(_._1).reduce((v1, v2) => (v1._1, v1._2, (v1._3.asBreeze + v2._3.asBreeze).fromBreeze))
      .map(g => LabeledVector(g._1, g._3))
      /*{
        //                      (pij - qij)   * qij
        (h, l) => (h._1, h._2, (h._3 - l._3) * l._3)
      }
      .join(sumAffinities).where(0).equalTo(0) {
      //         (pij - qij)* qij * Z
      (l, z) => (l._1, l._2, l._3 * z._3)
    }
      .join(distances).where(0).equalTo(0) {
      //     (pij - qij)* qij * Z * (yi - yj)
      (l, d) => (l._1, l._2, l._3 * d._4)
    }
      .groupBy(_._1)
      .reduce((g1, g2) => (g1._1, g1._2, g1._3 + g2._3))
      .map(g => LabeledVector(g._1, (4.0 * g._3).fromBreeze))*/
  }

  def centerEmbedding(embedding: DataSet[LabeledVector]): DataSet[LabeledVector] = {
    // center embedding
    val sumAndCount = embedding.map(x => (x.vector.asBreeze, 1)).reduce((x, y) => (x._1 + y._1, x._2 + y._2))


    embedding.mapWithBcVariable(sumAndCount) {
      (lv, sumAndCount) => LabeledVector(lv.label, (lv.vector.asBreeze - (sumAndCount._1 :/ sumAndCount._2.toDouble)).fromBreeze)
    }
  }
  
  
  def enrichGradientByMomentumAndGain(gradient: DataSet[LabeledVector], workingSet: DataSet[(Double, Vector, Vector, Vector)], minGain: Double, momentum: Double, eta: Double):
  DataSet[(Double, Vector, Vector)] = {
    gradient.map(t => (t.label, t.vector)).join(workingSet).where(0).equalTo(0) {
      (dY, rest) =>
        val gain = rest._4
        val lastGradient = rest._3
        val currentGradient = dY._2
        val dimensionality = gain.size

        var newGain = new Array[Double](dimensionality)
        var newGradient = new Array[Double](dimensionality)

        for (i <- 0 until dimensionality) {
          if (currentGradient (i) > 0.0 && lastGradient(i) > 0.0) {
            newGain(i) = Math.max(gain(i) * 0.8, minGain)
          } else {
            newGain(i) = Math.max(gain(i) + 0.2, minGain)
          }
          newGradient(i) = momentum * lastGradient(i) - eta * (newGain(i) * currentGradient(i))
        }
        (dY._1, DenseVector(newGradient), DenseVector(newGain))
    }
  }
  
  def iterationComputation (iterations: Int, momentum: Double, workingSet: DataSet[(Double, Vector, Vector, Vector)], highdimAffinites: DataSet[(Long, Long, Double)], metric: DistanceMetric, learningRate: Double) = {
    workingSet.iterate(iterations) { workingSet =>

      val currentEmbedding = workingSet.map { t => LabeledVector(t._1, t._2) }
      //val lastGradient = workingSet.map { t => (t._1, t._3) }
      //val currentGain = workingSet.map { t => (t._1, t._4) }

      // compute pairwise differences yi - yj
      val distances = computeDistances(currentEmbedding, metric)
      // Compute Q-matrix and normalization sum
      val results = computeLowDimAffinities(distances)

      val lowDimAffinities = results.q
      val sumAffinities = results.sumQ

      val dY = gradient(lowDimAffinities, highdimAffinites, sumAffinities, distances)


      /*
        newGains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
		    newGains[gains < min_gain] = min_gain;
       */

      val minGain = 0.01
      val eta = 500

      val regularizedGradient = enrichGradientByMomentumAndGain(dY, workingSet, minGain, momentum, eta)

      // compute new embedding by taking one step in gradient direction
      val newEmbedding = currentEmbedding.join(regularizedGradient).where(0).equalTo(0) {
        (c, g) =>
          val newY = (c.vector.asBreeze + g._2.asBreeze).fromBreeze
          LabeledVector(c.label, newY)
      }
      val centeredEmbedding = centerEmbedding(newEmbedding)

      val nextIteration = centeredEmbedding.join(regularizedGradient).where(0).equalTo(0) {
        (c, g) =>
          (c.label, c.vector, g._2.asInstanceOf[Vector], g._3.asInstanceOf[Vector])
      }

      nextIteration
    }
  }

  def optimize(highDimAffinities: DataSet[(Long, Long, Double)], initialWorkingSet: DataSet[(Double, Vector, Vector, Vector)],
               learningRate: Double, iterations: Int, metric: DistanceMetric,
               earlyExaggeration: Double, initialMomentum: Double, finalMomentum: Double):
  DataSet[LabeledVector] = {
    

    
    
    /*
    // take 25% of iterations with early exaggeration -> maybe parameter?
    val iterationsEarlyEx = (0.25 * iterations).toInt
    val normalIterations = iterations - iterationsEarlyEx
    //Compute exagggerted Embedding
    val exageratedAffinites = highDimAffinities.map(r => (r._1, r._2, r._3 * earlyExaggeration))
    */
    /*
    val exageratedResults = iterationComputation(iterationsEarlyEx, initialMomentum, initialEmbedding, exageratedAffinites)
    //normal computation
    val normalResults: DataSet[LabeledVector] = iterationComputation(normalIterations, finalMomentum, exageratedResults, highDimAffinities)
    normalResults
    */

    // First Compute with
    /*
    val exagarretedEmbedding: DataSet[LabeledVector] = initialEmbedding.iterate(iterationsEarlyEx) {
      currentEmbedding =>

        // compute pairwise differences yi - yj
        val distances = computeDistances(currentEmbedding, metric)

        // Compute Q-matrix and normalization sum
        val results = computeLowDimAffinities(distances)

        val lowDimAffinities = results.q
        val sumAffinities = results.sumQ

        val dY = gradient(lowDimAffinities, exageratedAffinites, sumAffinities, distances)

        // compute new embedding by taking one step in gradient direction
        val newEmbedding = currentEmbedding.join(dY).where(0).equalTo(0) {
          // TODO: add momentum and gains
          (c, g) => LabeledVector(c.label, (c.vector.asBreeze - (learningRate * g.vector.asBreeze)).fromBreeze)
        }

        centerEmbedding(newEmbedding)
    }

    exagarretedEmbedding.iterate(normalIterations) {
      currentEmbedding =>

        // compute pairwise differences yi - yj
        val distances = computeDistances(currentEmbedding, metric)

        // Compute Q-matrix and normalization sum
        val results = computeLowDimAffinities(distances)

        val lowDimAffinities = results.q
        val sumAffinities = results.sumQ

        val dY = gradient(lowDimAffinities, highDimAffinities, sumAffinities, distances)

        // compute new embedding by taking one step in gradient direction
        val newEmbedding = currentEmbedding.join(dY).where(0).equalTo(0) {
          (c, g) => LabeledVector(c.label, (c.vector.asBreeze - (learningRate * g.vector.asBreeze)).fromBreeze)
        }
        newEmbedding
    }*/

    null: DataSet[LabeledVector]
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
    if (isCloseEnough(h, previousH) || iterations == 0) {
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
