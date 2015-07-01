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
import org.apache.flink.api.java.utils.ParameterTool
import org.apache.flink.api.scala._
import org.apache.flink.ml._
import org.apache.flink.configuration.Configuration
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.{SparseVector, DenseVector}
import org.apache.flink.ml.metrics.distances._
import org.apache.flink.util.Collector
import org.apache.flink.ml.math.Breeze._

import scala.math._
import scala.util.Random


object Tsne {

  def main(args: Array[String]) {
    val parameters = ParameterTool.fromArgs(args)
    val env = ExecutionEnvironment.getExecutionEnvironment

    // get parameters from command line or use default
    val inputPath = parameters.getRequired("input")
    val outputPath = parameters.getRequired("output")

    val dimension = parameters.getRequired("dimension").toInt

    val metric = SquaredEuclideanDistanceMetric()
    val perplexity = parameters.getDouble("perplexity", 30.0)
    val nComponents = parameters.getLong("nComponents", 2).toInt
    //val earlyExaggeration = resultingParameters(EarlyExaggeration)
    val learningRate = parameters.getDouble("learningRate", 0.1)
    val iterations = parameters.getLong("iterations", 1000).toInt

    val randomState = parameters.getLong("randomState", 0).toInt
    val neighbors = parameters.getLong("neighbors", 3*perplexity.toInt)

    // this is for MNIST dataset! change in case of different dataset
    // label,
    val input = env.readCsvFile[(Int, Int, Double)](inputPath, includedFields = Array(1,2,3))
      .groupBy(_._1).reduceGroup(
        elements => {
          val elementsIterable = elements.toIterable
          val entries = elementsIterable.map(x => (x._2, x._3))
          LabeledVector(elementsIterable.head._1.toDouble, SparseVector.fromCOO(dimension, entries))
        })

    val result = computeEmbedding(input, metric, perplexity, nComponents, learningRate, iterations,
      randomState, neighbors)

    result.writeAsCsv(outputPath)

    env.execute("TSNE")
  }

  private def computeEmbedding(input: DataSet[LabeledVector], metric: DistanceMetric,
                               perplexity: Double, nComponents: Int, learningRate: Double,
                               iterations: Int, randomState: Int, neighbors: Int):
  DataSet[LabeledVector] = {

    // center data or do this somewhere before in the pipeline

    // compute k nearest neighbors for all points
    val knn = input
      .cross(input) {
      (v1, v2) =>
        //   i         j     /----------------- d ----------------\
        (v1.label, v2.label, metric.distance(v1.vector, v2.vector))
    }
      // remove distances == 0
      .filter(_._3 != 0)
      // group by i
      .groupBy(_._1)
      // sort
      .sortGroup(_._3, Order.ASCENDING)
      // either take the n nearest neighbors or take the 3u nearest neighbors by default
      .first(
        if (neighbors > 0) {
          neighbors
        }
        else {
          3 * perplexity.toInt
        })

    // compute pairwise affinities p_j|i
    val condProbabilities =
      knn
        // group on i
        .groupBy(_._1)
        // compute pairwise affinities for each point i
        // binary search for sigma_i and the result is p_j|i
        .reduceGroup {
        (knn, affinities: Collector[(Double, Double, Double)]) =>
          val knnSeq = knn.toSeq
          // do a binary search to find sigma_i resulting in given perplexity
          // return pairwise affinities
          val pwAffinities = binarySearch(knnSeq, perplexity)
          for (p <- pwAffinities) {
            affinities.collect(p)
          }
      }

    val jointDistribution =
      condProbabilities
        // join p_i|j and p_j|i to symmetrize the distribution
        .join(condProbabilities).where(0, 1).equalTo(1, 0) {
        (i, j) => (i._1, i._2, i._3 + j._3)
      }

    // collect the sum over the joint distribution for normalization
    val sumP = jointDistribution.sum(2)

    val highDimAffinities = jointDistribution.mapWithBcVariable(sumP){(p, sum) => (p._1, p._2, p._3 / sum._3)}

    // TODO: early exaggeration and early compression goes here

    // init Y (embedding) by sampling from N(0, 10e-4*I)
    val embedding = input
      .map(new RichMapFunction[LabeledVector, LabeledVector] {
      private var gaussian: Random = null
      private val sigma = 10e-2

      override def open(parameters: Configuration) {
        gaussian = new Random(randomState)
      }

      def map(inp: LabeledVector): LabeledVector = {
        // TODO: extend to higher dimensional cases
        //                i                                y0                         y1
        LabeledVector(inp.label, DenseVector(gaussian.nextDouble * sigma, gaussian.nextDouble * sigma))
      }
    })

    val finalEmbedding = embedding.iterate(iterations) {
      currentEmbedding =>

        // compute pairwise differences yi - yj
        val distances = currentEmbedding
          .cross(currentEmbedding) {
          (e1, e2) =>
            //   i         j      /----------------- d ---------------\ /---------- difference vector----------\
            (e1.label, e2.label, metric.distance(e1.vector, e2.vector), e1.vector.asBreeze - e2.vector.asBreeze)
        }// remove distances == 0
          .filter(_._3 != 0)

        // compute low dimensional affinities q_ij
        /*val distances = currentEmbedding
          .cross(currentEmbedding) {
          (e1, e2) =>
            //   i         j     /----------------- d ----------------\
            (e1.label, e2.label, metric.distance(e1.vector, e2.vector))
        }// remove distances == 0
          .filter(_._3 != 0)*/

        // unnormalized q_ij
        val unnormAffinities = distances
          .map { d =>
          // i     j   1 / (1 + dij)
          (d._1, d._2, 1 / (1 + d._3), 1)
        }

        // sum over q_i
        val sumAffinities = unnormAffinities
          .groupBy(_._1)
          //                     i       j     sum(q_i)
          .reduce((a1, a2) => (a1._1, a1._2, a1._3 + a2._3, 1))

        // make affinities a probability distribution by q_ij / sum(qi)
        val lowDimAffinities = unnormAffinities.join(sumAffinities).where(0).equalTo(0) {
          (aff, sum) => (aff._1, aff._2, aff._3 / sum._3)
        }

        // TODO: compute gradient dC / dy_i with Flink SGD
        // this is not the optimized version
        val gradient = highDimAffinities
          .join(lowDimAffinities).where(0, 1).equalTo(0, 1) {
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
          .map(g => LabeledVector(g._1, (4.0 * g._3).fromBreeze))

        // compute new embedding by taking one step in gradient direction
        val newEmbedding = currentEmbedding.join(gradient).where(0).equalTo(0) {
          (c, g) => LabeledVector(c.label, (c.vector.asBreeze - (learningRate * g.vector.asBreeze)).fromBreeze)
        }
        newEmbedding
    }
    finalEmbedding
  }

  private def binarySearch(distances: Seq[(Double, Double, Double)], perplexity: Double):
  Seq[(Double, Double, Double)] = {
    // try to approximate beta_i (1/sigma_i**2) so we can compute p_j|i
    // set target to log(perplexity) (entropy) to make computation simpler
    approximateBeta(1.0,
      distances, log(perplexity), Double.NegativeInfinity, Double.NegativeInfinity, Double.PositiveInfinity)
  }

  private def approximateBeta(beta: Double, distances: Seq[(Double, Double, Double)], targetH: Double,
                              previousH: Double, min: Double,
                              max: Double, iterations: Int = 50):
  Seq[(Double, Double, Double)] = {
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
        if(h - targetH > 0) {
          if (max == Double.PositiveInfinity || max == Double.NegativeInfinity) {
            beta * 2
          }
          else {
            (beta + max) / 2
          }
        }
        else {
          if(min == Double.PositiveInfinity || min == Double.NegativeInfinity) {
            beta / 2
          }
          else {
            (beta + min) / 2
          }
        }
      }
      if(h - targetH > 0) {
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

  private def computeH(distances: Seq[(Double, Double, Double)], beta: Double = 1.0): Double = {
    //                          d_i      exp(-d_i * beta)
    val p = distances.map(d => (d._3, exp(-d._3 * beta)))
    val sumP = p.map(_._2).sum
    log(sumP) + beta * p.map(p => p._1 * p._2).sum / sumP
  }

  private def computeP(distances: Seq[(Double, Double, Double)], beta: Double): Seq[(Double, Double, Double)] = {
    //                            i     j      exp(-d_i * beta)
    val p = distances.map(d => (d._1, d._2, exp(-d._3 * beta)))
    val sumP = p.map(_._3).sum
    //            i     j      p_i|j
    p.map(p => (p._1, p._2, p._3 / sumP))
  }
}
