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
import breeze.stats.distributions.Rand
import org.apache.flink.api.common.functions.{RichGroupReduceFunction, RichMapFunction}
import org.apache.flink.api.common.operators.Order
import org.apache.flink.api.scala._
import org.apache.flink.configuration.Configuration
import org.apache.flink.ml._
import org.apache.flink.ml.common.FlinkMLTools
import org.apache.flink.util.Collector

import scala.collection.JavaConverters._


object TsneHelpers {

  //============================= TSNE steps ===============================================//

  def kNearestNeighbors(input: DataSet[(Int, Vector[Double])], k: Int, metric: (Vector[Double], Vector[Double]) => Double):
  DataSet[(Int, Int, Double)] = {
    // compute k nearest neighbors for all points
    input
      .cross(input) {
      (v1, v2) =>
        //   i         j     /----------------- d ----------------\
        (v1._1, v2._1, metric(v1._2, v2._2))
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

  def partitionKnn(input: DataSet[(Int, Vector[Double])], k: Int, metric: (Vector[Double], Vector[Double]) => Double, blocks: Int):
  DataSet[(Int, Int, Double)] = {

    val partitioner = FlinkMLTools.ModuloKeyPartitioner
    val inputSplit = FlinkMLTools.block(input, blocks, Some(partitioner))

    val crossed = inputSplit.cross(inputSplit).mapPartition {
      (iter, out: Collector[(Int, Int, Double)]) => {
        for ((split1, split2) <- iter) {
          for (a <- split1.values; b <- split2.values) {
            if (a._1 != b._1) {
              out.collect(a._1, b._1, metric(a._2, b._2))
            }
          }
        }
      }
    }

    val result = crossed.groupBy(0).sortGroup(2, Order.ASCENDING).reduceGroup {
      (iter, out: Collector[(Int, Int, Double)]) => {
        if (iter.hasNext) {
          for (n <- iter.take(k)) {
            out.collect(n)
          }
        }
      }
    }

    result
  }

  def projectKnn(input: DataSet[(Int, Vector[Double])], k: Int, metric: (Vector[Double], Vector[Double]) => Double, dimension: Int,
                 iterations: Int): DataSet[(Int, Int, Double)] = {

    val randomVectors: Seq[DenseVector[Double]] = for (_ <- 1 until iterations) yield {
      DenseVector.rand[Double](dimension)
    }

    val nnInput = input.map(x => (x._1, x._2, x._2))

    var possibleNeighbors = findPossibleNeighbors(nnInput, k, metric)

    for (randomVector <- randomVectors) {
      val projectedVectors = nnInput.map(x => (x._1, x._2 + randomVector, x._2)).withForwardedFields("_1", "_2->_3")

      possibleNeighbors = possibleNeighbors
        .union(findPossibleNeighbors(projectedVectors, k, metric))
    }

    possibleNeighbors
      .groupBy(x => (x._1, x._2)).reduceGroup {
      (iter, out: Collector[(Int, Int, Vector[Double], Vector[Double])]) => {
        if (iter.hasNext) {
          out.collect(iter.next())
        }
      }
    }.withForwardedFields("_1", "_2", "_3", "_4").groupBy(_._1).reduceGroup {
      (iter, out: Collector[(Int, Int, Double)]) => {
        if (iter.hasNext) {
          val iterSeq = iter.toSeq
          val distances = iterSeq.map(x => (x._1, x._2, metric(x._3, x._4)))
          val kNeighbors = distances.sortBy(_._3).take(k)
          for (nn <- kNeighbors) {
            out.collect(nn)
          }
        }
      }
    }
  }

  private def findPossibleNeighbors(input: DataSet[(Int, Vector[Double], Vector[Double])], k: Int,
                            metric: (Vector[Double], Vector[Double]) => Double): DataSet[(Int, Int, Vector[Double], Vector[Double])] = {

    input.reduceGroup {
      (iter, out: Collector[(Int, Int, Vector[Double], Vector[Double])]) => {
        if (iter.hasNext) {
          val iterSeq = iter.toSeq
          val sortedSeq = iterSeq.sortWith((v1, v2) => !ZOrder.compareByZorder(v1._2, v2._2))

          for (i <- sortedSeq.indices) {
            // take k items to the left
            val left = sortedSeq.slice(i - k, i)
            val right = sortedSeq.slice(i + 1, i + k + 1)
            for (l <- left) {
              out.collect((sortedSeq(i)._1, l._1, sortedSeq(i)._3, l._3))
            }
            for (r <- right) {
              out.collect((sortedSeq(i)._1, r._1, sortedSeq(i)._3, r._3))
            }
          }
        }
      }
    }
  }

  def pairwiseAffinities(input: DataSet[(Int, Int, Double)], perplexity: Double):
  DataSet[(Int, Int, Double)] = {
    // compute pairwise affinities p_j|i
    input
      // group on i
      .groupBy(_._1)
      // compute pairwise affinities for each point i
      // binary search for sigma_i and the result is p_j|i
      .reduceGroup {
      (knn, affinities: Collector[(Int, Int, Double)]) =>
        val knnSeq = knn.toSeq
        // do a binary search to find sigma_i resulting in given perplexity
        // return pairwise affinities
        val pwAffinities = binarySearch(knnSeq, perplexity)
        for (p <- pwAffinities) {
          affinities.collect(p)
        }
    }
  }

  def jointDistribution(input: DataSet[(Int, Int, Double)]): DataSet[(Int, Int, Double)] = {

    val inputTransposed =
      input.map(x => (x._2, x._1, x._3))

    val jointDistrib =
      input.union(inputTransposed).groupBy(0, 1).reduce((x, y) => (x._1, x._2, x._3 + y._3))

    // collect the sum over the joint distribution for normalization
    val sumP = jointDistrib.sum(2).map(x => scala.math.max(x._3, Double.MinValue))

    jointDistrib.mapWithBcVariable(sumP) {
      (p, sumP) => (p._1, p._2, scala.math.max(p._3 / sumP, Double.MinValue))
    }
  }

  def initWorkingSet(input: DataSet[(Int, Vector[Double])], nComponents: Int, randomState: Int):
  DataSet[(Int, Vector[Double], Vector[Double], Vector[Double])] = {
    // init Y (embedding) by sampling from N(0, I)
    input
      .map(new RichMapFunction[(Int, Vector[Double]), (Int, Vector[Double], Vector[Double], Vector[Double])] {
      private var gaussian: Rand[Double] = null

      override def open(parameters: Configuration) {
        gaussian = Rand.gaussian(0, 1e-4)
      }

      def map(inp: (Int, Vector[Double])): (Int, Vector[Double], Vector[Double], Vector[Double]) = {

        val y = DenseVector.rand[Double](nComponents, gaussian)
        val lastGradient = DenseVector.fill(nComponents, 0.0)
        val gains = DenseVector.fill(nComponents, 1.0)
        
        (inp._1, y, lastGradient, gains)
      }
    })
  }

  def gradient(highDimAffinities: DataSet[(Int, Vector[Double])], embedding: DataSet[(Int, Vector[Double])],
               metric: (Vector[Double], Vector[Double]) => Double, theta: Double, dimension:Int, iterOffset: Int=0):
    DataSet[(Int, Vector[Double])] = {

    // find xMin, xMax, yMin and yMax, as well as meanX and meanY
    val boundaryAndMean = embedding.map(x => (x._2(0), x._2(0), x._2(1), x._2(1), x._2(0), x._2(1), 1))
      .reduce((x, y) => (scala.math.min(x._1, y._1), scala.math.max(x._2, y._2), scala.math.min(x._3, y._3), scala.math.max(x._4, y._4), x._5 + y._5, x._6 + y._6, x._7 + y._7))

    // compute repulsive forces
    val tree = embedding
      .reduceGroup(new RichGroupReduceFunction[(Int, Vector[Double]), QuadTree] {
      private var boundaryAndMean: (Double, Double, Double, Double, Double, Double, Int) = null

      override def open(parameters: Configuration) {
        boundaryAndMean = getRuntimeContext
          .getBroadcastVariable[(Double, Double, Double, Double, Double, Double, Int)]("boundaryAndMean").get(0)
      }

      def reduce(embedding: java.lang.Iterable[(Int, Vector[Double])],
                  out: Collector[QuadTree]) = {
        val (minX, maxX, minY, maxY, sumX, sumY, count) = boundaryAndMean

        val meanX = sumX / count
        val meanY = sumY / count
        val tree = QuadTree(None, Cell(meanX, meanY, scala.math.max(maxX - minX, maxY - minY)))

        for (v <- embedding.asScala) {
          tree.insert(v._2)
        }
        out.collect(tree)
      }
    }).withBroadcastSet(boundaryAndMean, "boundaryAndMean")

    // compute repulsive forces
    val repForcesAndSum = embedding
      .map(new RichMapFunction[(Int, Vector[Double]), (Int, Vector[Double], Double)] {
      private var tree: QuadTree = null

      override def open(parameters: Configuration) {
        tree = getRuntimeContext.getBroadcastVariable[QuadTree]("tree").get(0)
      }

      def map(vector: (Int, Vector[Double])): (Int, Vector[Double], Double) = {
        val index = vector._1
        val leftVector = vector._2


        val (repForce, sumQ) = tree.computeRepulsiveForce(leftVector, theta)
        (index, repForce, sumQ)
      }
    }).withBroadcastSet(tree, "tree")

    val sumQ = repForcesAndSum.map(x => x._3).reduce((x, y) => x + y)

    // compute attracting forces
    val attrForces = highDimAffinities
      .map(new RichMapFunction[(Int, Vector[Double]), (Int, Vector[Double])] {
      private var embedding: Map[Int, Vector[Double]] = null
      private val lossAccumulator = new MapAccumulator()
      private var currentIteration = 0
      private var sumQ = 0.0

      override def open(parameters: Configuration) {
        embedding = getRuntimeContext.getBroadcastVariable[(Int, Vector[Double])]("embedding")
          .asScala.map(x => x._1 -> x._2).toMap
        sumQ = getRuntimeContext.getBroadcastVariable[Double]("sumQ").get(0)
        currentIteration = getIterationRuntimeContext.getSuperstepNumber
        getRuntimeContext.addAccumulator("loss", lossAccumulator)
      }

      def map(pi: (Int, Vector[Double])): (Int, Vector[Double]) = {
        val i = pi._1
        val p = pi._2.asInstanceOf[SparseVector[Double]]

        var partialGradient = DenseVector.fill(dimension, 0.0)

        var offset = 0
        while(offset < p.activeSize) {
          val j = p.indexAt(offset)
          val pij = p.valueAt(offset)

          val qij = 1 / (1 + metric(embedding(i), embedding(j)))

          partialGradient += pij * qij * (embedding(i) - embedding(j))

          if ((currentIteration + iterOffset) % 10 == 0) {
            val partialLoss = pij * scala.math.log(pij / (qij / sumQ))
            lossAccumulator.add((currentIteration + iterOffset, partialLoss))
          }

          offset += 1
        }

        (i, partialGradient)
      }
    }).withBroadcastSet(embedding, "embedding")
      .withBroadcastSet(sumQ, "sumQ")
      //.groupBy(_._1).reduce((v1, v2) => (v1._1, v1._2 + v2._2))

    // put everything together
    attrForces.join(repForcesAndSum).where(0).equalTo(0)
      .map(new RichMapFunction[((Int, Vector[Double]), (Int, Vector[Double], Double)), (Int, Vector[Double])] {
      private var sumQ: Double = 0.0

      override def open(parameters: Configuration) {
        sumQ = getRuntimeContext.getBroadcastVariable[Double]("sumQ").get(0)
      }

      def map(vectors: ((Int, Vector[Double]), (Int, Vector[Double], Double))): (Int, Vector[Double]) = {
        val attrForce = vectors._1._2
        val repForce = vectors._2._2 / sumQ

        (vectors._1._1, attrForce - repForce)
      }
    }).withBroadcastSet(sumQ, "sumQ")
  }

  def centerEmbedding(embedding: DataSet[(Int, Vector[Double], Vector[Double], Vector[Double])]):
  DataSet[(Int, Vector[Double], Vector[Double], Vector[Double])] = {
    // center embedding
    val sumAndCount = embedding.map(x => (x._2, 1)).reduce((x, y) => (x._1 + y._1, x._2 + y._2))

    embedding.mapWithBcVariable(sumAndCount) {
      (v, sumAndCount) => (v._1, v._2 - (sumAndCount._1 :/ sumAndCount._2.toDouble), v._3, v._4)
    }
  }

  def centerInput(embedding: DataSet[(Int, Vector[Double])]): DataSet[(Int, Vector[Double])] = {
    // center embedding
    val sumAndCount = embedding.map(x => (x._2, 1)).reduce((x, y) => (x._1 + y._1, x._2 + y._2))


    embedding.mapWithBcVariable(sumAndCount) {
      (lv, sumAndCount) => (lv._1, lv._2 - sumAndCount._1 :/ sumAndCount._2.toDouble)
    }
  }
  
  def updateEmbedding(gradient: DataSet[(Int, Vector[Double])],
                      workingSet: DataSet[(Int, Vector[Double], Vector[Double], Vector[Double])],
                      minGain: Double, momentum: Double, learningRate: Double):
  DataSet[(Int, Vector[Double], Vector[Double], Vector[Double])] = {

    gradient.join(workingSet).where(0).equalTo(0) {
      (dY, rest) =>
        val currentEmbedding = rest._2
        val previousGradient = rest._3
        val gain = rest._4
        val currentGradient = dY._2

        val d = currentGradient.size
        val newEmbedding = new DenseVector[Double](d)
        val newGain = new DenseVector[Double](d)
        val newGradient = new DenseVector[Double](d)

        for (i <- 0 until d) {
          if ((currentGradient (i) > 0.0) == (previousGradient(i) > 0.0)) {
            newGain(i) = scala.math.max(gain(i) * 0.8, minGain)
          } else {
            newGain(i) = scala.math.max(gain(i) + 0.2, minGain)
          }
          newGradient(i) = momentum * previousGradient(i) - learningRate * newGain(i) * currentGradient(i)
          newEmbedding(i) = newGradient(i) + currentEmbedding(i)
        }
        (dY._1, newEmbedding, newGradient, newGain)
    }
  }
  
  def iterationComputation (iterations: Int, momentum: Double,
                            workingSet:DataSet[(Int, Vector[Double], Vector[Double], Vector[Double])],
                            highdimAffinites: DataSet[(Int, Vector[Double])], metric: (Vector[Double], Vector[Double]) => Double,
                            learningRate: Double, theta: Double, dimension: Int, iterOffset: Int) = {

    workingSet.iterate(iterations) {
      // (index, embedding, gradient, gains)
      workingSet =>

      val currentEmbedding = workingSet.map(t => (t._1, t._2))

      val dY = gradient(highdimAffinites, currentEmbedding, metric, theta, dimension ,iterOffset)

      val minGain = 0.01

      val updatedEmbedding = updateEmbedding(dY, workingSet, minGain, momentum, learningRate)

      val centeredEmbedding = centerEmbedding(updatedEmbedding)

      centeredEmbedding
    }
  }

  def optimize(highDimAffinities: DataSet[(Int, Vector[Double])],
               initialWorkingSet: DataSet[(Int, Vector[Double], Vector[Double], Vector[Double])],
               learningRate: Double, iterations: Int, metric: (Vector[Double], Vector[Double]) => Double,
               earlyExaggeration: Double, initialMomentum: Double, finalMomentum: Double, theta: Double, dimension: Int):
  DataSet[(Int, Vector[Double])] = {

    val iterInitMomentumExaggeration = scala.math.min(iterations, 20)
    val iterExaggeration = scala.math.min(iterations - iterInitMomentumExaggeration, 101-20)
    val iterWoExaggeration = iterations - iterExaggeration - iterInitMomentumExaggeration

    var embedding: DataSet[(Int, Vector[Double], Vector[Double], Vector[Double])] = null

    // early exaggeration
    val exaggeratedAffinities = highDimAffinities.map(x => (x._1, x._2 * earlyExaggeration))

    // iterate with initial momentum and exaggerated input
    embedding = iterationComputation(iterInitMomentumExaggeration, initialMomentum,
      initialWorkingSet, exaggeratedAffinities, metric, learningRate, theta, dimension, 0)

    if (iterExaggeration > 0) {
      // iterate with final momentum and exaggerated input
      embedding = iterationComputation(iterExaggeration, finalMomentum, embedding, exaggeratedAffinities,
        metric, learningRate, theta, dimension, iterInitMomentumExaggeration)
    }

    // iterate with final momentum and standard input
    if (iterWoExaggeration > 0) {
      embedding = iterationComputation(iterWoExaggeration, finalMomentum, embedding, highDimAffinities,
        metric, learningRate, theta, dimension, iterExaggeration + iterInitMomentumExaggeration)
    }

    embedding.map(x => (x._1, x._2))
  }

  //============================= binary search ===============================================//

  private def binarySearch(distances: Seq[(Int, Int, Double)], perplexity: Double):
  Seq[(Int, Int, Double)] = {
    // try to approximate beta_i (1/sigma_i**2) so we can compute p_j|i
    // set target to log(perplexity) (entropy) to make computation simpler
    approximateBeta(1.0,
      distances, scala.math.log(perplexity), Double.NegativeInfinity, Double.NegativeInfinity, Double.PositiveInfinity)
  }

  private def approximateBeta(beta: Double, distances: Seq[(Int, Int, Double)], targetH: Double,
                              previousH: Double, min: Double,
                              max: Double, iterations: Int = 50):
  Seq[(Int, Int, Double)] = {
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
    scala.math.abs(a - b) < tol
  }

  private def computeH(distances: Seq[(Int, Int, Double)], beta: Double = 1.0): Double = {
    //                          d_i      exp(-d_i * beta)
    val p = distances.map(d => (d._3, scala.math.exp(-d._3 * beta)))
    val sumP = if (p.map(_._2).sum == 0.0) 1e-7 else p.map(_._2).sum
    scala.math.log(sumP) + beta * p.map(p => p._1 * p._2).sum / sumP
  }

  private def computeP(distances: Seq[(Int, Int, Double)], beta: Double):
  Seq[(Int, Int, Double)] = {
    //                            i     j      exp(-d_i * beta)
    val p = distances.map(d => (d._1, d._2, scala.math.exp(-d._3 * beta)))
    val sumP = if (p.map(_._3).sum == 0.0) 1e-7 else p.map(_._3).sum
    //            i     j      p_i|j
    p.map(p => (p._1, p._2, p._3 / sumP))
  }
}
