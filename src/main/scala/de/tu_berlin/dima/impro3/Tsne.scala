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
import breeze.linalg.functions._
import de.tu_berlin.dima.impro3.TsneHelpers._
import org.apache.flink.api.java.utils.ParameterTool
import org.apache.flink.api.scala._
import org.apache.flink.core.fs.FileSystem.WriteMode

import java.io._


object Tsne {

  def main(args: Array[String]) {
    val parameters = ParameterTool.fromArgs(args)
    val env = ExecutionEnvironment.getExecutionEnvironment

    // ========================== Parameters ======================================================

    val getExecutionPlan = parameters.has("executionPlan")
    val inputDistanceMatrix = parameters.has("inputDistanceMatrix")

    val inputPath = parameters.getRequired("input")
    val outputPath = parameters.getRequired("output")

    val inputDimension = parameters.getRequired("dimension").toInt

    val metric = parameters.get("metric", "sqeuclidean")
    val perplexity = parameters.getDouble("perplexity", 30.0)
    val nComponents = parameters.getLong("nComponents", 2)
    val earlyExaggeration = parameters.getLong("earlyExaggeration", 4)
    val learningRate = parameters.getDouble("learningRate", 1000)
    val iterations = parameters.getLong("iterations", 300)

    val randomState = parameters.getLong("randomState", 0)
    val neighbors = parameters.getLong("neighbors", 3 * perplexity.toInt)

    val initialMomentum = parameters.getDouble("initialMomentum", 0.5)
    val finalMomentum = parameters.getDouble("finalMomentum", 0.8)
    val theta = parameters.getDouble("theta", 0.25)
    val lossFile = parameters.get("loss", "loss.txt")
    val knnIterations = parameters.getLong("knnIterations", 3)
    val knnMethod = parameters.getRequired("knnMethod")
    val knnBlocks = parameters.getLong("knnBlocks", env.getParallelism)

    // ===========================================================================================

    var knn: DataSet[(Int, Int, Double)] = null

    if (inputDistanceMatrix) {
      knn = readDistanceMatrix(inputPath, env, Array(0,1,2))
    } else {
      val input = readInput(inputPath, inputDimension, env, Array(0,1,2))
      val knnMetric = getMetric(metric)
      knn = knnMethod match {
        case "bruteforce" => kNearestNeighbors(input, neighbors, knnMetric)
        case "partition" => partitionKnn(input, neighbors, knnMetric, knnBlocks)
        case "project" => projectKnn(input, neighbors, knnMetric, inputDimension, knnIterations)
        case _ => throw new IllegalArgumentException(s"Knn method '$metric' not defined")
      }
    }

    val result = computeEmbedding(env, knn, getMetric(metric), perplexity, inputDimension,
      nComponents, learningRate, iterations, randomState, neighbors, earlyExaggeration,
      initialMomentum, finalMomentum, theta, knnMethod, knnIterations, knnBlocks)

    result.map(x=> (x._1, x._2(0), x._2(1))).writeAsCsv(outputPath, writeMode=WriteMode.OVERWRITE)

    // either output the execution plan as json or execute the dataflow
    if (getExecutionPlan) {
      val executionPlan = env.getExecutionPlan()

      val pw = new PrintWriter(new File("tsne_executionPlan.json"))
      pw.write(executionPlan)
      pw.close

    } else {
      val executionResult = env.execute("TSNE")

      val pw = new PrintWriter(new File(lossFile))
      pw.write(executionResult.getAccumulatorResult("loss").toString)
      pw.close
    }
  }

  private def computeEmbedding(env: ExecutionEnvironment, knn: DataSet[(Int, Int, Double)],
                               metric: (Vector[Double], Vector[Double]) => Double,
                               perplexity: Double, inputDimension: Int, nComponents: Int,
                               learningRate: Double, iterations: Int, randomState: Int,
                               neighbors: Int, earlyExaggeration: Double, initialMomentum: Double,
                               finalMomentum: Double, theta: Double, knnMethod: String,
                               knnIterations: Int, knnBlocks: Int):
  DataSet[(Int, Vector[Double])] = {

    // compute high dimensional affinities p_j|i
    val pwAffinities = pairwiseAffinities(knn, perplexity)
    // compute joint probabilities pij
    val jntDistribution = jointDistribution(pwAffinities)
    // put everything into breeze SparseVectors
    val svJntDistribution: DataSet[(Int, SparseVector[Double])] = jntDistribution.groupBy(0).reduceGroup {
      entries =>
        val vectorBuilder = new VectorBuilder[Double](inputDimension * inputDimension)
        val first = entries.next()
        vectorBuilder.add(first._2, first._3)
        while (entries.hasNext) {
          val entry = entries.next()
          vectorBuilder.add(entry._2, entry._3)
        }
        (first._1, vectorBuilder.toSparseVector)
    }

    // init Y
    val initialWorkingSet = initWorkingSet(svJntDistribution, nComponents, randomState)
    // optimize via gradient descent
    optimize(svJntDistribution, initialWorkingSet, learningRate, iterations, metric,
      earlyExaggeration, initialMomentum, finalMomentum, theta, nComponents)
  }

  def readInput(inputPath: String, dimension: Int, env: ExecutionEnvironment,
                fields: Array[Int]): DataSet[(Int, Vector[Double])] = {

    env.readCsvFile[(Int, Int, Double)](inputPath, includedFields = fields)
      .groupBy(_._1).reduceGroup {
      vectorEntries =>
        val vectorBuilder = new VectorBuilder[Double](dimension)
        val first = vectorEntries.next()
        vectorBuilder.add(first._2, first._3)
        while (vectorEntries.hasNext) {
          val vectorEntry = vectorEntries.next()
          vectorBuilder.add(vectorEntry._2, vectorEntry._3)
        }
        (first._1, vectorBuilder.toDenseVector)
    }
  }

  def readDistanceMatrix(inputPath: String, env: ExecutionEnvironment,
                fields: Array[Int]): DataSet[(Int, Int, Double)] = {

    env.readCsvFile[(Int, Int, Double)](inputPath, includedFields = fields)
  }

  def getMetric(metric: String): (Vector[Double], Vector[Double]) => Double = {
    metric match {
      case "sqeuclidean" => squaredDistance.apply[Vector[Double], Vector[Double], Double]
      case "euclidean" => euclideanDistance.apply[Vector[Double], Vector[Double], Double]
      case "cosine" => cosineDistance.apply[Vector[Double], Vector[Double], Double]
      case _ => throw new IllegalArgumentException(s"Metric '$metric' not defined")
    }
  }
}
