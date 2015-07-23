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
import org.apache.flink.api.java.utils.ParameterTool
import org.apache.flink.api.scala._
import org.apache.flink.core.fs.FileSystem.WriteMode
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.SparseVector
import org.apache.flink.ml.metrics.distances._


object Tsne {

  def main(args: Array[String]) {
    val parameters = ParameterTool.fromArgs(args)
    val env = ExecutionEnvironment.getExecutionEnvironment

    // get parameters from command line or use default
    val inputPath = parameters.getRequired("input")
    val outputPath = parameters.getRequired("output")

    val inputDimension = parameters.getRequired("dimension").toInt

    val metric = parameters.get("metric", "sqeucledian")
    val perplexity = parameters.getDouble("perplexity", 30.0)
    val nComponents = parameters.getLong("nComponents", 2)
    val earlyExaggeration = parameters.getLong("earlyExaggeration", 4)
    val learningRate = parameters.getDouble("learningRate", 1000)
    val iterations = parameters.getLong("iterations", 300)

    val randomState = parameters.getLong("randomState", 0)
    val neighbors = parameters.getLong("neighbors", 3 * perplexity.toInt)

    val initialMomentum = parameters.getDouble("initialMomentum", 0.5)
    val finalMomentum = parameters.getDouble("finalMomentum", 0.8)
    val theta = parameters.getDouble("theta", 0.5)
    val lossFile = parameters.get("loss", "loss.txt")
    val knnIterations = parameters.getLong("knnIterations", 3)
    val knnMethod = parameters.getRequired("knnMethod")

    val input = readInput(inputPath, inputDimension, env, Array(0,1,2))

    val result = computeEmbedding(env, input, getMetric(metric), perplexity, inputDimension, nComponents, learningRate, iterations,
      randomState, neighbors, earlyExaggeration, initialMomentum, finalMomentum, theta, knnMethod, knnIterations)

    result.map(x=> (x.label.toLong, x.vector(0), x.vector(1))).writeAsCsv(outputPath, writeMode=WriteMode.OVERWRITE)

    val executionResult = env.execute("TSNE")

    import java.io._
    val pw = new PrintWriter(new File(lossFile))
    pw.write(executionResult.getAccumulatorResult("loss").toString)
    pw.close
  }

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

  private def getMetric(metric: String): DistanceMetric = {
    metric match {
      case "sqeucledian" => SquaredEuclideanDistanceMetric()
      case "eucledian" => EuclideanDistanceMetric()
      case "cosine" => CosineDistanceMetric()
      case _ => throw new IllegalArgumentException(s"Metric '$metric' not defined")
    }
  }

  private def computeEmbedding(env: ExecutionEnvironment, input: DataSet[LabeledVector], metric: DistanceMetric,
                               perplexity: Double, inputDimension: Int, nComponents: Int, learningRate: Double,
                               iterations: Int, randomState: Int, neighbors: Int,
                               earlyExaggeration: Double, initialMomentum: Double,
                               finalMomentum: Double, theta: Double, knnMethod: String, knnIterations: Int):
  DataSet[LabeledVector] = {

    //val centeredInput = centerInput(input)
    //val knn = kNearestNeighbors(centeredInput, neighbors, metric)
    //val initialWorkingSet = initWorkingSet(centeredInput, nComponents, randomState)

    val knn = knnMethod match {
      case "bruteforce" => kNearestNeighbors(input, neighbors, metric)
      case "partition" => partitionKnn(input, neighbors, metric, env.getParallelism)
      case "project" => projectKnn(input, neighbors, metric, inputDimension, knnIterations)
      case _ => throw new IllegalArgumentException(s"Knn method '$metric' not defined")
    }

    val pwAffinities = pairwiseAffinities(knn, perplexity)

    val jntDistribution = jointDistribution(pwAffinities)

    val initialWorkingSet = initWorkingSet(input, nComponents, randomState)

    optimize(jntDistribution, initialWorkingSet, learningRate, iterations, metric, earlyExaggeration,
      initialMomentum, finalMomentum, theta)
  }
}
