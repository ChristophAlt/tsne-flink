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

import org.apache.flink.api.java.utils.ParameterTool
import org.apache.flink.api.scala._
import org.apache.flink.core.fs.FileSystem.WriteMode
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.SparseVector
import org.apache.flink.ml.metrics.distances._
import de.tu_berlin.dima.impro3.TsneHelpers._


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
    val nComponents = parameters.getLong("nComponents", 2)
    //val earlyExaggeration = parameters.getLong("earlyExaggeration")
    val learningRate = parameters.getDouble("learningRate", 1000)
    val iterations = parameters.getLong("iterations", 300)

    val randomState = parameters.getLong("randomState", 0)
    val neighbors = parameters.getLong("neighbors", 3 * perplexity.toInt)

    val input = readInput(inputPath, dimension, env, Array(0,1,2))

    val result = computeEmbedding(input, metric, perplexity, nComponents, learningRate, iterations,
      randomState, neighbors)

    result.map(x=> (x.label.toLong, x.vector(0), x.vector(1))).writeAsCsv(outputPath, writeMode=WriteMode.OVERWRITE)

    env.execute("TSNE")
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

  private def computeEmbedding(input: DataSet[LabeledVector], metric: DistanceMetric,
                               perplexity: Double, nComponents: Int, learningRate: Double,
                               iterations: Int, randomState: Int, neighbors: Int):
  DataSet[LabeledVector] = {

    // TODO: center data or do this somewhere before in the pipeline
    val knn = kNearestNeighbors(input, neighbors, metric)

    val pwAffinities = pairwiseAffinities(knn, perplexity)

    val jntDistribution = jointDistribution(pwAffinities)

    val initialEmbedding = initEmbedding(input, randomState)

    // TODO: early exaggeration and early compression
    optimize(jntDistribution, initialEmbedding, learningRate, iterations, metric)
  }
}