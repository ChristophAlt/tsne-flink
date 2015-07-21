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
import org.apache.flink.core.fs.FileSystem.WriteMode
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.{DenseVector, SparseVector}
import org.apache.flink.ml.metrics.distances.SquaredEuclideanDistanceMetric
import org.scalatest._


class TreeTestSuite extends FlatSpec with Matchers with Inspectors {
  "ZIndex" should "return the right tree mapping" in  {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val input = env.fromCollection(TreeTestSuite.y)
    
    val borders = Tree.getBorders(input)

    val results = Tree.mapToZIndex(input, 5, borders).collect()
    
    println(results)    
  }

  /*
  "gradient" should "return the right grad" in  {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val P = env.fromCollection(TsneHelpersTestSuite.Ppython)
    val y = env.fromCollection(TsneHelpersTestSuite.yPython)

    val gradient = Tree.gradient(y,SquaredEuclideanDistanceMetric(), 3, 2, P, 0.2, 2, env)
    
    println(gradient.collect())
  }*/

  "gradient" should "return the right grad" in  {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val input = Tsne.readInput("/home/felix/impro3/csv/test_act_1000.csv", 1024, env, Array(0,1,2))
    
    val inputDimensionalities = 1024
    val outputDimensionalities = 2
    val perplexity = 40
    val learningRate = 350
    val iterations = 500
    val randomstate = 0
    val neighbors = 150
    val earlyExageration = 4
    val initMom = 0.5
    val finalMom = 0.8
    val theta = 0.5

    val result = Tsne.computeEmbedding(
      env, 
      input, 
      SquaredEuclideanDistanceMetric(), 
      perplexity, 
      inputDimensionalities, 
      outputDimensionalities, 
      learningRate, iterations,
      randomstate, 
      neighbors, 
      earlyExageration, 
      initMom, 
      finalMom, 
      theta)

    result.map(x=> (x.label.toLong, x.vector(0), x.vector(1))).writeAsCsv("/home/felix/impro3/csv/out_act_1000.csv", writeMode=WriteMode.OVERWRITE)

    
    env.execute("TSNE")
    
  }

  
}

object TreeTestSuite {
  val y: Seq[LabeledVector] = List(
    LabeledVector(0, new DenseVector(Array(-0.00017189452155069314,-0.00004000535669646060))),
    LabeledVector(1, new DenseVector(Array(-0.00022717203149772645,0.00008663309627528085))),
    LabeledVector(2, new DenseVector(Array(-0.00010325765463991235,-0.00003582032920213515))),
    LabeledVector(3, new DenseVector(Array(-0.00011138115570423478,0.00000474475662365405))),
    LabeledVector(4, new DenseVector(Array(0.00008448473882524468,0.00022241394002237897))),
    LabeledVector(5, new DenseVector(Array(-0.00000064174153521060,-0.00007498357621804678))),
    LabeledVector(6, new DenseVector(Array(0.00001542291223429679,-0.00002279772806102704))),
    LabeledVector(7, new DenseVector(Array(-0.00013284059681958318,0.00000541931482402651))),
    LabeledVector(8, new DenseVector(Array(-0.00010620372323138086,0.00007868893550591110))),
    LabeledVector(9, new DenseVector(Array(-0.00000695497272360187,-0.00017049612819062857))),
    LabeledVector(10, new DenseVector(Array(-0.00006410186066559613,0.00001555884401596038))),
    LabeledVector(11, new DenseVector(Array(0.00000385880495284729,-0.00001055144283448880)))
  ).toSeq
}
