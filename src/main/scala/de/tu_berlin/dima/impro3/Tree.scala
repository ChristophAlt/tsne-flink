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
import org.apache.flink.api.scala._
import org.apache.flink.configuration.Configuration
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.Breeze._
import org.apache.flink.ml.math.{DenseVector, Vector}
import org.apache.flink.ml.metrics.distances.DistanceMetric
import org.apache.flink.util.Collector
import org.apache.flink.api.common.functions.Partitioner
import org.apache.flink.api.common.functions.RichJoinFunction
import org.apache.flink.ml._
import scala.collection.JavaConverters._

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

import breeze.linalg.{ DenseVector => BreezeDenseVector}


object Tree {

  //should we add the mean to the equation??
  def getBorders(y: DataSet[LabeledVector]) = {
    //get borders of the top-level quadrant
    val borders = y.map( y => (y.vector, y.vector))
        .reduce { (a, b) =>
      val yDimensions = a._2.size
      var minValues = new Array[Double](yDimensions)
      var maxValues = new Array[Double](yDimensions)
      for (i <- 0 until yDimensions) {
        minValues(i) = Math.min(a._1(i), b._1(i))
        maxValues(i) = Math.max(a._2(i), b._2(i))
      }
      (new DenseVector(minValues), new DenseVector(maxValues))
    }

    borders
  }

  def mapToZIndex(y: DataSet[LabeledVector], treeDepth : Int, borders: DataSet[(Vector, Vector)]) = {
    val z = y.map(new RichMapFunction[LabeledVector, (LabeledVector, String)]() {
      var mins : Vector = null
      var maxs : Vector = null

      def calcZindex(yPoint: LabeledVector, mins: Vector, maxs: Vector, zArray: String, depth: Int, maxTreeDepth: Int): String = {

        var zIndex = new Array[Boolean](mins.size)
        var separation = new Array[Double](mins.size)


        for (i <- 0 until mins.size) {
          separation(i) = ((maxs(i) - mins(i)) / 2) + mins(i)
          zIndex(i) = yPoint.vector(i) > separation(i)
        }

        var newMins : ArrayBuffer[Double] = new ArrayBuffer
        var newMaxs : ArrayBuffer[Double] = new ArrayBuffer

        var zSum = 0L
        for (i <- 0 until mins.size) {
          if (zIndex(i)) {
            newMins += separation(i)
            newMaxs += maxs(i)
          } else {
            newMaxs += separation(i)
            newMins += mins(i)
          }
          if (zIndex(i)) {
            zSum += Math.pow(2.0, i).toLong
          }
        }

        val newZstring = zArray + zSum + ","

        val result = if (depth == maxTreeDepth) {
          newZstring
        } else {
          calcZindex(yPoint, new DenseVector(newMins.toArray), new DenseVector(newMaxs.toArray), newZstring, depth + 1, maxTreeDepth)
        }
        return result
      }

      override def open(config: Configuration): Unit = {
        val borders = getRuntimeContext().getBroadcastVariable[(Vector, Vector)]("borders").get(0)
        mins = borders._1
        maxs = borders._2
      }

      def map(in: LabeledVector): (LabeledVector, String) = {
        return (in, calcZindex(in, mins, maxs, "",1, treeDepth))
      }
    }).withBroadcastSet(borders, "borders") // Broadcast the borders

    z

  }


  class TreePartitionerString(dimensions: Int) extends Partitioner[String] {

    def customlog(x: Int, base: Int): Int = {
      return (Math.log(x) / Math.log(base)).ceil.toInt
    }

    override def partition(key: String, numPartitions: Int): Int = {
      
      val indices = key.split(",")

      val base = Math.pow(2,dimensions).toInt
      
      val treelevels = customlog(numPartitions, base)
      
      //println("tree level: " + treelevels)     
      //println("string: " + key)

      // convert 2^dimensions number-system to decimal system
      var sum: Int = 0
      for (i <- 0 until treelevels) {
        //println("\tindex: " + indices(i).toInt)
        //println("\tsum: " + indices(i).toInt * Math.pow(base,(treelevels - i - 1)).toInt)
        
        sum += indices(i).toInt * Math.pow(base,(treelevels - i - 1)).toInt
      }
      
      //println("numPartitions: " + numPartitions)
      
      //println(key + " -> " + (sum % numPartitions))
      
      return sum % numPartitions
    }
  }

  
  def computeEdgeForces(P: DataSet[(Long, Long, Double)], Y: DataSet[LabeledVector], outputDim: Int, metric: DistanceMetric): DataSet[(Long, Vector)] = {
    
    P.map(new RichMapFunction[(Long,Long,Double), (Long,Vector)]() {
      
      var embedding:Map[Long,Vector] = Map()

      override def open(config: Configuration): Unit = {
        embedding = getRuntimeContext.getBroadcastVariable[LabeledVector]("y")
            .asScala.map(x => x.label.toLong -> x.vector).toMap
      }
      
      override def map(in: (Long, Long, Double)): (Long, Vector) = {
        val a = embedding(in._1)
        val b = embedding(in._2)
        val p = in._3
        
        val pos_f = (p / (1.0 + metric.distance(a, b))) * (a.asBreeze - b.asBreeze)
        
        (in._1, pos_f.fromBreeze)
      }
    }).withBroadcastSet(Y, "y")
    .groupBy(_._1).reduce{(a, b) => (a._1, (a._2.asBreeze + b._2.asBreeze).fromBreeze)}
  }
  
  
  def gradient(y: DataSet[LabeledVector],metric: DistanceMetric, treeDepth: Int, maxIterations: Int, p:DataSet[(Long, Long, Double)], theta: Double, yDimensions: Int): DataSet[LabeledVector] = {
    val borders = Tree.getBorders(y)

    val zValues = Tree.mapToZIndex(y, treeDepth, borders)
    
    val partitioned = zValues.partitionCustom(new TreePartitionerString(yDimensions), 1)
    
    val yCenterMass = partitioned.mapPartition{
      (values,out: Collector[(Long, DenseVector, Double)]) =>
        val pointList = values.toList
        
        val N = pointList.size
        
        //println(pointList)
        
        //println("N: " + N)
        //println("yDim: " + yDimensions)
        
        //create array representation
        //var dataArray = Array[Double](yDimensions * N)
        
        var dataArray = Array.fill[Double](yDimensions * N)(0.0)
        
        for (n <- 0 until N) {
          for (d <- 0 until yDimensions) {
            //println("l:" + pointList(n))
            //println("d:" + pointList(n)._1.vector(d))
            dataArray(n * yDimensions + d) = pointList(n)._1.vector(d)
          }
        }

        val tree = new SPTree(yDimensions, dataArray, N)
        
        //tree.print(0)
        
        
        // Compute all terms required for t-SNE gradient
        for(n <- 0 until N) {
          val centerMass = tree.computeNonEdgeForces(n, theta)
          
          if (centerMass == null) {
            out.collect((pointList(n)._1.label.toLong, DenseVector(Array.fill(yDimensions)(0.0)), 1.0))
          } else {
            out.collect((pointList(n)._1.label.toLong, DenseVector(centerMass.getNegf.asScala.map(t => t:Double).toArray[Double]), centerMass.getSumQ))
          }
        }
    }

    val sum_Q = yCenterMass.sum(2).map( t => t._3)

    val nonEdgeForces = yCenterMass.mapWithBcVariable[Double, (Long, Vector)](sum_Q) {
      (y, sumQ) =>
        (y._1, (y._2.asBreeze :/ sumQ).fromBreeze)
    }    

    val edgeForces = Tree.computeEdgeForces(p,y, yDimensions, metric)
    

    val gradient = edgeForces.join(nonEdgeForces).where(0).equalTo(0) {
      (pos_f,neg_f) =>
        LabeledVector(pos_f._1, (pos_f._2.asBreeze - neg_f._2.asBreeze).fromBreeze)
    }
    
    gradient
    
    
  }


  
}
