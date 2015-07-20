package de.tu_berlin.dima.impro3

import org.apache.flink.api.common.operators.Order
import org.apache.flink.api.scala._
import org.apache.flink.ml.common.LabeledVector

import scala.math._

/**
 * Created by jguenthe on 20.07.2015.
 */
class zScore extends Serializable {


  /**
   * Computers the z-scores for each entry of the input RDD of Vector of Long,
   * sorted in ascending order
   *
   * @param  data of Vector of Long
   * @return z-scores of the Dataset[( <line_no> , <z-value> )]
   */
  def computeScore(data: DataSet[LabeledVector]): DataSet[(Long, BigInt)] = {
    //Original code
    //    val score = data.map(word => scoreOfDataPoint(word._1) -> word._2).
    //      sortByKey(true).
    //      map(word => word._2 -> word._1)



    val score = data.map(word => (scoreOfDataPoint(word.vector).bigInteger, word.label)).
      sortPartition(0, order = Order.ASCENDING).map(word => (word._2.toLong, new scala.BigInt(word._1)))


    score
  }

  /**
   * Computes the z-score of a Vector
   *
   * @param Vector of Long
   * @return z-score of the vector
   */
  def scoreOfDataPoint(vector: org.apache.flink.ml.math.Vector): BigInt = {

    val x = vector.toArray.map(element => {
      element._2.toLong
    })

    var temp = 0L
    var score: BigInt = 0
    var counter = 0

    while (checkVectors(x) == 0) {
      for (i <- x.length - 1 to 0 by -1) {
        temp = x(i) & ((1 << 1) - 1)
        temp = temp << counter
        score = score + temp
        x(i) = x(i) >> 1
        counter = counter + 1
      }
    }
    score
  }


  /**
   * Checks if all entries within the array are 0 or not
   *
   * @param Array of Int
   * @return 1, if all elements are zero; else 0
   */
  def checkVectors(vector: Array[Long]): Int = {
    var flag = 1

    for (i <- 0 to vector.length - 1) {
      if (vector(i) != 0) {
        flag = 0
      }
    }

    return flag
  }


}
