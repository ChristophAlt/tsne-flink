package de.tu_berlin.dima.impro3

/**
 * Created by jguenthe on 20.07.2015.
 */

import org.apache.flink.api.common.functions.{RichFlatMapFunction, RichMapFunction}
import org.apache.flink.api.common.operators.Order
import org.apache.flink.api.scala.DataSetUtils._
import org.apache.flink.api.scala._
import org.apache.flink.configuration.Configuration
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.DenseVector
import org.apache.flink.ml.metrics.distances.{EuclideanDistanceMetric, DistanceMetric}
import org.apache.flink.util

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.util.Random


object zKnn extends Serializable {

  /**
   * Computes the nearest neighbors in the data-set for the data-point against which KNN
   * has to be applied
   *
   * @param input : RDD of Vectors of Int
   * @param k : Number of data-points of the dataSet on which knnJoin is to be done
   * @param iterations : the number of iterations which has to be carried out
   *
   * @return an RDD of Vectors of Int on which simple KNN needs to be applied with respect
   *         to the data-point
   */
  def knnJoin(input: DataSet[LabeledVector],
              k: Int,
              iterations: Int, metric: DistanceMetric): DataSet[(Long, Long, Double)] = {
    val initPoint = input.first(3).collect().toSeq(1)
    val size = initPoint.vector.size
    val rand = new Array[Int](size)
    val randomValue = new Random

    val model = zScore.computeScore(input)
    val dataScore = zScore.scoreOfDataPoint(initPoint.vector)

    for (count <- 0 to size - 1) rand(count) = 0

    var compute = knnJoin_perIteration(input, initPoint, DenseVector(rand), k, model, dataScore)
    val bcKey = "RAND"

    //TODO replace with iterate


    for (i <- 2 to iterations) {
      //random vector
      for (i <- 0 to size - 1) rand(i) = randomValue.nextInt(10000)
      var kLooped = -1
      val updatedData: DataSet[LabeledVector] = input.map(
        mapper = new RichMapFunction[LabeledVector, LabeledVector]() {
          var rand: mutable.Buffer[Array[Int]] = null

          override def open(config: Configuration): Unit = {
            rand = getRuntimeContext
              .getBroadcastVariable[Array[Int]]("RAND").asScala
          }

          def map(in: LabeledVector): LabeledVector = {
            kLooped = -1
            val size = in.vector.size
            val arr = rand.toArray.head
            val strings = in.vector.map(word => word._2 + arr({
              kLooped = kLooped + 1
              //Todo second boarcastset
              kLooped % size
            }));
            val newVector = strings.map(el => el.toDouble).toArray
            new LabeledVector(in.label, new DenseVector(newVector))
          }

        }
      ).withBroadcastSet(ExecutionEnvironment.getExecutionEnvironment.fromElements(rand), bcKey) // 2. Broadcast the DataSet


      val newDataValue = initPoint.vector.map(word => word._2 + rand({
        kLooped = kLooped + 1
        kLooped % size
      })).toArray

      val newDataPoint = new LabeledVector(initPoint.label, DenseVector(newDataValue))

      //
      val modelLooped = zScore.computeScore(updatedData)
      val dataScorelooped = zScore.scoreOfDataPoint(newDataPoint.vector)

      val looped = knnJoin_perIteration(updatedData, newDataPoint, DenseVector(rand),
        k, modelLooped, dataScorelooped)
      var index = -1;

      val cleaned = looped.map(mapper = new RichMapFunction[LabeledVector, LabeledVector] {
        var rand: mutable.Buffer[Array[Int]] = null

        override def open(config: Configuration): Unit = {
          rand = getRuntimeContext
            .getBroadcastVariable[Array[Int]]("RAND").asScala
        }

        override def map(line: LabeledVector): LabeledVector = {
          var index = -1
          val size = line.vector.size
          val arr = rand.toArray.head
          val vec = line.vector.map(word => word._2 - arr({
            index = index + 1
            index % size
          })).toArray
          new LabeledVector(line.label, DenseVector(vec))

        }
      }).withBroadcastSet(ExecutionEnvironment.getExecutionEnvironment.fromElements(rand), bcKey)

      compute = compute.union(cleaned)
    }

    compute = removeRedundantEntries(compute)


    val cof = new Configuration
    cof.setInteger("k", k)

    //TODO Performance: I honestly don't know how fast this is going to be
    val cleaned = input.flatMap(flatMapper = new RichFlatMapFunction[LabeledVector, (Long, Long, Double)] {
      var reducedData: List[LabeledVector] = null
      var k = 0
      var metric: DistanceMetric = new EuclideanDistanceMetric
      override def open(config: Configuration): Unit = {
        reducedData = getRuntimeContext
          .getBroadcastVariable[LabeledVector]("RESULT").asScala.toList
        k = config.getInteger("k", 0)

      }

      override def flatMap(dataPoint: LabeledVector, collector: util.Collector[(Long, Long, Double)]): Unit = {
        val a = reducedData.map(word => metric.distance(dataPoint.vector, word.vector) -> word).filter(el => el._1 > 0)
          .sortBy(el => el._1)
          .zipWithIndex.filter(wl => wl._2 < k).map(el => (dataPoint.label.toLong, el._1._2.label.toLong, el._1._1))
        for (el <- a) {
          collector.collect(el)
        }

      }
    }).withBroadcastSet(compute, "RESULT").withParameters(cof)


    cleaned
  }

  /**
   * Computes the nearest neighbors in the data-set for the data-point against which KNN
   * has to be applied for A SINGLE ITERATION
   *
   * @param data : Dataset of Vectors of Int, which is the data-set in which knnJoin has to be
   *             undertaken
   * @param dataPoint : Vector of Int, which is the data-point with which knnJoin is done
   *                  with the data-set
   * @param randPoint : Vector of Int, it's the random vector generated in each iteration
   * @param len : The number of data-points from the data-set on which knnJoin is to be done
   * @param zScore : RDD of (Long,Long), which is the ( <line_no> , <zscore> ) for each entry
   *               of the dataset
   * @param dataScore : Long value of z-score of the data-point
   *
   * @return an RDD of the nearest 2*len entries from the data-point on which KNN needs to be
   *         undertaken for that iteration
   */


  private def knnJoin_perIteration(data: DataSet[(LabeledVector)],
                                   dataPoint: LabeledVector,
                                   randPoint: org.apache.flink.ml.math.Vector,
                                   len: Int,
                                   zScore: DataSet[(Long, BigInt)],
                                   dataScore: BigInt): DataSet[LabeledVector] = {

    val greaterScoreTmp = zScore.filter(word => word._2 > dataScore).map(word => (word._2.bigInteger -> word._1)).
      sortPartition(0, Order.ASCENDING).map(w => (w._2))


    val greaterScore = greaterScoreTmp.zipWithIndex.map(el => el._2 -> el._1)


    val lesserScore = zScore.filter(word => word._2 <= dataScore).map(wr => wr._1).zipWithIndex


    if (greaterScore.count() > len && lesserScore.count() > len) {
      val trimtmp = greaterScore.filter(word => word._2 < len).map(word => word._1).
        union(lesserScore.filter(word => word._2 < len).map(word => word._1))


      val trim = trimtmp.map(el => (el, 1))
      val join = data.map(element => (element.label.toLong, element.vector)).join(trim).where(0).equalTo(0) {
        (left, right) => {
          new LabeledVector(right._1, left._2)
        }
      }
      join
    }
    else if (greaterScore.count() < len) {
      val lenMod = len + (len - greaterScore.count)
      val trimtmp = greaterScore.map(word => word._1)
        .union(lesserScore.filter(word => word._2 < lenMod)
        .map(word => word._1))

      val trim = trimtmp.map(el => (el, 1))
      val join = data.map(element => (element.label.toLong, element.vector)).join(trim).where(0).equalTo(0) {
        (left, right) => {
          new LabeledVector(right._1, left._2)
        }
      }
      join
    }
    else {
      val lenMod = len + (len - lesserScore.count)
      val trimtmp = greaterScore.filter(word => word._2 < lenMod).map(el => el._1)
        .union(lesserScore.map(el => (el._1)))

      val trim = trimtmp.map(el => (el, 1))
      val join = data.map(element => (element.label.toLong, element.vector)).join(trim).where(0).equalTo(0) {
        (left, right) => {
          new LabeledVector(right._1, left._2)
        }
      }
      join
    }

  }

  private def removeRedundantEntries(data: DataSet[LabeledVector]): DataSet[LabeledVector] = {
    data.map(vr => vr.label -> vr).distinct(0).map(el => el._2)
  }


  def neighbors(reducedData: DataSet[LabeledVector],
                dataPoint: org.apache.flink.ml.math.Vector
                , k: Int, metric: DistanceMetric): DataSet[(Long, Long, Double)] = {

    val distData = reducedData.map(word => metric.distance(dataPoint, word.vector) -> word)
      .sortPartition(0, Order.ASCENDING)
      .zipWithIndex.filter(wl => wl._1 < k).map(wl => (dataPoint.size.toLong, wl._2._2.label.toLong, wl._2._1))


    distData

    /** Spark Code
      * val distData = reducedData.map(word => euclideanDist(dataPoint, word) -> word)
              .sortByKey(true)
              .zipWithIndex()
              .filter(word => word._2 < k).map(word => word._1._2)
      distData
      */

  }

}
