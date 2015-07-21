import de.tu_berlin.dima.impro3.zKnn
import org.scalatest.{Matchers, FlatSpec}
import de.tu_berlin.dima.impro3.TsneHelpers._
import org.apache.flink.api.scala._
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.{DenseVector, SparseVector}
import org.apache.flink.ml.metrics.distances.SquaredEuclideanDistanceMetric
import org.scalatest._
import org.apache.flink.ml.math.Breeze._
import de.tu_berlin.dima.impro3.zScore._

import de.tu_berlin.dima.impro3.zKnn._

import scala.collection.mutable.ListBuffer
import scala.util.Random

/**
 * Created by jguenthe on 20.07.2015.
 */
class zKnnTest extends FlatSpec with Matchers {


  "compute ZScore" should "return 7 for example" in {
    val env = ExecutionEnvironment.getExecutionEnvironment
    val toTest = new ListBuffer[LabeledVector]
    for (a <- 0 until 8) {
      toTest.append(new LabeledVector(a, DenseVector(Array(a, a, a, a))))
    }
    val data = env.fromCollection(toTest)
    val test = computeScore(data).collect().toList


    test.head._2.intValue() should equal(0)

  }

  "test zKnn" should "return smth valid" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val toTest = new ListBuffer[LabeledVector]
    for (a <- 0 until 8) toTest.append(new LabeledVector(a, DenseVector(Array(a, a, a, a))))
    val data = env.fromCollection(toTest)

    val sample = new LabeledVector(Random.nextDouble(), DenseVector(Array(1, 5, 3,0)))

    val len = 4
    val iter = 4
    val test = zKnn.knnJoin(data, sample, len, iter, new SquaredEuclideanDistanceMetric).collect().toList

    test.head should equal(null)


  }


}
