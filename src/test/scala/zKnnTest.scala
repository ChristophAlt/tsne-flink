import de.tu_berlin.dima.impro3.zScore._
import de.tu_berlin.dima.impro3.{Tsne, zKnn}
import org.apache.flink.api.scala._
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.math.DenseVector
import org.apache.flink.ml.metrics.distances.SquaredEuclideanDistanceMetric
import org.joda.time.{Duration, DateTime}
import org.scalatest.{FlatSpec, Matchers}

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

  /**
  "test zKnn" should "return smth valid" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val toTest = new ListBuffer[LabeledVector]
    for (a <- 0 until 9) toTest.append(new LabeledVector(a, DenseVector(Array(a, a, a, a))))
    val data = env.fromCollection(toTest)

    val sample = new LabeledVector(Random.nextDouble(), DenseVector(Array(1, 1, 1, 0)))

    val len = 4
    val iter = 4
    val model = zKnn.knnJoin(data, sample, len, iter, new SquaredEuclideanDistanceMetric)

    val test1 = neighbors(model, DenseVector(1, 1, 1, 1), 4, new SquaredEuclideanDistanceMetric).collect().toList

    test1 should (contain(LabeledVector(1, DenseVector(0, 0, 0, 0))))


  } */


  "knnDescent" should "return the k  neighbors for each SparseVector" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    val neighbors = 80
    val metric = SquaredEuclideanDistanceMetric()
    val startTime = DateTime.now
    val input: DataSet[LabeledVector] = Tsne.readInput("D:\\Users\\jguenthe\\Downloads\\test_act.csv", 1024, env, null)

    val data = input.collect()

    val sample = new LabeledVector(Random.nextDouble(), DenseVector(Array(1, 1, 1, 0)))

    println("Starting")
    val results = zKnn.knnJoin(input, sample, 150, 6, new SquaredEuclideanDistanceMetric)


  //  val neighborsE = data.map(el => zKnn.neighbors(results, el.vector, 150, metric))


    val endTime=new Duration(DateTime.now,startTime)

    println("Run Time(Seconds):"+endTime.getStandardSeconds)


  }


}