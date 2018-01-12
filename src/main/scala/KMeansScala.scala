
import java.io.{BufferedWriter, FileOutputStream, OutputStreamWriter}

import au.com.bytecode.opencsv.CSVWriter
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}


object KMeansScala {

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("KMeansScala").setMaster("local"))


    val data = sc.textFile("/Users/shobh/Downloads/SparkTask/src/main/resources/mnist_test.csv")


    val parsedData = data.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()
    val numClusters = 10
    val numIterations = 400
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    val centroids = clusters.clusterCenters
    val file = "kmeans1.csv"
    val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file)))
    val w = new CSVWriter(writer)


    for (x <- centroids) {

      val s = x.toString
      writer.write(s.slice(1,s.length-2) + "\n")
    }

    writer.close()



  }

}
