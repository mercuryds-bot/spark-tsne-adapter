import breeze.linalg.DenseMatrix
import com.github.saurfang.spark.tsne.impl.BHTSNE
import org.apache.spark.mllib.linalg.distributed.RowMatrix

import scala.util.Random

object PysparkScalaTSNE {
  def tsne(rowMatrix: RowMatrix,
           maxIterations: Int = 1000,
           perplexity: Double = 30,
           theta: Double = 0.5,
           seed: Long = Random.nextLong()): DenseMatrix[Double] = {
    BHTSNE.tsne(rowMatrix,
      maxIterations = maxIterations,
      perplexity = perplexity,
      theta = theta,
      seed = seed)
  }
}
