package clustering;

import java.util.List;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;

/**
 * 
 * Superclass for clustering and regression algorithm, in fact after divided the
 * whole database into clusters, the best algorithm for classification or
 * decision is found on each cluster, and the result is displayed.
 * 
 * @author do390
 *
 */

public class RegressionClustering extends Clustering {

	protected String result;
	private final int rowLabel;
	private final double weight_training;
	private final double weight_test;

	/**
	 * 
	 * Constructor
	 * 
	 * @param sc
	 *            : The current JavaSparkContext.
	 * @param dataVector
	 *            : The dataBase in a list of Vector format.
	 * @param numberOfClusters
	 *            : Number of clusters we want.
	 * @param rowLabel
	 *            : Row of label we want to predict.
	 * @param weight_training
	 *            : between 0 and 1.
	 * @param weight_test
	 */

	public RegressionClustering(JavaSparkContext sc, List<Vector> dataVector,
			int numberOfClusters, int rowLabel, double weight_training,
			double weight_test) {
		super(sc, dataVector, numberOfClusters);
		this.rowLabel = rowLabel;
		this.weight_test = weight_test;
		this.weight_training = weight_training;

	}

	public String getResult() {
		return result;
	}

	public int getRowLabel() {
		return rowLabel;
	}

	public double getWeight_training() {
		return weight_training;
	}

	public double getWeight_test() {
		return weight_test;
	}

}
