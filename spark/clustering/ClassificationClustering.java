package clustering;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;

import parameters.Algorithm;
import parameters.DataBase;
import classification.BestModel;
import classification.ClassificationModel;

/**
 *
 * Clustering with classification on each cluster
 *
 * @author do390
 *
 */
public class ClassificationClustering extends RegressionClustering implements Algorithm {

	private static final String LINE = "--------------------------------------------------------------------------------------------------------------";

	private List<ClassificationModel> list; // After making the algorithm, we
											// recover a list with the bestmodel
											// for each cluster

	/**
	 *
	 * Constructor
	 *
	 * @param sc
	 * @param dataVector
	 * @param numberOfClusters
	 * @param rowLabel
	 * @param weight_training
	 * @param weight_test
	 */

	public ClassificationClustering(JavaSparkContext sc, List<Vector> dataVector, int numberOfClusters, int rowLabel,
			double weight_training, double weight_test) {
		super(sc, dataVector, numberOfClusters, rowLabel, weight_test, weight_test);
		this.clustering();
		this.list = new ArrayList<>();
	}

	/**
	 *
	 * Set the result, and the list of model
	 *
	 */

	public void algorithm() {

		String result = "";

		for (Cluster c : this.getClustersList()) {

			DataBase dataBase = new DataBase(this.getSc(), c.getList().toArray(new Vector[c.getList().size()]),
					this.getRowLabel(), this.getWeight_training(), this.getWeight_test());

			BestModel best = new BestModel(dataBase);
			best.algorithm();
			result += "\nCLUSTER NUMBER : " + c.getIndex() + "\nCENTER : " + c.getCenter() + best.getResult() + "\n\n"
					+ LINE;
			this.list.add(best);
		}
		this.result = result;

		System.out.println(this.getResult());

	}

	public List<ClassificationModel> getList() {
		return list;
	}
}
