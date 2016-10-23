package clustering;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.BisectingKMeans;
import org.apache.spark.mllib.clustering.BisectingKMeansModel;
import org.apache.spark.mllib.linalg.Vector;

/**
 *
 * Algorithm for clustering, which return a list of Cluster, after made the
 * algorithm on the dataBase
 *
 * @author do390
 *
 */

public class Clustering {

	private final JavaSparkContext sc;
	private final List<Vector> dataVector;// list of Vector data
	private ArrayList<Cluster> clustersList;// list of clusters obtained after
											// making the algorithm
	private final int numberOfClusters;
	private BisectingKMeansModel model;

	/**
	 *
	 * Constructor
	 *
	 * @param sc
	 * @param dataVector
	 * @param numberOfClusters
	 */

	public Clustering(JavaSparkContext sc, List<Vector> dataVector,
			int numberOfClusters) {
		this.sc = sc;
		this.dataVector = dataVector;
		this.numberOfClusters = numberOfClusters;
		this.clustersList = new ArrayList<>();
	}

	public JavaSparkContext getSc() {
		return sc;
	}

	public ArrayList<Cluster> getClustersList() {
		return clustersList;
	}

	public int getNumberOfClusters() {
		return numberOfClusters;
	}

	public List<Vector> getDataVector() {
		return dataVector;
	}

	public BisectingKMeansModel getModel() {
		return this.model;
	}

	/**
	 *
	 * Set the model with a trained bkmModel, who will predict what clustr a
	 * vector is from
	 *
	 */

	public void bkmModel() {
		BisectingKMeans bkm = new BisectingKMeans().setK(getNumberOfClusters());
		BisectingKMeansModel model = bkm.run(sc.parallelize(getDataVector()));
		this.model = model;

	}

	/**
	 *
	 * Place on the good Cluster found by the model each Vector from the
	 * datalist, then set the clusterList
	 *
	 */

	public void clustering() {
		this.bkmModel();

		Vector[] listCenters = getModel().clusterCenters();
		for (int indexCluster = 0; indexCluster != getNumberOfClusters(); indexCluster++) {
			this.clustersList.add(indexCluster, new Cluster(
					listCenters[indexCluster], indexCluster));
		}

		for (int indexVector = 0; indexVector != getDataVector().size(); indexVector++) {
			int indexCluster = getModel().predict(
					getDataVector().get(indexVector));
			this.getClustersList().get(indexCluster)
					.add(getDataVector().get(indexVector));

		}
	}

	/**
	 *
	 * Display the list of clusters
	 *
	 */

	public void displayClusters() {

		System.out.println("\n\n\nCompute Cost: "
				+ this.getModel().computeCost(sc.parallelize(dataVector).rdd())
				+ "\n\n\n");

		for (Cluster c : this.getClustersList()) {
			c.display();
		}
	}

}
