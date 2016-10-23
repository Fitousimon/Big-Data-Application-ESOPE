package clustering;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.mllib.linalg.Vector;

/**
 * 
 * Class Cluster, to define each group of data, by a center, a list of vector
 * and an index.
 * 
 * @author do390
 *
 */

public class Cluster {

	private Vector center; // Center of the Cluster, calculated by K-means
							// Algorithm

	private ArrayList<Vector> list;
	private int index;

	/**
	 * 
	 * Constructor, without list
	 * 
	 * @param center
	 * @param index
	 */

	public Cluster(Vector center, int index) {
		this.center = center;
		this.index = index;
		this.list = new ArrayList<>();
	}

	/**
	 * Constructor with data list
	 * 
	 * @param center
	 * @param list
	 * @param index
	 */

	public Cluster(Vector center, ArrayList<Vector> list, int index) {
		this.center = center;
		this.list = list;
		this.index = index;
	}

	public Vector getCenter() {
		return center;
	}

	public List<Vector> getList() {
		return list;
	}

	public int getIndex() {
		return this.index;
	}

	/**
	 * Add a vector to the data list
	 * 
	 * @param v
	 */

	public void add(Vector v) {
		this.list.add(v);
	}

	/**
	 * 
	 * Display the cluster
	 * 
	 */

	public void display() {

		System.out.println("CLUSTER " + this.getIndex() + " \nCENTER "
				+ this.getCenter() + " : " + "\n");

		for (Vector v : this.getList()) {
			System.out.println(v);
		}

		System.out.println("\n\n\n");
	}

}
