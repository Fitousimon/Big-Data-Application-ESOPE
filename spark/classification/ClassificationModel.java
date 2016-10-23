package classification;

import org.apache.spark.api.java.JavaPairRDD;

import parameters.DataBase;
import parameters.Model;

/**
 *
 * SuperClass for all the models in this package.
 *
 * @author do390
 *
 */

public class ClassificationModel extends Model {

	private int numClasses;

	/**
	 * Construct the ClassificationModel
	 *
	 * @param dataBase
	 * @param numClasses
	 */
	public ClassificationModel(DataBase dataBase, int numClasses) {
		super(dataBase);
		this.numClasses = numClasses;
	}

	public void setNumClasses(int n) {
		this.numClasses = n;
	}

	public int getNumClasses() {
		return this.numClasses;
	}

	/**
	 *
	 * Set the accuracy after calcul, it takes a JavaPairRDD<Double, Double> in
	 * parameter which is like a list of two columns, with predicted label and
	 * real label, and calculate the accuracy.
	 *
	 * @param predictionAndLabel
	 *            : JavaPairRDD<Double, Double>.
	 *
	 */

	public void accuracy(JavaPairRDD<Double, Double> predictionAndLabel) {
		double num = predictionAndLabel.filter(x -> x._1.equals(x._2)).count();
		double den = predictionAndLabel.count();

		double accuracy = 100 * num / den;

		// System.out.println("\n\nACCURACY = " + accuracy + "%");
		// System.out.println("Success = " + num + "\n" + "Fail = " + (den -
		// num)
		// + "\n" + "Total = " + den + "\n\n");
		this.accuracy = accuracy;
	}

}
