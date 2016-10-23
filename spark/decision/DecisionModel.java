package decision;

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

public class DecisionModel extends Model {

	/**
	 * 
	 * Construct the DecisionModel
	 * 
	 * @param dataBase
	 */

	public DecisionModel(DataBase dataBase) {
		super(dataBase);
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

	public void accuracy(JavaPairRDD<Double, Double> valuesAndPreds) {
		/*
		 * System.out.println("PREDICT VALUES AND REAL VALUES : \n");
		 * valuesAndPreds.foreach(f -> System.out.println(f));
		 */

		double accuracy = this.calculateAccuracy(valuesAndPreds);

		// System.out.println("ACCURACY = " + accuracy + " % " + "\n\n");
		this.accuracy = accuracy;
	}

	/**
	 * 
	 * Calculate the accuracy for a decision model, which is the mean of
	 * 100-relative error (x-y/y).
	 * 
	 * @param valuesAndPreds
	 * @return the accuracy
	 */

	public double calculateAccuracy(JavaPairRDD<Double, Double> valuesAndPreds) {
		double error = valuesAndPreds.filter(pair -> pair._2() != 0)
				.mapToDouble(pair -> Math.abs((pair._1 - pair._2) / (pair._2)))
				.mean();

		return (error < 1) ? 100 * (1 - error) : 0.0;
	}

}
