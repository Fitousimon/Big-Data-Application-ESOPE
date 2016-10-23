

package decision;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import org.apache.spark.mllib.util.Saveable;

import parameters.Algorithm;
import parameters.DataBase;
import scala.Tuple2;

/**
 *
 * Regression with SGD model.
 *
 * @author do390
 *
 */
public class RegressionSGD extends DecisionModel implements Algorithm {

	/**
	 * The algorithm needs some mathematical parameters, which are mainly used
	 * instanciated like this.
	 */

	public static final int NUMBER_OF_ITERATIONS = 10;
	public static final double STEP_SIZE_RESEARCH_BEGINNING = 10E-20;
	public static final double MAX_STEP_SIZE_ALLOWED = 1.0;
	public static final double STEP_SIZE_MULTIPLICATOR = 10.0;
	public static final double STEP_SIZE_REFINING = 10E-1;
	public static final double STEP_SIZE_DIV_MARGIN = 2.0;
	public static final double MAX_ERROR = 1.0;

	/**
	 *
	 * Constructor
	 *
	 * @param dataBase
	 */

	public RegressionSGD(DataBase dataBase) {
		super(dataBase);
		this.algorithm();
	}

	/**
	 *
	 * Return a stepsize, which is a fundamental parameter for this type of
	 * regression, so this algorithm tests every model for stepSize going from
	 * STEP_SIZE_RESEARCH_BEGINNING to MAX_STEP_SIZE_ALLOWED, by multiplying by
	 * STEP_SIZE_MULTIPLICATOR, for instance 0.0001 then 0.001, 0.01 etc.
	 *
	 * @return the best Stepsize found, according to the accuracy rate.
	 */

	public double stepSizeSearch() {

		double stepSize = STEP_SIZE_RESEARCH_BEGINNING;
		double accuracy = 0.0;

		Tuple2<Double, Double> stepsizeAndAccuracy = new Tuple2<Double, Double>(stepSize, accuracy);

		JavaPairRDD<Double, Double> valuesAndPreds;

		while (stepSize <= MAX_STEP_SIZE_ALLOWED) {
			Saveable model = LinearRegressionWithSGD.train(getDataBase().getTrainingData().rdd(), NUMBER_OF_ITERATIONS,
					stepSize, 1.0);

			valuesAndPreds = this.getDataBase().getTestData().mapToPair(point -> new Tuple2<Double, Double>(
					((LinearRegressionModel) model).predict(point.features()), point.label()));

			accuracy = calculateAccuracy(valuesAndPreds);
			if (accuracy > stepsizeAndAccuracy._2) {
				stepsizeAndAccuracy = new Tuple2<Double, Double>(stepSize, accuracy);
				this.accuracy = accuracy;
			}
			stepSize *= STEP_SIZE_MULTIPLICATOR;
		}
		// System.out
		// .println("STEP 1 : STEP_SIZE OPTIMUM FOUND \nREFINING STEP_SIZE");
		return stepsizeAndAccuracy._1();
	}

	/**
	 *
	 * When a good stepsize is found, this algorithm try to refine it, for
	 * instance we found 0.01 with the first algorithm, then now, this one will
	 * test from 0.005 to 0.015, by adding 0.001 on each step, then return the
	 * best one.
	 *
	 * @param stepSize
	 * @return the best stepSize
	 */

	public double stepSizeRefining(double stepSize) {

		Tuple2<Double, Double> stepsizeAndAccuracy = new Tuple2<Double, Double>(stepSize, getAccuracy());
		double accuracy = getAccuracy();
		double step = stepSize * STEP_SIZE_REFINING;
		double stepSizeInf = stepSize / STEP_SIZE_DIV_MARGIN;
		double stepSizeSup = stepSize * STEP_SIZE_DIV_MARGIN;

		JavaPairRDD<Double, Double> valuesAndPreds;
		while (stepSizeInf <= stepSizeSup) {

			Saveable model = LinearRegressionWithSGD.train(getDataBase().getTrainingData().rdd(), NUMBER_OF_ITERATIONS,
					stepSize, 1.0);

			valuesAndPreds = getDataBase().getTestData().mapToPair(point -> new Tuple2<Double, Double>(
					((LinearRegressionModel) model).predict(point.features()), point.label()));

			accuracy = calculateAccuracy(valuesAndPreds);
			if (accuracy > stepsizeAndAccuracy._2) {
				stepsizeAndAccuracy = new Tuple2<Double, Double>(stepSize, accuracy);
				this.accuracy = accuracy;
			}
			stepSizeInf += step;
		}
		// System.out.println("BEST STEP_SIZE = " + stepsizeAndAccuracy._1);
		return stepsizeAndAccuracy._1;

	}

	/**
	 *
	 * Set the accuracy, the name and the model after training and test.
	 *
	 */

	public void algorithm() {
		long duration = System.currentTimeMillis();

		double stepSize = this.stepSizeRefining(this.stepSizeSearch());

		new LinearRegressionWithSGD();
		LinearRegressionModel model = LinearRegressionWithSGD.train(this.getDataBase().getTrainingData().rdd(),
				NUMBER_OF_ITERATIONS, stepSize);

		JavaPairRDD<Double, Double> valuesAndPreds = getDataBase().getTestData().mapToPair(
				point -> new Tuple2<Double, Double>(((LinearRegressionModel) model).predict(point.features()),
						point.label()));

		this.duration = System.currentTimeMillis() - duration;
		this.accuracy(valuesAndPreds);
		this.model = model;
		this.nameAlgo = "LinearRegressionWithSGD";

	}

}
