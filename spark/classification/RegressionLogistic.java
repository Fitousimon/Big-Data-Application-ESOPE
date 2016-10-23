package classification;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;

import parameters.Algorithm;
import parameters.DataBase;
import scala.Tuple2;

/**
 *
 * Logistic regression model
 *
 * @author do390
 *
 */

public class RegressionLogistic extends ClassificationModel implements Algorithm {

	/**
	 *
	 * Construct the RegressionLogictic
	 *
	 * @param dataBase
	 */

	public RegressionLogistic(DataBase dataBase, int n) {
		super(dataBase, n);
		this.algorithm();
	}

	/**
	 *
	 * Set the accuracy, the name and the model after training and test.
	 *
	 */

	public void algorithm() {
		LogisticRegressionModel model;
		try {
			long duration = System.currentTimeMillis();

			model = new LogisticRegressionWithLBFGS().setNumClasses(getNumClasses())
					.run(getDataBase().getTrainingData().rdd());

			JavaPairRDD<Double, Double> predictionAndLabel = getDataBase().getTestData()
					.mapToPair(f -> new Tuple2<>(model.predict(f.features()), f.label()));
			this.duration = System.currentTimeMillis() - duration;
			this.accuracy(predictionAndLabel);
			this.model = model;
			this.nameAlgo = "LogisticRegressionWithLBFGS";
		} catch (Exception e) {
			setNumClasses(getNumClasses() + 1);
			algorithm();

		}
	}

}
