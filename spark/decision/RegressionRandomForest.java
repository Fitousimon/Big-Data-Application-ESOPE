package decision;

import java.util.HashMap;
import java.util.Map;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;

import parameters.Algorithm;
import parameters.DataBase;
import scala.Tuple2;

/**
 *
 * RandomForest Model
 *
 * @author do390
 *
 */

public class RegressionRandomForest extends DecisionModel implements Algorithm {

	/**
	 * The algorithm needs some mathematical parameters, which are mainly used
	 * instanciated like this.
	 */

	public static final Map<Integer, Integer> CATEGORICAL_FEATURES_INFO = new HashMap<Integer, Integer>();
	public static final int NUM_TREES = 15;
	public static final String FEATURES_SUBSET_STRATEGY = "auto";
	public static final String IMPURITY = "variance";
	public static final int MAX_DEPTH = 10;
	public static final int MAX_BINS = 100;
	public static final int SEED = 12345;

	/**
	 *
	 * Construct the randomForest model
	 *
	 * @param dataBase
	 */

	public RegressionRandomForest(DataBase dataBase) {
		super(dataBase);
		this.algorithm();
	}

	/**
	 *
	 * Set the accuracy, the name and the model after training and test.
	 *
	 */

	public void algorithm() {

		long duration = System.currentTimeMillis();
		RandomForestModel model = RandomForest.trainRegressor(getDataBase().getTrainingData(),
				CATEGORICAL_FEATURES_INFO, NUM_TREES, FEATURES_SUBSET_STRATEGY, IMPURITY, MAX_DEPTH, MAX_BINS, SEED);

		JavaPairRDD<Double, Double> predictionAndLabel = getDataBase().getTestData()
				.mapToPair(f -> new Tuple2<>(model.predict(f.features()), f.label()));

		// System.out.println("Learned classification tree model:\n" +
		// model.toDebugString());
		this.duration = System.currentTimeMillis() - duration;
		this.accuracy(predictionAndLabel);
		this.model = model;
		this.nameAlgo = "Random Forest";

	}

}
