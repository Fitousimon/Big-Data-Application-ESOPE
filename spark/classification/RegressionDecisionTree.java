package classification;

import java.util.HashMap;
import java.util.Map;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;

import parameters.Algorithm;
import parameters.DataBase;
import scala.Tuple2;

/**
 *
 * RegressionDecisionTree Model
 *
 * @author do390
 *
 */

public class RegressionDecisionTree extends ClassificationModel implements Algorithm {

	/**
	 * The algorithm needs some mathematical parameters, which are mainly used
	 * instanciated like this.
	 */

	private static final Map<Integer, Integer> CATEGORICAL_FEATURES_INFO = new HashMap<Integer, Integer>();
	private static final String IMPURITY = "gini";
	private static final int MAX_DEPTH = 30;
	private static final int MAX_BINS = 32;

	/**
	 *
	 * Construct the RegressionDecisionTree
	 *
	 * @param dataBase
	 */

	public RegressionDecisionTree(DataBase dataBase, int n) {
		super(dataBase, n);
		this.algorithm();
	}

	/**
	 *
	 * Set the accuracy, the name and the model after training and test.
	 *
	 */

	public void algorithm() {
		try {
			long duration = System.currentTimeMillis();
			DecisionTreeModel model = DecisionTree.trainClassifier(this.getDataBase().getTrainingData(),
					getNumClasses(), CATEGORICAL_FEATURES_INFO, IMPURITY, MAX_DEPTH, MAX_BINS);

			JavaPairRDD<Double, Double> predictionAndLabel = this.getDataBase().getTestData()
					.mapToPair(f -> new Tuple2<>(model.predict(f.features()), f.label()));

			System.out.println("Learned classification tree model:\n" + model.toDebugString());

			this.duration = System.currentTimeMillis() - duration;
			this.accuracy(predictionAndLabel);
			this.model = model;
			this.nameAlgo = "Decision Tree";
		} catch (Exception e) {
			setNumClasses(getNumClasses() + 1);
			algorithm();

		}
	}

}
