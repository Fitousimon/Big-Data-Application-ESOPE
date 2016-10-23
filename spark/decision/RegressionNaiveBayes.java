package decision;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.regression.LabeledPoint;

import parameters.Algorithm;
import parameters.DataBase;
import scala.Tuple2;

public class RegressionNaiveBayes extends DecisionModel implements Algorithm {

	/**
	 * 
	 * Construct the classification Naive Bayes model
	 * 
	 * @param dataBase
	 */

	public RegressionNaiveBayes(DataBase dataBase) {
		super(dataBase);
		this.algorithm();
	}

	/**
	 * 
	 * This algorithm cannot be used for dataBase with negative values, so it is
	 * necessary to check before.
	 * 
	 * @return true if naive bayes cannot be used.
	 */

	public boolean infeasibleNaiveBayes() {

		boolean infeasibleNaiveBayes = false;
		int index = 0;

		List<LabeledPoint> list = getDataBase().getParsedData().collect();
		List<Double> allValues = new ArrayList<>();

		while (!infeasibleNaiveBayes && index != list.size()) {
			allValues.add(list.get(index).label());
			for (Double value : list.get(index).features().toArray()) {
				allValues.add(value);
			}

			int indexListValues = 0;
			while (!infeasibleNaiveBayes && indexListValues != allValues.size()) {
				infeasibleNaiveBayes = allValues.get(indexListValues) < 0;
				indexListValues++;
			}
			index++;
			allValues.clear();
		}
		return infeasibleNaiveBayes;
	}

	/**
	 * 
	 * Set the accuracy, the name and the model after training and test.
	 * 
	 */

	public void algorithm() {

		if (!this.infeasibleNaiveBayes()) {
			long duration = System.currentTimeMillis();
			new NaiveBayes();
			NaiveBayesModel model = NaiveBayes.train(getDataBase()
					.getTrainingData().rdd());
			JavaPairRDD<Double, Double> predictionAndLabel = getDataBase()
					.getTestData().mapToPair(
							f -> new Tuple2<>((model).predict(f.features()), f
									.label()));
			this.duration = System.currentTimeMillis() - duration;
			this.accuracy(predictionAndLabel);
			this.model = model;
			this.nameAlgo = "NaiveBayes";
		} else {
			this.accuracy = 0.0;
			this.model = null;
			this.nameAlgo = "NaiveBayes impossible";
		}
	}

}