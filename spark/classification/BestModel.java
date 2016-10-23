package classification;

import java.util.ArrayList;

import parameters.Algorithm;
import parameters.DataBase;

/**
 * ClassificationModel which tests all the knowing classification algorithms and
 * return the best one (according to the calculated accuracy).
 *
 * @author do390
 *
 */

public class BestModel extends ClassificationModel implements Algorithm {

	/**
	 *
	 * Construct the bestModel, with the dataBase in parameters.
	 *
	 * @param dataBase
	 *
	 */

	public BestModel(DataBase dataBase) {
		super(dataBase, 1);

	}

	/**
	 *
	 * Tests all the known algorithms from Spark in classification domain
	 * (Regression logistic, Decision tree, Random forest, Naive Bayes) and will
	 * select the better one.
	 *
	 * It will set the model, and display the result
	 *
	 */

	public void algorithm() {

		ArrayList<ClassificationModel> listModel = new ArrayList<>();

		ClassificationModel model1 = new RegressionLogistic(getDataBase(), getNumClasses());
		System.out.println("LOGISTIC REGRESSION ACHIEVED");

		ClassificationModel model2 = new RegressionDecisionTree(getDataBase(), model1.getNumClasses());
		System.out.println("DECISION TREE ACHIEVED");

		ClassificationModel model3 = new RegressionRandomForest(getDataBase(), model1.getNumClasses());
		System.out.println("RANDOM FOREST ACHIEVED");

		ClassificationModel model4 = new RegressionNaiveBayes(getDataBase(), model1.getNumClasses());
		System.out.println("NAIVE-BAYES ACHIEVED");

		listModel.add(model1);
		listModel.add(model2);
		listModel.add(model3);
		listModel.add(model4);

		StringBuilder buffer = new StringBuilder(128);
		buffer.append("\n\n\n");
		for (ClassificationModel model : listModel) {
			model.setResult();
			buffer.append(model.getResult());
		}

		setMostEfficientModel(listModel);
		buffer.append(BEST_ALGO_TO_USE);
		buffer.append(getNameAlgo());
		buffer.append(ACCURACY);
		buffer.append(getAccuracy());
		buffer.append(PER_CENT);
		this.result = buffer.toString();
		displayResult();

	}

	/**
	 *
	 * Set the instance variables to those of the best model calculated. Set the
	 * accuracy, the nameAlgo and the model.
	 *
	 * @param list
	 *            : the list of ClassificationModel trained and tested on the
	 *            dataBase.
	 *
	 */

	public void setMostEfficientModel(ArrayList<ClassificationModel> list) {

		// Iterator<DecisionModel> i = list.iterator();
		// while (i.hasNext()) {
		// DecisionModel model = i.next();
		// }

		list.stream().max(BestModel::byAccuracy).ifPresent(model -> {
			this.accuracy = model.getAccuracy();
			this.nameAlgo = model.getNameAlgo();
			this.model = model.getModel();
		});

	}

	private static int byAccuracy(ClassificationModel model1, ClassificationModel model2) {
		return Double.compare(model1.getAccuracy(), model2.getAccuracy());
	}

}
