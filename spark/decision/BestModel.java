package decision;

import java.util.Arrays;

import parameters.Algorithm;
import parameters.DataBase;

/**
 * DecisionModel which tests all the knowing classification algorithms and
 * return the best one (according to the calculated accuracy).
 *
 * @author do390
 *
 */

public class BestModel extends DecisionModel implements Algorithm {

	/**
	 *
	 * Construct the bestModel, with the dataBase in parameters.
	 *
	 * @param dataBase
	 *
	 */

	public BestModel(DataBase dataBase) {
		super(dataBase);

	}

	/**
	 *
	 * Tests all the knowing algorithms from Spark in decision domain
	 * (Regression SGD, Regression lasso, Random forest, Naive Bayes) and will
	 * select the better one.
	 *
	 * It will set the model, and display the result
	 *
	 */

	public void algorithm() {

		DecisionModel[] listModel = new DecisionModel[4];
		DataBase dataBase = getDataBase();

		listModel[0] = new RegressionSGD(dataBase);
		System.out.println("MODEL SGD ACHIEVED");

		listModel[1] = new RegressionRandomForest(dataBase);
		System.out.println("MODEL RANDOM FOREST ACHIEVED");

		listModel[2] = new RegressionNaiveBayes(dataBase);
		System.out.println("MODEL NAIVE-BAYES ACHIEVED");

		listModel[3] = new RegressionLasso(dataBase);
		System.out.println("MODEL LASSO ACHIEVED");

		StringBuilder buffer = new StringBuilder(128);
		buffer.append("\n\n\n");
		for (DecisionModel model : listModel) {
			model.setResult();
			buffer.append(model.getResult());
		}

		setMostEfficientModel(listModel);
		buffer.append(BEST_ALGO_TO_USE);
		buffer.append(getNameAlgo());
		buffer.append(ACCURACY);
		buffer.append(getAccuracy());
		buffer.append(PER_CENT);

		// String text = String.format("\n\nFOR THIS CLUSTER, THE BEST ALGORITHM
		// TO USE IS : %s\nACCURACY = %.2f \\%\n\n", getNameAlgo(),
		// getAccuracy());
		//
		// MessageFormat format = new MessageFormat("FOR THIS CLUSTER, THE BEST
		// ALGORITHM TO USE IS : {0}\nACCURACY = {1,number}");
		// Object[] args = {getNameAlgo(), getAccuracy()};
		// buffer.append( format.format(args) );

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

	public void setMostEfficientModel(DecisionModel[] list) {

		// Iterator<DecisionModel> i = list.iterator();
		// while (i.hasNext()) {
		// DecisionModel model = i.next();
		// }

		Arrays.asList(list).stream().max(BestModel::byAccuracy).ifPresent(model -> {
			this.accuracy = model.getAccuracy();
			this.nameAlgo = model.getNameAlgo();
			this.model = model.getModel();
		});

	}

	private static int byAccuracy(DecisionModel model1, DecisionModel model2) {
		return Double.compare(model1.getAccuracy(), model2.getAccuracy());
	}

}
