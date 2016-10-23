package parameters;

import org.apache.spark.mllib.util.Saveable;

public class Model {
	public static final String METHOD = "\nMETHOD : ";
	public static final String ACCURACY = "\nACCURACY : ";
	public static final String TRAIN_AND_TEST = "\nTRAINING AND TEST DURATION : ";
	public static final String BEST_ALGO_TO_USE = "\n\nFOR THIS CLUSTER, THE BEST ALGORITHM TO USE IS : ";
	public static final String PER_CENT = " %";
	public static final String TIME = " min\n";

	private final DataBase dataBase;
	protected Double accuracy = 0.0;
	protected String nameAlgo;
	protected Saveable model;
	protected long duration;
	protected String result;

	/**
	 *
	 * Construct the DecisionModel
	 *
	 * @param dataBase
	 */

	public Model(DataBase dataBase) {
		this.dataBase = dataBase;
	}

	public String getResult() {
		return this.result;
	}

	public void setResult() {
		this.result = METHOD + getNameAlgo() + ACCURACY + getAccuracy() + PER_CENT + TRAIN_AND_TEST
				+ getDuration() / 60000.0 + TIME;
	}

	public DataBase getDataBase() {
		return this.dataBase;
	}

	public Double getAccuracy() {
		return accuracy;
	}

	public String getNameAlgo() {
		return nameAlgo;

	}

	public Saveable getModel() {
		return this.model;
	}

	public long getDuration() {
		return this.duration;
	}

	/**
	 *
	 * Display the result.
	 *
	 */

	public void displayResult() {
		System.out.println(getResult());
	}
}
