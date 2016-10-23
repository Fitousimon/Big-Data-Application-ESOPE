package parameters;

import java.util.Arrays;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

/**
 * Creation of the dataBase in a readable format for Spark (which is a JavaRDD
 * of labeledPoints). A JavaRdd is a dataBase format, it reacts like a list,
 * with some very useful methods from Spark such as : foreach, filter etc.
 *
 * A LabeledPoint is a tuple formed by a double label, which is the value we
 * want the model to predict, and a vector (Not a java.util vector, but a vector
 * from Spark) of features, the regression models use this vector to find the
 * label.
 *
 * The parsedData is also divided into two differents dataBases, one for the
 * training of the algorithm, and the other to test the algorithm in order to
 * find its accuracy
 *
 * @author do390
 *
 */
public class DataBase {

	private static final String LINE = "--------------------------------------------------------------------------------------------------------------";

	private JavaRDD<LabeledPoint> parsedData; /* The whole dataBase */
	private final JavaRDD<LabeledPoint> trainingData;
	private final JavaRDD<LabeledPoint> testData;
	private int NUM_PARTITIONS = 4;

	/**
	 *
	 * Constructor, which take some parameters to construct the dataBase with
	 * parsedData, trainingData and testData.
	 *
	 * @param sc
	 *            : the current JavaSparkContext.
	 * @param dataList
	 *            : the current dataBase in a List<Vector> format.
	 * @param rowLabel
	 *            : the row index the algorithm should predict.
	 * @param weight_training
	 *            : between 0 and 1.
	 * @param weight_test
	 *            : between 0 and 1 (1-weight_training in fact).
	 */

	public DataBase(JavaSparkContext sc, Vector[] dataList, int rowLabel, double weight_training, double weight_test) {
		this.parsedData = sc.parallelize(Arrays.asList(this.toLabeledPoint(dataList, rowLabel)), NUM_PARTITIONS);
		JavaRDD<LabeledPoint>[] splits = this.getParsedData()
				.randomSplit(new double[] { weight_training, weight_test });

		this.trainingData = splits[0].cache();
		this.testData = splits[1].cache();
	}

	public JavaRDD<LabeledPoint> getParsedData() {
		return parsedData;
	}

	public JavaRDD<LabeledPoint> getTrainingData() {
		return trainingData;
	}

	public JavaRDD<LabeledPoint> getTestData() {
		return testData;
	}

	/**
	 *
	 * Transform a list of Vector into a list of LabeledPoints, knowing the
	 * index of the label the algorithm will predict.
	 *
	 * For some reasons (all the regression algorithms do not act on databases
	 * in the same way) we need to put a 0.0 column in the vector of features.
	 * For example the vector [1.2,3.5,-5.0] will be transformed into
	 * (3.5,[0.0,1.2,-5.0]) if we take 1 as rowLabel.
	 *
	 * @param list
	 *            : the list of Vector we want to transform.
	 * @param rowLabel
	 *            : the index of column the algorithm should predict.
	 *
	 * @return : ArrayList<LabeledPoint>
	 */

	public LabeledPoint[] toLabeledPoint(Vector[] list, int rowLabel) {
		LabeledPoint[] dataLabeledPoints = new LabeledPoint[list.length];

		Vector v;
		double[] values;
		for (int index = 0; index != list.length; index++) {
			values = new double[list[0].size()];
			values[0] = 0.0;
			v = list[index];
			for (int i = 0; i != rowLabel; i++) {
				values[i + 1] = v.apply(i);
			}
			for (int i = rowLabel + 1; i != v.size(); i++) {
				values[i] = list[index].apply(i);
			}
			dataLabeledPoints[index] = new LabeledPoint(v.apply(rowLabel), Vectors.dense(values));
		}
		return dataLabeledPoints;

	}

	/**
	 * Display the dataBase
	 */

	public void displayDataBase() {

		System.out.println("\n\n" + LINE + "\n\nDATABASE : \n\n" + LINE + "\n\n");
		getParsedData().foreach(lp -> System.out.println(lp.toString()));

		System.out.println("\n\n" + LINE + "\n\nTRAINING DATA :\n\n" + LINE + "\n\n");
		getTrainingData().foreach(lp -> System.out.println(lp.toString()));

		System.out.println("\n\n" + LINE + "\n\nTEST DATA :\n\n" + LINE + "\n\n");
		getTestData().foreach(lp -> System.out.println(lp.toString()));

	}

}
