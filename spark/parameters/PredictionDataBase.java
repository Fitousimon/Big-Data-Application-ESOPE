package parameters;

import java.io.File;
import java.util.List;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class PredictionDataBase {
	private final List<Vector> dataVector;
	private final File file;

	public PredictionDataBase(String split, JavaSparkContext sc, File file) {

		this.file = file;
		// Initialize the logger and the JavaSparkContext

		// Create a list of vector from the dataset
		dataVector = sc.textFile(getFile().getPath()).map(s -> {
			String[] tab = s.split(split);
			double[] tabDouble = new double[tab.length + 1];
			tabDouble[0] = 0.0;
			for (int i = 0; i != tab.length; i++) {
				tabDouble[i + 1] = Double.parseDouble(tab[i]);
			}

			return Vectors.dense(tabDouble);
		}).collect();
	}

	public List<Vector> getDataVector() {
		return dataVector;
	}

	public File getFile() {
		return file;
	}

}
