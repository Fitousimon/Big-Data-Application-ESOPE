package parameters;

import org.apache.log4j.Level;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

/**
 * Initialization of the SparkContext, and the JavaSparkContext which are
 * necessary to use the methods from Spark libraries.
 *
 * @author do390
 * @version 1.0
 *
 */

public class Context {

	/**
	 * Initialize the logger, to ensure that the console is more readable, and
	 * return a JavaSparkContext.
	 *
	 *
	 * @return JavaSparkContext
	 */

	public static JavaSparkContext scAndLogger() {

		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);

		SparkConf sparkConf = new SparkConf().setAppName("SparkConf").setMaster("local");
		return new JavaSparkContext(sparkConf);
	}

}
