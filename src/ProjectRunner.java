import java.util.ArrayList;

import shared.DataSet;
import shared.filt.LabelSplitFilter;
import shared.reader.CSVDataSetReader;


public class ProjectRunner {
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// Becuase I'm lazy and eclipse won't run multiple jobs at once it seems
		//ArrayList<Thread> threads = new ArrayList<Thread>();
		/*threads[0] = new Thread(new ContinuousPeaks());
		threads[1] = new Thread(new OneMax());
		threads[2] = new Thread(new Knapsack());
		threads[3] = new Thread(new NeuralNetTest());*/
		int[] iterations = {64000};
    	DataSet set = null;
		try {
			set = (new CSVDataSetReader("data/hd_train.csv")).read();
		} catch (Exception e1) {
			e1.printStackTrace();
		}
		(new LabelSplitFilter()).filter(set);
		
		for (int i : iterations) {
			(new NeuralNetTest(i, set)).run();
		}
	}

}
