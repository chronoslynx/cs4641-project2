
public class ProjectRunner {
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// Becuase I'm lazy and eclipse won't run multiple jobs at once it seems
		Thread[] threads = new Thread[4];
		threads[0] = new Thread(new ContinuousPeaks());
		threads[1] = new Thread(new OneMax());
		threads[2] = new Thread(new Knapsack());
		threads[3] = new Thread(new NeuralNetTest());
		
		for (Thread t : threads) {
			t.start();
		}
		
		for (Thread t : threads) {
			try {
				t.join();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

}
