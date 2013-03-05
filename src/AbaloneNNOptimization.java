import java.io.File;

import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.FixedIterationTrainer;
import shared.Instance;
import shared.SumOfSquaresError;
import shared.reader.CSVDataSetReader;
import func.nn.FeedForwardNetwork;
import func.nn.FeedForwardNeuralNetworkFactory;


/**
 * Testing RandomizedHillClimbing, SimulatedAnnealing, and GeneticAlgorithms
 * as optimization functions the weights of a feed-forward NeuralNetwork
 * @author Tim Swihart
 * @date 2013-03-05
 */
public class AbaloneNNOptimization {
	/**
	 * Tests out the perceptron with the classic xor test
	 * @param args ignored
	 */
	public static void main(String[] args) {
	    DataSet set = null;

	    /*double[][][] data = {
	           { { 1, 1, 1, 1 }, { 0 } },
	           { { 1, 0, 1, 0 }, { 1 } },
	           { { 0, 1, 0, 1 }, { 1 } },
	           { { 0, 0, 0, 0 }, { 0 } }
	    };
	    Instance[] patterns = new Instance[data.length];
	    for (int i = 0; i < patterns.length; i++) {
	        patterns[i] = new Instance(data[i][0]);
	        patterns[i].setLabel(new Instance(data[i][1]));
	    }*/


	    try {
			set = (new CSVDataSetReader((new File(".")).getAbsolutePath()+"/data/abalone.data")).read();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    Instance[] patterns = set.getInstances();
	    
	    FeedForwardNeuralNetworkFactory factory = new FeedForwardNeuralNetworkFactory();
	    FeedForwardNetwork network = factory.createClassificationNetwork(new int[] { 8, 3, 1 });
	    ErrorMeasure measure = new SumOfSquaresError();
	    NeuralNetworkOptimizationProblem nno = new NeuralNetworkOptimizationProblem(
	        set, network, measure);
	    
	    OptimizationAlgorithm rhc = new RandomizedHillClimbing(nno);
	    OptimizationAlgorithm sa = new SimulatedAnnealing(1E11, 0.95, nno);
	    OptimizationAlgorithm ga = new StandardGeneticAlgorithm(400, 100, 10, nno);
	    
	    //RHC
	    System.out.println("\nRandomized Hill Climbing");
	    FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 5000);
	    fit.train();
	    Instance rhcOpt = rhc.getOptimal();
	    network.setWeights(rhcOpt.getData());
	    
	    for (int i = 0; i < patterns.length; i++) {
	        network.setInputValues(patterns[i].getData());
	        network.run();
	        System.out.println("~~");
	        System.out.println(patterns[i].getLabel());
	        System.out.println(network.getOutputValues());
	    }
	    
	    //SA
	    System.out.println("\nSimulated Annealing");
	    fit = new FixedIterationTrainer(rhc, 5000);
	    fit.train();
	    Instance saOpt = rhc.getOptimal();
	    network.setWeights(saOpt.getData());
	    for (int i = 0; i < patterns.length; i++) {
	        network.setInputValues(patterns[i].getData());
	        network.run();
	        System.out.println("~~");
	        System.out.println(patterns[i].getLabel());
	        System.out.println(network.getOutputValues());
	    }
	    
	    //GA
	    System.out.println("\nGenetic Algorithms");
	    fit = new FixedIterationTrainer(rhc, 5000);
	    fit.train();
	    Instance gaOpt = rhc.getOptimal();
	    network.setWeights(gaOpt.getData());
	    for (int i = 0; i < patterns.length; i++) {
	        network.setInputValues(patterns[i].getData());
	        network.run();
	        System.out.println("~~");
	        System.out.println(patterns[i].getLabel());
	        System.out.println(network.getOutputValues());
	    }
	}
}
