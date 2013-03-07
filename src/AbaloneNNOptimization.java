import java.io.File;

import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.DataSet;
import shared.DataSetDescription;
import shared.ErrorMeasure;
import shared.FixedIterationTrainer;
import shared.Instance;
import shared.SumOfSquaresError;
import shared.filt.LabelSplitFilter;
import shared.reader.ArffDataSetReader;
import shared.reader.CSVDataSetReader;
import shared.reader.DataSetLabelBinarySeperator;
import shared.tester.AccuracyTestMetric;
import shared.tester.ConfusionMatrixTestMetric;
import shared.tester.NeuralNetworkTester;
import shared.tester.TestMetric;
import shared.tester.Tester;
//import shared.tester.AccuracyTestMetric;
//import shared.tester.NeuralNetworkTester;
//import shared.tester.TestMetric;
//import shared.tester.Tester;
import func.nn.feedfwd.FeedForwardNetwork;
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory;


/**
 * Testing RandomizedHillClimbing, SimulatedAnnealing, and GeneticAlgorithms
 * as optimization functions the weights of a feed-forward NeuralNetwork
 * @author Tim Swihart
 * @date 2013-03-05
 */
public class AbaloneNNOptimization {
	public static void main(String[] args) {
	    DataSet set = null;
	    try {
			set = (new ArffDataSetReader((new File(".")).getAbsolutePath()+"/data/letter-recognition-testing.arff")).read();
			LabelSplitFilter lsf = new LabelSplitFilter();
			lsf.filter(set);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    System.out.println(set.getDescription());
	    //DataSetLabelBinarySeperator.seperateLabels(set);
	    
	    FeedForwardNeuralNetworkFactory factory = new FeedForwardNeuralNetworkFactory();
	    FeedForwardNetwork network = factory.createClassificationNetwork(new int[] { 16, 8, 5 });
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
	    
	    /*for (int i = 0; i < patterns.length; i++) {
	        network.setInputValues(patterns[i].getData());
	        network.run();
	        System.out.println("~~");
	        System.out.println(patterns[i].getLabel());
	        System.out.println(network.getOutputValues());
	    }*/
	    
	    network.setWeights(rhcOpt.getData());
	    TestMetric acc = new AccuracyTestMetric();
        //TestMetric cm  = new ConfusionMatrixTestMetric(set.getLabelDataSet().getInstances());
        Tester t = new NeuralNetworkTester(network, acc);
        t.test(set.getInstances());
        
        acc.printResults();
	    
	    /*//SA
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
	    }*/
	}
}
