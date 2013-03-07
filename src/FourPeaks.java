import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.FourPeaksEvaluationFunction;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.SingleCrossOver;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;


public class FourPeaks {
	// size of out bit strings
	private static final int N = 60;
	
	// the threshold value
	private static final int T = N/10;
	
	//number of times to run the evaluation
	private static final int times = 10;
	
	public static void main(String[] args) {
	    int[] ranges = new int[N];
	    Arrays.fill(ranges, 2);
	    EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
	    Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
	    
	    double rhcTime = 0.0;
	    double rhcPerf = 0.0;
	    
	    double saTime = 0.0;
	    double saPerf = 0.0;
	    
	    double gaTime = 0.0;
	    double gaPerf = 0.0;
	    
	    double mimTime = 0.0;
	    double mimPerf = 0.0;
	    double start = 0.0;
	    for (int i = 0; i < times; i++) {
			RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
			FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
			start = System.nanoTime();
			fit.train();
			rhcTime += (System.nanoTime()-start)/1000000000.0;
			rhcPerf += ef.value(rhc.getOptimal());
			
			SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
			fit = new FixedIterationTrainer(sa, 200000);
			start = System.nanoTime();
			fit.train();
			saTime += (System.nanoTime()-start)/1000000000.0;
			saPerf += ef.value(sa.getOptimal());
			//System.out.println(ef.value(sa.getOptimal()));
			
			StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
	        fit = new FixedIterationTrainer(ga, 1000);
			start = System.nanoTime();
			fit.train();
			gaTime += (System.nanoTime()-start)/1000000000.0;
			gaPerf += ef.value(ga.getOptimal());
			
			MIMIC mimic = new MIMIC(200, 100, pop);
	        fit = new FixedIterationTrainer(mimic, 1000);
			start = System.nanoTime();
			fit.train();
			mimTime += (System.nanoTime()-start)/1000000000.0;
			mimPerf += ef.value(mimic.getOptimal());
	    }
	    File f = new File((new File("")).getAbsolutePath() + "/four_peaks.md");
	    BufferedWriter w = null;
	    try {
			w = new BufferedWriter(new FileWriter(f));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    try {
			w.write("#FOUR PEAKS RESULTS");
			w.newLine();
			w.newLine();
			w.write("##Randomized Hill Climbing");
			w.newLine();
			w.newLine();
			w.write("Performance: ");
			w.write(String.valueOf(rhcPerf/times));
			w.newLine();
			w.write("Timing: ");
			w.write(String.valueOf(rhcTime/times));
			w.newLine();
			w.newLine();
			w.write("##Simulated Annealing");
			w.newLine();
			w.newLine();
			w.write("Performance: ");
			w.write(String.valueOf(saPerf/times));
			w.newLine();
			w.write("Timing: ");
			w.write(String.valueOf(saTime/times));
			w.newLine();
			w.newLine();
			w.write("##Genetic Algorithms");
			w.newLine();
			w.newLine();
			w.write("Performance: ");
			w.write(String.valueOf(gaPerf/times));
			w.newLine();
			w.write("Timing: ");
			w.write(String.valueOf(gaTime/times));
			w.newLine();
			w.newLine();
			w.write("##MIMIC");
			w.newLine();
			w.newLine();
			w.write("Performance: ");
			w.write(String.valueOf(mimPerf/times));
			w.newLine();
			w.write("Timing: ");
			w.write(String.valueOf(mimTime/times));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
	    try {
			w.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.out.println("Unable to close w");
		}
	    System.out.println("Done");
	}
}
