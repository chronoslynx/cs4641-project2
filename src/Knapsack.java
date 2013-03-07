import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.ContinuousPeaksEvaluationFunction;
import opt.example.CountOnesEvaluationFunction;
import opt.example.KnapsackEvaluationFunction;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.SingleCrossOver;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;


public class Knapsack implements Runnable {
	//number of times to run the evaluation
	private static final int times = 10;
    /** Random number generator */
    private static final Random random = new Random();
    /** The number of items */
    private static final int NUM_ITEMS = 40;
    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum volume for a single element */
    private static final double MAX_VOLUME = 50;
    /** The volume of the knapsack */
    private static final double KNAPSACK_VOLUME = 
         MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4;
	
    public static void main(String[] args) {
    	(new Knapsack()).run();
    }
    
	public void run() {
		int[] copies = new int[NUM_ITEMS];
        Arrays.fill(copies, COPIES_EACH);
        double[] weights = new double[NUM_ITEMS];
        double[] volumes = new double[NUM_ITEMS];
        for (int i = 0; i < NUM_ITEMS; i++) {
            weights[i] = random.nextDouble() * MAX_WEIGHT;
            volumes[i] = random.nextDouble() * MAX_VOLUME;
        }
         int[] ranges = new int[NUM_ITEMS];
        Arrays.fill(ranges, COPIES_EACH + 1);
        EvaluationFunction ef = new KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies);
        
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
	        fit = new FixedIterationTrainer(mimic, 100);
			start = System.nanoTime();
			fit.train();
			mimTime += (System.nanoTime()-start)/1000000000.0;
			mimPerf += ef.value(mimic.getOptimal());
	    }
	    File f = new File((new File("")).getAbsolutePath() + "/knapsack.md");
	    BufferedWriter w = null;
	    try {
			w = new BufferedWriter(new FileWriter(f));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    try {
			w.write("#KNAPSACK RESULTS\n\n");
			w.write("<table>\n");
			w.write("<tr>\n\t<td><strong>Algorithm</strong>\n\t<td><strong>Optimum</strong>\n\t<td><strong>Performance (s)</strong>\n</tr>\n");
			w.write("<tr>\n\t");
			w.write("<td><i>Randomized Hill Climbing</i>\n");
			w.write("\t<td>");
			w.write(String.valueOf(rhcPerf/times));
			w.newLine();
			w.write("\t<td>");
			w.write(String.valueOf(rhcTime/times));
			w.newLine();
			w.write("</tr><tr>\n\t");
			w.write("<td><i>Simulated Annealing</i>\n");
			w.write("\t<td>");
			w.write(String.valueOf(saPerf/times));
			w.newLine();
			w.write("\t<td>");
			w.write(String.valueOf(saTime/times));
			w.newLine();
			w.write("</tr><tr>\n\t");
			w.write("<td><i>Genetic Algorithms</i>\n");
			w.write("\t<td>");
			w.write(String.valueOf(gaPerf/times));
			w.newLine();
			w.write("\t<td>");
			w.write(String.valueOf(gaTime/times));
			w.newLine();
			w.write("</tr><tr>\n\t");
			w.write("<td><i>MIMIC</i>\n");
			w.write("\t<td>");
			w.write(String.valueOf(mimPerf/times));
			w.newLine();
			w.write("\t<td>");
			w.write(String.valueOf(mimTime/times));
			w.newLine();
			w.write("</tr></table>");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
	    try {
			w.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.out.println("Unable to close w (Knapsack)");
		}
	    System.out.println("Knapsack Done");
	}
}
