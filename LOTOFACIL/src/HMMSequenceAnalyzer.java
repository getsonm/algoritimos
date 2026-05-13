/**
 * Autor: getson@engineer.com
 */
import java.util.*;

public class HMMSequenceAnalyzer {
    
    private static final int NUM_HIDDEN_STATES = 5;
    private static final int MAX_NUMBER = 60;
    private static final double SMOOTHING = 0.01;
    
    private double[][] transitionMatrix;
    private double[][] emissionMatrix;
    private double[] initialStateDistribution;
    private int[] hiddenStates;
    private List<Set<Integer>> sequences;
    
    public HMMSequenceAnalyzer(List<Set<Integer>> historicalSequences) {
        this.sequences = new ArrayList<>(historicalSequences);
        this.hiddenStates = new int[NUM_HIDDEN_STATES];
        
        initializeMatrices();
        trainBaumWelch();
    }
    
    private void initializeMatrices() {
        transitionMatrix = new double[NUM_HIDDEN_STATES][NUM_HIDDEN_STATES];
        emissionMatrix = new double[NUM_HIDDEN_STATES][MAX_NUMBER + 1];
        initialStateDistribution = new double[NUM_HIDDEN_STATES];
        
        Random rand = new Random(42);
        for (int i = 0; i < NUM_HIDDEN_STATES; i++) {
            initialStateDistribution[i] = 1.0 / NUM_HIDDEN_STATES;
            for (int j = 0; j < NUM_HIDDEN_STATES; j++) {
                transitionMatrix[i][j] = rand.nextDouble();
            }
            for (int j = 1; j <= MAX_NUMBER; j++) {
                emissionMatrix[i][j] = rand.nextDouble();
            }
        }
        
        normalize();
    }
    
    private void normalize() {
        for (int i = 0; i < NUM_HIDDEN_STATES; i++) {
            double sum = 0;
            for (int j = 0; j < NUM_HIDDEN_STATES; j++) {
                sum += transitionMatrix[i][j];
            }
            for (int j = 0; j < NUM_HIDDEN_STATES; j++) {
                transitionMatrix[i][j] /= sum;
            }
        }
        
        for (int i = 0; i < NUM_HIDDEN_STATES; i++) {
            double sum = 0;
            for (int j = 1; j <= MAX_NUMBER; j++) {
                sum += emissionMatrix[i][j];
            }
            for (int j = 1; j <= MAX_NUMBER; j++) {
                emissionMatrix[i][j] /= sum;
            }
        }
        
        double sum = 0;
        for (int i = 0; i < NUM_HIDDEN_STATES; i++) {
            sum += initialStateDistribution[i];
        }
        for (int i = 0; i < NUM_HIDDEN_STATES; i++) {
            initialStateDistribution[i] /= sum;
        }
    }
    
    private void trainBaumWelch() {
        int iterations = 10;
        
        for (int iter = 0; iter < iterations; iter++) {
            double[][][] gamma = new double[sequences.size()][][];
            double[][][] xi = new double[sequences.size()][][];
            
            for (int seqIdx = 0; seqIdx < sequences.size(); seqIdx++) {
                List<Integer> sequence = new ArrayList<>(sequences.get(seqIdx));
                int T = sequence.size();
                
                double[][] alpha = forwardAlgorithm(sequence);
                double[][] beta = backwardAlgorithm(sequence);
                
                gamma[seqIdx] = new double[T][NUM_HIDDEN_STATES];
                xi[seqIdx] = new double[T - 1][NUM_HIDDEN_STATES * NUM_HIDDEN_STATES];
                
                double[] c = new double[T];
                for (int t = 0; t < T; t++) {
                    c[t] = 0;
                    for (int i = 0; i < NUM_HIDDEN_STATES; i++) {
                        gamma[seqIdx][t][i] = alpha[t][i] * beta[t][i];
                        c[t] += gamma[seqIdx][t][i];
                    }
                    if (c[t] > 0) {
                        for (int i = 0; i < NUM_HIDDEN_STATES; i++) {
                            gamma[seqIdx][t][i] /= c[t];
                        }
                    }
                }
                
                for (int t = 0; t < T - 1; t++) {
                    double denom = 0;
                    for (int i = 0; i < NUM_HIDDEN_STATES; i++) {
                        for (int j = 0; j < NUM_HIDDEN_STATES; j++) {
                            denom += alpha[t][i] * transitionMatrix[i][j] * 
                                    emissionMatrix[j][sequence.get(t + 1)] * beta[t + 1][j];
                        }
                    }
                    
                    for (int i = 0; i < NUM_HIDDEN_STATES; i++) {
                        for (int j = 0; j < NUM_HIDDEN_STATES; j++) {
                            if (denom > 0) {
                                xi[seqIdx][t][i * NUM_HIDDEN_STATES + j] = 
                                    (alpha[t][i] * transitionMatrix[i][j] * 
                                    emissionMatrix[j][sequence.get(t + 1)] * beta[t + 1][j]) / denom;
                            }
                        }
                    }
                }
            }
            
            updateParameters(gamma, xi);
        }
    }
    
    private double[][] forwardAlgorithm(List<Integer> sequence) {
        int T = sequence.size();
        double[][] alpha = new double[T][NUM_HIDDEN_STATES];
        
        for (int i = 0; i < NUM_HIDDEN_STATES; i++) {
            alpha[0][i] = initialStateDistribution[i] * emissionMatrix[i][sequence.get(0)];
        }
        
        for (int t = 1; t < T; t++) {
            for (int j = 0; j < NUM_HIDDEN_STATES; j++) {
                alpha[t][j] = 0;
                for (int i = 0; i < NUM_HIDDEN_STATES; i++) {
                    alpha[t][j] += alpha[t - 1][i] * transitionMatrix[i][j];
                }
                alpha[t][j] *= emissionMatrix[j][sequence.get(t)];
            }
        }
        
        return alpha;
    }
    
    private double[][] backwardAlgorithm(List<Integer> sequence) {
        int T = sequence.size();
        double[][] beta = new double[T][NUM_HIDDEN_STATES];
        
        for (int i = 0; i < NUM_HIDDEN_STATES; i++) {
            beta[T - 1][i] = 1.0;
        }
        
        for (int t = T - 2; t >= 0; t--) {
            for (int i = 0; i < NUM_HIDDEN_STATES; i++) {
                beta[t][i] = 0;
                for (int j = 0; j < NUM_HIDDEN_STATES; j++) {
                    beta[t][i] += transitionMatrix[i][j] * emissionMatrix[j][sequence.get(t + 1)] * beta[t + 1][j];
                }
            }
        }
        
        return beta;
    }
    
    private void updateParameters(double[][][] gamma, double[][][] xi) {
        double[] newInitial = new double[NUM_HIDDEN_STATES];
        double[][] newTransition = new double[NUM_HIDDEN_STATES][NUM_HIDDEN_STATES];
        double[][] newEmission = new double[NUM_HIDDEN_STATES][MAX_NUMBER + 1];
        
        for (int seqIdx = 0; seqIdx < sequences.size(); seqIdx++) {
            List<Integer> sequence = new ArrayList<>(sequences.get(seqIdx));
            int T = sequence.size();
            
            for (int i = 0; i < NUM_HIDDEN_STATES; i++) {
                newInitial[i] += gamma[seqIdx][0][i];
                
                for (int t = 0; t < T - 1; t++) {
                    double gammaSum = 0;
                    for (int k = 0; k < T - 1; k++) {
                        gammaSum += gamma[seqIdx][k][i];
                    }
                    if (gammaSum > 0) {
                        for (int j = 0; j < NUM_HIDDEN_STATES; j++) {
                            newTransition[i][j] += xi[seqIdx][t][i * NUM_HIDDEN_STATES + j];
                        }
                    }
                }
                
                for (int t = 0; t < T; t++) {
                    newEmission[i][sequence.get(t)] += gamma[seqIdx][t][i];
                }
            }
        }
        
        double sum = 0;
        for (int i = 0; i < NUM_HIDDEN_STATES; i++) {
            sum += newInitial[i];
        }
        if (sum > 0) {
            for (int i = 0; i < NUM_HIDDEN_STATES; i++) {
                initialStateDistribution[i] = (newInitial[i] + SMOOTHING) / (sum + NUM_HIDDEN_STATES * SMOOTHING);
            }
        }
        
        for (int i = 0; i < NUM_HIDDEN_STATES; i++) {
            sum = 0;
            for (int j = 0; j < NUM_HIDDEN_STATES; j++) {
                sum += newTransition[i][j];
            }
            if (sum > 0) {
                for (int j = 0; j < NUM_HIDDEN_STATES; j++) {
                    transitionMatrix[i][j] = (newTransition[i][j] + SMOOTHING) / (sum + NUM_HIDDEN_STATES * SMOOTHING);
                }
            }
        }
        
        for (int i = 0; i < NUM_HIDDEN_STATES; i++) {
            sum = 0;
            for (int num = 1; num <= MAX_NUMBER; num++) {
                sum += newEmission[i][num];
            }
            if (sum > 0) {
                for (int num = 1; num <= MAX_NUMBER; num++) {
                    emissionMatrix[i][num] = (newEmission[i][num] + SMOOTHING) / (sum + MAX_NUMBER * SMOOTHING);
                }
            }
        }
    }
    
    public int[] predictNextSequence(List<Integer> recentSequence, int length) {
        double[] stateProbs = new double[NUM_HIDDEN_STATES];
        
        if (recentSequence.isEmpty()) {
            System.arraycopy(initialStateDistribution, 0, stateProbs, 0, NUM_HIDDEN_STATES);
        } else {
            double[][] alpha = forwardAlgorithm(recentSequence);
            int T = recentSequence.size();
            double sum = 0;
            for (int i = 0; i < NUM_HIDDEN_STATES; i++) {
                stateProbs[i] = alpha[T - 1][i];
                sum += stateProbs[i];
            }
            if (sum > 0) {
                for (int i = 0; i < NUM_HIDDEN_STATES; i++) {
                    stateProbs[i] /= sum;
                }
            }
        }
        
        int[] prediction = new int[length];
        Random rand = new Random();
        Set<Integer> used = new HashSet<>();
        
        for (int step = 0; step < length; step++) {
            int bestNum = -1;
            double bestScore = -1;
            
            for (int num = 1; num <= MAX_NUMBER; num++) {
                if (used.contains(num)) continue;
                
                double score = 0;
                for (int i = 0; i < NUM_HIDDEN_STATES; i++) {
                    score += stateProbs[i] * emissionMatrix[i][num];
                }
                
                if (score > bestScore) {
                    bestScore = score;
                    bestNum = num;
                }
            }
            
            if (bestNum == -1) {
                for (int num = 1; num <= MAX_NUMBER; num++) {
                    if (!used.contains(num)) {
                        bestNum = num;
                        break;
                    }
                }
            }
            
            prediction[step] = bestNum;
            used.add(bestNum);
            
            double[] nextStateProbs = new double[NUM_HIDDEN_STATES];
            for (int j = 0; j < NUM_HIDDEN_STATES; j++) {
                for (int i = 0; i < NUM_HIDDEN_STATES; i++) {
                    nextStateProbs[j] += stateProbs[i] * transitionMatrix[i][j];
                }
                nextStateProbs[j] *= emissionMatrix[j][bestNum];
            }
            
            double sum = 0;
            for (int j = 0; j < NUM_HIDDEN_STATES; j++) {
                sum += nextStateProbs[j];
            }
            
            if (sum > 0) {
                for (int j = 0; j < NUM_HIDDEN_STATES; j++) {
                    stateProbs[j] = nextStateProbs[j] / sum;
                }
            }
        }
        
        Arrays.sort(prediction);
        return prediction;
    }
    
    public Map<Integer, Double> getNumberScoresFromHMM() {
        Map<Integer, Double> scores = new HashMap<>();
        
        for (int num = 1; num <= MAX_NUMBER; num++) {
            double score = 0;
            double weights = 0;
            
            for (int state = 0; state < NUM_HIDDEN_STATES; state++) {
                double weight = initialStateDistribution[state];
                score += weight * emissionMatrix[state][num];
                weights += weight;
            }
            
            scores.put(num, weights > 0 ? score / weights : 0);
        }
        
        return scores;
    }
    
    public double getSequenceLogLikelihood(List<Integer> sequence) {
        double[][] alpha = forwardAlgorithm(sequence);
        double likelihood = 0;
        for (int i = 0; i < NUM_HIDDEN_STATES; i++) {
            likelihood += alpha[sequence.size() - 1][i];
        }
        return likelihood > 0 ? java.lang.Math.log(likelihood) : Double.NEGATIVE_INFINITY;
    }
}
