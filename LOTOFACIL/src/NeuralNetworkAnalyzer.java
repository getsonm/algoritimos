/**
 * Autor: getson@engineer.com
 */
import java.util.*;

public class NeuralNetworkAnalyzer {
    private List<NeuralLayer> layers;
    private double learningRate;
    private int epochs;
    private double momentum = 0.9;
    private List<Double> trainingLoss;
    private int minNum = 1;
    private int maxNum = 25;
    
    public NeuralNetworkAnalyzer(int[] layerSizes, double learningRate, int epochs) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.layers = new ArrayList<>();
        this.trainingLoss = new ArrayList<>();
        
        ActivationFunction[] activations = {
            new ReLUActivation(),
            new ReLUActivation(),
            new SigmoidActivation()
        };
        
        for (int i = 0; i < layerSizes.length - 1; i++) {
            ActivationFunction activation = i < activations.length ? activations[i] : new ReLUActivation();
            layers.add(new NeuralLayer(layerSizes[i], layerSizes[i + 1], activation));
        }
    }
    
    public void train(List<Set<Integer>> sequences, int batchSize) {
        int totalSamples = sequences.size();
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double epochLoss = 0;
            int batchCount = 0;
            
            for (int start = 0; start < totalSamples; start += batchSize) {
                int end = (int) Math.min(start + batchSize, totalSamples);
                double batchLoss = 0;
                
                for (int i = start; i < end; i++) {
                    double[] input = sequenceToVector(sequences.get(i));
                    double[] target = input.clone();
                    
                    double[] output = forward(input);
                    double[] loss = new double[output.length];
                    
                    for (int j = 0; j < output.length; j++) {
                        loss[j] = output[j] - target[j];
                        batchLoss += loss[j] * loss[j];
                    }
                    
                    backward(loss);
                }
                
                batchLoss /= (end - start);
                epochLoss += batchLoss;
                batchCount++;
            }
            
            epochLoss /= Math.max(1, batchCount);
            trainingLoss.add(epochLoss);
            
            if ((epoch + 1) % (epochs / 10) == 0 || epoch == 0) {
                System.out.printf("Epoch %d/%d - Loss: %.6f\n", epoch + 1, epochs, epochLoss);
            }
        }
    }
    
    private double[] forward(double[] input) {
        double[] current = input;
        for (NeuralLayer layer : layers) {
            current = layer.forward(current);
        }
        return current;
    }
    
    private void backward(double[] outputError) {
        double[] error = outputError;
        
        for (int i = layers.size() - 1; i >= 0; i--) {
            error = layers.get(i).backward(error, learningRate);
        }
    }
    
    public Map<Integer, Double> predictNumberScores() {
        Map<Integer, Double> scores = new HashMap<>();
        
        for (int num = minNum; num <= maxNum; num++) {
            double[] input = new double[maxNum - minNum + 1];
            input[num - minNum] = 1.0;
            
            double[] output = forward(input);
            
            // Garantir que o índice está dentro dos limites
            double score = 0.0;
            if (num - minNum < output.length) {
                score = normalizeScore(output[num - minNum]);
            }
            scores.put(num, score);
        }
        
        return scores;
    }
    
    public Map<Integer, Double> predictWithContext(List<Set<Integer>> recentSequences) {
        Map<Integer, Double> scores = new HashMap<>();
        
        for (int num = minNum; num <= maxNum; num++) {
            scores.put(num, 0.0);
        }
        
        if (recentSequences.isEmpty()) {
            return scores;
        }
        
        try {
            for (int num = minNum; num <= maxNum; num++) {
                double sumScore = 0;
                int count = 0;
                
                for (Set<Integer> recent : recentSequences) {
                    double[] input = sequenceToVector(recent);
                    double[] output = forward(input);
                    
                    // Validação de índice
                    if (num - minNum >= 0 && num - minNum < output.length) {
                        sumScore += output[num - minNum];
                        count++;
                    }
                }
                
                double avgScore = count > 0 ? sumScore / count : 0;
                scores.put(num, normalizeScore(avgScore));
            }
        } catch (Exception e) {
            // Retornar scores padrão em caso de erro
            System.err.println("Erro em predictWithContext: " + e.getMessage());
        }
        
        return scores;
    }
    
    private double[] sequenceToVector(Set<Integer> sequence) {
        double[] vector = new double[maxNum - minNum + 1];
        for (Integer num : sequence) {
            if (num >= minNum && num <= maxNum) {
                vector[num - minNum] = 1.0;
            }
        }
        return vector;
    }
    
    private double normalizeScore(double score) {
        score = Math.max(0, Math.min(1, score));
        return score;
    }
    
    public List<Double> getTrainingLoss() {
        return trainingLoss;
    }
    
    public List<List<Integer>> generateBets(int count) {
        List<List<Integer>> bets = new ArrayList<>();
        Map<Integer, Double> scores = predictNumberScores();
        
        for (int b = 0; b < count; b++) {
            List<Integer> bet = new ArrayList<>();
            Set<Integer> used = new HashSet<>();
            
            List<Map.Entry<Integer, Double>> sorted = new ArrayList<>(scores.entrySet());
            sorted.sort((a, b1) -> Double.compare(b1.getValue(), a.getValue()));
            
            Random rand = new Random(b * 7 + System.currentTimeMillis());
            
            for (int i = 0; i < 15 && bet.size() < 15; i++) {
                int idx = i + rand.nextInt(Math.max(1, Math.min(5, sorted.size() - i)));
                Integer num = sorted.get(Math.min(idx, sorted.size() - 1)).getKey();
                if (!used.contains(num) && num >= 1 && num <= 25) {
                    bet.add(num);
                    used.add(num);
                }
            }
            
            while (bet.size() < 15) {
                int num = rand.nextInt(25) + 1;
                if (!used.contains(num)) {
                    bet.add(num);
                    used.add(num);
                }
            }
            
            Collections.sort(bet);
            bets.add(bet);
        }
        
        return bets;
    }

    
    public void printNetworkInfo() {
        System.out.println("\n=== NEURAL NETWORK ARCHITECTURE ===");
        System.out.println("Total Layers: " + layers.size());
        System.out.println("Learning Rate: " + learningRate);
        System.out.println("Epochs: " + epochs);
        System.out.println("Momentum: " + momentum);
        System.out.println();
        
        for (int i = 0; i < layers.size(); i++) {
            NeuralLayer layer = layers.get(i);
            System.out.printf("Layer %d: %d → %d (Activation: %s)\n",
                i + 1,
                layer.getInputSize(),
                layer.getOutputSize(),
                layer.getActivationName());
        }
        System.out.println();
    }
    
    public void printTrainingProgress() {
        if (trainingLoss.isEmpty()) {
            System.out.println("No training data available");
            return;
        }
        
        System.out.println("\n=== TRAINING PROGRESS ===");
        System.out.println("Final Loss: " + trainingLoss.get(trainingLoss.size() - 1));
        System.out.println("Initial Loss: " + trainingLoss.get(0));
        System.out.println("Improvement: " + 
            (1.0 - (trainingLoss.get(trainingLoss.size() - 1) / trainingLoss.get(0))) * 100 + "%");
        
        System.out.println("\nLoss by Epoch (every 10%):");
        int step = (int) Math.max(1, trainingLoss.size() / 10);
        for (int i = 0; i < trainingLoss.size(); i += step) {
            int epoch = i + 1;
            double loss = trainingLoss.get(i);
            System.out.printf("  Epoch %d: %.6f\n", epoch, loss);
        }
        System.out.println();
    }
}
