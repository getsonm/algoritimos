/**
 * Autor: getson@engineer.com
 */
import java.util.*;

/**
 * Analisador Gradient Boosting para loteria
 * Combina múltiplos modelos fracos (árvores de decisão simples) em um forte
 */
public class GradientBoostingAnalyzer {
    private List<Set<Integer>> historicalSequences;
    private int minNum;
    private int maxNum;
    private int tamanhoAposta;
    
    // Modelos base
    private List<SimpleDecisionTree> trees;
    private List<Double> treeWeights;
    private Map<Integer, Double> numberScores;
    
    // Parâmetros
    private int numTrees = 10;
    private double learningRate = 0.1;
    
    public GradientBoostingAnalyzer(List<Set<Integer>> sequences, int minNum, int maxNum, int betsSize) {
        this.historicalSequences = new ArrayList<>(sequences);
        this.minNum = minNum;
        this.maxNum = maxNum;
        this.tamanhoAposta = betsSize;
        this.trees = new ArrayList<>();
        this.treeWeights = new ArrayList<>();
        this.numberScores = new HashMap<>();
        
        trainBoosting();
    }
    
    private void trainBoosting() {
        // Inicializar scores com média simples
        Map<Integer, Double> scores = new HashMap<>();
        for (int num = minNum; num <= maxNum; num++) {
            double freq = 0;
            for (Set<Integer> seq : historicalSequences) {
                if (seq.contains(num)) freq++;
            }
            scores.put(num, freq / Math.max(1.0, historicalSequences.size()));
        }
        
        // Treinar T árvores sequencialmente
        for (int t = 0; t < numTrees; t++) {
            // Criar dados com pesos (para gradient boosting)
            List<DataPoint> trainingData = createWeightedDataset(scores);
            
            // Treinar árvore
            SimpleDecisionTree tree = new SimpleDecisionTree(trainingData, minNum, maxNum);
            trees.add(tree);
            
            // Calcular predições da árvore
            Map<Integer, Double> treePreds = tree.predictAllNumbers();
            
            // Atualizar scores com learning rate
            for (int num = minNum; num <= maxNum; num++) {
                double treePred = treePreds.getOrDefault(num, 0.0);
                scores.put(num, scores.get(num) + learningRate * treePred);
            }
            
            // Limitar scores entre 0 e 1
            for (int num = minNum; num <= maxNum; num++) {
                scores.put(num, Math.max(0.0, Math.min(1.0, scores.get(num))));
            }
            
            treeWeights.add(learningRate);
        }
        
        this.numberScores = new HashMap<>(scores);
    }
    
    private List<DataPoint> createWeightedDataset(Map<Integer, Double> scores) {
        List<DataPoint> data = new ArrayList<>();
        
        for (int i = 0; i < historicalSequences.size(); i++) {
            Set<Integer> seq = historicalSequences.get(i);
            
            for (int num = minNum; num <= maxNum; num++) {
                double target = seq.contains(num) ? 1.0 : 0.0;
                double weight = Math.abs(target - scores.getOrDefault(num, 0.5));
                
                // Feature simples: posição relativa e frequência
                double freq = scores.getOrDefault(num, 0.5);
                
                data.add(new DataPoint(num, freq, target, weight));
            }
        }
        
        return data;
    }
    
    public Map<Integer, Double> getNumberScores() {
        return new HashMap<>(numberScores);
    }
    
    public List<Integer> generateBetGradientBoosting(Random rand) {
        List<Map.Entry<Integer, Double>> ranked = new ArrayList<>(numberScores.entrySet());
        ranked.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        
        List<Integer> bet = new ArrayList<>();
        
        // Usar ruleta ponderada em vez de selecionar apenas os top
        double totalScore = ranked.stream().mapToDouble(e -> Math.max(0.01, e.getValue())).sum();
        
        Set<Integer> selected = new HashSet<>();
        while (selected.size() < tamanhoAposta && selected.size() < ranked.size()) {
            double random = rand.nextDouble() * totalScore;
            double accumulated = 0.0;
            
            for (Map.Entry<Integer, Double> entry : ranked) {
                double score = Math.max(0.01, entry.getValue());
                accumulated += score;
                
                if (random <= accumulated && !selected.contains(entry.getKey())) {
                    selected.add(entry.getKey());
                    bet.add(entry.getKey());
                    break;
                }
            }
        }
        
        Collections.sort(bet);
        return new ArrayList<>(bet.subList(0, Math.min(tamanhoAposta, bet.size())));
    }
    
    // Classe interna: Árvore de decisão simples
    private static class SimpleDecisionTree {
        private List<DataPoint> data;
        private int minNum;
        private int maxNum;
        private Node root;
        
        SimpleDecisionTree(List<DataPoint> data, int minNum, int maxNum) {
            this.data = new ArrayList<>(data);
            this.minNum = minNum;
            this.maxNum = maxNum;
            this.root = buildTree(data, 0);
        }
        
        private Node buildTree(List<DataPoint> subset, int depth) {
            if (subset.isEmpty() || depth >= 5) {
                return new Node(0.5, null, null, null);
            }
            
            double avgTarget = subset.stream()
                    .mapToDouble(d -> d.target)
                    .average().orElse(0.5);
            
            if (depth >= 3) {
                return new Node(avgTarget, null, null, null);
            }
            
            // Split simples: por threshold
            double bestThreshold = 0.5;
            double bestGain = 0;
            
            for (double threshold = 0.3; threshold <= 0.7; threshold += 0.1) {
                List<DataPoint> left = new ArrayList<>();
                List<DataPoint> right = new ArrayList<>();
                
                for (DataPoint p : subset) {
                    if (p.feature < threshold) left.add(p);
                    else right.add(p);
                }
                
                if (left.isEmpty() || right.isEmpty()) continue;
                
                double entropy = calculateEntropy(subset);
                double leftEntropy = calculateEntropy(left);
                double rightEntropy = calculateEntropy(right);
                
                double gain = entropy - (left.size() * leftEntropy + right.size() * rightEntropy) / subset.size();
                
                if (gain > bestGain) {
                    bestGain = gain;
                    bestThreshold = threshold;
                }
            }
            
            List<DataPoint> left = new ArrayList<>();
            List<DataPoint> right = new ArrayList<>();
            for (DataPoint p : subset) {
                if (p.feature < bestThreshold) left.add(p);
                else right.add(p);
            }
            
            if (left.isEmpty() || right.isEmpty()) {
                return new Node(avgTarget, null, null, null);
            }
            
            return new Node(avgTarget,
                    buildTree(left, depth + 1),
                    buildTree(right, depth + 1),
                    bestThreshold);
        }
        
        private double calculateEntropy(List<DataPoint> subset) {
            if (subset.isEmpty()) return 0;
            double pos = subset.stream().filter(d -> d.target > 0.5).count() / (double) subset.size();
            double neg = 1.0 - pos;
            double entropy = 0;
            if (pos > 0) entropy -= pos * Math.log(pos) / Math.log(2);
            if (neg > 0) entropy -= neg * Math.log(neg) / Math.log(2);
            return entropy;
        }
        
        Map<Integer, Double> predictAllNumbers() {
            Map<Integer, Double> preds = new HashMap<>();
            for (int num = minNum; num <= maxNum; num++) {
                double freq = 0.5; // default
                DataPoint dummy = new DataPoint(num, freq, 0, 1);
                preds.put(num, predictSingle(dummy));
            }
            return preds;
        }
        
        private double predictSingle(DataPoint p) {
            Node current = root;
            while (current != null && current.threshold != null) {
                if (p.feature < current.threshold) {
                    current = current.left;
                } else {
                    current = current.right;
                }
            }
            return current != null ? current.value : 0.5;
        }
        
        private static class Node {
            double value;
            Node left;
            Node right;
            Double threshold;
            
            Node(double value, Node left, Node right, Double threshold) {
                this.value = value;
                this.left = left;
                this.right = right;
                this.threshold = threshold;
            }
        }
    }
    
    private static class DataPoint {
        int number;
        double feature;
        double target;
        double weight;
        
        DataPoint(int number, double feature, double target, double weight) {
            this.number = number;
            this.feature = feature;
            this.target = target;
            this.weight = weight;
        }
    }
}
