/**
 * Autor: getson@engineer.com
 */
import java.util.*;

/**
 * Analisador de Regressão Logística Probabilística
 * Modela a probabilidade de cada número aparecer baseado em features
 * Combina múltiplas features em um score único
 */
public class LogisticRegressionAnalyzer {
    private List<Set<Integer>> historicalSequences;
    private int minNum;
    private int maxNum;
    private int tamanhoAposta;
    
    // Weights dos features (treinados por gradiente descendente simples)
    private Map<Integer, Double> weights;
    private Map<Integer, Double> numberScores;
    
    // Features para cada número
    private Map<Integer, double[]> features;
    
    public LogisticRegressionAnalyzer(List<Set<Integer>> sequences, int minNum, int maxNum, int betsSize) {
        this.historicalSequences = new ArrayList<>(sequences);
        this.minNum = minNum;
        this.maxNum = maxNum;
        this.tamanhoAposta = betsSize;
        this.weights = new HashMap<>();
        this.numberScores = new HashMap<>();
        this.features = new HashMap<>();
        
        // Extrair features e treinar
        extractFeatures();
        trainLogisticRegression();
        calculateScores();
    }
    
    private void extractFeatures() {
        // Features para cada número:
        // 0: frequência (aparece em % de sequências)
        // 1: recência (quanto tempo desde última aparição)
        // 2: ciclo regularidade (desvio padrão da distância entre aparições)
        // 3: coocorrência média (quantos números aparecem com este)
        // 4: bias (constante = 1)
        
        int n = historicalSequences.size();
        
        for (int num = minNum; num <= maxNum; num++) {
            final int numFinal = num;
            double[] feat = new double[5];
            
            // Feature 0: Frequência
            long count = historicalSequences.stream()
                    .filter(seq -> seq.contains(numFinal))
                    .count();
            feat[0] = count / (double) n;
            
            // Feature 1: Recência (normalizada)
            List<Integer> positions = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (historicalSequences.get(i).contains(num)) {
                    positions.add(i);
                }
            }
            if (!positions.isEmpty()) {
                int recency = n - positions.get(positions.size() - 1);
                feat[1] = recency / (double) n; // 0 = apareceu no último, 1 = nunca apareceu recentemente
            } else {
                feat[1] = 1.0;
            }
            
            // Feature 2: Regularidade do ciclo
            if (positions.size() > 1) {
                double[] gaps = new double[positions.size() - 1];
                for (int i = 1; i < positions.size(); i++) {
                    gaps[i - 1] = positions.get(i) - positions.get(i - 1);
                }
                double meanGap = Arrays.stream(gaps).average().orElse(0);
                double variance = 0;
                for (double gap : gaps) {
                    variance += (gap - meanGap) * (gap - meanGap);
                }
                variance /= Math.max(1, gaps.length);
                double stdDev = Math.sqrt(variance);
                feat[2] = 1.0 / (1.0 + stdDev); // normalizado entre 0 e 1
            } else {
                feat[2] = 0.5;
            }
            
            // Feature 3: Coocorrência média
            double cooccSum = 0;
            int cooccCount = 0;
            for (Set<Integer> seq : historicalSequences) {
                if (seq.contains(num)) {
                    cooccSum += seq.size();
                    cooccCount++;
                }
            }
            feat[3] = (cooccCount > 0) ? (cooccSum / cooccCount) / (double) (maxNum - minNum + 1) : 0.5;
            
            // Feature 4: Bias
            feat[4] = 1.0;
            
            features.put(num, feat);
        }
    }
    
    private void trainLogisticRegression() {
        // Inicializar pesos aleatoriamente próximo a 0
        Random rand = new Random(42);
        for (int num = minNum; num <= maxNum; num++) {
            weights.put(num, (rand.nextDouble() - 0.5) * 0.1);
        }
        
        // Gradient descent simples
        double learningRate = 0.01;
        int epochs = 50;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int num = minNum; num <= maxNum; num++) {
                double error = 0;
                double gradient = 0;
                
                for (Set<Integer> seq : historicalSequences) {
                    double[] feat = features.get(num);
                    double logit = weights.get(num) * feat[4];
                    for (int i = 0; i < 4; i++) {
                        logit += weights.getOrDefault("w" + i, 0.5) * feat[i];
                    }
                    
                    // Sigmoid
                    double pred = sigmoid(logit);
                    double target = seq.contains(num) ? 1.0 : 0.0;
                    
                    error += (pred - target) * (pred - target);
                    gradient += (pred - target) * feat[4];
                }
                
                // Atualizar peso
                double newWeight = weights.get(num) - learningRate * gradient;
                weights.put(num, newWeight);
            }
        }
    }
    
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    private void calculateScores() {
        for (int num = minNum; num <= maxNum; num++) {
            double[] feat = features.get(num);
            double logit = weights.get(num) * feat[4];
            
            // Adicionar componentes de features
            logit += feat[0] * 0.4;    // frequência
            logit += (1.0 - feat[1]) * 0.3;  // recência invertida
            logit += feat[2] * 0.2;    // regularidade
            logit += feat[3] * 0.1;    // coocorrência
            
            double prob = sigmoid(logit);
            numberScores.put(num, prob);
        }
    }
    
    public Map<Integer, Double> getNumberScores() {
        return new HashMap<>(numberScores);
    }
    
    public List<Integer> generateBetLogistic(Random rand) {
        // Gerar aposta usando roleta ponderada estocástica
        // Garante diversidade mesmo com mesmos dados históricos
        
        List<Integer> bet = new ArrayList<>();
        Set<Integer> chosen = new HashSet<>();
        
        // Calcular soma total de scores para normalização
        double totalScore = numberScores.values().stream().mapToDouble(Double::doubleValue).sum();
        
        if (totalScore <= 0) {
            // Fallback: seleção aleatória pura
            List<Integer> allNums = new ArrayList<>();
            for (int num = minNum; num <= maxNum; num++) {
                allNums.add(num);
            }
            Collections.shuffle(allNums, rand);
            return new ArrayList<>(allNums.subList(0, Math.min(tamanhoAposta, allNums.size())));
        }
        
        // Roulette wheel selection: escolher baseado em probabilidades
        while (bet.size() < tamanhoAposta) {
            double spin = rand.nextDouble() * totalScore;
            double accumulated = 0.0;
            Integer selected = null;
            
            for (int num = minNum; num <= maxNum; num++) {
                if (chosen.contains(num)) continue;
                
                double score = numberScores.getOrDefault(num, 0.0);
                accumulated += score;
                
                if (spin <= accumulated) {
                    selected = num;
                    break;
                }
            }
            
            // Fallback se nenhum foi escolhido (números já foram selecionados)
            if (selected == null) {
                for (int num = minNum; num <= maxNum; num++) {
                    if (!chosen.contains(num)) {
                        selected = num;
                        break;
                    }
                }
            }
            
            if (selected != null) {
                bet.add(selected);
                chosen.add(selected);
            }
        }
        
        Collections.sort(bet);
        return bet;
    }
}
