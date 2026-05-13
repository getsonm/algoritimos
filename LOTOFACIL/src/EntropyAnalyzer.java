/**
 * Autor: getson@engineer.com
 */
import java.util.*;

/**
 * Analisador de Entropia e Informação Mútua
 * Mede independência entre números e identifica relações informativas
 */
public class EntropyAnalyzer {
    private List<Set<Integer>> historicalSequences;
    private int minNum;
    private int maxNum;
    private int tamanhoAposta;
    
    // Matrizes de probabilidade
    private Map<Integer, Double> singleProbabilities; // P(X)
    private Map<String, Double> jointProbabilities;  // P(X,Y)
    private Map<Integer, Double> entropyPerNumber;   // H(X)
    private Map<String, Double> mutualInformation;   // I(X;Y)
    private Map<Integer, Double> numberScores;       // score final
    
    public EntropyAnalyzer(List<Set<Integer>> sequences, int minNum, int maxNum, int betsSize) {
        this.historicalSequences = new ArrayList<>(sequences);
        this.minNum = minNum;
        this.maxNum = maxNum;
        this.tamanhoAposta = betsSize;
        this.singleProbabilities = new HashMap<>();
        this.jointProbabilities = new HashMap<>();
        this.entropyPerNumber = new HashMap<>();
        this.mutualInformation = new HashMap<>();
        this.numberScores = new HashMap<>();
        
        calculateProbabilities();
        calculateEntropies();
        calculateMutualInformation();
        calculateFinalScores();
    }
    
    private void calculateProbabilities() {
        final int n = historicalSequences.size();
        
        // P(X) para cada número
        for (int num = minNum; num <= maxNum; num++) {
            final int numFinal = num;
            long count = historicalSequences.stream()
                    .filter(seq -> seq.contains(numFinal))
                    .count();
            singleProbabilities.put(num, count / (double) n);
        }
        
        // P(X,Y) para pares
        for (int i = minNum; i <= maxNum; i++) {
            final int iFinal = i;
            for (int j = i + 1; j <= maxNum; j++) {
                final int jFinal = j;
                long count = historicalSequences.stream()
                        .filter(seq -> seq.contains(iFinal) && seq.contains(jFinal))
                        .count();
                String key = i + "," + j;
                jointProbabilities.put(key, count / (double) n);
            }
        }
    }
    
    private void calculateEntropies() {
        // H(X) = -Σ P(x) * log2(P(x))
        for (int num = minNum; num <= maxNum; num++) {
            double p = singleProbabilities.get(num);
            double entropy = 0;
            if (p > 0 && p < 1) {
                entropy = -(p * log2(p) + (1 - p) * log2(1 - p));
            }
            entropyPerNumber.put(num, entropy);
        }
    }
    
    private void calculateMutualInformation() {
        // I(X;Y) = H(X) + H(Y) - H(X,Y)
        for (int i = minNum; i <= maxNum; i++) {
            for (int j = i + 1; j <= maxNum; j++) {
                double px = singleProbabilities.get(i);
                double py = singleProbabilities.get(j);
                double pxy = jointProbabilities.get(i + "," + j);
                
                double hx = entropyPerNumber.get(i);
                double hy = entropyPerNumber.get(j);
                
                // H(X,Y) = -Σ P(x,y) * log2(P(x,y))
                double hxy = 0;
                if (pxy > 0) {
                    hxy -= pxy * log2(pxy);
                    double pxny = px - pxy; // P(x, ¬y)
                    if (pxny > 0) hxy -= pxny * log2(pxny);
                    double pnxy = py - pxy; // P(¬x, y)
                    if (pnxy > 0) hxy -= pnxy * log2(pnxy);
                    double pnxny = 1.0 - px - py + pxy; // P(¬x, ¬y)
                    if (pnxny > 0) hxy -= pnxny * log2(pnxny);
                }
                
                double mi = hx + hy - hxy;
                String key = i + "," + j;
                mutualInformation.put(key, Math.max(0, mi));
            }
        }
    }
    
    private void calculateFinalScores() {
        // Score de cada número baseado em:
        // 1. Probabilidade individual (30%)
        // 2. Entropia (números com entropia moderada são melhores) (30%)
        // 3. Informação Mútua com outros (40%)
        
        for (int num = minNum; num <= maxNum; num++) {
            double p = singleProbabilities.get(num);
            double h = entropyPerNumber.get(num);
            
            // Entropia normalizada (máximo é log2(2) = 1)
            double hNorm = h / 1.0; // log2(2)
            
            // Informação Mútua média com todos os outros
            double miSum = 0;
            int miCount = 0;
            for (int other = minNum; other <= maxNum; other++) {
                if (other == num) continue;
                String key = Math.min(num, other) + "," + Math.max(num, other);
                double mi = mutualInformation.getOrDefault(key, 0.0);
                miSum += mi;
                miCount++;
            }
            double miAvg = miCount > 0 ? miSum / miCount : 0.0;
            
            // Preferir entropia moderada (0.4 a 0.6)
            double hScore = 1.0 - Math.abs(0.5 - hNorm);
            
            double score = (p * 0.30) + (hScore * 0.30) + (miAvg * 0.40);
            numberScores.put(num, score);
        }
    }
    
    private double log2(double x) {
        return Math.log(x) / Math.log(2);
    }
    
    public Map<Integer, Double> getNumberScores() {
        return new HashMap<>(numberScores);
    }
    
    public List<Integer> generateBetEntropy(Random rand) {
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
        
        // Roulette wheel selection: escolher baseado em scores de entropia
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
