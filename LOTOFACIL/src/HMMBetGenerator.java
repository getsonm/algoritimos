/**
 * Autor: getson@engineer.com
 */
import java.util.*;

public class HMMBetGenerator {
    
    private HMMSequenceAnalyzer hmm;
    private List<Set<Integer>> historicalSequences;
    private int tamanhoAposta;
    private int minNumero;
    private int maxNumero;
    private double hmmWeight;
    
    public HMMBetGenerator(List<Set<Integer>> sequences, int betsSize, int minNum, int maxNum, double weight) {
        this.historicalSequences = new ArrayList<>(sequences);
        this.tamanhoAposta = betsSize;
        this.minNumero = minNum;
        this.maxNumero = maxNum;
        this.hmmWeight = weight;
        
        this.hmm = new HMMSequenceAnalyzer(sequences);
    }
    
    public List<Integer> generateBetUsingHMM(int seed) {
        Random rand = new Random(seed);
        
        List<Integer> recentSequence = new ArrayList<>();
        if (historicalSequences.size() >= 5) {
            for (int i = historicalSequences.size() - 5; i < historicalSequences.size(); i++) {
                recentSequence.addAll(historicalSequences.get(i));
            }
        }
        
        Map<Integer, Double> hmmScores = hmm.getNumberScoresFromHMM();
        
        Map<Integer, Double> hybridScores = new HashMap<>();
        
        for (int num = minNumero; num <= maxNumero; num++) {
            double hmmScore = hmmScores.getOrDefault(num, 0.0);
            
            double frequencyScore = 0;
            for (Set<Integer> seq : historicalSequences) {
                if (seq.contains(num)) {
                    frequencyScore++;
                }
            }
            frequencyScore /= historicalSequences.size();
            
            double hybrid = (hmmWeight * hmmScore) + ((1 - hmmWeight) * frequencyScore);
            hybridScores.put(num, hybrid);
        }
        
        // Seleção estocástica usando roleta ponderada
        List<Integer> bet = new ArrayList<>();
        Set<Integer> used = new HashSet<>();
        
        double totalScore = hybridScores.values().stream().mapToDouble(Double::doubleValue).sum();
        
        while (bet.size() < tamanhoAposta && used.size() < (maxNumero - minNumero + 1)) {
            double spin = rand.nextDouble() * totalScore;
            double accumulated = 0.0;
            Integer selected = null;
            
            for (int num = minNumero; num <= maxNumero; num++) {
                if (used.contains(num)) continue;
                
                accumulated += hybridScores.getOrDefault(num, 0.0);
                if (spin <= accumulated) {
                    selected = num;
                    break;
                }
            }
            
            // Fallback se nenhum selecionado
            if (selected == null) {
                for (int num = minNumero; num <= maxNumero; num++) {
                    if (!used.contains(num)) {
                        selected = num;
                        break;
                    }
                }
            }
            
            if (selected != null) {
                bet.add(selected);
                used.add(selected);
            }
        }
        
        Collections.sort(bet);
        return bet;
    }
    
    public List<List<Integer>> generateMultipleBetsUsingHMM(int numBets, int diversityOverlap) {
        List<List<Integer>> bets = new ArrayList<>();
        Set<String> generated = new HashSet<>();
        
        for (int i = 0; i < numBets; i++) {
            int attempts = 0;
            List<Integer> bet = null;
            List<Integer> fallbackBet = null; // Fallback para aposta sem duplicata
            boolean found = false;
            
            // Tentar até encontrar uma aposta válida
            while (attempts < 500 && !found) {
                bet = generateBetUsingHMM((int)(System.nanoTime() + i * 1000L + attempts));
                String key = keyFromList(bet);
                
                // Verificar se já foi gerada (duplicata global)
                boolean notDuplicate = !generated.contains(key);
                
                // Guarda primeira aposta sem duplicata como fallback
                if (notDuplicate && fallbackBet == null) {
                    fallbackBet = new ArrayList<>(bet);
                }
                
                // Verificar restrição de diversidade com outras apostas
                boolean diverseEnough = bets.isEmpty() || checkDiversityConstraint(bets, bet, diversityOverlap);
                
                if (notDuplicate && diverseEnough) {
                    generated.add(key);
                    bets.add(bet);
                    found = true;
                }
                
                attempts++;
            }
            
            // Se não encontrou aposta diversa, usar fallback (sem duplicata, mas talvez com sobreposição)
            if (!found && fallbackBet != null) {
                String key = keyFromList(fallbackBet);
                generated.add(key);
                bets.add(fallbackBet);
                System.out.println("⚠ Aviso: Aposta " + (i + 1) + " adicionada com sobreposição > limite (após 500 tentativas)");
            } else if (!found && bet != null) {
                // Último recurso: adiciona a aposta mesmo que seja duplicata
                String key = keyFromList(bet);
                generated.add(key);
                bets.add(bet);
                System.out.println("⚠ Aviso: Aposta " + (i + 1) + " adicionada mesmo com restrições não atendidas (último recurso)");
            }
        }
        
        return bets;
    }
    
    public List<List<Integer>> analyzePatternsByHMM(int windowSize) {
        List<List<Integer>> patterns = new ArrayList<>();
        
        for (int state = 0; state < 5; state++) {
            List<Integer> pattern = new ArrayList<>();
            
            double[] stateVector = new double[maxNumero + 1];
            for (int num = 1; num <= maxNumero; num++) {
                for (Set<Integer> seq : historicalSequences.subList(
                    (int) java.lang.Math.max(0, historicalSequences.size() - windowSize),
                    historicalSequences.size())) {
                    if (seq.contains(num)) {
                        stateVector[num]++;
                    }
                }
            }
            
            for (int num = 1; num <= maxNumero; num++) {
                if (stateVector[num] > windowSize * 0.1) {
                    pattern.add(num);
                }
            }
            
            if (!pattern.isEmpty()) {
                Collections.sort(pattern);
                if (pattern.size() > 6) {
                    pattern = pattern.subList(0, 6);
                }
                patterns.add(pattern);
            }
        }
        
        return patterns;
    }
    
    private boolean checkDiversityConstraint(List<List<Integer>> existing, List<Integer> newBet, int maxOverlap) {
        for (List<Integer> bet : existing) {
            int overlap = 0;
            for (Integer num : newBet) {
                if (bet.contains(num)) {
                    overlap++;
                }
            }
            if (overlap > maxOverlap) {
                return false;
            }
        }
        return true;
    }
    
    private String keyFromList(List<Integer> list) {
        StringBuilder sb = new StringBuilder();
        for (Integer num : list) {
            if (sb.length() > 0) sb.append(",");
            sb.append(num);
        }
        return sb.toString();
    }
    
    public void printHMMAnalysis() {
        System.out.println("\n===== HMM SEQUENCE ANALYSIS =====");
        System.out.println("Hidden states: 5");
        
        Map<Integer, Double> scores = hmm.getNumberScoresFromHMM();
        List<Map.Entry<Integer, Double>> topNumbers = new ArrayList<>(scores.entrySet());
        topNumbers.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        
        System.out.println("\nTop 10 numbers by HMM score:");
        for (int i = 0; i < Math.min(10, topNumbers.size()); i++) {
            System.out.printf("  %2d: %.4f%n", topNumbers.get(i).getKey(), topNumbers.get(i).getValue());
        }
        
        List<List<Integer>> patterns = analyzePatternsByHMM(500);
        System.out.println("\nPattern states detected:");
        for (int i = 0; i < patterns.size(); i++) {
            System.out.printf("  State %d: %s%n", i, patterns.get(i));
        }
    }
    
    public Map<Integer, Double> getHMMNumberScores() {
        return hmm.getNumberScoresFromHMM();
    }
}
