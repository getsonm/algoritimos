/**
 * Autor: getson@engineer.com
 */
import java.util.*;

/**
 * Analisador de Cadeia de Markov de Ordem Superior
 * Estende HMM com dependências de múltiplos passos anteriores
 * Captura padrões sequenciais mais complexos
 */
public class HigherOrderMarkovAnalyzer {
    private List<Set<Integer>> historicalSequences;
    private int minNum;
    private int maxNum;
    private int tamanhoAposta;
    
    // Ordem do Markov: 1 (depende do anterior), 2 (depende dos 2 anteriores), etc
    private int markovOrder = 2;
    
    // Matriz de transição de ordem superior: (estado[t-k], ..., estado[t-1]) -> estado[t]
    private Map<String, Map<Integer, Integer>> transitionCounts;
    private Map<String, Map<Integer, Double>> transitionProbs;
    private Map<Integer, Double> numberScores;
    
    public HigherOrderMarkovAnalyzer(List<Set<Integer>> sequences, int minNum, int maxNum, int betsSize) {
        this.historicalSequences = new ArrayList<>(sequences);
        this.minNum = minNum;
        this.maxNum = maxNum;
        this.tamanhoAposta = betsSize;
        this.transitionCounts = new HashMap<>();
        this.transitionProbs = new HashMap<>();
        this.numberScores = new HashMap<>();
        
        // Converter sequências em cadeias de números
        List<List<Integer>> chains = convertToNumberChains();
        
        // Treinar modelo Markov de ordem 2
        trainMarkovModel(chains);
        
        // Calcular scores
        calculateScoresFromMarkov();
    }
    
    private List<List<Integer>> convertToNumberChains() {
        List<List<Integer>> chains = new ArrayList<>();
        for (Set<Integer> seq : historicalSequences) {
            List<Integer> chain = new ArrayList<>(seq);
            Collections.sort(chain);
            chains.add(chain);
        }
        return chains;
    }
    
    private void trainMarkovModel(List<List<Integer>> chains) {
        // Para cada sequência, extrair transições de ordem superior
        for (List<Integer> chain : chains) {
            if (chain.size() <= markovOrder) continue;
            
            for (int t = markovOrder; t < chain.size(); t++) {
                // Estado anterior: concatenar últimas k números
                StringBuilder stateKey = new StringBuilder();
                for (int i = 1; i <= markovOrder; i++) {
                    if (stateKey.length() > 0) stateKey.append("|");
                    stateKey.append(chain.get(t - i));
                }
                
                // Estado atual
                int nextNum = chain.get(t);
                
                // Contar transição
                Map<Integer, Integer> counts = transitionCounts.computeIfAbsent(
                        stateKey.toString(),
                        k -> new HashMap<>()
                );
                counts.put(nextNum, counts.getOrDefault(nextNum, 0) + 1);
            }
        }
        
        // Converter contagens em probabilidades
        for (Map.Entry<String, Map<Integer, Integer>> entry : transitionCounts.entrySet()) {
            String state = entry.getKey();
            Map<Integer, Integer> counts = entry.getValue();
            
            int total = counts.values().stream().mapToInt(Integer::intValue).sum();
            Map<Integer, Double> probs = new HashMap<>();
            
            for (Map.Entry<Integer, Integer> countEntry : counts.entrySet()) {
                double prob = countEntry.getValue() / (double) total;
                probs.put(countEntry.getKey(), prob);
            }
            
            transitionProbs.put(state, probs);
        }
    }
    
    private void calculateScoresFromMarkov() {
        // Para cada número, calcular probabilidade média de transição para ele
        for (int num = minNum; num <= maxNum; num++) {
            double totalProb = 0;
            int count = 0;
            
            for (Map<Integer, Double> probs : transitionProbs.values()) {
                if (probs.containsKey(num)) {
                    totalProb += probs.get(num);
                    count++;
                }
            }
            
            double avgProb = count > 0 ? totalProb / count : 0.5;
            numberScores.put(num, avgProb);
        }
    }
    
    public Map<Integer, Double> getNumberScores() {
        return new HashMap<>(numberScores);
    }
    
    public List<Integer> generateBetMarkov(Random rand) {
        // Gerar aposta selecionando números com maior score
        List<Map.Entry<Integer, Double>> ranked = new ArrayList<>(numberScores.entrySet());
        ranked.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        
        List<Integer> bet = new ArrayList<>();
        
        // Usar transições para gerar sequência coerente
        if (!transitionProbs.isEmpty()) {
            // Escolher um estado inicial aleatório
            List<String> states = new ArrayList<>(transitionProbs.keySet());
            String currentState = states.get(rand.nextInt(states.size()));
            
            // Próximo número baseado no estado
            Map<Integer, Double> probs = transitionProbs.get(currentState);
            if (probs != null && !probs.isEmpty()) {
                List<Map.Entry<Integer, Double>> options = new ArrayList<>(probs.entrySet());
                options.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
                for (Map.Entry<Integer, Double> opt : options) {
                    if (bet.size() < tamanhoAposta) {
                        bet.add(opt.getKey());
                    }
                }
            }
        }
        
        // Completar com números top-scored se necessário
        if (bet.size() < tamanhoAposta) {
            for (Map.Entry<Integer, Double> entry : ranked) {
                if (bet.size() >= tamanhoAposta) break;
                if (!bet.contains(entry.getKey())) {
                    bet.add(entry.getKey());
                }
            }
        }
        
        Collections.sort(bet);
        return new ArrayList<>(bet.subList(0, Math.min(tamanhoAposta, bet.size())));
    }
    
    // Gerar sequência por caminhada aleatória pelo Markov
    public List<Integer> generateBetByWalkingMarkov(Random rand) {
        List<Integer> bet = new ArrayList<>();
        Set<Integer> used = new HashSet<>();
        
        if (transitionProbs.isEmpty()) {
            // Fallback: usar roleta ponderada com scores
            double totalScore = numberScores.values().stream().mapToDouble(Double::doubleValue).sum();
            
            if (totalScore <= 0) {
                // Seleção aleatória pura
                List<Integer> allNums = new ArrayList<>();
                for (int num = minNum; num <= maxNum; num++) {
                    allNums.add(num);
                }
                Collections.shuffle(allNums, rand);
                return new ArrayList<>(allNums.subList(0, Math.min(tamanhoAposta, allNums.size())));
            }
            
            // Roulette wheel selection
            while (bet.size() < tamanhoAposta) {
                double spin = rand.nextDouble() * totalScore;
                double accumulated = 0.0;
                Integer selected = null;
                
                for (int num = minNum; num <= maxNum; num++) {
                    if (used.contains(num)) continue;
                    
                    double score = numberScores.getOrDefault(num, 0.0);
                    accumulated += score;
                    
                    if (spin <= accumulated) {
                        selected = num;
                        break;
                    }
                }
                
                if (selected == null) {
                    for (int num = minNum; num <= maxNum; num++) {
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
        } else {
            // Começar em um estado aleatório
            List<String> states = new ArrayList<>(transitionProbs.keySet());
            String state = states.get(rand.nextInt(states.size()));
            
            // Caminhar pelo grafo de transições com seleção ponderada
            int attempts = 0;
            while (bet.size() < tamanhoAposta && attempts < 1000) {
                Map<Integer, Double> probs = transitionProbs.get(state);
                if (probs == null || probs.isEmpty()) break;
                
                // Selecionar próximo número por roulette wheel (ponderado)
                double totalProb = probs.values().stream().mapToDouble(Double::doubleValue).sum();
                double spin = rand.nextDouble() * totalProb;
                double accumulated = 0.0;
                Integer nextNum = null;
                
                for (Map.Entry<Integer, Double> entry : probs.entrySet()) {
                    // Pular números já usados
                    if (used.contains(entry.getKey())) continue;
                    
                    accumulated += entry.getValue();
                    if (spin <= accumulated) {
                        nextNum = entry.getKey();
                        break;
                    }
                }
                
                // Fallback: escolher primeiro número disponível
                if (nextNum == null) {
                    for (int num : probs.keySet()) {
                        if (!used.contains(num)) {
                            nextNum = num;
                            break;
                        }
                    }
                }
                
                if (nextNum != null && used.add(nextNum)) {
                    bet.add(nextNum);
                }
                
                // Atualizar estado: shift e add novo número
                String[] parts = state.split("\\|");
                StringBuilder newState = new StringBuilder();
                for (int i = 1; i < parts.length; i++) {
                    if (newState.length() > 0) newState.append("|");
                    newState.append(parts[i]);
                }
                if (newState.length() > 0) newState.append("|");
                newState.append(nextNum);
                
                state = newState.toString();
                attempts++;
            }
        }
        
        // Completar com números restantes usando roleta ponderada
        if (bet.size() < tamanhoAposta) {
            double totalScore = numberScores.values().stream().mapToDouble(Double::doubleValue).sum();
            
            while (bet.size() < tamanhoAposta && totalScore > 0) {
                double spin = rand.nextDouble() * totalScore;
                double accumulated = 0.0;
                Integer selected = null;
                
                for (int num = minNum; num <= maxNum; num++) {
                    if (used.contains(num)) continue;
                    
                    double score = numberScores.getOrDefault(num, 0.0);
                    accumulated += score;
                    
                    if (spin <= accumulated) {
                        selected = num;
                        break;
                    }
                }
                
                if (selected == null) {
                    for (int num = minNum; num <= maxNum; num++) {
                        if (!used.contains(num)) {
                            selected = num;
                            break;
                        }
                    }
                }
                
                if (selected != null) {
                    bet.add(selected);
                    used.add(selected);
                } else {
                    break;
                }
            }
        }
        
        Collections.sort(bet);
        return new ArrayList<>(bet.subList(0, Math.min(tamanhoAposta, bet.size())));
    }
}
