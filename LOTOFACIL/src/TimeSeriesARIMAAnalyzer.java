/**
 * Autor: getson@engineer.com
 */
import java.util.*;

/**
 * Analisador de Séries Temporais com modelo ARIMA simplificado
 * Modela cada número como série temporal independente e prevê probabilidades
 */
public class TimeSeriesARIMAAnalyzer {
    private List<Set<Integer>> historicalSequences;
    private int minNum;
    private int maxNum;
    private int tamanhoAposta;
    
    // Parâmetros ARIMA
    private int p = 2; // AR order
    private int d = 1; // differencing
    private int q = 1; // MA order
    
    // Dados de séries para cada número
    private Map<Integer, List<Double>> numberSeries;
    private Map<Integer, Double> numberPredictions;
    
    public TimeSeriesARIMAAnalyzer(List<Set<Integer>> sequences, int minNum, int maxNum, int betsSize) {
        this.historicalSequences = new ArrayList<>(sequences);
        this.minNum = minNum;
        this.maxNum = maxNum;
        this.tamanhoAposta = betsSize;
        this.numberSeries = new HashMap<>();
        this.numberPredictions = new HashMap<>();
        
        buildTimeSeries();
        fitARIMA();
    }
    
    private void buildTimeSeries() {
        // Para cada número, criar série de frequências: 1 se aparece, 0 se não
        for (int num = minNum; num <= maxNum; num++) {
            List<Double> series = new ArrayList<>();
            for (Set<Integer> seq : historicalSequences) {
                series.add(seq.contains(num) ? 1.0 : 0.0);
            }
            numberSeries.put(num, series);
        }
    }
    
    private void fitARIMA() {
        // Ajusta modelo ARIMA para cada número
        for (int num = minNum; num <= maxNum; num++) {
            List<Double> series = numberSeries.get(num);
            if (series.size() < p + d + q + 5) {
                numberPredictions.put(num, 0.5);
                continue;
            }
            
            // Diferencing (d=1)
            List<Double> diffSeries = new ArrayList<>();
            for (int i = 1; i < series.size(); i++) {
                diffSeries.add(series.get(i) - series.get(i - 1));
            }
            
            // Calcular média móvel auto-regressiva
            double arComponent = estimateARComponent(series);
            double maComponent = estimateMaComponent(diffSeries);
            
            // Previsão: combina componentes
            double prediction = 0.5 + (arComponent * 0.4) + (maComponent * 0.3);
            prediction = Math.max(0.0, Math.min(1.0, prediction));
            numberPredictions.put(num, prediction);
        }
    }
    
    private double estimateARComponent(List<Double> series) {
        // Modelo AR simples: correlação com lags
        if (series.size() < p + 2) return 0.5;
        
        double sum = 0.0;
        double lastVal = series.get(series.size() - 1);
        for (int lag = 1; lag <= Math.min(p, series.size() - 1); lag++) {
            double prevVal = series.get(series.size() - 1 - lag);
            sum += prevVal * (1.0 / lag); // peso reduzido com lag maior
        }
        double mean = series.stream().mapToDouble(Double::doubleValue).average().orElse(0.5);
        return (sum / Math.max(1, p) + mean) / 2.0;
    }
    
    private double estimateMaComponent(List<Double> series) {
        // Componente MA: média do erro recente
        if (series.size() < q + 1) return 0.5;
        
        double sum = 0.0;
        for (int i = Math.max(0, series.size() - q); i < series.size(); i++) {
            sum += Math.abs(series.get(i));
        }
        double maAvg = sum / Math.max(1, q);
        return 0.5 + (maAvg * 0.2); // suavizado
    }
    
    public Map<Integer, Double> getNumberScores() {
        return new HashMap<>(numberPredictions);
    }
    
    public List<Integer> generateBetARIMA(Random rand) {
        // Gera aposta usando roleta ponderada estocástica
        // Garante números únicos e diversidade
        
        List<Integer> bet = new ArrayList<>();
        Set<Integer> used = new HashSet<>();
        
        // Calcular soma total de scores para normalização
        double totalScore = numberPredictions.values().stream()
                .mapToDouble(Double::doubleValue)
                .sum();
        
        if (totalScore <= 0) {
            // Fallback: seleção aleatória pura
            List<Integer> allNums = new ArrayList<>();
            for (int num = minNum; num <= maxNum; num++) {
                allNums.add(num);
            }
            Collections.shuffle(allNums, rand);
            return new ArrayList<>(allNums.subList(0, Math.min(tamanhoAposta, allNums.size())));
        }
        
        // Roulette wheel selection: escolher baseado em probabilidades ARIMA
        while (bet.size() < tamanhoAposta) {
            double spin = rand.nextDouble() * totalScore;
            double accumulated = 0.0;
            Integer selected = null;
            
            for (int num = minNum; num <= maxNum; num++) {
                if (used.contains(num)) continue;
                
                double score = numberPredictions.getOrDefault(num, 0.0);
                accumulated += score;
                
                if (spin <= accumulated) {
                    selected = num;
                    break;
                }
            }
            
            // Fallback se nenhum selecionado
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
        
        Collections.sort(bet);
        return bet;
    }
}
