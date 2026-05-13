/**
 * Autor: getson@engineer.com
 */
import java.util.*;
import java.util.stream.Collectors;

public class MetaEnsembleAnalyzer {
    private List<Set<Integer>> transactions;
    private int minNumber;
    private int maxNumber;
    private int betSize;
    private List<AlgorimoVoto> algoritmos;

    public MetaEnsembleAnalyzer(List<Set<Integer>> transactions, int minNumber, int maxNumber, int betSize) {
        this.transactions = transactions;
        this.minNumber = minNumber;
        this.maxNumber = maxNumber;
        this.betSize = betSize;
        this.algoritmos = new ArrayList<>();
    }

    public List<Integer> generateBetMetaEnsemble(Random random) {
        // Executar todos os algoritmos e coletar votos
        Map<Integer, Double> votes = new HashMap<>();
        for (int i = minNumber; i <= maxNumber; i++) {
            votes.put(i, 0.0);
        }

        // 1. ARIMA (série temporal)
        try {
            TimeSeriesARIMAAnalyzer arima = new TimeSeriesARIMAAnalyzer(transactions, minNumber, maxNumber, betSize);
            List<Integer> arimaResult = arima.generateBetARIMA(new Random(random.nextLong()));
            for (Integer num : arimaResult) {
                votes.put(num, votes.get(num) + 1.0);
            }
        } catch (Exception e) {
            System.err.println("Erro no voto ARIMA: " + e.getMessage());
        }

        // 2. Clustering (K-Means)
        try {
            ClusteringAnalyzer clustering = new ClusteringAnalyzer(transactions, minNumber, maxNumber, betSize);
            List<Integer> clusterResult = clustering.generateBetFromClusters(new Random(random.nextLong()));
            for (Integer num : clusterResult) {
                votes.put(num, votes.get(num) + 1.0);
            }
        } catch (Exception e) {
            System.err.println("Erro no voto Clustering: " + e.getMessage());
        }

        // 3. Gradient Boosting
        try {
            GradientBoostingAnalyzer boosting = new GradientBoostingAnalyzer(transactions, minNumber, maxNumber, betSize);
            List<Integer> boostResult = boosting.generateBetGradientBoosting(new Random(random.nextLong()));
            for (Integer num : boostResult) {
                votes.put(num, votes.get(num) + 1.0);
            }
        } catch (Exception e) {
            System.err.println("Erro no voto Gradient Boosting: " + e.getMessage());
        }

        // 4. Entropia
        try {
            EntropyAnalyzer entropy = new EntropyAnalyzer(transactions, minNumber, maxNumber, betSize);
            List<Integer> entropyResult = entropy.generateBetEntropy(new Random(random.nextLong()));
            for (Integer num : entropyResult) {
                votes.put(num, votes.get(num) + 1.0);
            }
        } catch (Exception e) {
            System.err.println("Erro no voto Entropia: " + e.getMessage());
        }

        // 5. Markov de Ordem Superior
        try {
            HigherOrderMarkovAnalyzer markov = new HigherOrderMarkovAnalyzer(transactions, minNumber, maxNumber, betSize);
            List<Integer> markovResult = markov.generateBetByWalkingMarkov(new Random(random.nextLong()));
            for (Integer num : markovResult) {
                votes.put(num, votes.get(num) + 1.0);
            }
        } catch (Exception e) {
            System.err.println("Erro no voto Markov: " + e.getMessage());
        }

        // 6. Regressão Logística
        try {
            LogisticRegressionAnalyzer logistic = new LogisticRegressionAnalyzer(transactions, minNumber, maxNumber, betSize);
            List<Integer> logisticResult = logistic.generateBetLogistic(new Random(random.nextLong()));
            for (Integer num : logisticResult) {
                votes.put(num, votes.get(num) + 1.0);
            }
        } catch (Exception e) {
            System.err.println("Erro no voto Regressão Logística: " + e.getMessage());
        }

        // 7. Random Forest
        try {
            RandomForestAnalyzer forest = new RandomForestAnalyzer(transactions, minNumber, maxNumber, betSize);
            List<Integer> forestResult = forest.generateBetRandomForest(new Random(random.nextLong()));
            for (Integer num : forestResult) {
                votes.put(num, votes.get(num) + 1.0);
            }
        } catch (Exception e) {
            System.err.println("Erro no voto Random Forest: " + e.getMessage());
        }

        // 8. Naive Bayes
        try {
            NaiveBayesAnalyzer naiveBayes = new NaiveBayesAnalyzer(transactions, minNumber, maxNumber, betSize);
            List<Integer> nbResult = naiveBayes.generateBetNaiveBayes(new Random(random.nextLong()));
            for (Integer num : nbResult) {
                votes.put(num, votes.get(num) + 1.0);
            }
        } catch (Exception e) {
            System.err.println("Erro no voto Naive Bayes: " + e.getMessage());
        }

        // 9. Frequência básica
        try {
            Map<Integer, Integer> frequency = calculateFrequency();
            List<Integer> freqResult = frequency.entrySet().stream()
                    .sorted((a, b) -> Integer.compare(b.getValue(), a.getValue()))
                    .limit(betSize)
                    .map(Map.Entry::getKey)
                    .collect(Collectors.toList());
            for (Integer num : freqResult) {
                votes.put(num, votes.get(num) + 1.0);
            }
        } catch (Exception e) {
            System.err.println("Erro no voto Frequência: " + e.getMessage());
        }

        // Selecionar os betSize números com maior votação
        List<Integer> selectedNumbers = votes.entrySet().stream()
                .sorted((a, b) -> Double.compare(b.getValue(), a.getValue()))
                .limit(betSize)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());

        // Preencher se necessário
        while (selectedNumbers.size() < betSize) {
            int num = minNumber + random.nextInt(maxNumber - minNumber + 1);
            if (!selectedNumbers.contains(num)) {
                selectedNumbers.add(num);
            }
        }

        Collections.sort(selectedNumbers);
        return selectedNumbers;
    }

    private Map<Integer, Integer> calculateFrequency() {
        Map<Integer, Integer> freq = new HashMap<>();
        for (int i = minNumber; i <= maxNumber; i++) {
            freq.put(i, 0);
        }
        for (Set<Integer> transaction : transactions) {
            for (Integer num : transaction) {
                if (num >= minNumber && num <= maxNumber) {
                    freq.put(num, freq.get(num) + 1);
                }
            }
        }
        return freq;
    }

    private static class AlgorimoVoto {
        String nome;
        double peso;

        AlgorimoVoto(String nome, double peso) {
            this.nome = nome;
            this.peso = peso;
        }
    }
}
