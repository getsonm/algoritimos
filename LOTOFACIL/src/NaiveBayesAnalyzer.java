/**
 * Autor: getson@engineer.com
 */
import java.util.*;
import java.util.stream.Collectors;

public class NaiveBayesAnalyzer {
    private List<Set<Integer>> transactions;
    private int minNumber;
    private int maxNumber;
    private int betSize;
    private Map<Integer, Double> numberProbability;

    public NaiveBayesAnalyzer(List<Set<Integer>> transactions, int minNumber, int maxNumber, int betSize) {
        this.transactions = transactions;
        this.minNumber = minNumber;
        this.maxNumber = maxNumber;
        this.betSize = betSize;
        this.numberProbability = new HashMap<>();
        trainNaiveBayes();
    }

    public List<Integer> generateBetNaiveBayes(Random random) {
        if (numberProbability.isEmpty()) {
            return generateRandomBet();
        }

        // Seleção estocástica: roleta ponderada pela probabilidade
        // Garante diversidade mesmo com mesmos dados históricos
        List<Integer> selectedNumbers = new ArrayList<>();
        Set<Integer> chosen = new HashSet<>();
        double totalProb = numberProbability.values().stream().mapToDouble(Double::doubleValue).sum();

        while (selectedNumbers.size() < betSize) {
            // Roulette wheel selection: escolhe aleatoriamente ponderado pela probabilidade
            double spin = random.nextDouble() * totalProb;
            double accumulated = 0.0;
            Integer selected = null;

            for (Map.Entry<Integer, Double> entry : numberProbability.entrySet()) {
                accumulated += entry.getValue();
                if (spin <= accumulated && !chosen.contains(entry.getKey())) {
                    selected = entry.getKey();
                    break;
                }
            }

            // Fallback se nenhum foi escolhido (números já foram selecionados)
            if (selected == null) {
                for (Map.Entry<Integer, Double> entry : numberProbability.entrySet()) {
                    if (!chosen.contains(entry.getKey())) {
                        selected = entry.getKey();
                        break;
                    }
                }
            }

            // Se ainda assim não encontrou, escolher aleatoriamente de não-selecionados
            if (selected == null) {
                for (int num = minNumber; num <= maxNumber; num++) {
                    if (!chosen.contains(num)) {
                        selected = num;
                        break;
                    }
                }
            }

            if (selected != null) {
                selectedNumbers.add(selected);
                chosen.add(selected);
            }
        }

        Collections.sort(selectedNumbers);
        return selectedNumbers;
    }

    private void trainNaiveBayes() {
        // P(número aparece) = frequência / total de transações
        for (int num = minNumber; num <= maxNumber; num++) {
            int count = 0;
            for (Set<Integer> transaction : transactions) {
                if (transaction.contains(num)) {
                    count++;
                }
            }
            double probability = count / (double) Math.max(1, transactions.size());
            numberProbability.put(num, probability);
        }

        // Calcular probabilidades condicionais considerando coocorrências
        calculateConditionalProbabilities();
    }

    private void calculateConditionalProbabilities() {
        // Matriz de coocorrência
        Map<String, Integer> cooccurrence = new HashMap<>();

        for (Set<Integer> transaction : transactions) {
            List<Integer> nums = new ArrayList<>(transaction);
            for (int i = 0; i < nums.size(); i++) {
                for (int j = i + 1; j < nums.size(); j++) {
                    int n1 = Math.min(nums.get(i), nums.get(j));
                    int n2 = Math.max(nums.get(i), nums.get(j));
                    String key = n1 + "_" + n2;
                    cooccurrence.put(key, cooccurrence.getOrDefault(key, 0) + 1);
                }
            }
        }

        // Atualizar probabilidades com bônus de coocorrência
        double maxCooccur = cooccurrence.values().stream()
                .mapToInt(Integer::intValue)
                .max()
                .orElse(1);

        for (String key : cooccurrence.keySet()) {
            String[] parts = key.split("_");
            int n1 = Integer.parseInt(parts[0]);
            int n2 = Integer.parseInt(parts[1]);
            double cooccurScore = cooccurrence.get(key) / maxCooccur;

            numberProbability.put(n1, numberProbability.get(n1) * (1.0 + cooccurScore * 0.3));
            numberProbability.put(n2, numberProbability.get(n2) * (1.0 + cooccurScore * 0.3));
        }

        // Normalizar probabilidades
        double total = numberProbability.values().stream().mapToDouble(Double::doubleValue).sum();
        if (total > 0) {
            for (Map.Entry<Integer, Double> e : numberProbability.entrySet()) {
                numberProbability.put(e.getKey(), e.getValue() / total);
            }
        }
    }

    private List<Integer> generateRandomBet() {
        Random rand = new Random();
        List<Integer> selected = new ArrayList<>();
        for (int num = minNumber; num <= maxNumber && selected.size() < betSize; num++) {
            if (rand.nextDouble() < 0.6) {
                selected.add(num);
            }
        }
        while (selected.size() < betSize) {
            int num = minNumber + rand.nextInt(maxNumber - minNumber + 1);
            if (!selected.contains(num)) {
                selected.add(num);
            }
        }
        Collections.sort(selected);
        return selected;
    }
}
