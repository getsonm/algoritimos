/**
 * Autor: getson@engineer.com
 */
import java.util.*;
import java.util.stream.Collectors;

public class RandomForestAnalyzer {
    private List<Set<Integer>> transactions;
    private int minNumber;
    private int maxNumber;
    private int betSize;
    private int numTrees = 100;
    private Random random;

    public RandomForestAnalyzer(List<Set<Integer>> transactions, int minNumber, int maxNumber, int betSize) {
        this.transactions = transactions;
        this.minNumber = minNumber;
        this.maxNumber = maxNumber;
        this.betSize = betSize;
        this.random = new Random();
    }

    public List<Integer> generateBetRandomForest(Random rand) {
        if (transactions == null || transactions.isEmpty()) {
            return new ArrayList<>();
        }

        // Treinar forest e obter votos
        Map<Integer, Integer> votes = trainForestAndVote();

        // Seleção estocástica usando roleta ponderada (não determinística)
        List<Integer> selectedNumbers = new ArrayList<>();
        Set<Integer> used = new HashSet<>();
        
        double totalVotes = votes.values().stream().mapToDouble(Integer::doubleValue).sum();
        if (totalVotes <= 0) totalVotes = 1.0;
        
        while (selectedNumbers.size() < betSize && used.size() < (maxNumber - minNumber + 1)) {
            double spin = rand.nextDouble() * totalVotes;
            double accumulated = 0.0;
            Integer selected = null;
            
            for (int num = minNumber; num <= maxNumber; num++) {
                if (used.contains(num)) continue;
                
                accumulated += votes.getOrDefault(num, 0);
                if (spin <= accumulated) {
                    selected = num;
                    break;
                }
            }
            
            // Fallback se nenhum selecionado
            if (selected == null) {
                for (int num = minNumber; num <= maxNumber; num++) {
                    if (!used.contains(num)) {
                        selected = num;
                        break;
                    }
                }
            }
            
            if (selected != null) {
                selectedNumbers.add(selected);
                used.add(selected);
            }
        }

        Collections.sort(selectedNumbers);
        return selectedNumbers;
    }

    private Map<Integer, Integer> trainForestAndVote() {
        Map<Integer, Integer> globalVotes = new HashMap<>();
        for (int i = minNumber; i <= maxNumber; i++) {
            globalVotes.put(i, 0);
        }

        // Treinar múltiplas árvores com bagging
        for (int t = 0; t < numTrees; t++) {
            // Criar bootstrap sample (amostragem com reposição)
            List<Set<Integer>> bootstrapSample = createBootstrapSample();

            // Treinar árvore de decisão
            DecisionNode tree = trainDecisionTree(bootstrapSample, 0, 5); // max depth = 5

            // Obter predições e contar votos
            Map<Integer, Integer> treeVotes = getTreeVotes(tree);
            for (Map.Entry<Integer, Integer> e : treeVotes.entrySet()) {
                globalVotes.put(e.getKey(), globalVotes.get(e.getKey()) + e.getValue());
            }
        }

        return globalVotes;
    }

    private List<Set<Integer>> createBootstrapSample() {
        List<Set<Integer>> sample = new ArrayList<>();
        int sampleSize = Math.max(10, transactions.size() / 2);
        for (int i = 0; i < sampleSize; i++) {
            int idx = random.nextInt(transactions.size());
            sample.add(new HashSet<>(transactions.get(idx)));
        }
        return sample;
    }

    private DecisionNode trainDecisionTree(List<Set<Integer>> data, int depth, int maxDepth) {
        if (data.isEmpty() || depth >= maxDepth) {
            return new DecisionNode(null, null, null, getMostCommonNumbers(data));
        }

        // Selecionar feature (número) aleatório para split
        int splitFeature = minNumber + random.nextInt(maxNumber - minNumber + 1);

        // Dividir dados em dois subconjuntos
        List<Set<Integer>> left = new ArrayList<>();
        List<Set<Integer>> right = new ArrayList<>();

        for (Set<Integer> transaction : data) {
            if (transaction.contains(splitFeature)) {
                left.add(transaction);
            } else {
                right.add(transaction);
            }
        }

        // Se split não é bom, retorna folha
        if (left.isEmpty() || right.isEmpty()) {
            return new DecisionNode(null, null, null, getMostCommonNumbers(data));
        }

        // Recursivamente treinar subárvores
        DecisionNode leftChild = trainDecisionTree(left, depth + 1, maxDepth);
        DecisionNode rightChild = trainDecisionTree(right, depth + 1, maxDepth);

        return new DecisionNode(splitFeature, leftChild, rightChild, null);
    }

    private Map<Integer, Integer> getTreeVotes(DecisionNode node) {
        Map<Integer, Integer> votes = new HashMap<>();
        if (node.isLeaf()) {
            for (Integer num : node.leafNumbers) {
                votes.put(num, votes.getOrDefault(num, 0) + 1);
            }
        } else {
            votes.putAll(getTreeVotes(node.leftChild));
            votes.putAll(getTreeVotes(node.rightChild));
        }
        return votes;
    }

    private List<Integer> getMostCommonNumbers(List<Set<Integer>> data) {
        Map<Integer, Integer> frequency = new HashMap<>();
        for (int i = minNumber; i <= maxNumber; i++) {
            frequency.put(i, 0);
        }

        for (Set<Integer> transaction : data) {
            for (Integer num : transaction) {
                if (num >= minNumber && num <= maxNumber) {
                    frequency.put(num, frequency.get(num) + 1);
                }
            }
        }

        return frequency.entrySet().stream()
                .sorted((a, b) -> Integer.compare(b.getValue(), a.getValue()))
                .limit(Math.min(betSize, 10))
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
    }

    private static class DecisionNode {
        Integer splitFeature;
        DecisionNode leftChild;
        DecisionNode rightChild;
        List<Integer> leafNumbers;

        DecisionNode(Integer splitFeature, DecisionNode leftChild, DecisionNode rightChild, List<Integer> leafNumbers) {
            this.splitFeature = splitFeature;
            this.leftChild = leftChild;
            this.rightChild = rightChild;
            this.leafNumbers = leafNumbers;
        }

        boolean isLeaf() {
            return leafNumbers != null;
        }
    }
}
