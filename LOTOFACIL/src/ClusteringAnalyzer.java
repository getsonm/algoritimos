/**
 * Autor: getson@engineer.com
 */
import java.util.*;

/**
 * Analisador de Clustering com K-Means
 * Agrupa números por padrões similares de coocorrência
 * Números no mesmo cluster tendem a aparecer juntos
 */
public class ClusteringAnalyzer {
    private List<Set<Integer>> historicalSequences;
    private int minNum;
    private int maxNum;
    private int tamanhoAposta;
    private int numClusters;
    
    // Matriz de coocorrência e distâncias
    private double[][] cooccurrenceMatrix;
    private double[][] distanceMatrix;
    
    // Atribuições de cluster e centros
    private int[] numberToCluster;
    private double[][] clusterCenters;
    private Map<Integer, Double> clusterQuality;
    
    public ClusteringAnalyzer(List<Set<Integer>> sequences, int minNum, int maxNum, int betsSize) {
        this.historicalSequences = new ArrayList<>(sequences);
        this.minNum = minNum;
        this.maxNum = maxNum;
        this.tamanhoAposta = betsSize;
        this.numClusters = Math.max(3, tamanhoAposta / 5); // 3 a 5 clusters
        this.numberToCluster = new int[maxNum - minNum + 1];
        this.clusterQuality = new HashMap<>();
        
        buildCooccurrenceMatrix();
        buildDistanceMatrix();
        runKMeans();
    }
    
    private void buildCooccurrenceMatrix() {
        int size = maxNum - minNum + 1;
        cooccurrenceMatrix = new double[size][size];
        
        // Contar coocorrências
        for (Set<Integer> seq : historicalSequences) {
            List<Integer> nums = new ArrayList<>(seq);
            for (int i = 0; i < nums.size(); i++) {
                for (int j = i + 1; j < nums.size(); j++) {
                    int n1 = nums.get(i) - minNum;
                    int n2 = nums.get(j) - minNum;
                    cooccurrenceMatrix[n1][n2]++;
                    cooccurrenceMatrix[n2][n1]++;
                }
            }
        }
        
        // Normalizar
        double max = Arrays.stream(cooccurrenceMatrix)
                .flatMapToDouble(Arrays::stream)
                .max().orElse(1.0);
        if (max > 0) {
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    cooccurrenceMatrix[i][j] /= max;
                }
            }
        }
    }
    
    private void buildDistanceMatrix() {
        int size = maxNum - minNum + 1;
        distanceMatrix = new double[size][size];
        
        for (int i = 0; i < size; i++) {
            for (int j = i; j < size; j++) {
                if (i == j) {
                    distanceMatrix[i][j] = 0;
                } else {
                    // Distância euclidiana simples: 1 - coocorrência
                    double dist = 1.0 - cooccurrenceMatrix[i][j];
                    distanceMatrix[i][j] = dist;
                    distanceMatrix[j][i] = dist;
                }
            }
        }
    }
    
    private void runKMeans() {
        int size = maxNum - minNum + 1;
        Random rand = new Random(42);
        
        // Inicializar centros aleatoriamente
        clusterCenters = new double[numClusters][size];
        for (int k = 0; k < numClusters; k++) {
            int idx = rand.nextInt(size);
            for (int i = 0; i < size; i++) {
                clusterCenters[k][i] = cooccurrenceMatrix[idx][i];
            }
        }
        
        // Iterar K-Means
        int maxIter = 20;
        for (int iter = 0; iter < maxIter; iter++) {
            // Atribuir pontos ao cluster mais próximo
            for (int i = 0; i < size; i++) {
                double minDist = Double.MAX_VALUE;
                int bestCluster = 0;
                for (int k = 0; k < numClusters; k++) {
                    double dist = euclideanDistance(cooccurrenceMatrix[i], clusterCenters[k]);
                    if (dist < minDist) {
                        minDist = dist;
                        bestCluster = k;
                    }
                }
                numberToCluster[i] = bestCluster;
            }
            
            // Recalcular centros
            for (int k = 0; k < numClusters; k++) {
                Arrays.fill(clusterCenters[k], 0);
                int count = 0;
                for (int i = 0; i < size; i++) {
                    if (numberToCluster[i] == k) {
                        for (int j = 0; j < size; j++) {
                            clusterCenters[k][j] += cooccurrenceMatrix[i][j];
                        }
                        count++;
                    }
                }
                if (count > 0) {
                    for (int j = 0; j < size; j++) {
                        clusterCenters[k][j] /= count;
                    }
                }
            }
        }
        
        // Calcular qualidade de cada cluster
        for (int k = 0; k < numClusters; k++) {
            double quality = 0;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (numberToCluster[i] == k) {
                    quality += 1.0 - euclideanDistance(cooccurrenceMatrix[i], clusterCenters[k]);
                    count++;
                }
            }
            clusterQuality.put(k, count > 0 ? quality / count : 0);
        }
    }
    
    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < Math.min(a.length, b.length); i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
    
    public List<Integer> generateBetFromClusters(Random rand) {
        // Gera aposta selecionando números de clusters de alta qualidade
        // Prioriza clusters com melhor coesão
        List<Map.Entry<Integer, Double>> clusterRanked = new ArrayList<>(clusterQuality.entrySet());
        clusterRanked.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        
        List<Integer> bet = new ArrayList<>();
        int numbersPerCluster = Math.max(1, tamanhoAposta / numClusters);
        
        for (Map.Entry<Integer, Double> clusterEntry : clusterRanked) {
            int clusterId = clusterEntry.getKey();
            
            // Encontrar números neste cluster
            List<Integer> clusterNumbers = new ArrayList<>();
            for (int i = 0; i < numberToCluster.length; i++) {
                if (numberToCluster[i] == clusterId) {
                    clusterNumbers.add(i + minNum);
                }
            }
            
            // Adicionar números deste cluster à aposta
            for (int i = 0; i < Math.min(numbersPerCluster, clusterNumbers.size()) && bet.size() < tamanhoAposta; i++) {
                bet.add(clusterNumbers.get(i));
            }
            
            if (bet.size() >= tamanhoAposta) break;
        }
        
        // Completar com números aleatórios se necessário
        while (bet.size() < tamanhoAposta) {
            int num = minNum + rand.nextInt(maxNum - minNum + 1);
            if (!bet.contains(num)) {
                bet.add(num);
            }
        }
        
        Collections.sort(bet);
        return new ArrayList<>(bet.subList(0, tamanhoAposta));
    }
}
