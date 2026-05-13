/**
 * Autor: getson@engineer.com
 */
import java.util.*;

/**
 * Otimizador de Apostas usando Algoritmo Genético
 * Evolui uma população de apostas para encontrar combinações ótimas
 */
public class GeneticAlgorithmOptimizer {
    private static final int MIN_NUM = 1;
    private static final int MAX_NUM = 25;
    private static final int APOSTA_SIZE = 15;
    
    private int populationSize;
    private int generations;
    private double mutationRate;
    private double eliteRate;
    private Random random;
    private List<Chromosome> population;
    private List<Chromosome> history;
    private Map<Integer, Double> numberScores;
    private List<Set<Integer>> historicalData;

    public GeneticAlgorithmOptimizer(int populationSize, int generations, double mutationRate) {
        this.populationSize = populationSize;
        this.generations = generations;
        this.mutationRate = mutationRate;
        this.eliteRate = 0.1; // 10% elite preservation
        this.random = new Random();
        this.population = new ArrayList<>();
        this.history = new ArrayList<>();
        this.numberScores = new HashMap<>();
        this.historicalData = new ArrayList<>();
        initializeNumberScores();
    }

    /**
     * Construtor com dados históricos para otimização
     */
    public GeneticAlgorithmOptimizer(List<Set<Integer>> historicalData, int populationSize, int generations) {
        this.historicalData = new ArrayList<>(historicalData);
        this.populationSize = populationSize;
        this.generations = generations;
        this.mutationRate = 0.25; // Aumentado de 0.15 para 0.25 (mais diversidade)
        this.eliteRate = 0.1;
        this.random = new Random();
        this.population = new ArrayList<>();
        this.history = new ArrayList<>();
        this.numberScores = new HashMap<>();
        initializeNumberScoresFromHistory();
    }

    /**
     * Inicializa scores baseado em dados históricos
     */
    private void initializeNumberScoresFromHistory() {
        // Contar frequências
        Map<Integer, Integer> freq = new HashMap<>();
        for (int i = MIN_NUM; i <= MAX_NUM; i++) {
            freq.put(i, 0);
        }
        
        for (Set<Integer> seq : historicalData) {
            for (Integer num : seq) {
                freq.put(num, freq.getOrDefault(num, 0) + 1);
            }
        }
        
        // Normalizar para scores 0-1
        double maxFreq = historicalData.size() * APOSTA_SIZE;
        for (int i = MIN_NUM; i <= MAX_NUM; i++) {
            double score = freq.get(i) / maxFreq;
            numberScores.put(i, score + 0.2); // Adicionar baseline
        }
    }

    /**
     * Inicializa scores padrão para números (pode ser sobrescrito)
     */
    private void initializeNumberScores() {
        for (int i = MIN_NUM; i <= MAX_NUM; i++) {
            numberScores.put(i, java.lang.Math.random() * 0.8 + 0.2); // 0.2-1.0
        }
    }

    /**
     * Define scores customizados (ex: de outro algoritmo)
     */
    public void setNumberScores(Map<Integer, Double> scores) {
        if (scores != null) {
            this.numberScores.putAll(scores);
        }
    }

    /**
     * Classe interna: Cromossomo (representa uma aposta)
     */
    public static class Chromosome implements Comparable<Chromosome> {
        public List<Integer> aposta; // 6 números
        public double fitness;

        public Chromosome() {
            this.aposta = new ArrayList<>();
            this.fitness = 0;
        }

        public Chromosome(List<Integer> aposta) {
            this.aposta = new ArrayList<>(aposta);
            this.fitness = 0;
        }

        @Override
        public int compareTo(Chromosome other) {
            return Double.compare(other.fitness, this.fitness); // Descending
        }

        @Override
        public String toString() {
            return String.format("Aposta: %s | Fitness: %.4f", aposta, fitness);
        }
    }

    /**
     * Cria população inicial aleatória
     */
    public void initializePopulation() {
        population.clear();

        for (int i = 0; i < populationSize; i++) {
            Chromosome chrom = generateRandomChromosome();
            population.add(chrom);
        }

        System.out.println("✓ População inicial criada: " + populationSize + " apostas");
    }

    /**
     * Gera cromossomo aleatório
     */
    private Chromosome generateRandomChromosome() {
        Chromosome chrom = new Chromosome();
        Set<Integer> chosen = new HashSet<>();

        while (chosen.size() < APOSTA_SIZE) {
            int num = MIN_NUM + random.nextInt(MAX_NUM - MIN_NUM + 1);
            if (!chosen.contains(num)) {
                chrom.aposta.add(num);
                chosen.add(num);
            }
        }

        Collections.sort(chrom.aposta);
        return chrom;
    }

    /**
     * Calcula fitness de um cromossomo
     * Baseado na soma de scores dos números + bônus por diversidade
     */
    private double calculateFitness(Chromosome chrom) {
        double fitness = 0;

        // Parte 1: Score dos números
        for (Integer num : chrom.aposta) {
            fitness += numberScores.getOrDefault(num, 0.5);
        }

        // Parte 2: Bônus de diversidade (números espalhados)
        double diversity = calculateDiversity(chrom.aposta);
        fitness += diversity * 2; // 2x peso

        // Parte 3: Penalidade para números consecutivos (reduz chances)
        double consecutive = countConsecutives(chrom.aposta);
        fitness -= consecutive * 0.5;

        return fitness;
    }

    /**
     * Calcula diversidade (espalhamento entre números)
     */
    private double calculateDiversity(List<Integer> aposta) {
        if (aposta.size() < 2) return 0;

        double sumDistances = 0;
        for (int i = 0; i < aposta.size() - 1; i++) {
            sumDistances += aposta.get(i + 1) - aposta.get(i);
        }

        return sumDistances / (aposta.size() * MAX_NUM); // Normalizado
    }

    /**
     * Conta números consecutivos na aposta
     */
    private int countConsecutives(List<Integer> aposta) {
        int count = 0;
        for (int i = 0; i < aposta.size() - 1; i++) {
            if (aposta.get(i + 1) - aposta.get(i) == 1) {
                count++;
            }
        }
        return count;
    }

    /**
     * Avalia toda população
     */
    private void evaluatePopulation() {
        for (Chromosome chrom : population) {
            chrom.fitness = calculateFitness(chrom);
        }
        Collections.sort(population);
    }

    /**
     * Seleção por torneio
     */
    private Chromosome tournamentSelection() {
        int tournamentSize = 5;
        Chromosome best = population.get(random.nextInt(populationSize));

        for (int i = 0; i < tournamentSize; i++) {
            Chromosome candidate = population.get(random.nextInt(populationSize));
            if (candidate.fitness > best.fitness) {
                best = candidate;
            }
        }

        return best;
    }

    /**
     * Crossover (recombinação): une dois pais
     */
    private Chromosome crossover(Chromosome parent1, Chromosome parent2) {
        Chromosome child = new Chromosome();
        Set<Integer> chosen = new HashSet<>();

        // Herança do pai 1: primeira metade dos números
        int halfSize = (int) Math.ceil(APOSTA_SIZE / 2.0);
        for (int i = 0; i < halfSize && i < parent1.aposta.size(); i++) {
            child.aposta.add(parent1.aposta.get(i));
            chosen.add(parent1.aposta.get(i));
        }

        // Herança do pai 2: números que não estão no filho
        for (Integer num : parent2.aposta) {
            if (child.aposta.size() >= APOSTA_SIZE) break;
            if (!chosen.contains(num)) {
                child.aposta.add(num);
                chosen.add(num);
            }
        }

        // Preencher com aleatórios se necessário
        while (child.aposta.size() < APOSTA_SIZE) {
            int num = MIN_NUM + random.nextInt(MAX_NUM - MIN_NUM + 1);
            if (!chosen.contains(num)) {
                child.aposta.add(num);
                chosen.add(num);
            }
        }

        Collections.sort(child.aposta);
        return child;
    }

    /**
     * Mutação: altera aleatoriamente um número
     */
    private void mutate(Chromosome chrom) {
        if (random.nextDouble() < mutationRate) {
            // Escolher posição aleatória
            int pos = random.nextInt(APOSTA_SIZE);

            // Gerar número novo que não está na aposta
            Integer newNum;
            do {
                newNum = MIN_NUM + random.nextInt(MAX_NUM - MIN_NUM + 1);
            } while (chrom.aposta.contains(newNum));

            chrom.aposta.set(pos, newNum);
            Collections.sort(chrom.aposta);
        }
    }

    /**
     * Executa uma geração: crossover + mutação + seleção
     */
    private void evolveGeneration() {
        List<Chromosome> newPopulation = new ArrayList<>();
        Set<String> uniqueApostas = new HashSet<>();

        // Elite: preserva melhores (sem duplicatas)
        int eliteSize = (int) (populationSize * eliteRate);
        for (int i = 0; i < eliteSize && i < population.size(); i++) {
            String key = population.get(i).aposta.toString();
            if (!uniqueApostas.contains(key)) {
                newPopulation.add(new Chromosome(population.get(i).aposta));
                uniqueApostas.add(key);
            }
        }

        // Gerar resto da população com garantia de diversidade
        int attempts = 0;
        while (newPopulation.size() < populationSize && attempts < populationSize * 10) {
            Chromosome parent1 = tournamentSelection();
            Chromosome parent2 = tournamentSelection();

            Chromosome child = crossover(parent1, parent2);
            mutate(child);

            String key = child.aposta.toString();
            if (!uniqueApostas.contains(key)) {
                newPopulation.add(child);
                uniqueApostas.add(key);
            }
            
            attempts++;
        }

        // Se ainda tiver espaço, gerar aleatórios
        while (newPopulation.size() < populationSize) {
            Chromosome random = generateRandomChromosome();
            String key = random.aposta.toString();
            if (!uniqueApostas.contains(key)) {
                newPopulation.add(random);
                uniqueApostas.add(key);
            }
        }

        population = newPopulation;
    }

    /**
     * Executa evolução completa com saída de progresso
     */
    public List<Chromosome> optimize() {
        System.out.println("\n╔════════════════════════════════════════════════════╗");
        System.out.println("║  ALGORITMO GENÉTICO - OTIMIZAÇÃO DE APOSTAS     ║");
        System.out.println("╚════════════════════════════════════════════════════╝\n");

        initializePopulation();

        System.out.println("Evoluindo " + generations + " gerações...\n");

        for (int gen = 1; gen <= generations; gen++) {
            evaluatePopulation();
            history.add(new Chromosome(population.get(0).aposta));

            if (gen % 5 == 0 || gen == 1 || gen == generations) {
                System.out.printf("Geração %3d | Melhor Fitness: %.6f | Top Aposta: %s\n",
                    gen, population.get(0).fitness, population.get(0).aposta);
            }

            if (gen < generations) {
                evolveGeneration();
            }
        }

        // Avaliação final
        evaluatePopulation();

        System.out.println("\n✓ Otimização concluída!");
        return getTopApostas(10);
    }

    /**
     * Executa evolução sem exibir progresso (silencioso)
     * Usado para comparações de performance
     */
    public List<Chromosome> optimizeSilent() {
        initializePopulation();

        for (int gen = 1; gen <= generations; gen++) {
            evaluatePopulation();
            history.add(new Chromosome(population.get(0).aposta));

            if (gen < generations) {
                evolveGeneration();
            }
        }

        evaluatePopulation();
        return getTopApostas(10);
    }

    /**
     * Retorna as N melhores apostas
     */
    public List<Chromosome> getTopApostas(int n) {
        evaluatePopulation();
        List<Chromosome> top = new ArrayList<>();

        for (int i = 0; i < Math.min(n, population.size()); i++) {
            top.add(population.get(i));
        }

        return top;
    }

    /**
     * Exibe resultados
     */
    public void printResults() {
        evaluatePopulation();

        System.out.println("\n" + "═".repeat(70));
        System.out.println("TOP 10 APOSTAS OTIMIZADAS - ALGORITMO GENÉTICO");
        System.out.println("═".repeat(70));

        for (int i = 0; i < Math.min(10, population.size()); i++) {
            Chromosome chrom = population.get(i);
            System.out.printf("Aposta %2d: ", i + 1);
            for (Integer num : chrom.aposta) {
                System.out.printf("%02d ", num);
            }
            System.out.printf("| Fitness: %.6f\n", chrom.fitness);
        }

        System.out.println("═".repeat(70));
    }

    /**
     * Estatísticas de evolução
     */
    public void printEvolutionStats() {
        if (history.isEmpty()) return;

        System.out.println("\n" + "═".repeat(70));
        System.out.println("ESTATÍSTICAS DE EVOLUÇÃO");
        System.out.println("═".repeat(70));

        double initialFitness = calculateFitness(history.get(0));
        double finalFitness = calculateFitness(history.get(history.size() - 1));
        double improvement = ((finalFitness - initialFitness) / initialFitness) * 100;

        System.out.printf("Fitness Inicial:  %.6f\n", initialFitness);
        System.out.printf("Fitness Final:    %.6f\n", finalFitness);
        System.out.printf("Melhoria:         %.2f%%\n", improvement);
        System.out.printf("Gerações:         %d\n", history.size());
        System.out.printf("População:        %d\n", populationSize);
        System.out.printf("Taxa Mutação:     %.2f%%\n", mutationRate * 100);

        System.out.println("═".repeat(70));
    }

    /**
     * Obter as N melhores apostas (garantindo diversidade)
     */
    public List<List<Integer>> getBestBets(int count) {
        // Executar otimização
        optimize();
        
        List<List<Integer>> bets = new ArrayList<>();
        Set<String> used = new HashSet<>();
        evaluatePopulation();
        
        // Tentar conseguir apostas diversas (sem duplicatas)
        for (int i = 0; i < population.size() && bets.size() < count; i++) {
            List<Integer> aposta = new ArrayList<>(population.get(i).aposta);
            String key = keyFromList(aposta);
            
            if (!used.contains(key)) {
                bets.add(aposta);
                used.add(key);
            }
        }
        
        // Se não conseguiu aproveitar apostas distintas da população, gerar novas aleatoriamente
        if (bets.size() < count) {
            for (int i = bets.size(); i < count; i++) {
                List<Integer> newBet = generateRandomBet();
                String key = keyFromList(newBet);
                
                if (!used.contains(key)) {
                    bets.add(newBet);
                    used.add(key);
                }
            }
        }
        
        return bets;
    }
    
    /**
     * Gera uma aposta aleatória (fallback para garantir diversidade)
     */
    private List<Integer> generateRandomBet() {
        List<Integer> bet = new ArrayList<>();
        List<Integer> nums = new ArrayList<>();
        for (int i = MIN_NUM; i <= MAX_NUM; i++) {
            nums.add(i);
        }
        Collections.shuffle(nums, random);
        for (int i = 0; i < APOSTA_SIZE; i++) {
            bet.add(nums.get(i));
        }
        Collections.sort(bet);
        return bet;
    }
    
    /**
     * Converte aposta para chave única (string)
     */
    private String keyFromList(List<Integer> aposta) {
        List<Integer> ord = new ArrayList<>(aposta);
        Collections.sort(ord);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < ord.size(); i++) {
            if (i > 0) sb.append(' ');
            sb.append(String.format("%02d", ord.get(i)));
        }
        return sb.toString();
    }

    /**
     * Teste do algoritmo
     */
    public static void main(String[] args) {
        GeneticAlgorithmOptimizer optimizer = new GeneticAlgorithmOptimizer(50, 30, 0.15);

        // Simular scores de outro algoritmo
        Map<Integer, Double> customScores = new HashMap<>();
        for (int i = 1; i <= 60; i++) {
            customScores.put(i, 0.4 + java.lang.Math.random() * 0.6); // 0.4-1.0
        }
        optimizer.setNumberScores(customScores);

        optimizer.optimize();
        optimizer.printResults();
        optimizer.printEvolutionStats();
    }
}
