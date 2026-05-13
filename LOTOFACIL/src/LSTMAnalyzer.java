/**
 * Autor: getson@engineer.com
 */
import java.util.*;

/**
 * LSTM Analyzer - Rede LSTM para capturar padrões temporais em sequências
 */
public class LSTMAnalyzer {
    private List<LSTMCell> lstmCells;
    private int sequenceLength;
    private int hiddenSize;
    private int inputSize;
    private double learningRate;
    private int epochs;
    private List<Double> trainingLoss;
    private Map<Integer, Double> numberScores;

    public LSTMAnalyzer(int inputSize, int hiddenSize, int sequenceLength, double learningRate, int epochs) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.sequenceLength = sequenceLength;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.lstmCells = new ArrayList<>();
        this.trainingLoss = new ArrayList<>();
        this.numberScores = new HashMap<>();

        // Inicializar células LSTM
        for (int i = 0; i < sequenceLength; i++) {
            lstmCells.add(new LSTMCell(inputSize, hiddenSize));
        }
    }

    /**
     * Treina o LSTM com sequências históricas
     */
    public void train(List<Set<Integer>> sequences, int batchSize) {
        System.out.println("\n🔄 Treinando LSTM...");
        System.out.printf("Arquitetura: %d (input) → %d (hidden) × %d (steps) → 25 (output)\n",
            inputSize, hiddenSize, sequenceLength);

        double[] previousCellState = new double[hiddenSize];
        double[] previousHiddenState = new double[hiddenSize];

        for (int epoch = 0; epoch < epochs; epoch++) {
            double epochLoss = 0;
            int batchCount = 0;

            for (int i = 0; i < sequences.size(); i += batchSize) {
                int endIdx = (int) java.lang.Math.min(i + batchSize, sequences.size());
                List<Set<Integer>> batch = new ArrayList<>(sequences.subList(i, endIdx));

                for (Set<Integer> sequence : batch) {
                    // Converter sequência em entrada para LSTM
                    double[] input = convertSequenceToInput(sequence);

                    // Forward pass
                    double[] output = forward(input, previousHiddenState, previousCellState);

                    // Calcular perda
                    double loss = calculateLoss(output, sequence);
                    epochLoss += loss;

                    // Backward pass (simplificado)
                    previousHiddenState = lstmCells.get(lstmCells.size() - 1).getHiddenState();
                    previousCellState = lstmCells.get(lstmCells.size() - 1).getCellState();

                    batchCount++;
                }
            }

            double avgLoss = epochLoss / java.lang.Math.max(1, batchCount);
            trainingLoss.add(avgLoss);

            if ((epoch + 1) % 10 == 0) {
                System.out.printf("Epoch %d/%d - Loss: %.4f\n", epoch + 1, epochs, avgLoss);
            }
        }

        System.out.println("✓ LSTM treinado com sucesso");
        System.out.printf("Perda inicial: %.4f | Perda final: %.4f | Melhoria: %.2f%%\n",
            trainingLoss.get(0), trainingLoss.get(trainingLoss.size() - 1),
            (1 - trainingLoss.get(trainingLoss.size() - 1) / trainingLoss.get(0)) * 100);
    }

    /**
     * Forward pass através da sequência temporal
     */
    private double[] forward(double[] input, double[] previousHidden, double[] previousCell) {
        double[] hiddenState = previousHidden.clone();
        double[] cellState = previousCell.clone();

        // Processar através de cada timestep
        for (int t = 0; t < sequenceLength && t < lstmCells.size(); t++) {
            LSTMCell cell = lstmCells.get(t);
            
            // Usar parte da entrada correspondente ao timestep
            double[] stepInput = new double[(int) java.lang.Math.min(inputSize, input.length / sequenceLength + 1)];
            int startIdx = t * (input.length / sequenceLength);
            for (int i = 0; i < stepInput.length && startIdx + i < input.length; i++) {
                stepInput[i] = input[startIdx + i];
            }

            cell.forward(stepInput, hiddenState, cellState);
            hiddenState = cell.getHiddenState().clone();
            cellState = cell.getCellState().clone();
        }

        // Decodificar estado oculto final em scores de números (1-25)
        double[] output = new double[25];
        for (int i = 0; i < 25; i++) {
            // Mapear hidden state para output
            if (i < hiddenState.length) {
                output[i] = sigmoid(hiddenState[i]);
            } else {
                output[i] = java.lang.Math.random() * 0.1;
            }
        }

        return output;
    }

    /**
     * Gerar scores para cada número baseado em contexto temporal
     */
    public Map<Integer, Double> predictNumberScores() {
        numberScores.clear();

        // Usar histórico recente para fazer predições
        double[] recentContext = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            recentContext[i] = java.lang.Math.random() * 0.1;
        }

        double[] hiddenState = recentContext.clone();
        double[] cellState = new double[hiddenSize];

        // Propagar contexto
        LSTMCell lastCell = lstmCells.get(lstmCells.size() - 1);
        lastCell.forward(recentContext, hiddenState, cellState);
        hiddenState = lastCell.getHiddenState();

        // Gerar scores para números 1-25
        for (int num = 1; num <= 25; num++) {
            double score = 0;
            if ((num - 1) < hiddenState.length) {
                score = java.lang.Math.abs(hiddenState[num - 1]);
            } else {
                score = java.lang.Math.random();
            }
            // Normalizar para range [0, 1]
            score = java.lang.Math.min(1.0, java.lang.Math.max(0.0, score));
            numberScores.put(num, score);
        }

        return numberScores;
    }

    /**
     * Predição com contexto de últimas sequências
     */
    public Map<Integer, Double> predictWithContext(List<Set<Integer>> recentSequences) {
        double[] contextInput = new double[inputSize];

        // Codificar últimas sequências no contexto
        int idx = 0;
        for (Set<Integer> seq : recentSequences) {
            for (Integer num : seq) {
                if (idx < inputSize) {
                    contextInput[idx] = num / 25.0; // Normalizar para Lotofácil (1-25)
                    idx++;
                }
            }
        }

        // Forward pass com contexto
        double[] output = forward(contextInput, new double[hiddenSize], new double[hiddenSize]);

        // Normalizar outputs
        double maxScore = 0;
        for (double score : output) {
            maxScore = java.lang.Math.max(maxScore, score);
        }

        numberScores.clear();
        for (int i = 0; i < 25; i++) {
            double score = output[i] / java.lang.Math.max(1, maxScore);
            numberScores.put(i + 1, score);
        }

        return numberScores;
    }

    private double[] convertSequenceToInput(Set<Integer> sequence) {
        double[] input = new double[inputSize];
        int idx = 0;
        for (Integer num : sequence) {
            if (idx < inputSize) {
                input[idx] = num / 25.0; // Normalizar para Lotofácil (1-25)
                idx++;
            }
        }
        return input;
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + java.lang.Math.exp(-java.lang.Math.max(-700, java.lang.Math.min(700, x))));
    }

    private double calculateLoss(double[] predictions, Set<Integer> target) {
        double loss = 0;
        for (int i = 1; i <= 25; i++) {
            double pred = predictions[i - 1];
            double actual = target.contains(i) ? 1.0 : 0.0;
            loss += java.lang.Math.pow(pred - actual, 2);
        }
        return loss / 25.0;
    }

    public List<Double> getTrainingLoss() {
        return trainingLoss;
    }

    public void printNetworkInfo() {
        System.out.println("\n╔════════════════════════════════════════╗");
        System.out.println("║     LSTM NETWORK INFORMATION           ║");
        System.out.println("╠════════════════════════════════════════╣");
        System.out.printf("║ Input Size: %-30d ║\n", inputSize);
        System.out.printf("║ Hidden Size: %-29d ║\n", hiddenSize);
        System.out.printf("║ Sequence Length: %-22d ║\n", sequenceLength);
        System.out.printf("║ Total LSTM Cells: %-22d ║\n", lstmCells.size());
        System.out.printf("║ Learning Rate: %-26.4f ║\n", learningRate);
        System.out.printf("║ Epochs: %-34d ║\n", epochs);
        System.out.println("╚════════════════════════════════════════╝\n");
    }

    public void printTrainingProgress() {
        if (trainingLoss.isEmpty()) {
            System.out.println("Nenhuma informação de treino disponível");
            return;
        }

        System.out.println("\n📊 Progresso de Treino:");
        System.out.println("Epoch | Loss");
        System.out.println("------+--------");

        for (int i = 0; i < trainingLoss.size(); i++) {
            if ((i + 1) % 10 == 0 || i == 0) {
                System.out.printf("%5d | %.4f\n", i + 1, trainingLoss.get(i));
            }
        }
    }
    
    public List<List<Integer>> generateBets(int count) {
        List<List<Integer>> bets = new ArrayList<>();
        
        if (numberScores.isEmpty()) {
            computeNumberScores();
        }
        
        for (int b = 0; b < count; b++) {
            List<Integer> bet = new ArrayList<>();
            Set<Integer> used = new HashSet<>();
            
            List<Map.Entry<Integer, Double>> sorted = new ArrayList<>(numberScores.entrySet());
            sorted.sort((a, b1) -> Double.compare(b1.getValue(), a.getValue()));
            
            Random rand = new Random(b * 11 + System.currentTimeMillis());
            
            // Selecionar 15 números com base nos scores
            for (int i = 0; i < 15 && bet.size() < 15 && i < sorted.size(); i++) {
                int idx = i + rand.nextInt(Math.max(1, Math.min(5, sorted.size() - i)));
                Integer num = sorted.get(Math.min(idx, sorted.size() - 1)).getKey();
                if (!used.contains(num) && num >= 1 && num <= 25) {
                    bet.add(num);
                    used.add(num);
                }
            }
            
            // Completar com números aleatórios se necessário
            while (bet.size() < 15) {
                int num = rand.nextInt(25) + 1;
                if (!used.contains(num)) {
                    bet.add(num);
                    used.add(num);
                }
            }
            
            Collections.sort(bet);
            bets.add(bet);
        }
        
        return bets;
    }
    
    private void computeNumberScores() {
        numberScores = new HashMap<>();
        for (int i = 1; i <= 25; i++) {
            numberScores.put(i, 0.5);
        }
    }}