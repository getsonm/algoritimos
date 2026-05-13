/**
 * Autor: getson@engineer.com
 */
import java.io.*;
import java.util.*;

public class Soduko {

    // ── Caminhos do arquivo ───────────────────────────────────────────
    static final String[] CAMINHOS_ARQUIVO = {
        "arquivos/soduko.csv",
        "../arquivos/soduko.csv",
        "src/../arquivos/soduko.csv",
    };

    // ── Constantes de validação ───────────────────────────────────────
    static final int    DEZENAS_LINHA = 15;
    static final int    MIN_DEZENA    = 1;
    static final int    MAX_DEZENA    = 25;

    // ── Constantes RF ─────────────────────────────────────────────────
    static final int    CELLS             = 25;
    static final int    PREDICT_COUNT     = 19;   // próxima linha: 19 dezenas
    static final int    N_TREES           = 300;
    static final int    MAX_DEPTH         = Integer.MAX_VALUE; // RF padrão: árvores completas
    static final int    MIN_SAMPLES_SPLIT = 4;
    static final int    FEATURES_PER_NODE = 5;    // sqrt(25) ≈ 5
    static final long   SEED              = 2026L;
    static final int    N_GLOBAL_QUERIES  = 300;  // queries p/ distribuição global
    static final int    N_RECENT_ROWS     = 40;   // linhas recentes p/ contexto
    static final double RECENT_WEIGHT     = 0.25; // peso do contexto recente

    // ════════════════════════════════════════════════════════════════
    //  Nó da Árvore
    // ════════════════════════════════════════════════════════════════
    static class Node {
        int      splitFeature = -1;
        Node     left, right;
        double[] leafProbs;
        boolean isLeaf() { return leafProbs != null; }
    }

    // ════════════════════════════════════════════════════════════════
    //  Árvore de Decisão binária multi-label
    // ════════════════════════════════════════════════════════════════
    static class DecisionTree {
        Node root;
        private final Random rng;
        private final double[] prior; // frequência marginal real do dataset

        DecisionTree(int treeIndex, double[] prior) {
            this.rng   = new Random(SEED + treeIndex);
            this.prior = prior;
        }

        void fit(int[][] data) { root = build(data, 0); }

        double[] predict(int[] x) {
            Node n = root;
            while (!n.isLeaf())
                n = (x[n.splitFeature] == 0) ? n.left : n.right;
            return n.leafProbs;
        }

        private Node build(int[][] data, int depth) {
            Node node = new Node();
            if (depth >= MAX_DEPTH || data.length < MIN_SAMPLES_SPLIT) {
                node.leafProbs = calcLeafProbs(data); return node;
            }
            int[] features = randomSubset(CELLS, FEATURES_PER_NODE);
            int bestFeat = -1; double bestGini = Double.MAX_VALUE;
            int[][] bestLeft = null, bestRight = null;
            for (int f : features) {
                List<int[]> L = new ArrayList<>(), R = new ArrayList<>();
                for (int[] row : data) (row[f] == 0 ? L : R).add(row);
                if (L.isEmpty() || R.isEmpty()) continue;
                double g = weightedGini(L, R, data.length);
                if (g < bestGini) {
                    bestGini = g; bestFeat = f;
                    bestLeft  = L.toArray(new int[0][]);
                    bestRight = R.toArray(new int[0][]);
                }
            }
            if (bestFeat == -1) { node.leafProbs = calcLeafProbs(data); return node; }
            node.splitFeature = bestFeat;
            node.left  = build(bestLeft,  depth + 1);
            node.right = build(bestRight, depth + 1);
            return node;
        }

        private double[] calcLeafProbs(int[][] data) {
            // Folha vazia → usa prior real do dataset (frequência marginal)
            if (data.length == 0) return Arrays.copyOf(prior, CELLS);
            double[] p = new double[CELLS];
            for (int[] row : data) for (int j = 0; j < CELLS; j++) p[j] += row[j];
            for (int j = 0; j < CELLS; j++) p[j] /= data.length;
            return p;
        }

        private double weightedGini(List<int[]> L, List<int[]> R, int total) {
            return (giniML(L) * L.size() + giniML(R) * R.size()) / total;
        }

        private double giniML(List<int[]> S) {
            double g = 0; int n = S.size();
            for (int j = 0; j < CELLS; j++) {
                double p1 = 0;
                for (int[] r : S) p1 += r[j];
                p1 /= n; g += 2.0 * p1 * (1.0 - p1);
            }
            return g / CELLS;
        }

        private int[] randomSubset(int n, int k) {
            int[] a = new int[n];
            for (int i = 0; i < n; i++) a[i] = i;
            for (int i = 0; i < k; i++) {
                int j = i + rng.nextInt(n - i);
                int t = a[i]; a[i] = a[j]; a[j] = t;
            }
            return Arrays.copyOf(a, k);
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Random Forest
    // ════════════════════════════════════════════════════════════════
    static class RandomForest {
        DecisionTree[] trees;

        void fit(int[][] data) {
            trees = new DecisionTree[N_TREES];
            // Calcula prior real uma vez: frequência marginal de cada célula
            double[] prior = new double[CELLS];
            for (int[] row : data) for (int j = 0; j < CELLS; j++) prior[j] += row[j];
            for (int j = 0; j < CELLS; j++) prior[j] /= data.length;
            for (int t = 0; t < N_TREES; t++) {
                trees[t] = new DecisionTree(t, prior);
                trees[t].fit(bootstrap(data, t));
            }
        }

        double[] inferir(int[] query) {
            double[] p = new double[CELLS];
            for (DecisionTree tree : trees) {
                double[] lp = tree.predict(query);
                for (int j = 0; j < CELLS; j++) p[j] += lp[j];
            }
            for (int j = 0; j < CELLS; j++) p[j] /= N_TREES;
            return p;
        }

        private int[][] bootstrap(int[][] data, int treeIndex) {
            Random rng = new Random(SEED + N_TREES + treeIndex);
            int n = data.length;
            int[][] s = new int[n][];
            for (int i = 0; i < n; i++) s[i] = data[rng.nextInt(n)];
            return s;
        }
    }

// ════════════════════════════════════════════════════════════════
    //  PROGRAMA PRINCIPAL
    // ════════════════════════════════════════════════════════════════
    public static void main(String[] args) throws IOException {

        // ── 1. Localizar arquivo ───────────────────────────────────────
        String arquivo = null;
        for (String caminho : CAMINHOS_ARQUIVO) {
            if (new java.io.File(caminho).exists()) { arquivo = caminho; break; }
        }
        if (arquivo == null) {
            System.out.println("Erro: soduko.csv não encontrado. Diretório atual: "
                    + new java.io.File(".").getAbsolutePath());
            return;
        }

        // ── 2. Validação de integridade ────────────────────────────────
        List<String> erros = new ArrayList<>();
        int totalLinhas = 0, linhasValidas = 0, linhasComErro = 0, linhasComentario = 0;

        try (BufferedReader br = new BufferedReader(new FileReader(arquivo))) {
            String linha; int numLinha = 0;
            while ((linha = br.readLine()) != null) {
                numLinha++; linha = linha.trim();
                if (linha.isEmpty()) continue;
                if (linha.startsWith(";")) { linhasComentario++; continue; }
                totalLinhas++;
                List<String> el = validarLinha(linha, numLinha);
                if (el.isEmpty()) linhasValidas++; else { linhasComErro++; erros.addAll(el); }
            }
        }

        System.out.println("=".repeat(55));
        System.out.println("  VALIDAÇÃO — soduko.csv");
        System.out.printf("  Arquivo           : %s%n", new java.io.File(arquivo).getAbsolutePath());
        System.out.println("=".repeat(55));
        System.out.printf("  Total de linhas   : %d%n", totalLinhas);
        System.out.printf("  Linhas de coment. : %d%n", linhasComentario);
        System.out.printf("  Linhas válidas    : %d%n", linhasValidas);
        System.out.printf("  Linhas com erro   : %d%n", linhasComErro);
        System.out.println("=".repeat(55));
        if (erros.isEmpty()) System.out.println("  Arquivo ÍNTEGRO — nenhum erro encontrado.");
        else { System.out.printf("  %d erro(s) detectado(s):%n", erros.size()); for (String e : erros) System.out.println("  " + e); }
        System.out.println("=".repeat(55));

        if (linhasComErro > 0) { System.out.println("  Corrija os erros antes de prosseguir."); return; }

        // ── 3. Carregar dataset como matriz binária ────────────────────
        int[][] data = carregarMatriz(arquivo);

        // ── 4. Treinar Random Forest ───────────────────────────────────
        System.out.printf("%n  Treinando Random Forest (%d árvores, %d linhas de histórico)...%n",
                N_TREES, data.length);
        RandomForest rf = new RandomForest();
        rf.fit(data);
        System.out.println("  Treinamento concluído.");

        // ── 5. Estimar distribuição de probabilidades RF ───────────────
        double[] probGlobal  = estimarGlobal(rf, data);
        double[] probRecente = estimarRecente(rf, data);
        double[] probFinal   = blend(probGlobal, probRecente);

        // ── 6. Ranking das 25 células por P_final (decrescente) ────────
        Integer[] rank = new Integer[CELLS];
        for (int i = 0; i < CELLS; i++) rank[i] = i;
        Arrays.sort(rank, (a, b) -> Double.compare(probFinal[b], probFinal[a]));

        // ── 7. Top-19 selecionadas ─────────────────────────────────────
        List<Integer> selecionadas = new ArrayList<>();
        for (int i = 0; i < PREDICT_COUNT; i++) selecionadas.add(rank[i] + 1);
        Collections.sort(selecionadas);

        // ════════════════════════════════════════════════════════════
        //  RESULTADOS  (a partir daqui — linha ~70 de main)
        // ════════════════════════════════════════════════════════════
        System.out.println("\n" + "=".repeat(55));
        System.out.printf("  PREVISÃO RF — PRÓXIMA LINHA (linha %d do arquivo)%n", data.length + 1);
        System.out.printf("  Dataset utilizado : %d linhas de histórico%n", data.length);
        System.out.println("=".repeat(55));
        System.out.printf("  19 dezenas previstas: %s%n%n", selecionadas);

        System.out.println("  DETALHAMENTO — Células selecionadas e motivo RF:");
        System.out.println("  " + "-".repeat(53));
        System.out.printf("  %-8s %-11s %-12s %-11s %-5s%n",
                "Dezena", "P_global", "P_recente", "P_final", "Rank");
        System.out.println("  " + "-".repeat(53));
        for (int i = 0; i < PREDICT_COUNT; i++) {
            int cel    = rank[i];        // índice 0-24
            int dezena = cel + 1;        // dezena 1-25
            System.out.printf("  %-8d %-11.4f %-12.4f %-11.4f #%-5d%n",
                    dezena, probGlobal[cel], probRecente[cel], probFinal[cel], i + 1);
        }
        System.out.println("  " + "-".repeat(53));

        System.out.println("\n  COMO O RF SELECIONOU CADA DEZENA:");
        System.out.println("  O Random Forest treinou " + N_TREES + " árvores de decisão completas");
        System.out.println("  (sem limite de profundidade — padrão RF clássico),");
        System.out.println("  cada uma sobre uma amostra bootstrap do dataset.");
        System.out.println("  Em cada nó, " + FEATURES_PER_NODE + " células são testadas; o split que");
        System.out.println("  minimiza o índice Gini multi-label é escolhido.");
        System.out.println("  Folhas vazias usam o prior real do dataset (freq. marginal).");
        System.out.println("  A previsão de cada árvore é a média das folhas.");
        System.out.println("  P_global  = média de P_RF sobre " + N_GLOBAL_QUERIES + " queries do dataset");
        System.out.println("  P_recente = média P_RF das últimas " + N_RECENT_ROWS + " linhas (tendência)");
        System.out.printf("  P_final   = %.0f%% P_global + %.0f%% P_recente%n",
                (1.0 - RECENT_WEIGHT) * 100, RECENT_WEIGHT * 100);
        System.out.println("  As 19 dezenas de maior P_final são as escolhidas.");
        System.out.println("  Nenhum valor é fixo: o modelo se adapta a cada nova");
        System.out.println("  linha adicionada ao arquivo soduko.csv.");
        System.out.println("=".repeat(55));

        System.out.println("\n  Células NÃO selecionadas (menor P_final — descartadas):");
        System.out.printf("  %-8s %-11s %-5s%n", "Dezena", "P_final", "Rank");
        System.out.println("  " + "-".repeat(28));
        for (int i = PREDICT_COUNT; i < CELLS; i++) {
            System.out.printf("  %-8d %-11.4f #%-5d%n",
                    rank[i] + 1, probFinal[rank[i]], i + 1);
        }
        System.out.println("=".repeat(55));
    }

    // ════════════════════════════════════════════════════════════════
    //  Distribuição GLOBAL: média P_RF sobre N_GLOBAL_QUERIES queries
    // ════════════════════════════════════════════════════════════════
    static double[] estimarGlobal(RandomForest rf, int[][] data) {
        Random r = new Random(SEED + 1L);
        double[] p = new double[CELLS];
        for (int q = 0; q < N_GLOBAL_QUERIES; q++) {
            double[] lp = rf.inferir(data[r.nextInt(data.length)]);
            for (int j = 0; j < CELLS; j++) p[j] += lp[j];
        }
        for (int j = 0; j < CELLS; j++) p[j] /= N_GLOBAL_QUERIES;
        return p;
    }

    // ════════════════════════════════════════════════════════════════
    //  Distribuição RECENTE: P_RF das últimas N_RECENT_ROWS linhas
    // ════════════════════════════════════════════════════════════════
    static double[] estimarRecente(RandomForest rf, int[][] data) {
        double[] p = new double[CELLS];
        int start = Math.max(0, data.length - N_RECENT_ROWS);
        for (int i = start; i < data.length; i++) {
            double[] lp = rf.inferir(data[i]);
            for (int j = 0; j < CELLS; j++) p[j] += lp[j];
        }
        int cnt = data.length - start;
        for (int j = 0; j < CELLS; j++) p[j] /= cnt;
        return p;
    }

    // ════════════════════════════════════════════════════════════════
    //  Blend: (1 - RECENT_WEIGHT) * global + RECENT_WEIGHT * recente
    // ════════════════════════════════════════════════════════════════
    static double[] blend(double[] global, double[] recent) {
        double[] b = new double[CELLS];
        for (int j = 0; j < CELLS; j++)
            b[j] = (1.0 - RECENT_WEIGHT) * global[j] + RECENT_WEIGHT * recent[j];
        return b;
    }

    // ════════════════════════════════════════════════════════════════
    //  Carregar CSV → matriz binária [n][25]
    // ════════════════════════════════════════════════════════════════
    static int[][] carregarMatriz(String path) throws IOException {
        List<int[]> rows = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                if (line.startsWith(";")) continue;
                int[] v = new int[CELLS];
                for (String s : line.split(","))
                    v[Integer.parseInt(s.trim()) - 1] = 1;
                rows.add(v);
            }
        }
        return rows.toArray(new int[0][]);
    }

    // ════════════════════════════════════════════════════════════════
    //  Validação de linha CSV
    // ════════════════════════════════════════════════════════════════
    static List<String> validarLinha(String linha, int numLinha) {
        List<String> erros = new ArrayList<>();
        String[] partes = linha.split(",", -1);
        if (partes.length != DEZENAS_LINHA) {
            erros.add(String.format("Linha %d: esperado %d dezenas, encontrado %d → \"%s\"",
                    numLinha, DEZENAS_LINHA, partes.length, linha));
            return erros;
        }
        Set<Integer> vistas = new LinkedHashSet<>();
        for (int i = 0; i < partes.length; i++) {
            String token = partes[i].trim();
            if (token.matches("0\\d+")) {
                erros.add(String.format("Linha %d, pos %d: zero à esquerda → \"%s\"", numLinha, i + 1, token));
                continue;
            }
            int valor;
            try { valor = Integer.parseInt(token); }
            catch (NumberFormatException e) {
                erros.add(String.format("Linha %d, pos %d: valor não numérico → \"%s\"", numLinha, i + 1, token));
                continue;
            }
            if (valor < MIN_DEZENA || valor > MAX_DEZENA) {
                erros.add(String.format("Linha %d, pos %d: valor %d fora de [%d,%d]",
                        numLinha, i + 1, valor, MIN_DEZENA, MAX_DEZENA));
                continue;
            }
            if (!vistas.add(valor))
                erros.add(String.format("Linha %d, pos %d: dezena %d duplicada", numLinha, i + 1, valor));
        }
        return erros;
    }
}
