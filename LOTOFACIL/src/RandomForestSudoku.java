/**
 * Autor: getson@engineer.com
 */
import java.io.*;
import java.util.*;

/**
 * RandomForestSudoku.java
 * ─────────────────────────────────────────────────────────────────────────
 *  Gera UMA previsão determinística de configuração Sudoku 5×5 via RF.
 *
 *  Método:
 *  1. Distribuição GLOBAL: média de P_RF sobre N_GLOBAL_QUERIES queries
 *     aleatórias do dataset → estima E[P_RF(j)] real, sem viés de query.
 *  2. Distribuição RECENTE: P_RF médio das últimas N_RECENT_ROWS linhas
 *     → captura tendências recentes sem fixar em um único registro.
 *  3. Blend: (1-w)*global + w*recente → probabilidades finais.
 *  4. TOP-15 DETERMINÍSTICO: seleciona as 15 células de maior P_blend.
 *     Sem ruído — resultado é sempre o mesmo para o mesmo dataset.
 * ─────────────────────────────────────────────────────────────────────────
 */
public class RandomForestSudoku {

    // ── Hiperparâmetros RF ────────────────────────────────────────────────
    static final int    CELLS             = 25;
    static final int    REVEALED_COUNT    = 15;
    static final int    N_TREES           = 300;
    static final int    MAX_DEPTH         = 12;
    static final int    MIN_SAMPLES_SPLIT = 4;
    static final int    FEATURES_PER_NODE = 5;   // sqrt(25) = 5
    static final long   SEED              = 2026L;

    // ── Parâmetros de estimação ───────────────────────────────────────────
    static final int    N_GLOBAL_QUERIES  = 300;  // queries p/ distribuição global
    static final int    N_RECENT_ROWS     = 40;   // linhas recentes p/ contexto
    static final double RECENT_WEIGHT     = 0.25; // peso do contexto recente

    // ════════════════════════════════════════════════════════════════════
    //  Nó da Árvore
    // ════════════════════════════════════════════════════════════════════
    static class Node {
        int      splitFeature = -1;
        Node     left, right;
        double[] leafProbs;
        boolean isLeaf() { return leafProbs != null; }
    }

    // ════════════════════════════════════════════════════════════════════
    //  Árvore de Decisão binária multi-label
    // ════════════════════════════════════════════════════════════════════
    static class DecisionTree {
        Node root;
        private final Random rng;

        DecisionTree(int treeIndex) {
            this.rng = new Random(SEED + treeIndex);
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
            double[] p = new double[CELLS];
            if (data.length == 0) { Arrays.fill(p, (double) REVEALED_COUNT / CELLS); return p; }
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

    // ════════════════════════════════════════════════════════════════════
    //  Random Forest
    // ════════════════════════════════════════════════════════════════════
    static class RandomForest {
        DecisionTree[] trees;
        double[]       marginalProbs;

        void fit(int[][] data, boolean verbose) {
            trees         = new DecisionTree[N_TREES];
            marginalProbs = new double[CELLS];
            for (int[] row : data)
                for (int j = 0; j < CELLS; j++) marginalProbs[j] += row[j];
            for (int j = 0; j < CELLS; j++) marginalProbs[j] /= data.length;
            for (int t = 0; t < N_TREES; t++) {
                trees[t] = new DecisionTree(t);
                trees[t].fit(bootstrap(data, t));
                if (verbose && (t + 1) % 100 == 0)
                    System.out.printf("      [%3d / %d arvores]%n", t + 1, N_TREES);
            }
        }

        /** Inferência: P_RF(j) para o vetor query — média das N_TREES folhas */
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

    // ════════════════════════════════════════════════════════════════════
    //  PROGRAMA PRINCIPAL
    // ════════════════════════════════════════════════════════════════════
    public static void main(String[] args) throws IOException {

        // ── 1. Carregar dados ──────────────────────────────────────────
        int[][] data = carregarDados("soduko.csv");
        int N = data.length;

        // ── 2. Treinar RF ──────────────────────────────────────────────
        RandomForest rf = new RandomForest();
        rf.fit(data, false);

        // ── 3. Estimar distribuições ───────────────────────────────────
        double[] globalProbs = estimarDistribuicaoGlobal(rf, data);
        double[] recentProbs = estimarDistribuicaoRecente(rf, data);
        double[] baseProbs   = blendear(globalProbs, recentProbs);

        // ── 4. Ranking completo das 25 células por P_blend ──────────────────
        //  Pred 1: ranks 1–15  (top-15 puro)
        //  Pred 2: ranks 1–12 + ranks 16–18  (substitui as 3 menos prováveis pelas próximas 3)
        //  Pred 3: ranks 1–12 + ranks 19–21  (substitui pelas 3 seguintes)
        Integer[] rank = new Integer[CELLS];
        for (int i = 0; i < CELLS; i++) rank[i] = i;
        Arrays.sort(rank, (a, b) -> Double.compare(baseProbs[b], baseProbs[a]));

        int[][] preds = new int[3][CELLS];
        // Pred 1: ranks 0–14
        for (int i = 0; i < 15; i++)          preds[0][rank[i]] = 1;
        // Pred 2: ranks 0–11 + 15–17
        for (int i = 0; i < 12; i++)          preds[1][rank[i]] = 1;
        for (int i = 15; i < 18; i++)         preds[1][rank[i]] = 1;
        // Pred 3: ranks 0–11 + 18–20
        for (int i = 0; i < 12; i++)          preds[2][rank[i]] = 1;
        for (int i = 18; i < 21; i++)         preds[2][rank[i]] = 1;

        // ── 5. Exibir resultado ────────────────────────────────────────
        System.out.println("=".repeat(50));
        System.out.println("  PREVISOES RANDOM FOREST — SUDOKU 5x5");
        System.out.println("=".repeat(50));
        System.out.printf("  Registro %d (top-15):          %s%n", N + 1, toList(preds[0]));
        System.out.printf("  Registro %d (top-12 + alt1):  %s%n", N + 2, toList(preds[1]));
        System.out.printf("  Registro %d (top-12 + alt2):  %s%n", N + 3, toList(preds[2]));
        System.out.println("=".repeat(50));
    }

    // ════════════════════════════════════════════════════════════════════
    //  Distribuição GLOBAL
    //  Média de P_RF sobre N_GLOBAL_QUERIES queries aleatórias.
    //  Estima E[P_RF(j)] = P(célula_j=revelada | distribuição do dataset).
    //  Elimina o viés da query única que causava o "ponto fixo".
    // ════════════════════════════════════════════════════════════════════
    static double[] estimarDistribuicaoGlobal(RandomForest rf, int[][] data) {
        Random r = new Random(SEED + 1L);
        double[] p = new double[CELLS];
        for (int q = 0; q < N_GLOBAL_QUERIES; q++) {
            double[] lp = rf.inferir(data[r.nextInt(data.length)]);
            for (int j = 0; j < CELLS; j++) p[j] += lp[j];
        }
        for (int j = 0; j < CELLS; j++) p[j] /= N_GLOBAL_QUERIES;
        return p;
    }

    // ════════════════════════════════════════════════════════════════════
    //  Distribuição RECENTE
    //  Média de P_RF sobre as últimas N_RECENT_ROWS linhas do dataset.
    //  Captura tendências recentes sem fixar em uma linha específica.
    // ════════════════════════════════════════════════════════════════════
    static double[] estimarDistribuicaoRecente(RandomForest rf, int[][] data) {
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

    // ════════════════════════════════════════════════════════════════════
    //  Blend: (1 - RECENT_WEIGHT) * global + RECENT_WEIGHT * recente
    // ════════════════════════════════════════════════════════════════════
    static double[] blendear(double[] global, double[] recent) {
        double[] b = new double[CELLS];
        for (int j = 0; j < CELLS; j++)
            b[j] = (1.0 - RECENT_WEIGHT) * global[j] + RECENT_WEIGHT * recent[j];
        return b;
    }

    // ════════════════════════════════════════════════════════════════════
    //  Top-15 determinístico: seleciona as 15 células de maior P_blend
    // ════════════════════════════════════════════════════════════════════
    static int[] top15(double[] probs) {
        Integer[] idx = new Integer[CELLS];
        for (int i = 0; i < CELLS; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Double.compare(probs[b], probs[a]));
        int[] v = new int[CELLS];
        for (int i = 0; i < REVEALED_COUNT; i++) v[idx[i]] = 1;
        return v;
    }

    // ════════════════════════════════════════════════════════════════════
    //  Carregamento de dados CSV → matriz binária [n][25]
    // ════════════════════════════════════════════════════════════════════
    static int[][] carregarDados(String path) throws IOException {
        List<int[]> rows = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                int[] v = new int[CELLS];
                for (String s : line.split(","))
                    v[Integer.parseInt(s.trim()) - 1] = 1;
                rows.add(v);
            }
        }
        return rows.toArray(new int[0][]);
    }

    // ════════════════════════════════════════════════════════════════════
    //  Auxiliares
    // ════════════════════════════════════════════════════════════════════
    static List<Integer> toList(int[] v) {
        List<Integer> l = new ArrayList<>();
        for (int i = 0; i < v.length; i++) if (v[i] == 1) l.add(i + 1);
        return l;
    }

    static String toKey(int[] v) {
        StringBuilder sb = new StringBuilder();
        for (int x : v) sb.append(x);
        return sb.toString();
    }
}
