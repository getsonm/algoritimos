/**
 * Autor: getson@engineer.com
 */
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Versão aprimorada baseada em Mineração de Dados (Apriori + Regras de Associação)
 *
 * Melhorias em relação ao LotofacilDM:
 * - Suporte ponderado por recência (EWMA) para valorizar sorteios recentes
 * - Parâmetros configuráveis por linha de comando (suporte, confiança, maxK, apostas, seed)
 * - Leitura opcional de histórico via arquivo (CSV/espacos) com validação
 * - Geração com diversidade (limita sobreposição entre apostas)
 * - Restrições simples (mín/máx de pares) e correções automáticas
 * - Opção de salvar as apostas em arquivo
 *
 * Exemplos:
 *  java LotofacilDMPlus --bets 12 --size 15 --minSupport 0.07 --minConfidence 0.55 --maxK 3 --seed 123
 *  java LotofacilDMPlus --history historico.csv --out apostas.txt --evenMin 6 --evenMax 9 --minOverlap 9
 */
public class LotofacilDMPlus {

    // Parâmetros padrão
    private int TAMANHO_APOSTA = 15;
    private int MIN_NUMERO = 1;
    private int MAX_NUMERO = 25;
    private int NUMERO_APOSTAS = 3;

    private double minSupport = 0.068;      // suporte mínimo para itemsets
    private double minConfidence = 0.48;   // confiança mínima para regras
    private int maxK = 1;                  // tamanho máximo do itemset no Apriori (aumentado para capturar quartetos)
    private double recencyAlpha = 0.967;    // fator de decaimento (0..1) para ponderar recência (mais agressivo)
    private int windowSize = 300; // usar 500 por padrão
    private double assocBoost = 0.24; // aumentado para reforço muito mais agressivo (0.4 -> 0.85)
    private int minDiversityOverlap = 7; // reduzido para mais diversidade

    // Restrições simples de composição
    private int evenMin = 6;
    private int evenMax = 9;

    // Diversidade entre apostas (máximo de números em comum)
    

    private Long seed = null; // se null, usa currentTimeMillis

    // Cache para análise de padrões cíclicos
    private final Map<Integer, Double> numeroFrequenciaMedia = new HashMap<>();
    private final Map<Integer, Integer> numeroCiclo = new HashMap<>();

    private String historyPath = null; // histórico externo opcional
    private int historyStartLine = 1;   // primeira linha (1-based) a considerar do histórico
    private int historyEndLine = Integer.MAX_VALUE; // última linha (inclusiva)

    // Random principal é derivado da "seed" quando necessário nas rotinas

    // Estruturas geradas (usar bitmasks para eficiência)
    private final Map<Integer, Double> supportMapMask = new HashMap<>(); // chave = bitmask (bits 0..24 -> números 1..25)
    private final List<AssociationRule> regras = new ArrayList<>();

    // Cache simples
    private List<Set<Integer>> transacoes = null;      // subconjunto efetivo usado para treinar (após janela)
    private List<Set<Integer>> transacoesTodas = null; // todas as transações do range solicitado
    private List<Double> pesosTransacoes = null; // peso por transação (recência)
    private double pesoTotal = 0.0;
    // Representações em bitmask (1..25 -> bits 0..24) para acelerar operações de inclusão/conjuntos
    private List<Integer> transacoesMask = null;
    private List<Integer> transacoesTodasMask = null;
    // Combinações já existentes para evitar repetir
    private Set<String> combosExistentes = null;

    // Backtest walk-forward
    private boolean backtest = false;
    private int btStart = 200;     // começar a partir desta linha (1-based dentro do range carregado)
    private int btStride = 1;      // pular passos

    // Configuração de GPU
    private boolean useGPU = true; // usar GPU para acelerar cálculos (requer placa compatível)

    // Lista de configurações (permitido até 6)
    private List<Config> configuracoes = new ArrayList<>();
    private int configuracaoAtual = 1;
    // Alvo opcional para forçar uma combinação específica nesta configuração
    private List<Integer> targetCombo = null;
    private boolean ensureTargetCombo = false;

    public LotofacilDMPlus() {
        // Configuração 0: PERSONALISADA (otimizada com pesos rebalanceados)
        Config cfg = new Config();
        cfg.TAMANHO_APOSTA = 15;
        cfg.NUMERO_APOSTAS = this.NUMERO_APOSTAS;
        cfg.MIN_NUMERO = 1;
        cfg.MAX_NUMERO = 25;

        cfg.minSupport = this.minSupport;    // muito alto para acelerar significativamente
        cfg.minConfidence = this.minConfidence; // confiança alta
        cfg.maxK = this.maxK;             // apenas pares
        cfg.recencyAlpha = this.recencyAlpha;  // recência moderada
        cfg.windowSize = this.windowSize;     // janela muito pequena para acelerar drasticamente
        cfg.assocBoost = this.assocBoost;    // reforço muito agressivo de associações
        cfg.minDiversityOverlap = this.minDiversityOverlap; // reduzido para explorar combinações diferentes

        cfg.evenMin = 6;
        cfg.evenMax = 9;

        cfg.useGPU = true;

        cfg.historyPath = "/home/getson/glm/getson/projetos/bkp/LOTOFACIL/arquivos/lotofacil.csv";
        cfg.historyStartLine = 1;
        cfg.historyEndLine = Integer.MAX_VALUE;

        cfg.seed = null;

        cfg.backtest = false;
        cfg.btStart = 2500;
        cfg.btStride = 1;

        addConfiguracao(cfg);
        usarConfiguracao(0);
    }

    // Executa múltiplas gerações variando a seed e retorna um mapa de frequência das combinações geradas
    public Map<String, Integer> runMultipleSeeds(int runs, long startSeed, int stride) {
        // prepara modelo e scores uma vez para a configuração atual
        executarMineracao();
        Map<Integer, Double> scores = pontuarNumeros();

        Map<String, Integer> freq = new HashMap<>();

        // guardar combosExistentes original para restaurar depois
        Set<String> originalExists = (combosExistentes == null) ? null : new HashSet<>(combosExistentes);

        for (int i = 0; i < runs; i++) {
            long s = startSeed + (long) i * stride;
            Long oldSeed = this.seed;
            this.seed = s;

            // usar cópia temporária de combosExistentes para evitar que gerações anteriores contaminem as próximas
            Set<String> tempExists = (originalExists == null) ? new HashSet<>() : new HashSet<>(originalExists);
            this.combosExistentes = tempExists;

            List<List<Integer>> apostas = gerarApostas(scores);

            for (List<Integer> a : apostas) {
                String k = keyFromList(a);
                freq.put(k, freq.getOrDefault(k, 0) + 1);
            }

            // restaurar seed (guardaremos original no loop para segurança)
            this.seed = oldSeed;
        }

        // restaurar combosExistentes original
        this.combosExistentes = (originalExists == null) ? null : new HashSet<>(originalExists);

        return freq;
    }

    // ============================ Pipeline ============================

    private void prepararTransacoes() {
        if (transacoes != null) return;
        // carregar histórico — requer um arquivo externo válido
        if (historyPath == null) {
            throw new IllegalStateException("Nenhum historyPath configurado: forneça cfgInicial.historyPath apontando para o CSV de histórico.");
        }
        int[][] hist;
        try {
            hist = carregarHistorico(historyPath);
        } catch (IOException ex) {
            throw new IllegalStateException("Falha ao carregar histórico de '" + historyPath + "': " + ex.getMessage(), ex);
        }
        // Converter todas as linhas do range em transações (ordem preservada)
        transacoesTodas = new ArrayList<>();
        transacoesTodasMask = new ArrayList<>();
        for (int[] sorteio : hist) {
            Set<Integer> s = new HashSet<>();
            int mask = 0;
            for (int num : sorteio) {
                if (num >= MIN_NUMERO && num <= MAX_NUMERO) {
                    s.add(num);
                    mask |= (1 << (num - 1));
                }
            }
            if (!s.isEmpty()) {
                transacoesTodas.add(s);
                transacoesTodasMask.add(mask);
            }
        }
        // Aplicar janela de treinamento se configurada
        if (windowSize > 0 && transacoesTodas.size() > windowSize) {
            transacoes = new ArrayList<>(transacoesTodas.subList(transacoesTodas.size() - windowSize, transacoesTodas.size()));
            transacoesMask = new ArrayList<>(transacoesTodasMask.subList(transacoesTodasMask.size() - windowSize, transacoesTodasMask.size()));
        } else {
            transacoes = new ArrayList<>(transacoesTodas);
            transacoesMask = new ArrayList<>(transacoesTodasMask);
        }

        // pesos por recência: transacoes[0] = mais antigo, transacoes[n-1] = mais recente
        int n = transacoes.size();
        pesosTransacoes = new ArrayList<>(n);
        pesoTotal = 0.0;
        for (int i = 0; i < n; i++) {
            int age = (n - 1) - i; // 0 para o mais recente
            double w = Math.pow(recencyAlpha, age);
            pesosTransacoes.add(w);
            pesoTotal += w;
        }
    }

    private void executarMineracao() {
        // garantir estado limpo de suportes/regras antes de minerar
        supportMapMask.clear();
        regras.clear();
        prepararTransacoes();
        int n = transacoes.size();
        if (n == 0) throw new IllegalStateException("Sem transações no histórico");

        // Apriori ponderado por recência
        // k = 1
        Map<Set<Integer>, Double> contagem1 = new HashMap<>();
        for (int t = 0; t < n; t++) {
            Set<Integer> trans = transacoes.get(t);
            double w = pesosTransacoes.get(t);
            for (Integer item : trans) {
                Set<Integer> s = Collections.singleton(item);
                contagem1.put(s, contagem1.getOrDefault(s, 0.0) + w);
            }
        }
        List<Set<Integer>> frequentesK = new ArrayList<>();
        for (Map.Entry<Set<Integer>, Double> e : contagem1.entrySet()) {
            double sup = e.getValue() / pesoTotal;
            if (sup >= minSupport) {
                int mask = maskFromSet(e.getKey());
                supportMapMask.put(mask, sup);
                frequentesK.add(e.getKey());
            }
        }

        // k = 2..maxK
        List<Set<Integer>> prevFrequent = new ArrayList<>(frequentesK);
        for (int k = 2; k <= Math.max(2, maxK); k++) {
            Set<Set<Integer>> candidates = gerarCandidatos(prevFrequent, k);
            if (candidates.isEmpty()) break;
            Map<Set<Integer>, Double> contagem = new HashMap<>();
            // cache bitmask dos candidatos para testes rápidos
            Map<Set<Integer>, Integer> candMask = new HashMap<>();
            for (Set<Integer> c : candidates) { contagem.put(c, 0.0); candMask.put(c, maskFromSet(c)); }

            for (int t = 0; t < n; t++) {
                // usar representação em bitmask quando disponível
                int transM = (transacoesMask != null && transacoesMask.size() > t) ? transacoesMask.get(t) : maskFromSet(transacoes.get(t));
                double w = pesosTransacoes.get(t);
                for (Set<Integer> c : candidates) {
                    int cm = candMask.get(c);
                    if ((transM & cm) == cm) {
                        contagem.put(c, contagem.get(c) + w);
                    }
                }
            }

            List<Set<Integer>> frequentK = new ArrayList<>();
            for (Map.Entry<Set<Integer>, Double> e : contagem.entrySet()) {
                double sup = e.getValue() / pesoTotal;
                if (sup >= minSupport) {
                    int mask = maskFromSet(e.getKey());
                    supportMapMask.put(mask, sup);
                    frequentK.add(e.getKey());
                }
            }

            if (frequentK.isEmpty()) break;
            prevFrequent = frequentK;
        }

        // Regras de associação (usar suporte armazenado por mask)
        regras.clear();
        for (Map.Entry<Integer, Double> e : supportMapMask.entrySet()) {
            int itemsetMask = e.getKey();
            Set<Integer> itemset = setFromMask(itemsetMask);
            if (itemset.size() < 2) continue;

            List<Set<Integer>> subsets = gerarSubconjuntosProprios(itemset);
            for (Set<Integer> antecedente : subsets) {
                Set<Integer> consequente = new HashSet<>(itemset);
                consequente.removeAll(antecedente);
                if (consequente.isEmpty()) continue;

                int maskAnte = maskFromSet(antecedente);
                Double supItemset = supportMapMask.get(itemsetMask);
                Double supAnte = supportMapMask.get(maskAnte);
                if (supItemset == null || supAnte == null || supAnte == 0) continue;

                double conf = supItemset / supAnte;
                if (conf >= minConfidence) {
                    int maskCons = maskFromSet(consequente);
                    regras.add(new AssociationRule(maskAnte, maskCons, supItemset, conf));
                }
            }
        }
    }

    private Set<Set<Integer>> gerarCandidatos(List<Set<Integer>> prev, int k) {
        Set<Set<Integer>> candidatos = new HashSet<>();
        int m = prev.size();
        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j < m; j++) {
                Set<Integer> a = prev.get(i);
                Set<Integer> b = prev.get(j);
                Set<Integer> union = new HashSet<>(a);
                union.addAll(b);
                if (union.size() == k) candidatos.add(union);
            }
        }
        return candidatos;
    }

    private List<Set<Integer>> gerarSubconjuntosProprios(Set<Integer> s) {
        List<Integer> items = new ArrayList<>(s);
        int n = items.size();
        List<Set<Integer>> resultado = new ArrayList<>();
        int max = (1 << n);
        for (int mask = 1; mask < max - 1; mask++) {
            Set<Integer> sub = new HashSet<>();
            for (int i = 0; i < n; i++) if ((mask & (1 << i)) != 0) sub.add(items.get(i));
            resultado.add(sub);
        }
        return resultado;
    }

    // Converte um conjunto de números (1..25) para representação em bitmask (bit 0 -> número 1)
    private static int maskFromSet(Set<Integer> s) {
        int m = 0;
        for (Integer v : s) {
            if (v != null && v >= 1 && v <= 25) m |= (1 << (v - 1));
        }
        return m;
    }

    // Converte bitmask para Set<Integer>
    private static Set<Integer> setFromMask(int mask) {
        Set<Integer> out = new HashSet<>();
        for (int i = 0; i < 25; i++) {
            if (((mask >> i) & 1) != 0) out.add(i + 1);
        }
        return out;
    }

    private Map<Integer, Double> pontuarNumeros() {
        Map<Integer, Double> scores = new HashMap<>();
        for (int i = MIN_NUMERO; i <= MAX_NUMERO; i++) scores.put(i, 0.0);

        // Analisar padrões cíclicos primeiro
        analisarPadroesCiclicos();

        // 1) suporte dos singletons (peso 30% - reduzido de 35%)
        for (int i = MIN_NUMERO; i <= MAX_NUMERO; i++) {
            int m = (1 << (i - 1));
            Double sup = supportMapMask.getOrDefault(m, 0.0);
            scores.put(i, scores.get(i) + sup * 0.30);
        }

        // 2) confiança média das regras onde o número está no consequente (peso 35% - reduzido de 40%)
        Map<Integer, List<Double>> confs = new HashMap<>();
        for (AssociationRule r : regras) {
            int cm = r.consequenteMask;
            for (int b = 0; b < 25; b++) if (((cm >> b) & 1) != 0) {
                int num = b + 1;
                confs.computeIfAbsent(num, k -> new ArrayList<>()).add(r.confidence);
            }
        }
        for (Map.Entry<Integer, List<Double>> e : confs.entrySet()) {
            double avg = e.getValue().stream().mapToDouble(d -> d).average().orElse(0.0);
            scores.put(e.getKey(), scores.get(e.getKey()) + avg * 0.35);
        }

        // 3) bônus de recência: frequência nas últimas X transações (peso 20% - aumentado de 15%)
        int n = transacoes.size();
        int janela = Math.max(10, Math.min(100, n / 2)); // janela maior
        int inicio = Math.max(0, n - janela);
        Map<Integer, Integer> freqRecentes = new HashMap<>();
        for (int i = MIN_NUMERO; i <= MAX_NUMERO; i++) freqRecentes.put(i, 0);
        for (int t = inicio; t < n; t++) {
            for (Integer num : transacoes.get(t)) {
                freqRecentes.put(num, freqRecentes.get(num) + 1);
            }
        }
        for (int i = MIN_NUMERO; i <= MAX_NUMERO; i++) {
            double fr = freqRecentes.get(i) / (double) Math.max(1, (n - inicio));
            scores.put(i, scores.get(i) + fr * 0.20);
        }

        // 4) bônus de padrões cíclicos (peso 15% - aumentado de 10%)
        for (int i = MIN_NUMERO; i <= MAX_NUMERO; i++) {
            Double cicloScore = numeroFrequenciaMedia.getOrDefault(i, 0.0);
            scores.put(i, scores.get(i) + cicloScore * 0.15);
        }

        return scores;
    }

    private List<List<Integer>> gerarApostas(Map<Integer, Double> scores) {
        List<List<Integer>> apostas = new ArrayList<>();
        Random baseRnd = (seed == null) ? new Random(System.currentTimeMillis()) : new Random(seed);
        if (combosExistentes == null) combosExistentes = carregarCombosExistentes();
        Set<String> geradasNorm = new HashSet<>();

        // Tenta incluir uma combinação alvo fixa (se configurada)
        if (ensureTargetCombo && targetCombo != null && targetCombo.size() == TAMANHO_APOSTA) {
            List<Integer> alvo = new ArrayList<>(targetCombo);
            Collections.sort(alvo);
            if (diversidadeOk(apostas, alvo) && !jaExiste(alvo, geradasNorm)) {
                String k = keyFromList(alvo);
                geradasNorm.add(k);
                combosExistentes.add(k);
                apostas.add(alvo);
            }
        }

        for (int i = apostas.size(); i < NUMERO_APOSTAS; i++) {
            int attempts = 0;
            List<Integer> aposta;
            do {
                aposta = gerarApostaUnica(scores, new Random(baseRnd.nextLong()));
                attempts++;
                if (attempts > 500) break; // aumentado de 200 para 500
            } while (!diversidadeOk(apostas, aposta) || jaExiste(aposta, geradasNorm));

            // Se passou do limite, tenta fallback aleatório respeitando tamanho
            int safety = 0;
            // Removido fallback aleatório: não geramos apostas puramente aleatórias.
            // Caso exceda tentativas, mantemos a última combinação gerada pelo modelo.

            String k = keyFromList(aposta);
            geradasNorm.add(k);
            combosExistentes.add(k);
            apostas.add(aposta);
        }
        
        return apostas;
    }

    private boolean diversidadeOk(List<List<Integer>> existentes, List<Integer> nova) {
        for (List<Integer> a : existentes) {
            int overlap = intersecao(a, nova);
            if (overlap > minDiversityOverlap) return false;
        }
        return true;
    }

    private int intersecao(List<Integer> a, List<Integer> b) {
        int i = 0, j = 0, c = 0;
        while (i < a.size() && j < b.size()) {
            int x = a.get(i), y = b.get(j);
            if (x == y) { c++; i++; j++; }
            else if (x < y) i++; else j++;
        }
        return c;
    }

    // Analisa padrões cíclicos dos números para detectar ciclos de aparição
    private void analisarPadroesCiclicos() {
        numeroFrequenciaMedia.clear();
        numeroCiclo.clear();

        if (transacoesTodas == null || transacoesTodas.isEmpty()) return;

        // Para cada número, calcular frequência e ciclo médio
        for (int num = MIN_NUMERO; num <= MAX_NUMERO; num++) {
            List<Integer> posicoes = new ArrayList<>();
            for (int t = 0; t < transacoesTodas.size(); t++) {
                if (transacoesTodas.get(t).contains(num)) {
                    posicoes.add(t);
                }
            }

            if (posicoes.isEmpty()) {
                numeroFrequenciaMedia.put(num, 0.0);
                numeroCiclo.put(num, Integer.MAX_VALUE);
                continue;
            }

            // Frequência relativa
            double freq = posicoes.size() / (double) transacoesTodas.size();
            numeroFrequenciaMedia.put(num, freq);

            // Ciclo médio (distância média entre aparições)
            if (posicoes.size() > 1) {
                double cicloMedio = 0.0;
                for (int i = 1; i < posicoes.size(); i++) {
                    cicloMedio += (posicoes.get(i) - posicoes.get(i - 1));
                }
                cicloMedio /= (posicoes.size() - 1);
                numeroCiclo.put(num, (int) Math.round(cicloMedio));
            } else {
                numeroCiclo.put(num, Integer.MAX_VALUE);
            }
        }
    }

    private List<Integer> gerarApostaUnica(Map<Integer, Double> scores, Random r) {
        // Pré-índice de regras por consequente para reforço incremental
        Map<Integer, List<AssociationRule>> regrasPorConsequente = new HashMap<>();
        for (AssociationRule ar : regras) {
            int cm = ar.consequenteMask;
            for (int b = 0; b < 25; b++) if (((cm >> b) & 1) != 0) {
                int num = b + 1;
                regrasPorConsequente.computeIfAbsent(num, k -> new ArrayList<>()).add(ar);
            }
        }

        // construir base de pesos com variação aleatória reduzida (mais consistência)
        Map<Integer, Double> basePeso = new HashMap<>();
        for (int i = MIN_NUMERO; i <= MAX_NUMERO; i++) {
            double base = scores.getOrDefault(i, 0.0);
            double variacao = 0.95 + r.nextDouble() * 0.10; // 0.95..1.05 (mais consistente)
            basePeso.put(i, Math.max(0.0, base * variacao));
        }

        List<Integer> aposta = new ArrayList<>();
        Set<Integer> escolhidos = new HashSet<>();
        int escolhidosMask = 0;

        while (aposta.size() < TAMANHO_APOSTA) {
            // calcular pesos dinâmicos com reforço DE ASSOCIAÇÃO MUITO AGRESSIVO
            double soma = 0.0;
            List<NumeroPeso> candidatos = new ArrayList<>();
            for (int num = MIN_NUMERO; num <= MAX_NUMERO; num++) {
                if (escolhidos.contains(num)) continue;
                double w = basePeso.getOrDefault(num, 0.0);
                if (w <= 0) continue;
                double boost = 0.0;
                
                // Reforço por regras: números que aparecem em regras satisfeitas
                List<AssociationRule> rel = regrasPorConsequente.get(num);
                if (rel != null && !rel.isEmpty()) {
                    double acum = 0.0; int cnt = 0;
                    for (AssociationRule ar : rel) {
                        if ((escolhidosMask & ar.antecedenteMask) == ar.antecedenteMask) {
                            acum += ar.confidence * ar.confidence; // quadrado para mais peso
                            cnt++;
                        }
                    }
                    if (cnt > 0) boost += (acum / cnt) * 2.0; // multiplicador agressivo
                }
                
                // Reforço por coocorrência: números que aparecem com escolhidos
                if (!escolhidos.isEmpty()) {
                    double supAcum = 0.0; int sc = 0;
                    for (Integer e : escolhidos) {
                        int pairMask = (1 << (num - 1)) | (1 << (e - 1));
                        Double sup = supportMapMask.get(pairMask);
                        if (sup != null) { 
                            supAcum += sup * sup; // quadrado para favor pares com alto suporte
                            sc++; 
                        }
                    }
                    if (sc > 0) boost += (supAcum / sc) * 1.5;
                }
                
                // Reforço por frequência recente (números que já apareceram recentemente)
                int n = transacoes.size();
                int janela = Math.max(5, n / 4);
                int freqRec = 0;
                for (int t = Math.max(0, n - janela); t < n; t++) {
                    if (transacoes.get(t).contains(num)) freqRec++;
                }
                double freqScore = freqRec / (double) Math.max(1, janela);
                boost += freqScore * 0.8;
                
                double wFinal = w * (1.0 + assocBoost * boost);
                if (wFinal < 1e-9) wFinal = 1e-9;
                candidatos.add(new NumeroPeso(num, wFinal));
                soma += wFinal;
            }

            if (candidatos.isEmpty()) break;

            // roleta proporcional ao peso final (mantém estocástica)
            double alvo = r.nextDouble() * soma;
            double acc = 0.0;
            int idx = -1;
            for (int j = 0; j < candidatos.size(); j++) {
                acc += candidatos.get(j).peso;
                if (alvo <= acc) { idx = j; break; }
            }
            if (idx == -1) idx = candidatos.size() - 1;
            int escolhido = candidatos.get(idx).numero;
            escolhidos.add(escolhido);
            aposta.add(escolhido);
            escolhidosMask |= (1 << (escolhido - 1));
        }

        Collections.sort(aposta);
        // corrigir paridade caso necessário
        List<NumeroPeso> poolForParity = new ArrayList<>();
        for (Map.Entry<Integer, Double> e : basePeso.entrySet()) poolForParity.add(new NumeroPeso(e.getKey(), e.getValue()));
        ajustarParidade(aposta, poolForParity, r);
        Collections.sort(aposta);
        return aposta;
    }

    private List<Integer> gerarApostaFallback(Random baseRnd) {
        // Fallback: gera combinação aleatória simples de 15 números distintos 1..25, ordenada
        Random r = new Random(baseRnd.nextLong());
        List<Integer> nums = new ArrayList<>();
        for (int i = MIN_NUMERO; i <= MAX_NUMERO; i++) nums.add(i);
        Collections.shuffle(nums, r);
        List<Integer> aposta = new ArrayList<>(nums.subList(0, Math.min(TAMANHO_APOSTA, nums.size())));
        Collections.sort(aposta);
        return aposta;
    }

    private boolean jaExiste(List<Integer> aposta, Set<String> geradasNorm) {
        String k = keyFromList(aposta);
        if (geradasNorm.contains(k)) return true; // já gerada nesta execução
        if (combosExistentes != null && combosExistentes.contains(k)) return true; // já existe em arquivos
        return false;
    }

    private String keyFromList(List<Integer> aposta) {
        // garante ordenação e formata como "01 02 ... 15"
        List<Integer> ord = new ArrayList<>(aposta);
        Collections.sort(ord);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < ord.size(); i++) {
            if (i > 0) sb.append(' ');
            sb.append(String.format("%02d", ord.get(i)));
        }
        return sb.toString();
    }

    private Set<String> carregarCombosExistentes() {
        Set<String> out = new HashSet<>();
        // 1) ler todas as linhas do  do histórico (arquivo completo, ignorando range)
        if (historyPath != null) {
            try (BufferedReader br = new BufferedReader(new FileReader(historyPath))) {
                String line;
                while ((line = br.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty() || line.startsWith("#")) continue;
                    String[] toks = line.split("[^0-9]+");
                    List<Integer> nums = new ArrayList<>();
                    for (String tk : toks) {
                        if (tk.isEmpty()) continue;
                        try {
                            int v = Integer.parseInt(tk);
                            if (v >= MIN_NUMERO && v <= MAX_NUMERO) nums.add(v);
                        } catch (NumberFormatException ignore) {}
                    }
                    if (!nums.isEmpty()) {
                        // normalizar: únicos + sort
                        Set<Integer> s = new TreeSet<>(nums);
                        if (s.size() == TAMANHO_APOSTA) {
                            List<Integer> ord = new ArrayList<>(s);
                            out.add(keyFromList(ord));
                        }
                    }
                }
                } catch (IOException ex) {
                    System.err.println("Aviso: falha lendo combos do histórico: " + ex.getMessage());
                }
        }

        // NOTE: reading previously-saved output (`apostas_dmplus.txt`) was removed —
        // the program no longer relies on that file to avoid duplicates.
        return out;
    }

    private void ajustarParidade(List<Integer> aposta, List<NumeroPeso> pool, Random r) {
        int pares = (int) aposta.stream().filter(n -> n % 2 == 0).count();
        if (pares >= evenMin && pares <= evenMax) return;

        Set<Integer> escolhidos = new HashSet<>(aposta);
        List<NumeroPeso> candidatosAdicionar = new ArrayList<>();
        List<Integer> candidatosRemover = new ArrayList<>();

        boolean precisaMaisPares = pares < evenMin;
        for (NumeroPeso np : pool) {
            if (escolhidos.contains(np.numero)) continue;
            if (precisaMaisPares && np.numero % 2 == 0) candidatosAdicionar.add(np);
            if (!precisaMaisPares && np.numero % 2 != 0) candidatosAdicionar.add(np);
        }
        for (Integer n : aposta) {
            if (precisaMaisPares && n % 2 != 0) candidatosRemover.add(n);
            if (!precisaMaisPares && n % 2 == 0) candidatosRemover.add(n);
        }
        // ordenar por peso desc para adicionar, e por peso asc para remover (aprox.)
        candidatosAdicionar.sort((a, b) -> Double.compare(b.peso, a.peso));
        candidatosRemover.sort(Comparator.comparingInt(n -> {
            // peso aproximado baseado em posição na pool
            for (NumeroPeso np : pool) if (np.numero == n) return (int) (-np.peso * 100000);
            return 0;
        }));

        int targetMin = evenMin;
        int targetMax = evenMax;
        int maxTrocas = 10;
        int trocas = 0;
        while ((pares < targetMin || pares > targetMax) && trocas < maxTrocas && !candidatosAdicionar.isEmpty() && !candidatosRemover.isEmpty()) {
            NumeroPeso add = candidatosAdicionar.remove(0);
            Integer rem = candidatosRemover.remove(r.nextInt(Math.min(3, candidatosRemover.size()))); // remove um dos piores
            if (aposta.remove(rem)) {
                aposta.add(add.numero);
                pares = (int) aposta.stream().filter(n -> n % 2 == 0).count();
                trocas++;
            }
        }
    }

    // ============================ IO / Util ============================

    private int[][] carregarHistorico(String path) throws IOException {
        List<int[]> linhas = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            int lineNo = 0;
                while ((line = br.readLine()) != null) {
                lineNo++;
                if (lineNo < historyStartLine) continue;
                if (lineNo > historyEndLine) { break; }
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;
                String[] toks = line.split("[^0-9]+"); // separa por qualquer não-dígito
                List<Integer> nums = new ArrayList<>();
                for (String tk : toks) {
                    if (tk.isEmpty()) continue;
                    int v = Integer.parseInt(tk);
                    if (v >= MIN_NUMERO && v <= MAX_NUMERO) nums.add(v);
                }
                if (!nums.isEmpty()) {
                    // garantir únicos e ordenar
                    Set<Integer> s = new TreeSet<>(nums);
                    int[] arr = new int[s.size()];
                    int idx = 0;
                    for (Integer v : s) arr[idx++] = v;
                    linhas.add(arr);
                }
            }
        } catch (IOException ex) {
            // Propagar exceção para o chamador tratar — não usar fallback embutido
            throw ex;
        }
        if (linhas.isEmpty()) {
            throw new IOException("Histórico vazio ou inválido em '" + path + "'.");
        }
        int[][] out = new int[linhas.size()][];
        for (int i = 0; i < linhas.size(); i++) out[i] = linhas.get(i);
        System.out.println("+---------------------------------------------------+");
        System.out.println("|-------------- Carreguei " + out.length + " linhas --------------|");
        System.out.println("+---------------------------------------------------+");
        return out;
    }

    // método de gravação removido intencionalmente — o programa não grava arquivos

    private void exibirResumo(Map<Integer, Double> scores, int topN) {
        System.out.println("\n--- Top números por score ---");
        scores.entrySet().stream()
                .sorted((a, b) -> Double.compare(b.getValue(), a.getValue()))
                .limit(topN)
                .forEach(e -> System.out.printf("N%02d : %.4f\n", e.getKey(), e.getValue()));

        System.out.println("\n--- Regras de associação (top 20 por confiança) ---");
        regras.stream()
                .sorted((a, b) -> Double.compare(b.confidence, a.confidence))
                .limit(20)
                .forEach(r -> System.out.println(r));
    }

    // ============================ Exec / CLI ============================

    public void exec() {
        if (configuracoes.isEmpty()) {
            System.err.println("Nenhuma configuração cadastrada.");
            return;
        }
        // Executa a geração consolidada: 6 apostas por algoritmo (sem Aleatório)
        executarTodosAlgoritmosLotofacil();
    }
    
    // ==================== ALGORITMOS ADICIONAIS ====================
    
    private void executarTodosAlgoritmosLotofacil() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("EXECUTANDO TODOS OS ALGORITMOS - 3 APOSTAS CADA");
        System.out.println("=".repeat(80) + "\n");
        
        // Carregar transações uma vez para todos
        prepararTransacoes();
        if (transacoesTodas == null || transacoesTodas.isEmpty()) {
            System.err.println("Erro: histórico vazio ou inválido.");
            return;
        }
        
        Random baseRnd = new Random(System.currentTimeMillis());
        
        // 1. APRIORI + REGRAS DE ASSOCIAÇÃO (já implementado no DM Plus)
        try {
            System.out.println("\n" + "-".repeat(80));
            System.out.println("1. APRIORI + REGRAS DE ASSOCIAÇÃO (DM Plus)");
            System.out.println("-".repeat(80));
            executarMineracao();
            Map<Integer, Double> scores = pontuarNumeros();
            List<List<Integer>> apostasApriori = gerarApostas(scores);
            exibirApostas("Apriori + Regras de Associação", apostasApriori);
            exibirTopApostasTamanho("Apriori + Regras de Associação", rankingPorScore(scores), TAMANHO_APOSTA + 1, 3);
        } catch (Exception ex) {
            System.err.println("Erro no Apriori: " + ex.getMessage());
        }
        
        // 2. SÉRIES TEMPORAIS (ARIMA)
        try {
            System.out.println("\n" + "-".repeat(80));
            System.out.println("2. ANÁLISE DE SÉRIES TEMPORAIS (ARIMA)");
            System.out.println("-".repeat(80));
            TimeSeriesARIMAAnalyzer arimaAnalyzer = new TimeSeriesARIMAAnalyzer(
                    transacoesTodas, MIN_NUMERO, MAX_NUMERO, TAMANHO_APOSTA);
            List<List<Integer>> apostasARIMA = new ArrayList<>();
            for (int i = 0; i < 3; i++) {
                apostasARIMA.add(arimaAnalyzer.generateBetARIMA(new Random(baseRnd.nextLong())));
            }
            exibirApostas("ARIMA", apostasARIMA);
            exibirTopApostasTamanho("ARIMA", rankingPorFrequencia(apostasARIMA), TAMANHO_APOSTA + 1, 3);
        } catch (Exception ex) {
            System.err.println("Erro no ARIMA: " + ex.getMessage());
        }
        
        // 3. CLUSTERING (K-Means)
        try {
            System.out.println("\n" + "-".repeat(80));
            System.out.println("3. CLUSTERING (K-Means)");
            System.out.println("-".repeat(80));
            ClusteringAnalyzer clusterAnalyzer = new ClusteringAnalyzer(
                    transacoesTodas, MIN_NUMERO, MAX_NUMERO, TAMANHO_APOSTA);
            List<List<Integer>> apostasClusters = new ArrayList<>();
            for (int i = 0; i < 3; i++) {
                apostasClusters.add(clusterAnalyzer.generateBetFromClusters(new Random(baseRnd.nextLong())));
            }
            exibirApostas("K-Means Clustering", apostasClusters);
            exibirTopApostasTamanho("K-Means Clustering", rankingPorFrequencia(apostasClusters), TAMANHO_APOSTA + 1, 3);
        } catch (Exception ex) {
            System.err.println("Erro no Clustering: " + ex.getMessage());
        }
        
        // 4. GRADIENT BOOSTING
        try {
            System.out.println("\n" + "-".repeat(80));
            System.out.println("4. GRADIENT BOOSTING");
            System.out.println("-".repeat(80));
            GradientBoostingAnalyzer boostAnalyzer = new GradientBoostingAnalyzer(
                    transacoesTodas, MIN_NUMERO, MAX_NUMERO, TAMANHO_APOSTA);
            List<List<Integer>> apostasBoost = new ArrayList<>();
            for (int i = 0; i < 3; i++) {
                apostasBoost.add(boostAnalyzer.generateBetGradientBoosting(new Random(baseRnd.nextLong())));
            }
            exibirApostas("Gradient Boosting", apostasBoost);
            exibirTopApostasTamanho("Gradient Boosting", rankingPorFrequencia(apostasBoost), TAMANHO_APOSTA + 1, 3);
        } catch (Exception ex) {
            System.err.println("Erro no Gradient Boosting: " + ex.getMessage());
        }
        
        // 5. ANÁLISE DE ENTROPIA
        try {
            System.out.println("\n" + "-".repeat(80));
            System.out.println("5. ANÁLISE DE ENTROPIA E INFORMAÇÃO MÚTUA");
            System.out.println("-".repeat(80));
            EntropyAnalyzer entropyAnalyzer = new EntropyAnalyzer(
                    transacoesTodas, MIN_NUMERO, MAX_NUMERO, TAMANHO_APOSTA);
            List<List<Integer>> apostasEntropy = new ArrayList<>();
            for (int i = 0; i < 3; i++) {
                apostasEntropy.add(entropyAnalyzer.generateBetEntropy(new Random(baseRnd.nextLong())));
            }
            exibirApostas("Análise de Entropia", apostasEntropy);
            exibirTopApostasTamanho("Análise de Entropia", rankingPorFrequencia(apostasEntropy), TAMANHO_APOSTA + 1, 3);
        } catch (Exception ex) {
            System.err.println("Erro na Análise de Entropia: " + ex.getMessage());
        }
        
        // 6. MARKOV DE ORDEM SUPERIOR
        try {
            System.out.println("\n" + "-".repeat(80));
            System.out.println("6. CADEIA DE MARKOV DE ORDEM SUPERIOR");
            System.out.println("-".repeat(80));
            HigherOrderMarkovAnalyzer markovAnalyzer = new HigherOrderMarkovAnalyzer(
                    transacoesTodas, MIN_NUMERO, MAX_NUMERO, TAMANHO_APOSTA);
            List<List<Integer>> apostasMarkov = new ArrayList<>();
            for (int i = 0; i < 3; i++) {
                apostasMarkov.add(markovAnalyzer.generateBetByWalkingMarkov(new Random(baseRnd.nextLong())));
            }
            exibirApostas("Markov Ordem Superior", apostasMarkov);
            exibirTopApostasTamanho("Markov Ordem Superior", rankingPorFrequencia(apostasMarkov), TAMANHO_APOSTA + 1, 3);
        } catch (Exception ex) {
            System.err.println("Erro no Markov: " + ex.getMessage());
        }
        
        // 7. REGRESSÃO LOGÍSTICA
        try {
            System.out.println("\n" + "-".repeat(80));
            System.out.println("7. REGRESSÃO LOGÍSTICA PROBABILÍSTICA");
            System.out.println("-".repeat(80));
            LogisticRegressionAnalyzer logisticAnalyzer = new LogisticRegressionAnalyzer(
                    transacoesTodas, MIN_NUMERO, MAX_NUMERO, TAMANHO_APOSTA);
            List<List<Integer>> apostasLogistic = new ArrayList<>();
            for (int i = 0; i < 3; i++) {
                apostasLogistic.add(logisticAnalyzer.generateBetLogistic(new Random(baseRnd.nextLong())));
            }
            exibirApostas("Regressão Logística", apostasLogistic);
            exibirTopApostasTamanho("Regressão Logística", rankingPorFrequencia(apostasLogistic), TAMANHO_APOSTA + 1, 3);
        } catch (Exception ex) {
            System.err.println("Erro na Regressão Logística: " + ex.getMessage());
        }
        
        // 8. RANDOM FOREST
        try {
            System.out.println("\n" + "-".repeat(80));
            System.out.println("8. RANDOM FOREST (Ensemble Bagging)");
            System.out.println("-".repeat(80));
            RandomForestAnalyzer forestAnalyzer = new RandomForestAnalyzer(
                    transacoesTodas, MIN_NUMERO, MAX_NUMERO, TAMANHO_APOSTA);
            List<List<Integer>> apostasForest = new ArrayList<>();
            for (int i = 0; i < 3; i++) {
                apostasForest.add(forestAnalyzer.generateBetRandomForest(new Random(baseRnd.nextLong())));
            }
            exibirApostas("Random Forest", apostasForest);
            exibirTopApostasTamanho("Random Forest", rankingPorFrequencia(apostasForest), TAMANHO_APOSTA + 1, 3);
        } catch (Exception ex) {
            System.err.println("Erro no Random Forest: " + ex.getMessage());
        }
        
        // 9. NAIVE BAYES PROBABILÍSTICO
        try {
            System.out.println("\n" + "-".repeat(80));
            System.out.println("9. NAIVE BAYES PROBABILÍSTICO");
            System.out.println("-".repeat(80));
            NaiveBayesAnalyzer nbAnalyzer = new NaiveBayesAnalyzer(
                    transacoesTodas, MIN_NUMERO, MAX_NUMERO, TAMANHO_APOSTA);
            List<List<Integer>> apostasNB = new ArrayList<>();
            for (int i = 0; i < 3; i++) {
                apostasNB.add(nbAnalyzer.generateBetNaiveBayes(new Random(baseRnd.nextLong())));
            }
            exibirApostas("Naive Bayes", apostasNB);
            exibirTopApostasTamanho("Naive Bayes", rankingPorFrequencia(apostasNB), TAMANHO_APOSTA + 1, 3);
        } catch (Exception ex) {
            System.err.println("Erro no Naive Bayes: " + ex.getMessage());
        }
        
        // 10. META-ENSEMBLE (Votação de todos os algoritmos)
        try {
            System.out.println("\n" + "-".repeat(80));
            System.out.println("10. META-ENSEMBLE (VOTAÇÃO DE TODOS OS 9 ALGORITMOS)");
            System.out.println("-".repeat(80));
            MetaEnsembleAnalyzer metaAnalyzer = new MetaEnsembleAnalyzer(
                    transacoesTodas, MIN_NUMERO, MAX_NUMERO, TAMANHO_APOSTA);
            List<List<Integer>> apostasMeta = new ArrayList<>();
            for (int i = 0; i < 3; i++) {
                apostasMeta.add(metaAnalyzer.generateBetMetaEnsemble(new Random(baseRnd.nextLong())));
            }
            exibirApostas("Meta-Ensemble (Votação)", apostasMeta);
            exibirTopApostasTamanho("Meta-Ensemble (Votação)", rankingPorFrequencia(apostasMeta), TAMANHO_APOSTA + 1, 3);
        } catch (Exception ex) {
            System.err.println("Erro no Meta-Ensemble: " + ex.getMessage());
        }
        
        // 11. HMM (Hidden Markov Model)
        try {
            System.out.println("\n" + "-".repeat(80));
            System.out.println("11. HMM (HIDDEN MARKOV MODEL)");
            System.out.println("-".repeat(80));
            HMMBetGenerator hmmGenerator = new HMMBetGenerator(
                    transacoesTodas, TAMANHO_APOSTA, MIN_NUMERO, MAX_NUMERO, 0.7);
            List<List<Integer>> apostasHMM = hmmGenerator.generateMultipleBetsUsingHMM(3, minDiversityOverlap);
            exibirApostas("HMM (Hidden Markov Model)", apostasHMM);
            exibirTopApostasTamanho("HMM (Hidden Markov Model)", rankingPorFrequencia(apostasHMM), TAMANHO_APOSTA + 1, 3);
        } catch (Exception ex) {
            System.err.println("Erro no HMM: " + ex.getMessage());
        }
        
        // 12. LSTM (Long Short-Term Memory)
        try {
            System.out.println("\n" + "-".repeat(80));
            System.out.println("12. LSTM (LONG SHORT-TERM MEMORY)");
            System.out.println("-".repeat(80));
            LSTMAnalyzer lstmAnalyzer = new LSTMAnalyzer(25, 32, 5, 0.001, 10);
            lstmAnalyzer.train(transacoesTodas, 32);
            List<List<Integer>> apostasLSTM = lstmAnalyzer.generateBets(3);
            exibirApostas("LSTM (Long Short-Term Memory)", apostasLSTM);
            exibirTopApostasTamanho("LSTM (Long Short-Term Memory)", rankingPorFrequencia(apostasLSTM), TAMANHO_APOSTA + 1, 3);
        } catch (Exception ex) {
            System.err.println("Erro no LSTM: " + ex.getMessage());
        }
        
        // 13. REDE NEURAL ARTIFICIAL
        try {
            System.out.println("\n" + "-".repeat(80));
            System.out.println("13. REDE NEURAL ARTIFICIAL");
            System.out.println("-".repeat(80));
            int[] layerSizes = {25, 64, 32, 25};
            NeuralNetworkAnalyzer neuralAnalyzer = new NeuralNetworkAnalyzer(layerSizes, 0.01, 10);
            neuralAnalyzer.train(transacoesTodas, 32);
            List<List<Integer>> apostasNeural = neuralAnalyzer.generateBets(3);
            exibirApostas("Rede Neural Artificial", apostasNeural);
            exibirTopApostasTamanho("Rede Neural Artificial", rankingPorFrequencia(apostasNeural), TAMANHO_APOSTA + 1, 3);
        } catch (Exception ex) {
            System.err.println("Erro na Rede Neural: " + ex.getMessage());
        }
        
        // 14. ALGORITMO GENÉTICO
        try {
            System.out.println("\n" + "-".repeat(80));
            System.out.println("14. ALGORITMO GENÉTICO");
            System.out.println("-".repeat(80));
            GeneticAlgorithmOptimizer gaOptimizer = new GeneticAlgorithmOptimizer(
                    transacoesTodas, 50, 30);
            List<List<Integer>> apostasGA = gaOptimizer.getBestBets(3);
            exibirApostas("Algoritmo Genético", apostasGA);
            exibirTopApostasTamanho("Algoritmo Genético", rankingPorFrequencia(apostasGA), TAMANHO_APOSTA + 1, 3);
        } catch (Exception ex) {
            System.err.println("Erro no Algoritmo Genético: " + ex.getMessage());
        }
        
        // 15. APOSTA PONDERADA POR CONFIANÇA
        try {
            System.out.println("\n" + "-".repeat(80));
            System.out.println("15. APOSTA PONDERADA POR CONFIANÇA (ELITE)");
            System.out.println("-".repeat(80));
            executarMineracao();
            Map<Integer, Double> confiancaScores = new HashMap<>();
            for (int i = MIN_NUMERO; i <= MAX_NUMERO; i++) {
                confiancaScores.put(i, 0.0);
            }
            // Usar apenas confiança das regras (peso 100%)
            Map<Integer, List<Double>> confs = new HashMap<>();
            for (AssociationRule r : regras) {
                int cm = r.consequenteMask;
                for (int b = 0; b < 25; b++) if (((cm >> b) & 1) != 0) {
                    int num = b + 1;
                    confs.computeIfAbsent(num, k -> new ArrayList<>()).add(r.confidence);
                }
            }
            for (Map.Entry<Integer, List<Double>> e : confs.entrySet()) {
                double avg = e.getValue().stream().mapToDouble(d -> d).average().orElse(0.0);
                confiancaScores.put(e.getKey(), avg);
            }
            List<List<Integer>> apostasPonderada = gerarApostas(confiancaScores);
            exibirApostas("Aposta Ponderada por Confiança", apostasPonderada);
            exibirTopApostasTamanho("Aposta Ponderada por Confiança", rankingPorScore(confiancaScores), TAMANHO_APOSTA + 1, 3);
        } catch (Exception ex) {
            System.err.println("Erro na Aposta Ponderada: " + ex.getMessage());
        }
        
        // 16. APOSTA HÍBRIDA (ELITE)
        try {
            System.out.println("\n" + "-".repeat(80));
            System.out.println("16. APOSTA HÍBRIDA (ELITE) - Combinação de Apriori + Genético");
            System.out.println("-".repeat(80));
            executarMineracao();
            Map<Integer, Double> hybridScores = pontuarNumeros();
            GeneticAlgorithmOptimizer hybridOptimizer = new GeneticAlgorithmOptimizer(transacoesTodas, 100, 50);
            hybridOptimizer.setNumberScores(hybridScores);
            List<List<Integer>> apostasHibrida = hybridOptimizer.getBestBets(3);
            exibirApostas("Aposta Híbrida (Elite)", apostasHibrida);
            exibirTopApostasTamanho("Aposta Híbrida (Elite)", rankingPorScore(hybridScores), TAMANHO_APOSTA + 1, 3);
        } catch (Exception ex) {
            System.err.println("Erro na Aposta Híbrida: " + ex.getMessage());
        }
        
        // 17. APOSTA CONSENSUAL
        try {
            System.out.println("\n" + "-".repeat(80));
            System.out.println("17. APOSTA CONSENSUAL - Votação de Todos os 14 Algoritmos");
            System.out.println("-".repeat(80));
            // Gerar apostas de cada algoritmo e contar frequência de cada número
            Map<Integer, Integer> consensoVotos = new HashMap<>();
            for (int i = MIN_NUMERO; i <= MAX_NUMERO; i++) {
                consensoVotos.put(i, 0);
            }
            
            Random consensoRnd = new Random(System.currentTimeMillis());
            
            // Contar votos de cada algoritmo
            try {
                prepararTransacoes();
                executarMineracao();
                Map<Integer, Double> scores = pontuarNumeros();
                List<List<Integer>> apostasApriori = gerarApostas(scores);
                for (List<Integer> aposta : apostasApriori) {
                    for (int num : aposta) consensoVotos.put(num, consensoVotos.get(num) + 1);
                }
            } catch (Exception e) {}
            
            try {
                TimeSeriesARIMAAnalyzer arima = new TimeSeriesARIMAAnalyzer(transacoesTodas, MIN_NUMERO, MAX_NUMERO, TAMANHO_APOSTA);
                for (int i = 0; i < 3; i++) {
                    List<Integer> aposta = arima.generateBetARIMA(new Random(consensoRnd.nextLong()));
                    for (int num : aposta) consensoVotos.put(num, consensoVotos.get(num) + 1);
                }
            } catch (Exception e) {}
            
            try {
                NaiveBayesAnalyzer nb = new NaiveBayesAnalyzer(transacoesTodas, MIN_NUMERO, MAX_NUMERO, TAMANHO_APOSTA);
                for (int i = 0; i < 3; i++) {
                    List<Integer> aposta = nb.generateBetNaiveBayes(new Random(consensoRnd.nextLong()));
                    for (int num : aposta) consensoVotos.put(num, consensoVotos.get(num) + 1);
                }
            } catch (Exception e) {}
            
            try {
                RandomForestAnalyzer forest = new RandomForestAnalyzer(transacoesTodas, MIN_NUMERO, MAX_NUMERO, TAMANHO_APOSTA);
                for (int i = 0; i < 3; i++) {
                    List<Integer> aposta = forest.generateBetRandomForest(new Random(consensoRnd.nextLong()));
                    for (int num : aposta) consensoVotos.put(num, consensoVotos.get(num) + 1);
                }
            } catch (Exception e) {}
            
            // Selecionar 15 números com mais votos
            List<Integer> apostaConsensual = consensoVotos.entrySet().stream()
                    .sorted((a, b) -> Integer.compare(b.getValue(), a.getValue()))
                    .limit(TAMANHO_APOSTA)
                    .map(Map.Entry::getKey)
                    .collect(java.util.stream.Collectors.toList());
            Collections.sort(apostaConsensual);
            
            List<List<Integer>> apostasConsensual = new ArrayList<>();
            apostasConsensual.add(apostaConsensual);
            exibirApostas("Aposta Consensual", apostasConsensual);

            // Também gerar TOP 3 APOSTA CONSENSUAL com 16 dezenas (sem alterar as de 15 acima)
            int tamanho16 = Math.min(MAX_NUMERO - MIN_NUMERO + 1, TAMANHO_APOSTA + 1);
            int qtd16 = 3;
            if (tamanho16 > TAMANHO_APOSTA) {
                // Ranking por votos (desc) e, em empate, pelo número (asc)
                List<Integer> rankingVotos = consensoVotos.entrySet().stream()
                        .sorted((a, b) -> {
                            int cmp = Integer.compare(b.getValue(), a.getValue());
                            if (cmp != 0) return cmp;
                            return Integer.compare(a.getKey(), b.getKey());
                        })
                        .map(Map.Entry::getKey)
                        .collect(java.util.stream.Collectors.toList());

                System.out.println("\n" + "-".repeat(80));
                System.out.println("TOP " + qtd16 + " APOSTA CONSENSUAL (" + tamanho16 + " dezenas)");
                System.out.println("-".repeat(80));

                for (int i = 0; i < qtd16; i++) {
                    int start = Math.min(i, Math.max(0, rankingVotos.size() - tamanho16));
                    List<Integer> aposta16 = new ArrayList<>(rankingVotos.subList(start, start + tamanho16));
                    Collections.sort(aposta16);
                    System.out.printf("Aposta %d (16 dezenas): ", i + 1);
                    for (int num : aposta16) System.out.printf("%02d ", num);
                    System.out.println();
                }
            }
        } catch (Exception ex) {
            System.err.println("Erro na Aposta Consensual: " + ex.getMessage());
        }
        
        // 18. GETSON
        try {
            System.out.println("\n" + "-".repeat(80));
            System.out.println("18. GETSON - Duas apostas de 16 dezenas (8 pares, 8 ímpares; 5 ímpares primos) baseadas na Rede Neural Artificial");
            System.out.println("-".repeat(80));

            // Reutiliza o algoritmo 13 (Rede Neural Artificial) como base de pontuação
            int[] layerSizes = {25, 64, 32, 25};
            NeuralNetworkAnalyzer neuralAnalyzerGetson = new NeuralNetworkAnalyzer(layerSizes, 0.01, 10);
            neuralAnalyzerGetson.train(transacoesTodas, 32);
            Map<Integer, Double> nnScores = neuralAnalyzerGetson.predictNumberScores();

            Random r1 = new Random(System.currentTimeMillis());
            Random r2 = new Random(System.currentTimeMillis() + 1337);
            List<Integer> aposta1 = gerarApostaGetson(nnScores, r1);
            List<Integer> aposta2 = gerarApostaGetson(nnScores, r2);

            System.out.printf("Aposta 1 (16 dezenas): ");
            for (int num : aposta1) System.out.printf("%02d ", num);
            System.out.println();
            System.out.printf("Aposta 2 (16 dezenas): ");
            for (int num : aposta2) System.out.printf("%02d ", num);
            System.out.println();
        } catch (Exception ex) {
            System.err.println("Erro na Aposta GETSON: " + ex.getMessage());
        }
        
        System.out.println("\n" + "=".repeat(80));
        System.out.println("EXECUÇÃO COMPLETA - 14 ALGORITMOS + 3 ESTRATÉGIAS ESPECIAIS (45+ APOSTAS TOTAIS)");
        System.out.println("=".repeat(80) + "\n");
    }
    
    private void exibirApostas(String nomeAlgoritmo, List<List<Integer>> apostas) {
        // Garantir que temos exatamente 3 apostas
        List<List<Integer>> apostasCompletas = new ArrayList<>(apostas);
        Random fallbackRnd = new Random(System.nanoTime());
        
        while (apostasCompletas.size() < NUMERO_APOSTAS) {
            // Gerar aposta fallback aleatória se necessário
            List<Integer> falbackAposta = new ArrayList<>();
            List<Integer> nums = new ArrayList<>();
            for (int i = MIN_NUMERO; i <= MAX_NUMERO; i++) {
                nums.add(i);
            }
            Collections.shuffle(nums, fallbackRnd);
            for (int i = 0; i < TAMANHO_APOSTA; i++) {
                falbackAposta.add(nums.get(i));
            }
            Collections.sort(falbackAposta);
            apostasCompletas.add(falbackAposta);
        }
        
        System.out.println("Apostas geradas: " + apostasCompletas.size());
        for (int i = 0; i < Math.min(NUMERO_APOSTAS, apostasCompletas.size()); i++) {
            List<Integer> aposta = apostasCompletas.get(i);
            System.out.printf("Aposta %d: ", i + 1);
            for (int num : aposta) {
                System.out.printf("%02d ", num);
            }
            System.out.println();
        }
    }

    private void exibirTopApostasTamanho(String nomeAlgoritmo, List<Integer> rankingNumeros, int tamanho, int quantidade) {
        int maxRange = (MAX_NUMERO - MIN_NUMERO + 1);
        int tamanhoEfetivo = Math.min(maxRange, tamanho);
        if (tamanhoEfetivo <= TAMANHO_APOSTA) return; // só interessa quando aumenta (ex: 16)
        if (rankingNumeros == null || rankingNumeros.size() < tamanhoEfetivo) return;

        System.out.println("\n" + "-".repeat(80));
        System.out.println("TOP " + quantidade + " " + nomeAlgoritmo.toUpperCase() + " (" + tamanhoEfetivo + " dezenas)");
        System.out.println("-".repeat(80));

        int qtd = Math.max(0, quantidade);
        for (int i = 0; i < qtd; i++) {
            int start = Math.min(i, Math.max(0, rankingNumeros.size() - tamanhoEfetivo));
            List<Integer> aposta = new ArrayList<>(rankingNumeros.subList(start, start + tamanhoEfetivo));
            Collections.sort(aposta);
            System.out.printf("Aposta %d (%d dezenas): ", i + 1, tamanhoEfetivo);
            for (int num : aposta) System.out.printf("%02d ", num);
            System.out.println();
        }
    }

    private List<Integer> rankingPorScore(Map<Integer, Double> scores) {
        Map<Integer, Double> safe = new HashMap<>();
        for (int i = MIN_NUMERO; i <= MAX_NUMERO; i++) {
            safe.put(i, (scores != null) ? scores.getOrDefault(i, 0.0) : 0.0);
        }
        List<Map.Entry<Integer, Double>> entries = new ArrayList<>(safe.entrySet());
        entries.sort((a, b) -> {
            int cmp = Double.compare(b.getValue(), a.getValue());
            if (cmp != 0) return cmp;
            return Integer.compare(a.getKey(), b.getKey());
        });
        List<Integer> ranking = new ArrayList<>(entries.size());
        for (Map.Entry<Integer, Double> e : entries) ranking.add(e.getKey());
        return ranking;
    }

    private List<Integer> rankingPorFrequencia(List<List<Integer>> apostas) {
        Map<Integer, Integer> freq = new HashMap<>();
        for (int i = MIN_NUMERO; i <= MAX_NUMERO; i++) freq.put(i, 0);
        if (apostas != null) {
            for (List<Integer> a : apostas) {
                if (a == null) continue;
                for (Integer n : a) {
                    if (n == null) continue;
                    if (n >= MIN_NUMERO && n <= MAX_NUMERO) freq.put(n, freq.getOrDefault(n, 0) + 1);
                }
            }
        }
        List<Map.Entry<Integer, Integer>> entries = new ArrayList<>(freq.entrySet());
        entries.sort((a, b) -> {
            int cmp = Integer.compare(b.getValue(), a.getValue());
            if (cmp != 0) return cmp;
            return Integer.compare(a.getKey(), b.getKey());
        });
        List<Integer> ranking = new ArrayList<>(entries.size());
        for (Map.Entry<Integer, Integer> e : entries) ranking.add(e.getKey());
        return ranking;
    }

    private boolean isPrimeGetson(int n) {
        // Considera primos no intervalo 1..25
        return n == 2 || n == 3 || n == 5 || n == 7 || n == 11 || n == 13 || n == 17 || n == 19 || n == 23;
    }

    private List<Integer> gerarApostaGetson(Map<Integer, Double> scores, Random rnd) {
        // Ordena números por score da Rede Neural (desc) e desempate pelo número (asc)
        List<Map.Entry<Integer, Double>> entries = new ArrayList<>(scores.entrySet());
        entries.sort((a, b) -> {
            int cmp = Double.compare(b.getValue(), a.getValue());
            if (cmp != 0) return cmp;
            return Integer.compare(a.getKey(), b.getKey());
        });

        List<Integer> evens = new ArrayList<>();
        List<Integer> oddPrimes = new ArrayList<>();
        List<Integer> oddComposites = new ArrayList<>();

        for (Map.Entry<Integer, Double> e : entries) {
            int n = e.getKey();
            if (n % 2 == 0) {
                evens.add(n);
            } else {
                if (isPrimeGetson(n)) oddPrimes.add(n); else oddComposites.add(n);
            }
        }

        // Seleção alvo: 8 pares, 8 ímpares (dos quais pelo menos 5 devem ser primos)
        // Estratégia: garantir exatamente 5 ímpares primos quando possível, e 3 ímpares compostos.
        List<Integer> chosen = new ArrayList<>();

        // Auxiliar para seleção com leve variação entre as duas apostas
        java.util.function.BiFunction<List<Integer>, Integer, List<Integer>> pickTopN = (src, n) -> {
            List<Integer> copy = new ArrayList<>(src);
            int window = Math.min(copy.size(), Math.max(n, n + 2));
            // Embaralha levemente a janela superior para variar escolhas
            List<Integer> windowList = new ArrayList<>(copy.subList(0, window));
            Collections.shuffle(windowList, rnd);
            List<Integer> picked = new ArrayList<>();
            for (int i = 0; i < n && i < windowList.size(); i++) picked.add(windowList.get(i));
            return picked;
        };

        // 1) 5 ímpares primos
        List<Integer> pickedOddPrimes = pickTopN.apply(oddPrimes, 5);
        // 2) 8 pares
        List<Integer> pickedEvens = pickTopN.apply(evens, 8);
        // 3) 3 ímpares compostos
        List<Integer> pickedOddComposites = pickTopN.apply(oddComposites, 3);

        chosen.addAll(pickedOddPrimes);
        chosen.addAll(pickedOddComposites);
        chosen.addAll(pickedEvens);

        // Caso falte algo por algum motivo, completa respeitando as cotas
        int cntOdd = (int) chosen.stream().filter(x -> x % 2 != 0).count();
        int cntEven = chosen.size() - cntOdd;
        int needOdd = 8 - cntOdd;
        int needEven = 16 - chosen.size() - Math.max(0, needOdd);
        if (needOdd > 0) {
            // Primeiro tenta mais ímpares compostos, depois primos
            for (int v : oddComposites) if (needOdd > 0 && !chosen.contains(v)) { chosen.add(v); needOdd--; }
            for (int v : oddPrimes) if (needOdd > 0 && !chosen.contains(v)) { chosen.add(v); needOdd--; }
        }
        if (needEven > 0) {
            for (int v : evens) if (needEven > 0 && !chosen.contains(v)) { chosen.add(v); needEven--; }
        }

        // Ajuste final defensivo para garantir tamanho 16
        // Se sobrar a mais (não deveria), corta mantendo ordenação por score
        if (chosen.size() > 16) {
            // Ordena escolhidos pela pontuação desc e corta
            chosen.sort((a, b) -> {
                int cmp = Double.compare(scores.getOrDefault(b, 0.0), scores.getOrDefault(a, 0.0));
                if (cmp != 0) return cmp;
                return Integer.compare(a, b);
            });
            chosen = new ArrayList<>(chosen.subList(0, 16));
        }

        // Ordena crescente para exibição
        Collections.sort(chosen);
        return chosen;
    }
    private void executarMetaApostaAlternativa(int idx, String tipo) {
        try {
            usarConfiguracao(idx);
            resetarCacheModelo();
            System.out.println("\n--- " + tipo + " ---");
            
            executarMineracao();
            Map<Integer, Double> scores = pontuarNumeros();
            List<List<Integer>> apostas = gerarApostas(scores);
            
            System.out.println("Meta-aposta " + tipo + ":");
            for (int i = 0; i < apostas.size(); i++) {
                List<Integer> a = apostas.get(i);
                System.out.printf("Meta %s %d: ", tipo, i + 1);
                for (int num : a) System.out.printf("%02d ", num);
                System.out.println();
            }
        } catch (Exception ex) {
            System.err.println("Falha na meta-aposta " + tipo + ": " + ex.getMessage());
        }
    }

    // Removido parsing de linha de comando; parâmetros são fixos no construtor

    public static void main(String[] args) {
        LotofacilDMPlus app = new LotofacilDMPlus();
        if (args != null && args.length > 0) {
            String cmd = args[0].toLowerCase();
            try {
                if ("sweep".equals(cmd)) {
                    app.executarSweep();
                    return;
                } else if ("random".equals(cmd) || "randsearch".equals(cmd)) {
                    int iters = 200;
                    // GETSON
                    // int evalLimit = 200;
                    int evalLimit = 1500;
                    if (args.length >= 2) iters = Integer.parseInt(args[1]);
                    if (args.length >= 3) evalLimit = Integer.parseInt(args[2]);
                    app.executarRandomSearch(iters, evalLimit);
                    return;
                } else if ("autoadapt".equals(cmd) || "adapt".equals(cmd)) {
                    int evalLimit = 300; // avaliações máximas por backtest na adaptação
                    if (args.length >= 2) evalLimit = Integer.parseInt(args[1]);
                    app.autoAdaptarDepoisDeAtualizarHistorico(evalLimit);
                    return;
                } else if ("estrategia1".equals(cmd) || "confianca".equals(cmd)) {
                    app.estrategiaApostaPonderadaConfianca();
                    return;
                } else if ("estrategia2".equals(cmd) || "hibrida".equals(cmd) || "elite".equals(cmd)) {
                    app.estrategiaApostaHibridaElite();
                    return;
                } else if ("estrategia3".equals(cmd) || "consensual".equals(cmd)) {
                    app.estrategiaApostaConsensual();
                    return;
                } else if ("todasEstrategias".equals(cmd) || "all".equals(cmd)) {
                    System.out.println("\n" + "=".repeat(80));
                    System.out.println("EXECUTANDO TODAS AS 3 ESTRATÉGIAS");
                    System.out.println("=".repeat(80));
                    app.estrategiaApostaPonderadaConfianca();
                    app.estrategiaApostaHibridaElite();
                    app.estrategiaApostaConsensual();
                    System.out.println("\n" + "=".repeat(80));
                    System.out.println("TODAS AS 3 ESTRATÉGIAS EXECUTADAS");
                    System.out.println("=".repeat(80) + "\n");
                    return;
                } else if ("completo".equals(cmd) || "full".equals(cmd) || "tudo".equals(cmd)) {
                    System.out.println("\n" + "╔" + "=".repeat(78) + "╗");
                    System.out.println("║" + " ".repeat(15) + "EXECUÇÃO COMPLETA: 14 ALGORITMOS + 3 ESTRATÉGIAS" + " ".repeat(17) + "║");
                    System.out.println("╚" + "=".repeat(78) + "╝\n");
                    
                    // Executar 14 algoritmos
                    app.executarTodosAlgoritmosLotofacil();
                    
                    // Executar 3 estratégias
                    System.out.println("\n\n" + "╔" + "=".repeat(78) + "╗");
                    System.out.println("║" + " ".repeat(25) + "AGORA EXECUTANDO AS 3 ESTRATÉGIAS" + " ".repeat(20) + "║");
                    System.out.println("╚" + "=".repeat(78) + "╝\n");
                    
                    app.estrategiaApostaPonderadaConfianca();
                    app.estrategiaApostaHibridaElite();
                    app.estrategiaApostaConsensual();
                    
                    System.out.println("\n" + "╔" + "=".repeat(78) + "╗");
                    System.out.println("║" + " ".repeat(18) + "EXECUÇÃO COMPLETA FINALIZADA COM SUCESSO!" + " ".repeat(18) + "║");
                    System.out.println("║" + " ".repeat(8) + "✓ 14 Algoritmos | ✓ 3 Estratégias | ✓ Total: 17 Métodos de Análise" + " ".repeat(2) + "║");
                    System.out.println("╚" + "=".repeat(78) + "╝\n");
                    return;
                }
            } catch (Exception ex) {
                System.err.println("Erro ao executar comando: " + ex.getMessage());
                ex.printStackTrace();
            }
        }
        new LotofacilDMPlus().exec();
    }

    // Busca aleatória simples sobre hiperparâmetros usando backtest como métrica
    public void executarRandomSearch(int iterations, int evalLimit) {
        System.out.println("=== INICIANDO RANDOM SEARCH ===");
        Random rnd = (this.seed == null) ? new Random(123456789L) : new Random(this.seed);

        class RRes { double metric; String cfg; }
        List<RRes> results = new ArrayList<>();

        for (int it = 1; it <= iterations; it++) {
            double s = 0.02 + rnd.nextDouble() * (0.12 - 0.02);
            double c = 0.40 + rnd.nextDouble() * (0.75 - 0.40);
            int k = 1 + rnd.nextInt(4); // 1..4
            double rAlpha = 0.90 + rnd.nextDouble() * (0.995 - 0.90);
            int[] windows = new int[]{100, 300, 500, 800};
            int w = windows[rnd.nextInt(windows.length)];
            double b = rnd.nextDouble();
            int o = 5 + rnd.nextInt(7); // 5..11

            Config cfg = new Config();
            cfg.TAMANHO_APOSTA = this.TAMANHO_APOSTA;
            cfg.NUMERO_APOSTAS = this.NUMERO_APOSTAS;
            cfg.MIN_NUMERO = this.MIN_NUMERO;
            cfg.MAX_NUMERO = this.MAX_NUMERO;
            cfg.minSupport = s;
            cfg.minConfidence = c;
            cfg.maxK = k;
            cfg.recencyAlpha = rAlpha;
            cfg.evenMin = this.evenMin;
            cfg.evenMax = this.evenMax;
            cfg.minDiversityOverlap = o;
            cfg.assocBoost = b;
            cfg.windowSize = w;
            cfg.historyPath = this.historyPath;
            cfg.historyStartLine = this.historyStartLine;
            cfg.historyEndLine = this.historyEndLine;
            cfg.seed = this.seed;
            cfg.backtest = true;
            cfg.btStart = 1; // for small histories ensure backtest can run
            cfg.btStride = Math.max(1, this.btStride);

            aplicarConfiguracao(cfg);
            resetarCacheModelo();

            double metric = realizarBacktestMetric(evalLimit);

            RRes r = new RRes(); r.metric = metric; r.cfg = String.format("s=%.3f c=%.2f k=%d r=%.3f w=%d b=%.2f o=%d", s, c, k, rAlpha, w, b, o);
            results.add(r);

            if (it % 20 == 0 || it == 1 || it == iterations) {
                System.out.printf("[%d/%d] %s -> %.3f\n", it, iterations, r.cfg, r.metric);
            }
        }

        // ordenar e mostrar top 10
        results.sort((a, b) -> Double.compare(b.metric, a.metric));
        System.out.println("\n=== TOP 10 (RANDOM SEARCH) ===");
        for (int i = 0; i < Math.min(10, results.size()); i++) {
            RRes rr = results.get(i);
            System.out.printf("%2d) %.3f  %s\n", i + 1, rr.metric, rr.cfg);
        }
        System.out.println("=== RANDOM SEARCH CONCLUÍDO ===");
    }

    // ============================ Auto-Adaptação Pós-Atualização ============================
    // Ajusta parâmetros automaticamente com base em backtest nas linhas mais recentes do histórico
    public void autoAdaptarDepoisDeAtualizarHistorico(int evalLimit) {
        System.out.println("=== AUTO-ADAPTAÇÃO PÓS-ATUALIZAÇÃO DE HISTÓRICO ===\n" +
                "Objetivo: ajustar suporte/confiança/janela/recência com base nos concursos mais recentes.");

        // Garantir que conseguimos medir o tamanho do histórico atual
        try {
            resetarCacheModelo();
            prepararTransacoes();
        } catch (Exception ex) {
            System.err.println("Falha ao preparar transações: " + ex.getMessage());
            return;
        }

        int nAll = (transacoesTodas != null) ? transacoesTodas.size() : (transacoes != null ? transacoes.size() : 0);
        if (nAll < 100) {
            System.err.println("Histórico muito curto para auto-adaptação (mín. 100 linhas).");
            return;
        }

        // Janela de backtest focada na parte mais recente do histórico
        int btRecent = Math.max(200, Math.min(800, nAll / 3));
        int btStartLine = Math.max(1, nAll - btRecent);

        // Espaço de busca limitado ao redor dos parâmetros atuais
        double baseS = this.minSupport;
        double baseC = this.minConfidence;
        int baseK = this.maxK;
        double baseR = this.recencyAlpha;
        int baseW = this.windowSize;
        double baseB = this.assocBoost;
        int baseO = this.minDiversityOverlap;

        double[] supports = new double[]{
                clamp(baseS * 0.85, 0.02, 0.20),
                clamp(baseS, 0.02, 0.20),
                clamp(baseS * 1.15, 0.02, 0.20)
        };
        double[] confidences = new double[]{
                clamp(baseC * 0.90, 0.40, 0.90),
                clamp(baseC, 0.40, 0.90),
                clamp(baseC * 1.10, 0.40, 0.90)
        };
        int[] maxKs = new int[]{ Math.max(1, baseK - 1), baseK, Math.min(4, baseK + 1) };
        double[] recencies = new double[]{
                clamp(baseR * 0.98, 0.90, 0.999),
                clamp(baseR, 0.90, 0.999),
                clamp(baseR * 1.01, 0.90, 0.999)
        };
        int[] windows = new int[]{
                Math.max(100, (int) (baseW * 0.8)),
                baseW,
                Math.min(1200, (int) (baseW * 1.2))
        };
        double[] boosts = new double[]{
                clamp(baseB * 0.8, 0.0, 1.5),
                clamp(baseB, 0.0, 1.5),
                clamp(baseB * 1.2, 0.0, 1.5)
        };
        int[] overlaps = new int[]{ Math.max(5, baseO - 1), baseO, Math.min(12, baseO + 1) };

        class ARes { double metric; String cfg; Config c; }
        List<ARes> resultados = new ArrayList<>();

        int total = supports.length * confidences.length * maxKs.length * recencies.length * windows.length * boosts.length * overlaps.length;
        int seen = 0;

        for (double s : supports) for (double c : confidences) for (int k : maxKs) for (double r : recencies) for (int w : windows) for (double b : boosts) for (int o : overlaps) {
            seen++;
            Config cfg = new Config();
            cfg.TAMANHO_APOSTA = this.TAMANHO_APOSTA;
            cfg.NUMERO_APOSTAS = this.NUMERO_APOSTAS;
            cfg.MIN_NUMERO = this.MIN_NUMERO;
            cfg.MAX_NUMERO = this.MAX_NUMERO;
            cfg.minSupport = s;
            cfg.minConfidence = c;
            cfg.maxK = k;
            cfg.recencyAlpha = r;
            cfg.evenMin = this.evenMin;
            cfg.evenMax = this.evenMax;
            cfg.minDiversityOverlap = o;
            cfg.assocBoost = b;
            cfg.windowSize = w;
            cfg.useGPU = this.useGPU;
            cfg.historyPath = this.historyPath;
            cfg.historyStartLine = this.historyStartLine;
            cfg.historyEndLine = this.historyEndLine;
            cfg.seed = this.seed;
            cfg.backtest = true;
            cfg.btStart = btStartLine;
            cfg.btStride = Math.max(1, this.btStride);

            aplicarConfiguracao(cfg);
            resetarCacheModelo();
            double metric = realizarBacktestMetric(evalLimit);

            ARes rres = new ARes();
            rres.metric = metric;
            rres.cfg = String.format("s=%.3f c=%.2f k=%d r=%.3f w=%d b=%.2f o=%d | btStart=%d",
                    s, c, k, r, w, b, o, btStartLine);
            rres.c = cfg;
            resultados.add(rres);

            if (seen % 20 == 0 || seen == 1 || seen == total) {
                System.out.printf("[%d/%d] %s -> %.3f\n", seen, total, rres.cfg, rres.metric);
            }
        }

        // Selecionar melhor
        resultados.sort((a, b) -> Double.compare(b.metric, a.metric));
        if (resultados.isEmpty() || Double.isNaN(resultados.get(0).metric)) {
            System.err.println("Não foi possível encontrar configuração melhor.");
            // restaurar configuração atual
            usarConfiguracao(Math.max(0, configuracaoAtual));
            resetarCacheModelo();
            return;
        }

        System.out.println("\n=== TOP 5 CONFIGURAÇÕES (AUTO-ADAPT) ===");
        for (int i = 0; i < Math.min(5, resultados.size()); i++) {
            ARes rr = resultados.get(i);
            System.out.printf("%2d) %.3f  %s\n", i + 1, rr.metric, rr.cfg);
        }

        // Aplicar melhor configuração e gerar apostas consolidadas
        Config melhor = resultados.get(0).c;
        // manter a posição atual na lista, se existir
        if (configuracaoAtual >= 0 && configuracaoAtual < configuracoes.size()) {
            configuracoes.set(configuracaoAtual, melhor);
        }
        aplicarConfiguracao(melhor);
        resetarCacheModelo();

        System.out.println("\n=== MELHOR CONFIGURAÇÃO APLICADA ===");
        System.out.println(resultados.get(0).cfg);
        System.out.println("Gerando apostas com parâmetros ajustados...");
        executarTodosAlgoritmosLotofacil();
        System.out.println("=== AUTO-ADAPTAÇÃO CONCLUÍDA ===");
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    // ============================ Tipos auxiliares ============================

    private static class NumeroPeso { final int numero; final double peso; NumeroPeso(int n, double p) { numero = n; peso = p; } }

    private static class AssociationRule {
        final int antecedenteMask;
        final int consequenteMask;
        final double support;
        final double confidence;

        AssociationRule(int aMask, int cMask, double s, double conf) {
            this.antecedenteMask = aMask; this.consequenteMask = cMask; this.support = s; this.confidence = conf;
        }

        Set<Integer> getAntecedenteNumbers() { return setFromMask(antecedenteMask); }

        Set<Integer> getConsequenteNumbers() { return setFromMask(consequenteMask); }

        public String toString() {
            return String.format("%s => %s (sup=%.3f, conf=%.3f)", setFromMask(antecedenteMask), setFromMask(consequenteMask), support, confidence);
        }
    }

    // Executa o pipeline completo para uma configuração específica
    private void executarParaConfiguracao(int idx) {
        try {
            usarConfiguracao(idx);
            resetarCacheModelo();
            System.out.println("\n==============================");
            System.out.println("Executando configuração " + idx);
            System.out.println("==============================");

            if (backtest) {
                realizarBacktest();
                return;
            }

            executarMineracao();
            Map<Integer, Double> scores = pontuarNumeros();
            List<List<Integer>> apostas = gerarApostas(scores);

            System.out.println("=== GERANDO APOSTAS (DM Plus) ===");
            for (int i = 0; i < apostas.size(); i++) {
                List<Integer> a = apostas.get(i);
                System.out.printf("Aposta %2d: ", i + 1);
                for (int num : a) System.out.printf("%02d ", num);
                System.out.println();
            }

            exibirResumo(scores, 16);
        } catch (Exception ex) {
            System.err.println("Falha na configuração " + idx + ": " + ex.getMessage());
        }
    }

    // Limpa caches/dados derivados para respeitar a configuração selecionada
    private void resetarCacheModelo() {
        this.transacoes = null;
        this.transacoesTodas = null;
        this.transacoesMask = null;
        this.transacoesTodasMask = null;
        this.pesosTransacoes = null;
        this.pesoTotal = 0.0;
        this.supportMapMask.clear();
        this.regras.clear();
        // combosExistentes permanece para evitar duplicações entre configurações na mesma execução
    }

    // Representa um conjunto de parâmetros fixos
    private static class Config {
        int TAMANHO_APOSTA;
        int MIN_NUMERO;
        int MAX_NUMERO;
        int NUMERO_APOSTAS;
        double minSupport;
        double minConfidence;
        int maxK;
        double recencyAlpha;
        int evenMin;
        int evenMax;
        int minDiversityOverlap;
        double assocBoost;
        int windowSize;
        boolean useGPU;
        String historyPath;
        int historyStartLine;
        int historyEndLine;
        Long seed;
        boolean backtest;
        int btStart;
        int btStride;
        List<Integer> targetCombo;
        boolean ensureTargetCombo;
    }

    // Adiciona uma nova configuração (máximo 6)
    public void addConfiguracao(Config c) {
        if (configuracoes.size() >= 6) throw new IllegalStateException("Limite de 6 configurações atingido");
        configuracoes.add(c);
    }

    // Seleciona e aplica configuração existente
    public void usarConfiguracao(int idx) {
        if (idx < 0 || idx >= configuracoes.size()) throw new IllegalArgumentException("Índice de configuração inválido: " + idx);
        aplicarConfiguracao(configuracoes.get(idx));
        this.configuracaoAtual = idx;
    }

    // Copia valores da configuração para os campos ativos
    private void aplicarConfiguracao(Config c) {
        this.TAMANHO_APOSTA = c.TAMANHO_APOSTA;
        this.MIN_NUMERO = c.MIN_NUMERO;
        this.MAX_NUMERO = c.MAX_NUMERO;
        this.NUMERO_APOSTAS = c.NUMERO_APOSTAS;
        this.minSupport = c.minSupport;
        this.minConfidence = c.minConfidence;
        this.maxK = c.maxK;
        this.recencyAlpha = c.recencyAlpha;
        this.evenMin = c.evenMin;
        this.evenMax = c.evenMax;
        this.minDiversityOverlap = c.minDiversityOverlap;
        this.assocBoost = c.assocBoost;
        this.windowSize = c.windowSize;
        this.useGPU = c.useGPU;
        this.historyPath = c.historyPath;
        this.historyStartLine = c.historyStartLine;
        this.historyEndLine = c.historyEndLine;
        this.seed = c.seed;
        this.backtest = c.backtest;
        this.btStart = c.btStart;
        this.btStride = c.btStride;
        this.targetCombo = (c.targetCombo != null) ? new ArrayList<>(c.targetCombo) : null;
        this.ensureTargetCombo = c.ensureTargetCombo;
    }

        // HISTORICO_SORTEIOS removed — external  is required now

    // ============================ Backtest ============================
    private void realizarBacktest() {
        // Carrega todas as transações do arquivo/range solicitado
        transacoes = null; transacoesTodas = null; pesosTransacoes = null; pesoTotal = 0.0;
        prepararTransacoes(); // popula transacoesTodas com o range; transacoes = janela (iremos ignorar janela aqui)
        List<Set<Integer>> all = transacoesTodas != null ? transacoesTodas : transacoes;
        if (all == null || all.size() < btStart + 1) {
            System.err.println("Backtest: histórico insuficiente para iniciar em linha " + btStart);
            return;
        }

        int startIdx = Math.max(1, btStart) - 1; // 0-based índice da linha de previsão (usa prefixo até startIdx-1)
        int n = all.size();
        int avaliacoes = 0;
        int[] histAcertos = new int[TAMANHO_APOSTA + 1];
        double somaMelhor = 0.0;

        Random baseRnd = (seed == null) ? new Random(123456789L) : new Random(seed);

        long t0 = System.currentTimeMillis();
        for (int t = startIdx; t < n; t += Math.max(1, btStride)) {
            List<Set<Integer>> prefixo = new ArrayList<>(all.subList(0, t));
            if (windowSize > 0 && prefixo.size() > windowSize) {
                prefixo = new ArrayList<>(prefixo.subList(prefixo.size() - windowSize, prefixo.size()));
            }
            if (prefixo.isEmpty()) continue;

            // Construir modelo neste prefixo (sem vazamento de futuro)
            construirModelo(prefixo);
            Map<Integer, Double> scores = pontuarNumeros();
            // gerar apostas usando semente determinística incremental para reprodutibilidade
            Long s = baseRnd.nextLong();
            Long oldSeed = this.seed; this.seed = s;
            List<List<Integer>> apostas = gerarApostas(scores);
            this.seed = oldSeed;

            // Avaliar contra a linha real t
            Set<Integer> real = all.get(t);
            int melhor = 0;
            for (List<Integer> a : apostas) {
                int acertos = 0;
                for (Integer x : a) if (real.contains(x)) acertos++;
                if (acertos > melhor) melhor = acertos;
            }
            if (melhor >= 0 && melhor <= TAMANHO_APOSTA) histAcertos[melhor]++;
            somaMelhor += melhor;
            avaliacoes++;

            if (avaliacoes % 500 == 0) {
                System.out.printf("Backtest: %d/%d avaliados (linha %d)\n", avaliacoes, n - startIdx, t + 1);
            }
        }
        long t1 = System.currentTimeMillis();

        System.out.println("=== BACKTEST (DM Plus) ===");
        System.out.printf("Linhas: %d..%d | Avaliações: %d | Tempo: %.2fs\n",
                historyStartLine, Math.min(historyEndLine, historyStartLine - 1 + (all != null ? all.size() : 0)),
                avaliacoes, (t1 - t0) / 1000.0);
        System.out.printf("Média do melhor acerto (entre %d apostas): %.3f\n", NUMERO_APOSTAS, (avaliacoes > 0 ? somaMelhor / avaliacoes : 0.0));
        System.out.println("Distribuição do melhor acerto:");
        for (int k = 0; k <= TAMANHO_APOSTA; k++) {
            if (histAcertos[k] > 0) System.out.printf("%2d acertos: %d\n", k, histAcertos[k]);
        }
    }

    // Variante do backtest que retorna a média do melhor acerto (útil para varreduras)
    private double realizarBacktestMetric(int maxEvaluations) {
        // Carrega todas as transações do arquivo/range solicitado
        transacoes = null; transacoesTodas = null; pesosTransacoes = null; pesoTotal = 0.0;
        prepararTransacoes(); // popula transacoesTodas com o range; transacoes = janela (iremos ignorar janela aqui)
        List<Set<Integer>> all = transacoesTodas != null ? transacoesTodas : transacoes;
        if (all == null || all.size() < btStart + 1) {
            return Double.NaN;
        }

        int startIdx = Math.max(1, btStart) - 1; // 0-based índice da linha de previsão (usa prefixo até startIdx-1)
        int n = all.size();
        int avaliacoes = 0;
        double somaMelhor = 0.0;

        Random baseRnd = (seed == null) ? new Random(123456789L) : new Random(seed);

        for (int t = startIdx; t < n; t += Math.max(1, btStride)) {
            List<Set<Integer>> prefixo = new ArrayList<>(all.subList(0, t));
            if (windowSize > 0 && prefixo.size() > windowSize) {
                prefixo = new ArrayList<>(prefixo.subList(prefixo.size() - windowSize, prefixo.size()));
            }
            if (prefixo.isEmpty()) continue;

            construirModelo(prefixo);
            Map<Integer, Double> scores = pontuarNumeros();
            Long s = baseRnd.nextLong();
            Long oldSeed = this.seed; this.seed = s;
            List<List<Integer>> apostas = gerarApostas(scores);
            this.seed = oldSeed;

            Set<Integer> real = all.get(t);
            int melhor = 0;
            for (List<Integer> a : apostas) {
                int acertos = 0;
                for (Integer x : a) if (real.contains(x)) acertos++;
                if (acertos > melhor) melhor = acertos;
            }
            somaMelhor += melhor;
            avaliacoes++;

            if (maxEvaluations > 0 && avaliacoes >= maxEvaluations) break;
        }

        return (avaliacoes > 0) ? (somaMelhor / avaliacoes) : Double.NaN;
    }

    // Executa uma varredura (sweep) simples sobre grades de hiperparâmetros e imprime os melhores resultados
    public void executarSweep() {
        System.out.println("=== INICIANDO SWEEP DE PARÂMETROS ===");

        double[] supports = new double[]{0.06, 0.08, 0.10};
        double[] confidences = new double[]{0.50, 0.60};
        int[] maxKs = new int[]{2, 3};
        double[] recencies = new double[]{0.95, 0.97};
        int[] windows = new int[]{300, 500};
        double[] boosts = new double[]{0.0, 0.5};
        int[] overlaps = new int[]{7, 9};

        class Result { double metric; String cfg; }
        List<Result> results = new ArrayList<>();

        int total = supports.length * confidences.length * maxKs.length * recencies.length * windows.length * boosts.length * overlaps.length;
        int seen = 0;
        int evalLimit = 200; // limitar avaliações por backtest para manter sweep rápido

        for (double s : supports) for (double c : confidences) for (int k : maxKs) for (double r : recencies) for (int w : windows) for (double b : boosts) for (int o : overlaps) {
            seen++;
            Config cfg = new Config();
            cfg.TAMANHO_APOSTA = this.TAMANHO_APOSTA;
            cfg.NUMERO_APOSTAS = this.NUMERO_APOSTAS;
            cfg.MIN_NUMERO = this.MIN_NUMERO;
            cfg.MAX_NUMERO = this.MAX_NUMERO;
            cfg.minSupport = s;
            cfg.minConfidence = c;
            cfg.maxK = k;
            cfg.recencyAlpha = r;
            cfg.evenMin = this.evenMin;
            cfg.evenMax = this.evenMax;
            cfg.minDiversityOverlap = o;
            cfg.assocBoost = b;
            cfg.windowSize = w;
            cfg.historyPath = this.historyPath;
            cfg.historyStartLine = this.historyStartLine;
            cfg.historyEndLine = this.historyEndLine;
            cfg.seed = this.seed;
            cfg.backtest = true;
            cfg.btStart = 1; // use start=1 so sweep works on small histories
            cfg.btStride = Math.max(1, this.btStride);

            aplicarConfiguracao(cfg);
            resetarCacheModelo();

            double metric = realizarBacktestMetric(evalLimit);

            Result res = new Result();
            res.metric = metric;
            res.cfg = String.format("s=%.3f c=%.2f k=%d r=%.3f w=%d b=%.2f o=%d", s, c, k, r, w, b, o);
            results.add(res);

            System.out.printf("[%d/%d] %s -> %.3f\n", seen, total, res.cfg, res.metric);
        }

        // ordenar e mostrar top 10
        results.sort((a, b) -> Double.compare(b.metric, a.metric));
        System.out.println("\n=== TOP 10 CONFIGURAÇÕES (SWEEP) ===");
        for (int i = 0; i < Math.min(10, results.size()); i++) {
            Result r = results.get(i);
            System.out.printf("%2d) %.3f  %s\n", i + 1, r.metric, r.cfg);
        }
        System.out.println("=== SWEEP CONCLUÍDO ===");
    }

    private void construirModelo(List<Set<Integer>> dataset) {
        // Recalcula pesos e supportMap/regras usando dataset fornecido
        this.transacoes = new ArrayList<>(dataset);
        // construir máscaras correspondentes para acesso rápido
        this.transacoesMask = new ArrayList<>(transacoes.size());
        for (Set<Integer> s : this.transacoes) this.transacoesMask.add(maskFromSet(s));
        int n = transacoes.size();
        this.pesosTransacoes = new ArrayList<>(n);
        this.pesoTotal = 0.0;
        for (int i = 0; i < n; i++) {
            int age = (n - 1) - i;
            double w = Math.pow(recencyAlpha, age);
            pesosTransacoes.add(w);
            pesoTotal += w;
        }

        supportMapMask.clear();
        regras.clear();

        // k=1
        Map<Set<Integer>, Double> contagem1 = new HashMap<>();
        for (int t = 0; t < n; t++) {
            Set<Integer> trans = transacoes.get(t);
            double w = pesosTransacoes.get(t);
            for (Integer item : trans) {
                Set<Integer> s = Collections.singleton(item);
                contagem1.put(s, contagem1.getOrDefault(s, 0.0) + w);
            }
        }
        List<Set<Integer>> frequentesK = new ArrayList<>();
        for (Map.Entry<Set<Integer>, Double> e : contagem1.entrySet()) {
            double sup = e.getValue() / pesoTotal;
            if (sup >= minSupport) {
                int mask = maskFromSet(e.getKey());
                supportMapMask.put(mask, sup);
                frequentesK.add(e.getKey());
            }
        }
        // k=2..maxK
        List<Set<Integer>> prevFrequent = new ArrayList<>(frequentesK);
        for (int k = 2; k <= Math.max(2, maxK); k++) {
            Set<Set<Integer>> candidates = gerarCandidatos(prevFrequent, k);
            if (candidates.isEmpty()) break;
            Map<Set<Integer>, Double> contagem = new HashMap<>();
            Map<Set<Integer>, Integer> candMask = new HashMap<>();
            for (Set<Integer> c : candidates) { contagem.put(c, 0.0); candMask.put(c, maskFromSet(c)); }
            for (int t = 0; t < n; t++) {
                int transM = (transacoesMask != null && transacoesMask.size() > t) ? transacoesMask.get(t) : maskFromSet(transacoes.get(t));
                double w = pesosTransacoes.get(t);
                for (Set<Integer> c : candidates) if ((transM & candMask.get(c)) == candMask.get(c)) contagem.put(c, contagem.get(c) + w);
            }
            List<Set<Integer>> frequentK = new ArrayList<>();
            for (Map.Entry<Set<Integer>, Double> e : contagem.entrySet()) {
                double sup = e.getValue() / pesoTotal;
                if (sup >= minSupport) {
                    int mask = maskFromSet(e.getKey());
                    supportMapMask.put(mask, sup);
                    frequentK.add(e.getKey());
                }
            }
            if (frequentK.isEmpty()) break;
            prevFrequent = frequentK;
        }
        // regras
        for (Map.Entry<Integer, Double> e : supportMapMask.entrySet()) {
            int itemsetMask = e.getKey();
            Set<Integer> itemset = setFromMask(itemsetMask);
            if (itemset.size() < 2) continue;
            List<Set<Integer>> subsets = gerarSubconjuntosProprios(itemset);
            for (Set<Integer> antecedente : subsets) {
                Set<Integer> consequente = new HashSet<>(itemset);
                consequente.removeAll(antecedente);
                if (consequente.isEmpty()) continue;
                int maskAnte = maskFromSet(antecedente);
                Double supItemset = supportMapMask.get(itemsetMask);
                Double supAnte = supportMapMask.get(maskAnte);
                if (supItemset == null || supAnte == null || supAnte == 0) continue;
                double conf = supItemset / supAnte;
                if (conf >= minConfidence) regras.add(new AssociationRule(maskAnte, maskFromSet(consequente), supItemset, conf));
            }
        }
    }
    
    // ============================ ESTRATÉGIAS DE APOSTAS AVANÇADAS ============================
    
    /**
     * ESTRATÉGIA 1: APOSTA PONDERADA POR CONFIANÇA
     * Utiliza regras de associação com peso baseado em confiança
     * Números são selecionados se aparecem em regras de alta confiança
     */
    public List<List<Integer>> estrategiaApostaPonderadaConfianca() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("ESTRATÉGIA 1: APOSTA PONDERADA POR CONFIANÇA");
        System.out.println("=".repeat(80));
        
        executarMineracao();
        
        if (regras.isEmpty()) {
            System.out.println("Nenhuma regra encontrada. Usando fallback...");
            List<Integer> fallback = gerarApostaFallback(new Random(System.nanoTime()));
            return Arrays.asList(fallback);
        }
        
        // Calcular peso de confiança para cada número
        Map<Integer, Double> pesoConfianca = new HashMap<>();
        Map<Integer, Integer> frequenciaRegras = new HashMap<>();
        
        for (AssociationRule regra : regras) {
            // Números no consequente recebem peso baseado na confiança
            Set<Integer> consequentes = regra.getConsequenteNumbers();
            double pesoRegra = regra.confidence * regra.support; // combinação de confiança e suporte
            
            for (Integer num : consequentes) {
                pesoConfianca.put(num, pesoConfianca.getOrDefault(num, 0.0) + pesoRegra);
                frequenciaRegras.put(num, frequenciaRegras.getOrDefault(num, 0) + 1);
            }
            
            // Números no antecedente também recebem peso (pré-requisitos)
            Set<Integer> antecedentes = regra.getAntecedenteNumbers();
            double pesoAntecedente = pesoRegra * 0.7; // 70% do peso da regra
            
            for (Integer num : antecedentes) {
                pesoConfianca.put(num, pesoConfianca.getOrDefault(num, 0.0) + pesoAntecedente);
                frequenciaRegras.put(num, frequenciaRegras.getOrDefault(num, 0) + 1);
            }
        }
        
        // Normalizar pesos pela frequência
        Map<Integer, Double> pesoNormalizado = new HashMap<>();
        for (Map.Entry<Integer, Double> e : pesoConfianca.entrySet()) {
            int freq = frequenciaRegras.getOrDefault(e.getKey(), 1);
            double pesoNorm = e.getValue() / Math.sqrt(freq); // dividir por sqrt da frequência
            pesoNormalizado.put(e.getKey(), pesoNorm);
        }
        
        // Gerar apostas selecionando números com maior peso de confiança
        List<List<Integer>> apostas = new ArrayList<>();
        Random rng = new Random(System.nanoTime());
        
        for (int i = 0; i < NUMERO_APOSTAS; i++) {
            List<Integer> aposta = new ArrayList<>();
            Set<Integer> selecionados = new HashSet<>();
            
            // Ordenar números por peso de confiança
            List<Map.Entry<Integer, Double>> ranking = new ArrayList<>(pesoNormalizado.entrySet());
            ranking.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
            
            // Adicionar números com maior peso de confiança
            for (Map.Entry<Integer, Double> entry : ranking) {
                if (aposta.size() >= TAMANHO_APOSTA) break;
                if (!selecionados.contains(entry.getKey())) {
                    aposta.add(entry.getKey());
                    selecionados.add(entry.getKey());
                }
            }
            
            // Completar com números aleatórios se necessário
            while (aposta.size() < TAMANHO_APOSTA) {
                int num = MIN_NUMERO + rng.nextInt(MAX_NUMERO - MIN_NUMERO + 1);
                if (!selecionados.contains(num)) {
                    aposta.add(num);
                    selecionados.add(num);
                }
            }
            
            Collections.sort(aposta);
            apostas.add(aposta);
        }
        
        // Exibir resultados
        for (int i = 0; i < apostas.size(); i++) {
            List<Integer> aposta = apostas.get(i);
            System.out.printf("Aposta %d: ", i + 1);
            for (int num : aposta) System.out.printf("%02d ", num);
            System.out.println();
        }
        
        return apostas;
    }
    
    /**
     * ESTRATÉGIA 2: APOSTA HÍBRIDA (ELITE)
     * Combina múltiplas fontes de scoring:
     * - Frequência de aparição
     * - Suporte em regras de associação
     * - Coocorrências
     * - Padrões cíclicos
     */
    public List<List<Integer>> estrategiaApostaHibridaElite() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("ESTRATÉGIA 2: APOSTA HÍBRIDA (ELITE)");
        System.out.println("=".repeat(80));
        
        executarMineracao();
        Map<Integer, Double> scores = pontuarNumeros();
        analisarPadroesCiclicos();
        
        // Combinar múltiplas fontes de scoring
        Map<Integer, Double> scoreHibrido = new HashMap<>();
        
        for (int num = MIN_NUMERO; num <= MAX_NUMERO; num++) {
            double scoreBase = scores.getOrDefault(num, 0.0);
            
            // Componente 1: Frequência (40%)
            double freq = numeroFrequenciaMedia.getOrDefault(num, 0.0);
            
            // Componente 2: Suporte em itemsets (30%)
            int numMask = (1 << (num - 1));
            double suporteNum = supportMapMask.entrySet().stream()
                .filter(e -> (e.getKey() & numMask) != 0)
                .mapToDouble(Map.Entry::getValue)
                .average()
                .orElse(0.0);
            
            // Componente 3: Coocorrências (20%)
            double coocorrencia = 0.0;
            int countCoocorr = 0;
            for (int outro = MIN_NUMERO; outro <= MAX_NUMERO; outro++) {
                if (outro == num) continue;
                int pairMask = numMask | (1 << (outro - 1));
                Double suportePar = supportMapMask.get(pairMask);
                if (suportePar != null) {
                    coocorrencia += suportePar;
                    countCoocorr++;
                }
            }
            if (countCoocorr > 0) coocorrencia /= countCoocorr;
            
            // Componente 4: Ciclos (10%)
            double cicloScore = 0.0;
            int ciclo = numeroCiclo.getOrDefault(num, Integer.MAX_VALUE);
            if (ciclo != Integer.MAX_VALUE && ciclo > 0) {
                cicloScore = 1.0 / (1.0 + ciclo / 100.0); // números com ciclos menores são melhores
            }
            
            // Combinar componentes com pesos
            double scoreTotal = (freq * 0.40) + (suporteNum * 0.30) + (coocorrencia * 0.20) + (cicloScore * 0.10);
            scoreHibrido.put(num, scoreTotal);
        }
        
        // Gerar apostas usando score híbrido
        List<List<Integer>> apostas = new ArrayList<>();
        Random rng = new Random(System.nanoTime());
        
        for (int i = 0; i < NUMERO_APOSTAS; i++) {
            List<Integer> aposta = new ArrayList<>();
            Set<Integer> selecionados = new HashSet<>();
            
            // Ordenar por score híbrido
            List<Map.Entry<Integer, Double>> ranking = new ArrayList<>(scoreHibrido.entrySet());
            ranking.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
            
            // Selecionar top números com variação (elite + exploradores)
            int topElite = (int) Math.ceil(TAMANHO_APOSTA * 0.7); // 70% são os melhores
            int exploradores = TAMANHO_APOSTA - topElite;
            
            // Elite (top 70%)
            for (int j = 0; j < Math.min(topElite, ranking.size()); j++) {
                aposta.add(ranking.get(j).getKey());
                selecionados.add(ranking.get(j).getKey());
            }
            
            // Exploradores (30% aleatórios dos restantes)
            List<Integer> restantes = new ArrayList<>();
            for (Map.Entry<Integer, Double> entry : ranking) {
                if (!selecionados.contains(entry.getKey())) {
                    restantes.add(entry.getKey());
                }
            }
            Collections.shuffle(restantes, rng);
            for (int j = 0; j < exploradores && j < restantes.size(); j++) {
                aposta.add(restantes.get(j));
            }
            
            // Completar se necessário
            while (aposta.size() < TAMANHO_APOSTA) {
                int num = MIN_NUMERO + rng.nextInt(MAX_NUMERO - MIN_NUMERO + 1);
                if (!selecionados.contains(num)) {
                    aposta.add(num);
                    selecionados.add(num);
                }
            }
            
            Collections.sort(aposta);
            apostas.add(aposta);
        }
        
        // Exibir resultados
        for (int i = 0; i < apostas.size(); i++) {
            List<Integer> aposta = apostas.get(i);
            System.out.printf("Aposta %d: ", i + 1);
            for (int num : aposta) System.out.printf("%02d ", num);
            System.out.println();
        }
        
        return apostas;
    }
    
    /**
     * ESTRATÉGIA 3: APOSTA CONSENSUAL
     * Seleciona números que aparecem consistentemente em vários algoritmos
     * Realiza múltiplas gerações e encontra o consenso (números mais frequentes)
     */
    public List<List<Integer>> estrategiaApostaConsensual() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("ESTRATÉGIA 3: APOSTA CONSENSUAL");
        System.out.println("=".repeat(80));
        
        executarMineracao();
        Map<Integer, Double> scores = pontuarNumeros();
        
        // Gerar múltiplas apostas e contar frequência de números
        Map<Integer, Integer> frequenciaNumerosGerados = new HashMap<>();
        int numGereacoes = 20; // gerar 20 apostas para consenso
        Random rng = new Random(System.nanoTime());
        
        System.out.println("Gerando " + numGereacoes + " apostas para análise de consenso...");
        
        for (int i = 0; i < numGereacoes; i++) {
            List<Integer> aposta = gerarApostaUnica(scores, rng);
            for (Integer num : aposta) {
                frequenciaNumerosGerados.put(num, frequenciaNumerosGerados.getOrDefault(num, 0) + 1);
            }
        }
        
        // Calcular score de consenso (quanto mais aparições, melhor)
        Map<Integer, Double> scoreConsensual = new HashMap<>();
        for (Map.Entry<Integer, Integer> e : frequenciaNumerosGerados.entrySet()) {
            double consenso = e.getValue() / (double) numGereacoes; // 0..1
            scoreConsensual.put(e.getKey(), consenso);
        }
        
        // Gerar apostas usando consenso
        List<List<Integer>> apostas = new ArrayList<>();
        
        for (int i = 0; i < NUMERO_APOSTAS; i++) {
            List<Integer> aposta = new ArrayList<>();
            Set<Integer> selecionados = new HashSet<>();
            
            // Ordenar por consenso
            List<Map.Entry<Integer, Double>> ranking = new ArrayList<>(scoreConsensual.entrySet());
            ranking.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
            
            // Threshold: selecionar números que apareceram em pelo menos 50% das gerações
            double threshold = 0.5;
            for (Map.Entry<Integer, Double> entry : ranking) {
                if (entry.getValue() >= threshold && aposta.size() < TAMANHO_APOSTA) {
                    aposta.add(entry.getKey());
                    selecionados.add(entry.getKey());
                }
            }
            
            // Se não houver números com consenso > 50%, adicionar os com maior consenso
            if (aposta.size() < TAMANHO_APOSTA) {
                for (Map.Entry<Integer, Double> entry : ranking) {
                    if (!selecionados.contains(entry.getKey()) && aposta.size() < TAMANHO_APOSTA) {
                        aposta.add(entry.getKey());
                        selecionados.add(entry.getKey());
                    }
                }
            }
            
            // Completar com aleatórios se necessário
            while (aposta.size() < TAMANHO_APOSTA) {
                int num = MIN_NUMERO + rng.nextInt(MAX_NUMERO - MIN_NUMERO + 1);
                if (!selecionados.contains(num)) {
                    aposta.add(num);
                    selecionados.add(num);
                }
            }
            
            Collections.sort(aposta);
            apostas.add(aposta);
        }
        
        // Exibir estatísticas de consenso
        System.out.println("\nEstatísticas de Consenso:");
        scoreConsensual.entrySet().stream()
            .sorted((a, b) -> Double.compare(b.getValue(), a.getValue()))
            .limit(10)
            .forEach(e -> System.out.printf("Número %02d: %.1f%% de consenso\n", e.getKey(), e.getValue() * 100));
        
        // Exibir apostas
        for (int i = 0; i < apostas.size(); i++) {
            List<Integer> aposta = apostas.get(i);
            System.out.printf("\nAposta %d: ", i + 1);
            for (int num : aposta) System.out.printf("%02d ", num);
            System.out.println();
        }

        // Gerar também TOP 3 apostas consensuais com 16 dezenas (mantendo as de 15 acima)
        int tamanho16 = Math.min(MAX_NUMERO - MIN_NUMERO + 1, TAMANHO_APOSTA + 1);
        int qtd16 = 3;
        if (tamanho16 > TAMANHO_APOSTA) {
            System.out.println("\n" + "-".repeat(80));
            System.out.println("TOP " + qtd16 + " APOSTA CONSENSUAL (" + tamanho16 + " dezenas)");
            System.out.println("-".repeat(80));

            // Ranking por consenso (reutiliza o scoreConsensual já calculado)
            List<Map.Entry<Integer, Double>> ranking = new ArrayList<>(scoreConsensual.entrySet());
            ranking.sort((a, b) -> {
                int cmp = Double.compare(b.getValue(), a.getValue());
                if (cmp != 0) return cmp;
                return Integer.compare(a.getKey(), b.getKey());
            });

            for (int i = 0; i < qtd16; i++) {
                List<Integer> aposta16 = new ArrayList<>();
                Set<Integer> selecionados16 = new HashSet<>();

                // Threshold: selecionar números que apareceram em pelo menos 50% das gerações
                double threshold = 0.5;
                for (Map.Entry<Integer, Double> entry : ranking) {
                    if (entry.getValue() >= threshold && aposta16.size() < tamanho16) {
                        aposta16.add(entry.getKey());
                        selecionados16.add(entry.getKey());
                    }
                }

                // Completar com os de maior consenso (se necessário)
                if (aposta16.size() < tamanho16) {
                    for (Map.Entry<Integer, Double> entry : ranking) {
                        if (!selecionados16.contains(entry.getKey()) && aposta16.size() < tamanho16) {
                            aposta16.add(entry.getKey());
                            selecionados16.add(entry.getKey());
                        }
                    }
                }

                // Completar com aleatórios se necessário
                while (aposta16.size() < tamanho16) {
                    int num = MIN_NUMERO + rng.nextInt(MAX_NUMERO - MIN_NUMERO + 1);
                    if (!selecionados16.contains(num)) {
                        aposta16.add(num);
                        selecionados16.add(num);
                    }
                }

                Collections.sort(aposta16);
                System.out.printf("\nAposta %d (16 dezenas): ", i + 1);
                for (int num : aposta16) System.out.printf("%02d ", num);
                System.out.println();
            }
        }
        
        return apostas;
    }
    
    // ============================ Métodos auxiliares ============================
}
