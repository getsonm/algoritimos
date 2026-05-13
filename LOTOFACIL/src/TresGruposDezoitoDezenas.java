/**
 * Autor: getson@engineer.com
 */
import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class TresGruposDezoitoDezenas {

    private static final int LAST_N_LINES = 6;
    private static final int MAX_CONSECUTIVE_HOURS = 3; // maximo de horas seguidas permitidas
    private static final Path OUTPUT_PATH = Paths.get("arquivos", "TGDDSaida.txt");

    public static void main(String[] args) {
        Path csvPath = resolveCsvPath(args);

        try {
            List<String> lastLines = readLastNonEmptyLines(csvPath, LAST_N_LINES);
            if (lastLines.isEmpty()) {
                System.out.println("Arquivo vazio (ou sem linhas válidas): " + csvPath.toAbsolutePath());
                return;
            }

            // No arquivo: grave SOMENTE as 6 ultimas linhas encontradas no CSV
            Files.write(OUTPUT_PATH, lastLines, StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);

            // Na tela: calcula e imprime a frequencia das dezenas nas ultimas 6 linhas
            List<String> drawLines = extractLastDrawLines(lastLines, LAST_N_LINES);
            int[] freq = countFrequencies(drawLines);
            List<String> freqReport = formatFrequencyReport(freq, drawLines.size());

            for (String line : freqReport) {
                System.out.println(line);
            }

            List<String> percentileReport = formatPercentileReport(freq, drawLines.size());
            for (String line : percentileReport) {
                System.out.println(line);
            }

            List<String> suggestion = formatSuggestionReport(freq, drawLines.size());
            for (String line : suggestion) {
                System.out.println(line);
            }
        } catch (IOException e) {
            System.err.println("Falha ao ler o arquivo: " + csvPath.toAbsolutePath());
            System.err.println("Erro: " + e.getMessage());
            System.exit(1);
        }
    }

    private static List<String> extractLastDrawLines(List<String> lines, int n) {
        if (lines == null || lines.isEmpty() || n <= 0) {
            return List.of();
        }

        List<String> drawLines = new ArrayList<>(n);
        for (int i = lines.size() - 1; i >= 0 && drawLines.size() < n; i--) {
            String t = lines.get(i) == null ? "" : lines.get(i).trim();
            if (t.isEmpty()) {
                continue;
            }
            // Linha de dezenas: somente numeros e virgulas
            if (t.matches("\\d+(\\s*,\\s*\\d+)+")) {
                drawLines.add(0, t);
            }
        }
        return drawLines;
    }

    private static int[] countFrequencies(List<String> drawLines) {
        int[] freq = new int[26]; // 1..25
        if (drawLines == null) {
            return freq;
        }

        for (String line : drawLines) {
            if (line == null) {
                continue;
            }
            String[] parts = line.split(",");
            for (String p : parts) {
                String s = p.trim();
                if (s.isEmpty()) {
                    continue;
                }
                int v;
                try {
                    v = Integer.parseInt(s);
                } catch (NumberFormatException ex) {
                    continue;
                }
                if (v >= 1 && v <= 25) {
                    freq[v]++;
                }
            }
        }

        return freq;
    }

    private static List<String> formatFrequencyReport(int[] freq, int totalLines) {
        if (freq == null || freq.length < 26) {
            return List.of("Relatorio de frequencia indisponivel.");
        }

        Map<Integer, List<Integer>> byCount = new HashMap<>();
        int max = 0;
        for (int d = 1; d <= 25; d++) {
            int c = freq[d];
            if (c <= 0) {
                continue;
            }
            max = Math.max(max, c);
            byCount.computeIfAbsent(c, k -> new ArrayList<>()).add(d);
        }

        List<String> out = new ArrayList<>();
        out.add("Frequencia das dezenas nas ultimas " + totalLines + " linhas do arquivo:");

        for (int c = max; c >= 1; c--) {
            List<Integer> dezenas = byCount.get(c);
            if (dezenas == null || dezenas.isEmpty()) {
                continue;
            }
            dezenas.sort(Integer::compareTo);
            out.add("qtd=" + c + ": " + joinInts(dezenas));
        }

        return out;
    }

    private static List<String> formatPercentileReport(int[] freq, int totalLines) {
        if (freq == null || freq.length < 26) {
            return List.of("Relatorio de percentil indisponivel.");
        }

        List<String> out = new ArrayList<>();
        out.add("Percentil por dezena nas ultimas " + totalLines + " linhas do arquivo:");

        record DezenaPct(int dezena, int count, double pct) {}

        List<DezenaPct> items = new ArrayList<>(25);
        for (int d = 1; d <= 25; d++) {
            double pct = (totalLines <= 0) ? 0.0 : (freq[d] * 100.0) / totalLines;
            items.add(new DezenaPct(d, freq[d], pct));
        }

        items.sort((a, b) -> {
            int byCount = Integer.compare(b.count(), a.count());
            if (byCount != 0) {
                return byCount;
            }
            return Integer.compare(a.dezena(), b.dezena());
        });

        Locale ptBr = Locale.forLanguageTag("pt-BR");
        for (DezenaPct it : items) {
            out.add(String.format(ptBr, "dezena=%d: %.2f%% (%d/%d)", it.dezena(), it.pct(), it.count(), totalLines));
        }

        return out;
    }

    private static List<String> formatSuggestionReport(int[] freq, int totalLines) {
        if (freq == null || freq.length < 26) {
            return List.of("Sugestao indisponivel.");
        }

        record DezenaCnt(int dezena, int count) {}

        List<DezenaCnt> descartadas100 = new ArrayList<>();
        List<DezenaCnt> candidatas = new ArrayList<>(25);
        for (int d = 1; d <= 25; d++) {
            DezenaCnt item = new DezenaCnt(d, freq[d]);
            if (totalLines > 0 && freq[d] == totalLines) {
                descartadas100.add(item);
            } else {
                candidatas.add(item);
            }
        }

        // Ordena por percentil desc, desempate dezena asc
        candidatas.sort((a, b) -> {
            int byCount = Integer.compare(b.count(), a.count());
            return byCount != 0 ? byCount : Integer.compare(a.dezena(), b.dezena());
        });

        // Selecao greedy: evita sequencias de mais de MAX_CONSECUTIVE_HOURS horas seguidas
        List<Integer> top15 = new ArrayList<>(15);
        List<Integer> puladasConsec = new ArrayList<>();
        for (DezenaCnt c : candidatas) {
            if (top15.size() >= 15) break;
            int d = c.dezena();
            int runAbaixo = 0;
            for (int prev = d - 1; prev >= 1 && top15.contains(prev); prev--) {
                runAbaixo++;
            }
            int runAcima = 0;
            for (int next = d + 1; next <= 25 && top15.contains(next); next++) {
                runAcima++;
            }
            if (runAbaixo + 1 + runAcima <= MAX_CONSECUTIVE_HOURS) {
                top15.add(d);
            } else {
                puladasConsec.add(d);
            }
        }
        top15.sort(Integer::compareTo);
        puladasConsec.sort(Integer::compareTo);

        descartadas100.sort((a, b) -> Integer.compare(a.dezena(), b.dezena()));
        List<Integer> descartadasNums = new ArrayList<>(descartadas100.size());
        for (DezenaCnt d : descartadas100) {
            descartadasNums.add(d.dezena());
        }

        List<String> out = new ArrayList<>();
        out.add("---");
        out.add("Sugestao para a linha " + (totalLines + 1)
                + " baseada no percentil (max " + MAX_CONSECUTIVE_HOURS + "h seguidas):");
        if (!descartadasNums.isEmpty()) {
            out.add("Descartadas (100%): " + joinInts(descartadasNums));
        }
        if (!puladasConsec.isEmpty()) {
            out.add("Puladas (consecutividade): " + joinInts(puladasConsec));
        }
        out.add("Top " + top15.size() + ": " + joinInts(top15));
        return out;
    }

    private static String joinInts(List<Integer> values) {
        if (values == null || values.isEmpty()) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < values.size(); i++) {
            if (i > 0) {
                sb.append(',');
            }
            sb.append(values.get(i));
        }
        return sb.toString();
    }

    private static Path resolveCsvPath(String[] args) {
        if (args != null && args.length > 0 && args[0] != null && !args[0].isBlank()) {
            return Paths.get(args[0].trim());
        }
        // padrão: arquivo do workspace
        return Paths.get("arquivos", "lotofacil.csv");
    }

    private static List<String> readLastNonEmptyLines(Path file, int n) throws IOException {
        if (n <= 0) {
            return List.of();
        }

        Deque<String> deque = new ArrayDeque<>(n);
        try (BufferedReader br = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
            String line;
            while ((line = br.readLine()) != null) {
                String trimmed = line.trim();
                if (trimmed.isEmpty()) {
                    continue;
                }
                if (deque.size() == n) {
                    deque.removeFirst();
                }
                deque.addLast(trimmed);
            }
        }

        return new ArrayList<>(deque);
    }
}
