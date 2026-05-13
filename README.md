# LotoFacil

Laboratório Java para análise histórica, mineração de padrões e geração de apostas da Lotofácil.

Projeto focado em experimentar diferentes abordagens de análise de sorteios, combinando mineração de dados, modelos estatísticos, redes neurais e heurísticas de seleção de dezenas.

---

## Visão Geral

Este repositório reúne um conjunto de analisadores para Lotofácil. O ponto de entrada principal é a classe [LotofacilDMPlus](src/LotofacilDMPlus.java), que lê o histórico de sorteios, identifica padrões frequentes e gera apostas com base em pontuação combinada.

Além dela, o projeto inclui implementações experimentais como HMM, LSTM, Random Forest, Naive Bayes, Regressão Logística, ARIMA, Apriori, Agrupamento e Algoritmo Genético. Na prática, trata-se de um ambiente de pesquisa e comparação de estratégias para seleção de dezenas.

Este projeto foi escrito totalmente em Java puro (JDK), sem uso de frameworks ou bibliotecas externas de machine learning.

## O Que O Projeto Faz

- Lê o histórico de sorteios da Lotofácil a partir de um arquivo CSV.
- Minera itemsets frequentes e regras de associação.
- Pesa os dados por recência para valorizar sorteios mais recentes.
- Calcula pontuações para as dezenas com base em suporte, confiança, padrões cíclicos e frequência recente.
- Gera apostas com restrições de diversidade e composição.
- Oferece modos extras para varredura de parâmetros, busca aleatória, adaptação automática e estratégias alternativas.

## Algoritmos E Componentes

A tabela abaixo reúne todos os algoritmos, modelos e classes de apoio presentes em src, incluindo os módulos experimentais e as estruturas que sustentam a rede neural e as estratégias de ensemble.

| Arquivo | Categoria | Papel no projeto |
| --- | --- | --- |
| [src/LotofacilDMPlus.java](src/LotofacilDMPlus.java) | Núcleo principal | Motor principal de mineração de dados com Apriori, regras de associação, recência e geração de apostas |
| [src/ClusteringAnalyzer.java](src/ClusteringAnalyzer.java) | Clustering | Agrupamento K-Means baseado em coocorrência entre dezenas |
| [src/EntropyAnalyzer.java](src/EntropyAnalyzer.java) | Estatística da informação | Cálculo de entropia, probabilidade conjunta e informação mútua |
| [src/GradientBoostingAnalyzer.java](src/GradientBoostingAnalyzer.java) | Ensemble boosting | Boosting com árvores de decisão simples treinadas em sequência |
| [src/GeneticAlgorithmOptimizer.java](src/GeneticAlgorithmOptimizer.java) | Otimização evolutiva | Busca de apostas por algoritmo genético com população, mutação e elitismo |
| [src/HMMSequenceAnalyzer.java](src/HMMSequenceAnalyzer.java) | Modelo sequencial | Cadeia de Markov Oculta com treino Baum-Welch para sequências históricas |
| [src/HMMBetGenerator.java](src/HMMBetGenerator.java) | Gerador híbrido | Combina scores do HMM com frequência histórica para montar apostas |
| [src/HigherOrderMarkovAnalyzer.java](src/HigherOrderMarkovAnalyzer.java) | Modelo sequencial | Cadeia de Markov de ordem superior para capturar dependências entre sorteios |
| [src/LogisticRegressionAnalyzer.java](src/LogisticRegressionAnalyzer.java) | Classificação | Regressão logística aplicada à seleção de dezenas |
| [src/LSTMAnalyzer.java](src/LSTMAnalyzer.java) | Rede recorrente | LSTM para capturar padrões temporais em sequências históricas |
| [src/MetaEnsembleAnalyzer.java](src/MetaEnsembleAnalyzer.java) | Ensemble meta | Votação entre múltiplos analisadores para produzir uma aposta consolidada |
| [src/NaiveBayesAnalyzer.java](src/NaiveBayesAnalyzer.java) | Classificação probabilística | Classificador Naive Bayes para estimar scores de dezenas |
| [src/NeuralNetworkAnalyzer.java](src/NeuralNetworkAnalyzer.java) | Rede neural | Rede neural feedforward treinada com backpropagation |
| [src/RandomForestAnalyzer.java](src/RandomForestAnalyzer.java) | Ensemble de árvores | Random Forest com bagging e votação para ranking de dezenas |
| [src/TimeSeriesARIMAAnalyzer.java](src/TimeSeriesARIMAAnalyzer.java) | Série temporal | ARIMA para análise temporal do histórico |
| [src/TresGruposDezoitoDezenas.java](src/TresGruposDezoitoDezenas.java) | Heurística de filtro | Análise das últimas linhas do histórico e sugestão por frequência e consecutividade |
| [src/ActivationFunction.java](src/ActivationFunction.java) | Apoio neural | Interface base para funções de ativação |
| [src/ReLUActivation.java](src/ReLUActivation.java) | Apoio neural | Implementação da ativação ReLU |
| [src/SigmoidActivation.java](src/SigmoidActivation.java) | Apoio neural | Implementação da ativação sigmoid |
| [src/TanhActivation.java](src/TanhActivation.java) | Apoio neural | Implementação da ativação tanh |
| [src/NeuralLayer.java](src/NeuralLayer.java) | Apoio neural | Camada densa com forward e backward para a rede neural |
| [src/LSTMCell.java](src/LSTMCell.java) | Apoio neural | Célula LSTM com portas e estados internos |
| [src/RandomForestSudoku.java](src/RandomForestSudoku.java) | Demonstração paralela | Exemplo de Random Forest multi-label aplicado ao Sudoku |
| [src/Soduko.java](src/Soduko.java) | Demonstração/validação | Programa auxiliar para validação e previsão sobre arquivo soduko.csv |

## Observações Técnicas Sobre Os Algoritmos

Nem todos os módulos têm o mesmo nível de completude estatística. Este repositório é experimental, então alguns componentes foram implementados como aproximações heurísticas ou versões simplificadas de modelos clássicos.

### Módulos Mais Completos Do Ponto De Vista Estrutural

- [src/LotofacilDMPlus.java](src/LotofacilDMPlus.java): pipeline principal com mineração Apriori, regras de associação, ponderação por recência, diversidade entre apostas e backtest adaptativo.
- [src/GeneticAlgorithmOptimizer.java](src/GeneticAlgorithmOptimizer.java): algoritmo genético com população, seleção, mutação, elitismo e cálculo de fitness.
- [src/ClusteringAnalyzer.java](src/ClusteringAnalyzer.java): K-Means aplicado sobre matriz de coocorrência.
- [src/RandomForestAnalyzer.java](src/RandomForestAnalyzer.java): bagging com árvores de decisão e votação agregada.
- [src/MetaEnsembleAnalyzer.java](src/MetaEnsembleAnalyzer.java): ensemble por votação entre múltiplos analisadores.

### Módulos Simplificados Ou Heurísticos

- [src/HMMSequenceAnalyzer.java](src/HMMSequenceAnalyzer.java): HMM com treino Baum-Welch simplificado, número fixo de estados ocultos e poucas iterações. É útil como experimento, mas não como HMM acadêmico completo.
- [src/HMMBetGenerator.java](src/HMMBetGenerator.java): usa o HMM como fonte de score, mas faz seleção final por roleta e fusão com frequência histórica.
- [src/HigherOrderMarkovAnalyzer.java](src/HigherOrderMarkovAnalyzer.java): modelo de Markov de ordem superior baseado em contagens observadas e probabilidades médias, sem calibração estatística avançada.
- [src/TimeSeriesARIMAAnalyzer.java](src/TimeSeriesARIMAAnalyzer.java): ARIMA simplificado, com parâmetros fixos e estimação aproximada dos componentes AR e MA.
- [src/GradientBoostingAnalyzer.java](src/GradientBoostingAnalyzer.java): boosting experimental com árvores simples e atualização iterativa leve, sem implementação completa de gradientes de perda real.
- [src/LogisticRegressionAnalyzer.java](src/LogisticRegressionAnalyzer.java): regressão logística heurística, usando features derivadas do histórico e ajuste simples por gradiente descendente.
- [src/NaiveBayesAnalyzer.java](src/NaiveBayesAnalyzer.java): Naive Bayes adaptado para frequência e coocorrência, com seleção final probabilística.
- [src/LSTMAnalyzer.java](src/LSTMAnalyzer.java): rede LSTM experimental com treino e retropropagação simplificados.
- [src/NeuralNetworkAnalyzer.java](src/NeuralNetworkAnalyzer.java): rede feedforward com backpropagation básica, usada como baseline neural.

### Módulos De Apoio Ou Demonstração

- [src/ActivationFunction.java](src/ActivationFunction.java), [src/ReLUActivation.java](src/ReLUActivation.java), [src/SigmoidActivation.java](src/SigmoidActivation.java), [src/TanhActivation.java](src/TanhActivation.java), [src/NeuralLayer.java](src/NeuralLayer.java), [src/LSTMCell.java](src/LSTMCell.java): infraestrutura neural de suporte.
- [src/TresGruposDezoitoDezenas.java](src/TresGruposDezoitoDezenas.java): heurística de análise das últimas linhas do CSV com filtro por frequência e consecutividade.
- [src/RandomForestSudoku.java](src/RandomForestSudoku.java) e [src/Soduko.java](src/Soduko.java): programas paralelos de demonstração/validação, fora do fluxo principal da Lotofácil.

### Resumo Prático

- O projeto é totalmente em Java puro, mas vários algoritmos são aproximações experimentais.
- A loteria Lotofácil impõe um espaço pequeno de busca, então vários modelos foram adaptados para ranking de dezenas, não para previsão estatística rigorosa.
- Os módulos de ensemble e mineração tendem a ser os mais úteis para exploração prática dentro deste código.

## Parâmetros, Execução E Organização Dos Experimentos

O projeto trabalha em dois níveis de ajuste:

1. Parâmetros globais do motor principal, definidos em [LotofacilDMPlus.java](src/LotofacilDMPlus.java).
2. Parâmetros internos de cada algoritmo, que na maioria dos casos estão codificados como defaults no próprio arquivo da classe.

Na prática, isso significa que você pode rodar o projeto com a configuração padrão, testar variações com `random` e `sweep`, consolidar uma faixa boa com `autoadapt` e depois executar `full` ou uma estratégia específica com os melhores parâmetros encontrados.

### Parâmetros Globais Principais

| Parâmetro | Onde atua | Impacto direto | Impacto indireto |
| --- | --- | --- | --- |
| `TAMANHO_APOSTA` | Todos os geradores | Define quantas dezenas por aposta | Muda a dificuldade de diversificação e o espaço de busca |
| `NUMERO_APOSTAS` | Todos os geradores | Define quantas apostas a execução emite | Aumenta a variância e a cobertura de combinações |
| `minSupport` | Apriori / regras | Filtra itemsets frequentes | Afeta os scores do motor principal e a relevância das regras |
| `minConfidence` | Regras de associação | Filtra relações fortes | Reduz ruído e reforça padrões estáveis |
| `maxK` | Apriori | Controla tamanho máximo dos itemsets | Aumenta expressividade, custo e esparsidade |
| `recencyAlpha` | Peso temporal | Dá mais ou menos peso aos concursos recentes | Muda a leitura de curto prazo versus histórico amplo |
| `windowSize` | Treino do motor principal | Limita a janela usada para treinar | Acelera execução quando menor e amplia contexto quando maior |
| `assocBoost` | Motor principal | Reforça regras de associação | Intensifica combinações recorrentes |
| `minDiversityOverlap` | Geradores finais | Limita sobreposição entre apostas | Controla quão parecidas as apostas saem |
| `evenMin` / `evenMax` | Validação de composição | Define faixa de pares por aposta | Evita composições extremas |
| `seed` | Aleatoriedade | Fixa resultados reproduzíveis | Facilita comparação entre execuções |
| `historyPath` | Leitura do histórico | Define o CSV de entrada | Sem isso o motor principal não treina |
| `historyStartLine` / `historyEndLine` | Leitura do histórico | Recorta o range usado | Útil para backtest e cenários controlados |
| `backtest` / `btStart` / `btStride` | Avaliação | Ativa walk-forward e define passo | Permite medir estabilidade da configuração |
| `useGPU` | Execução | Indica intenção de aceleração | O projeto continua rodando sem libs externas |

### Algoritmo X Parâmetros X Efeito

| Algoritmo | Parâmetros relevantes | Efeito direto | Custo / observação | Uso recomendado |
| --- | --- | --- | --- | --- |
| [LotofacilDMPlus.java](src/LotofacilDMPlus.java) | `minSupport`, `minConfidence`, `maxK`, `recencyAlpha`, `windowSize`, `assocBoost`, `minDiversityOverlap`, `seed` | É o núcleo de pontuação, mineração e geração | Mais sensível a tuning; bom ponto de partida do projeto | Melhor módulo para começar e para backtest |
| [GeneticAlgorithmOptimizer.java](src/GeneticAlgorithmOptimizer.java) | scores de entrada, população, gerações, mutação | Evolui apostas por fitness | Alto custo se aumentar gerações/população | Bom para refinamento e comparação |
| [MetaEnsembleAnalyzer.java](src/MetaEnsembleAnalyzer.java) | qualidade dos modelos-base | Vota entre métodos | Depende da qualidade dos módulos individuais | Melhor quando os modelos-base já estão calibrados |
| [ClusteringAnalyzer.java](src/ClusteringAnalyzer.java) | histórico, tamanho da aposta, número de clusters | Agrupa por coocorrência | Leve e interpretável | Útil para visão estrutural do histórico |
| [EntropyAnalyzer.java](src/EntropyAnalyzer.java) | histórico, min/max | Calcula entropia e informação mútua | Leve, mas sensível à qualidade do histórico | Bom para detectar relações entre dezenas |
| [GradientBoostingAnalyzer.java](src/GradientBoostingAnalyzer.java) | número de árvores, learning rate | Faz boosting simples | Experimental e moderadamente caro | Bom para ranking alternativo |
| [HMMSequenceAnalyzer.java](src/HMMSequenceAnalyzer.java) | estados ocultos, iterações, histórico | Treina HMM simplificado | Mais experimental e custoso | Melhor como experimento comparativo |
| [HMMBetGenerator.java](src/HMMBetGenerator.java) | peso do HMM, histórico | Combina HMM com frequência | Depende da qualidade do HMM | Útil como ensemble híbrido |
| [HigherOrderMarkovAnalyzer.java](src/HigherOrderMarkovAnalyzer.java) | ordem de Markov, histórico | Usa transições de ordem superior | Melhor com histórico amplo | Bom para capturar dependências sequenciais |
| [LogisticRegressionAnalyzer.java](src/LogisticRegressionAnalyzer.java) | features, épocas, learning rate | Ajusta score por regressão logística | Simples e rápido | Bom baseline probabilístico |
| [LSTMAnalyzer.java](src/LSTMAnalyzer.java) | input/hidden/steps/epochs | Modela sequência temporal | Mais caro e mais experimental | Útil para comparação, não para execução leve |
| [NaiveBayesAnalyzer.java](src/NaiveBayesAnalyzer.java) | histórico, bet size | Estima probabilidade por frequência | Muito leve | Bom baseline rápido |
| [NeuralNetworkAnalyzer.java](src/NeuralNetworkAnalyzer.java) | arquitetura, learning rate, epochs | Rede feedforward | Treino básico e rápido | Útil como baseline neural |
| [RandomForestAnalyzer.java](src/RandomForestAnalyzer.java) | número de árvores, amostragem | Bagging com votação | Custo moderado | Um dos melhores equilíbrios entre custo e estabilidade |
| [TimeSeriesARIMAAnalyzer.java](src/TimeSeriesARIMAAnalyzer.java) | p, d, q, histórico | Série temporal simplificada | Experimental e leve/moderado | Bom para experimentar comportamento temporal |
| [TresGruposDezoitoDezenas.java](src/TresGruposDezoitoDezenas.java) | range do CSV | Filtra e resume últimas linhas | Leve | Útil como pré-análise do arquivo |

### Como Organizar Os Experimentos

1. Comece com a configuração padrão do projeto.
2. Rode `random` para explorar faixas de `minSupport`, `minConfidence`, `maxK`, `recencyAlpha`, `windowSize`, `assocBoost` e `minDiversityOverlap`.
3. Use `sweep` quando quiser comparar perfis fixos em bloco.
4. Use `autoadapt` para refinar o que funcionou melhor no histórico recente.
5. Finalize com `full`, `all` ou uma estratégia específica.

### Perfis Recomendados

| Perfil | Característica | Comando exemplo | Melhor uso |
| --- | --- | --- | --- |
| Conservador | `minSupport` e `minConfidence` altos, `maxK` baixo, `windowSize` moderado | `java -jar target/LotoFacil-1.0.1.jar estrategia1` | Mais estabilidade e menos combinações raras |
| Equilibrado | Defaults próximos do padrão, janela média, diversidade intermediária | `java -jar target/LotoFacil-1.0.1.jar full` | Melhor ponto de partida para comparar tudo |
| Exploratório | `maxK` mais alto, `assocBoost` maior, janela menor, recência mais agressiva | `java -jar target/LotoFacil-1.0.1.jar random 200 1500` | Buscar combinações mais agressivas |

### Melhores Exemplos De Execução

| Comando | Por que é útil | Custo |
| --- | --- | --- |
| `java -jar target/LotoFacil-1.0.1.jar full` | Melhor visão geral do projeto inteiro | Alto |
| `java -jar target/LotoFacil-1.0.1.jar all` | Compara só as estratégias principais | Médio |
| `java -jar target/LotoFacil-1.0.1.jar random 200 1500` | Explora hiperparâmetros e mede backtest | Alto |
| `java -jar target/LotoFacil-1.0.1.jar autoadapt 300` | Refina a configuração com base no histórico recente | Médio/alto |

### Fluxo Prático Recomendado

```bash
# 1. Explorar
java -jar target/LotoFacil-1.0.1.jar random 150 1200

# 2. Refinar
java -jar target/LotoFacil-1.0.1.jar autoadapt 300

# 3. Consolidar
java -jar target/LotoFacil-1.0.1.jar full
```

Esse fluxo combina busca ampla, ajuste local e execução consolidada, e costuma ser o mais produtivo neste repositório.

## Requisitos

- JDK Java 25 ou superior (recomendado para compilar e executar).
- Maven 3.8+.
- Arquivo de histórico em arquivos/lotofacil.csv.

> O projeto usa o diretório src na raiz como código-fonte, em vez do layout padrão src/main/java.

## Como Executar

### 1. Compilar o projeto

```bash
mvn clean package
```

### 2. Executar a aplicação principal

```bash
java -jar target/LotoFacil-1.0.1.jar
```

Isso executa o modo padrão da classe principal.

## Modos De Execução

A classe principal aceita alguns comandos na linha de comando:

- sweep: executa varredura de configurações.
- random [iteracoes] [limite]: faz busca aleatória de hiperparâmetros.
- autoadapt [limite] ou adapt [limite]: adapta parâmetros após atualizar o histórico.
- estrategia1 ou confianca: executa a estratégia ponderada por confiança.
- estrategia2, hibrida ou elite: executa a estratégia híbrida elite.
- estrategia3 ou consensual: executa a estratégia consensual.
- all ou todasEstrategias: executa as três estratégias.
- full, tudo ou completo: executa todos os algoritmos e depois as três estratégias.

### Exemplos

```bash
java -jar target/LotoFacil-1.0.1.jar full
java -jar target/LotoFacil-1.0.1.jar all
java -jar target/LotoFacil-1.0.1.jar random 200 1500
java -jar target/LotoFacil-1.0.1.jar estrategia2
```

## Fonte Dos Dados

O histórico padrão é carregado de:

```text
arquivos/lotofacil.csv
```

Esse arquivo precisa existir e conter os sorteios no formato esperado pelo carregador do projeto.

Seguindo o formato:

    1,2,4,5,6,7,9,10,12,13,14,18,20,23,25 -
    3,4,05,6,8,9,10,13,17,19,20,22,23,24,25 -
    1,4,5,9,10,11,13,14,19,20,21,22,23,24,25 - ...

## Observações

- Este projeto é experimental e voltado a análise estatística, aprendizado de algorítimos e geração heurística de apostas.
- Ele não garante previsões vencedoras.
- Algumas classes são independentes e podem ser usadas como base para testes e estudos comparativos.
- Próximos Pareto,  RANDU...

## Licença

Este projeto está licenciado sob a MIT License.

- Texto completo: [LICENSE](LICENSE)
- Referência oficial: [MIT License (opensource.org)](https://opensource.org/license/mit)
