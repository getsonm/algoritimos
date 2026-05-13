/**
 * Autor: getson@engineer.com
 */
/**
 * LSTM Cell - Unidade básica de uma rede recorrente LSTM
 * Captura dependências temporais de longo prazo
 */
public class LSTMCell {
    // Pesos para forget gate
    private double[] wf, uf, bf;
    
    // Pesos para input gate
    private double[] wi, ui, bi;
    
    // Pesos para candidate memory
    private double[] wc, uc, bc;
    
    // Pesos para output gate
    private double[] wo, uo, bo;
    
    // Estados
    private double[] cellState;      // C_t
    private double[] hiddenState;    // h_t
    private double[] inputData;      // x_t
    
    // Gradientes
    private double[] wfGrad, ufGrad, bfGrad;
    private double[] wiGrad, uiGrad, biGrad;
    private double[] wcGrad, ucGrad, bcGrad;
    private double[] woGrad, uoGrad, boGrad;
    
    public LSTMCell(int inputSize, int hiddenSize) {
        // Inicializar pesos com small random values
        double scale = Math.sqrt(2.0 / (inputSize + hiddenSize));
        
        // Forget gate
        wf = initializeWeights(inputSize, scale);
        uf = initializeWeights(hiddenSize, scale);
        bf = new double[hiddenSize];
        
        // Input gate
        wi = initializeWeights(inputSize, scale);
        ui = initializeWeights(hiddenSize, scale);
        bi = new double[hiddenSize];
        
        // Candidate memory
        wc = initializeWeights(inputSize, scale);
        uc = initializeWeights(hiddenSize, scale);
        bc = new double[hiddenSize];
        
        // Output gate
        wo = initializeWeights(inputSize, scale);
        uo = initializeWeights(hiddenSize, scale);
        bo = new double[hiddenSize];
        
        // Estados
        this.cellState = new double[hiddenSize];
        this.hiddenState = new double[hiddenSize];
        
        // Gradientes
        this.wfGrad = new double[inputSize];
        this.ufGrad = new double[hiddenSize];
        this.bfGrad = new double[hiddenSize];
        
        this.wiGrad = new double[inputSize];
        this.uiGrad = new double[hiddenSize];
        this.biGrad = new double[hiddenSize];
        
        this.wcGrad = new double[inputSize];
        this.ucGrad = new double[hiddenSize];
        this.bcGrad = new double[hiddenSize];
        
        this.woGrad = new double[inputSize];
        this.uoGrad = new double[hiddenSize];
        this.boGrad = new double[hiddenSize];
    }

    private double[] initializeWeights(int size, double scale) {
        double[] weights = new double[size];
        for (int i = 0; i < size; i++) {
            weights[i] = (Math.random() - 0.5) * scale;
        }
        return weights;
    }

    /**
     * Forward pass through LSTM cell
     */
    public void forward(double[] input, double[] previousHidden, double[] previousCell) {
        this.inputData = input;
        int hiddenSize = hiddenState.length;
        int inputSize = input.length;
        
        // Forget gate: f_t = sigmoid(W_f * x_t + U_f * h_{t-1} + b_f)
        double[] forgetGate = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            double z = bf[i];
            for (int j = 0; j < inputSize && j < wf.length; j++) {
                z += wf[j] * input[j];
            }
            for (int j = 0; j < hiddenSize && j < uf.length; j++) {
                z += uf[j] * previousHidden[j];
            }
            forgetGate[i] = sigmoid(z);
        }
        
        // Input gate: i_t = sigmoid(W_i * x_t + U_i * h_{t-1} + b_i)
        double[] inputGate = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            double z = bi[i];
            for (int j = 0; j < inputSize && j < wi.length; j++) {
                z += wi[j] * input[j];
            }
            for (int j = 0; j < hiddenSize && j < ui.length; j++) {
                z += ui[j] * previousHidden[j];
            }
            inputGate[i] = sigmoid(z);
        }
        
        // Candidate memory: C'_t = tanh(W_c * x_t + U_c * h_{t-1} + b_c)
        double[] candidateMemory = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            double z = bc[i];
            for (int j = 0; j < inputSize && j < wc.length; j++) {
                z += wc[j] * input[j];
            }
            for (int j = 0; j < hiddenSize && j < uc.length; j++) {
                z += uc[j] * previousHidden[j];
            }
            candidateMemory[i] = Math.tanh(z);
        }
        
        // Cell state: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C'_t
        for (int i = 0; i < hiddenSize; i++) {
            cellState[i] = forgetGate[i] * previousCell[i] + inputGate[i] * candidateMemory[i];
        }
        
        // Output gate: o_t = sigmoid(W_o * x_t + U_o * h_{t-1} + b_o)
        double[] outputGate = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            double z = bo[i];
            for (int j = 0; j < inputSize && j < wo.length; j++) {
                z += wo[j] * input[j];
            }
            for (int j = 0; j < hiddenSize && j < uo.length; j++) {
                z += uo[j] * previousHidden[j];
            }
            outputGate[i] = sigmoid(z);
        }
        
        // Hidden state: h_t = o_t ⊙ tanh(C_t)
        for (int i = 0; i < hiddenSize; i++) {
            hiddenState[i] = outputGate[i] * Math.tanh(cellState[i]);
        }
    }

    /**
     * Backward pass through LSTM cell
     */
    public double[] backward(double[] dh, double[] dc, double[] previousHidden, double[] previousCell, double learningRate) {
        int hiddenSize = hiddenState.length;
        int inputSize = inputData.length;
        
        // Gradiente com relação à entrada
        double[] dInput = new double[inputSize];
        
        // Recomputar valores para backward pass
        // (simplificado para demonstração)
        double[] forgetGate = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            double z = bf[i];
            for (int j = 0; j < inputSize && j < wf.length; j++) {
                z += wf[j] * inputData[j];
            }
            forgetGate[i] = sigmoid(z);
        }
        
        // Atualizar pesos com gradientes
        for (int i = 0; i < inputSize && i < wf.length; i++) {
            wfGrad[i] += dh[i % hiddenSize];
            wf[i] -= learningRate * wfGrad[i];
        }
        
        return dInput;
    }

    private double sigmoid(double x) {
        // Clipping para evitar overflow
        x = java.lang.Math.max(-700, java.lang.Math.min(700, x));
        return 1.0 / (1.0 + java.lang.Math.exp(-x));
    }

    public double[] getHiddenState() {
        return hiddenState;
    }

    public double[] getCellState() {
        return cellState;
    }

    public void setCellState(double[] cellState) {
        this.cellState = cellState;
    }

    public void setHiddenState(double[] hiddenState) {
        this.hiddenState = hiddenState;
    }

    public void resetGradients() {
        // Reset gradientes
        wfGrad = new double[wf.length];
        ufGrad = new double[uf.length];
        bfGrad = new double[bf.length];
        
        wiGrad = new double[wi.length];
        uiGrad = new double[ui.length];
        biGrad = new double[bi.length];
        
        wcGrad = new double[wc.length];
        ucGrad = new double[uc.length];
        bcGrad = new double[bc.length];
        
        woGrad = new double[wo.length];
        uoGrad = new double[uo.length];
        boGrad = new double[bo.length];
    }
}
