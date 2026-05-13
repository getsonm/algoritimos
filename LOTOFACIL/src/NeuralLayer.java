/**
 * Autor: getson@engineer.com
 */
import java.util.*;

public class NeuralLayer {
    private double[][] weights;
    private double[] biases;
    private double[][] weightGradients;
    private double[] biasGradients;
    private int inputSize;
    private int outputSize;
    private ActivationFunction activation;
    private double[] lastInput;
    private double[] lastOutput;
    private double[] lastActivated;
    
    public NeuralLayer(int inputSize, int outputSize, ActivationFunction activation) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.activation = activation;
        
        this.weights = new double[inputSize][outputSize];
        this.biases = new double[outputSize];
        this.weightGradients = new double[inputSize][outputSize];
        this.biasGradients = new double[outputSize];
        
        initializeWeights();
    }
    
    private void initializeWeights() {
        Random random = new Random(42);
        double scale = Math.sqrt(2.0 / inputSize);
        
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] = random.nextGaussian() * scale;
            }
        }
        
        for (int j = 0; j < outputSize; j++) {
            biases[j] = 0.01;
        }
    }
    
    public double[] forward(double[] input) {
        this.lastInput = input;
        double[] z = new double[outputSize];
        
        for (int j = 0; j < outputSize; j++) {
            z[j] = biases[j];
            for (int i = 0; i < inputSize; i++) {
                z[j] += input[i] * weights[i][j];
            }
        }
        
        this.lastOutput = z;
        this.lastActivated = new double[outputSize];
        for (int j = 0; j < outputSize; j++) {
            lastActivated[j] = activation.activate(z[j]);
        }
        
        return lastActivated;
    }
    
    public double[] backward(double[] outputGradients, double learningRate) {
        Arrays.fill(biasGradients, 0);
        for (int i = 0; i < inputSize; i++) {
            Arrays.fill(weightGradients[i], 0);
        }
        
        double[] inputGradients = new double[inputSize];
        
        for (int j = 0; j < outputSize; j++) {
            double delta = outputGradients[j] * activation.derivative(lastOutput[j]);
            biasGradients[j] = delta;
            
            for (int i = 0; i < inputSize; i++) {
                weightGradients[i][j] = delta * lastInput[i];
                inputGradients[i] += delta * weights[i][j];
            }
        }
        
        updateWeights(learningRate);
        
        return inputGradients;
    }
    
    private void updateWeights(double learningRate) {
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] -= learningRate * weightGradients[i][j];
            }
        }
        
        for (int j = 0; j < outputSize; j++) {
            biases[j] -= learningRate * biasGradients[j];
        }
    }
    
    public double[] getWeights(int index) {
        return weights[index];
    }
    
    public void setWeights(int index, double[] newWeights) {
        weights[index] = newWeights;
    }
    
    public int getInputSize() {
        return inputSize;
    }
    
    public int getOutputSize() {
        return outputSize;
    }
    
    public String getActivationName() {
        return activation.getName();
    }
}
