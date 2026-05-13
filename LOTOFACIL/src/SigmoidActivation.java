/**
 * Autor: getson@engineer.com
 */
public class SigmoidActivation implements ActivationFunction {
    @Override
    public double activate(double x) {
        double clipped = java.lang.Math.max(-700, java.lang.Math.min(700, x));
        return 1.0 / (1.0 + java.lang.Math.exp(-clipped));
    }
    
    @Override
    public double derivative(double x) {
        double sig = activate(x);
        return sig * (1.0 - sig);
    }
    
    @Override
    public String getName() {
        return "Sigmoid";
    }
}
