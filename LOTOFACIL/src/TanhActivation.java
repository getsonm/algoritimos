/**
 * Autor: getson@engineer.com
 */
public class TanhActivation implements ActivationFunction {
    @Override
    public double activate(double x) {
        double clipped = Math.max(-700, Math.min(700, x));
        return Math.tanh(clipped);
    }
    
    @Override
    public double derivative(double x) {
        double tanh = activate(x);
        return 1.0 - (tanh * tanh);
    }
    
    @Override
    public String getName() {
        return "Tanh";
    }
}
