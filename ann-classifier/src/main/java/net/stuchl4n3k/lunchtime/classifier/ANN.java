package net.stuchl4n3k.lunchtime.classifier;

/**
 * Artificial neural network contract.
 *
 * @author petr.stuchlik
 */
public interface ANN {

    /**
     * Adds a given {@code sample} to the training set.
     */
    void addTrainingSample(Sample sample);

    /**
     * Invokes training on all previously given training samples.
     * @return Number of performed iterations during training
     */
    int train();

    /**
     * Predicts a label for a given {@code features} vector.
     */
    Label predict(Features features);
}
