package net.stuchl4n3k.lunchtime.classifier;

/**
 * A tuple of {@link Features} and a corresponding {@link Label}.
 * <p>
 *     Typically used to represent a training data sample.
 * </p>
 *
 * @author petr.stuchlik
 */
public class Sample {

    private final Features features;
    private final Label label;

    public Sample(Features features, Label label) {
        this.features = features;
        this.label = label;
    }

    public Features getFeatures() {
        return features;
    }

    public Label getLabel() {
        return label;
    }
}
