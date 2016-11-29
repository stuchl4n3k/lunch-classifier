package net.stuchl4n3k.lunchtime.classifier;

/**
 * @author petr.stuchlik
 */
public interface SampleFactory {

    Sample createSample(String path, int width, int height);

    Sample createLabeledSample(String path, int width, int height);

}
