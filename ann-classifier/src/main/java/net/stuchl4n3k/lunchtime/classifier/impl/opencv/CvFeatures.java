package net.stuchl4n3k.lunchtime.classifier.impl.opencv;

import lombok.ToString;
import net.stuchl4n3k.lunchtime.classifier.Features;
import org.opencv.core.Mat;

/**
 * @author petr.stuchlik
 */
@ToString
public class CvFeatures implements Features {

    private final Mat value;

    public CvFeatures(Mat value) {
        this.value = value;
    }

    @Override
    public Mat getValue() {
        return value;
    }
}
