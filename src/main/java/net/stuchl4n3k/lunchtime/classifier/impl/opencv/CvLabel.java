package net.stuchl4n3k.lunchtime.classifier.impl.opencv;

import lombok.ToString;
import net.stuchl4n3k.lunchtime.classifier.Label;
import org.opencv.core.Mat;

/**
 * @author petr.stuchlik
 */
@ToString
public class CvLabel implements Label {

    private final Mat value;

    public CvLabel(Mat value) {
        this.value = value;
    }

    @Override
    public Mat getValue() {
        return value;
    }
}
