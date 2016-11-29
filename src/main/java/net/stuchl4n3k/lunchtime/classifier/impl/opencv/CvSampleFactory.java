package net.stuchl4n3k.lunchtime.classifier.impl.opencv;

import net.stuchl4n3k.lunchtime.classifier.Features;
import net.stuchl4n3k.lunchtime.classifier.Label;
import net.stuchl4n3k.lunchtime.classifier.impl.AbstractSampleFactory;
import net.stuchl4n3k.lunchtime.classifier.util.CvUtils;

/**
 * @author petr.stuchlik
 */
public class CvSampleFactory extends AbstractSampleFactory {

    @Override
    public Features createFeatures(String path, int width, int height) {
        return new CvFeatures(CvUtils.loadImage(path, width, height));
    }

    @Override
    public Label createLabel(String path) {
        return new CvLabel(CvUtils.toMat(adaptClassToOpenCv(getClassOf(path))));
    }

    public double adaptClassToOpenCv(int cls) {
        return cls > 0 ? 1 : -1;
    }
}
