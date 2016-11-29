package net.stuchl4n3k.lunchtime.classifier.impl.opencv;

import java.util.ArrayList;
import java.util.List;
import lombok.extern.slf4j.Slf4j;
import net.stuchl4n3k.lunchtime.classifier.ANN;
import net.stuchl4n3k.lunchtime.classifier.Features;
import net.stuchl4n3k.lunchtime.classifier.Label;
import net.stuchl4n3k.lunchtime.classifier.Sample;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.ml.CvANN_MLP;

/**
 * Artificial neural network representation.
 * <p>
 * Note: this is abstraction over OpenCV MLP.<br>
 * Note: this class is not thread-safe.
 * </p>
 *
 * @author petr.stuchlik
 */
@Slf4j
public class CvANN implements ANN {

    public static final int NATIVE_DATA_TYPE = CvType.CV_32F;

    protected final int[] numNeuronsInLayers;
    protected final int numNeuronsInOutput;
    protected final List<Sample> trainingSamples;
    protected final CvANN_MLP mlp;

    public CvANN(int numNeuronsInput, int numNumNeuronsHidden, int numNeuronsOutput) {
        this(new int[]{numNeuronsInput, numNumNeuronsHidden, numNeuronsOutput});
    }

    public CvANN(int[] numNeuronsInLayers) {
        this.numNeuronsInLayers = numNeuronsInLayers;
        this.numNeuronsInOutput = numNeuronsInLayers[numNeuronsInLayers.length - 1];
        this.trainingSamples = new ArrayList<>();
        this.mlp = new CvANN_MLP();

        // Init MLP.
        MatOfInt layerSizes = new MatOfInt();
        layerSizes.fromArray(numNeuronsInLayers);
        mlp.create(layerSizes);
    }

    @Override
    public void addTrainingSample(Sample sample) {
        trainingSamples.add(sample);
    }

    @Override
    public int train() {
        Mat inputRowVectors = new Mat();
        Mat outputRowVectors = new Mat();
        Mat sampleWeightVectors = Mat.ones(trainingSamples.size(), 1, NATIVE_DATA_TYPE);

        for (int i = 0; i < trainingSamples.size(); i++) {
            Sample sample = trainingSamples.get(i);

            inputRowVectors.push_back((Mat) sample.getFeatures().getValue());
            outputRowVectors.push_back((Mat) sample.getLabel().getValue());

            // Adjust weight of this sample.
            double sampleWeight = adjustTrainingSampleWeight(sample);
            sampleWeightVectors.put(i, 0, sampleWeight);
        }

        trainingSamples.clear();

        LOG.debug("inputRowVectors: \n{}", toString(inputRowVectors));
        LOG.debug("outputRowVectors: \n{}", toString(outputRowVectors));

        return mlp.train(inputRowVectors, outputRowVectors, sampleWeightVectors);
    }

    @Override
    public Label predict(Features features) {
        CvLabel label = new CvLabel(Mat.zeros(1, numNeuronsInOutput, NATIVE_DATA_TYPE));
        mlp.predict((Mat) features.getValue(), label.getValue());

        int adaptedClass = adaptOpenCvClassToLabel(label.getValue());
        label.getValue().put(0, 0, adaptedClass);
        return label;
    }

    protected int adaptOpenCvClassToLabel(Mat openCvClass) {
        double adaptedClass = openCvClass.get(0, 0)[0];
        if (adaptedClass > 0) {
            return 1;
        } else {
            return -1;
        }
    }

    protected double adjustTrainingSampleWeight(Sample sample) {
//        return sample.getLabel().get(0, 0)[0] == 1 ? 0.9 : 0.1;
        return 1;
    }

    protected String toString(Mat mat) {
        return toString(mat, false);
    }

    protected String toString(Mat mat, boolean withContent) {
        StringBuilder strb = new StringBuilder();
        strb.append(mat.toString()).append("\n");
        if (withContent) {
            for (int i = 0; i < mat.rows(); i++) {
                for (int j = 0; j < mat.cols(); j++) {
                    strb.append(String.format("%.2f ", mat.get(i, j)[0]));
                }
                strb.append("\n");
            }
        }
        return strb.toString();
    }
}
