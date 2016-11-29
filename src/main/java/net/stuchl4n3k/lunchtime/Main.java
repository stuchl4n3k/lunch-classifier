package net.stuchl4n3k.lunchtime;

import java.io.File;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import lombok.extern.slf4j.Slf4j;
import net.stuchl4n3k.lunchtime.classifier.ANN;
import net.stuchl4n3k.lunchtime.classifier.Label;
import net.stuchl4n3k.lunchtime.classifier.Sample;
import net.stuchl4n3k.lunchtime.classifier.SampleFactory;
import net.stuchl4n3k.lunchtime.classifier.impl.opencv.CvANN;
import net.stuchl4n3k.lunchtime.classifier.impl.opencv.CvSampleFactory;
import net.stuchl4n3k.lunchtime.classifier.util.IoUtils;
import nu.pattern.OpenCV;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

/**
 * LunchCam classifier using CvANN (OpenCV implementation).
 * <p>
 * Based on <a href="http://docs.opencv.org/2.4/modules/ml/doc/neural_networks.html">http://docs.opencv.org/2.4/modules/ml/doc/neural_networks.html</a>
 * and <a href="http://docs.opencv.org/3.1.0/dc/dd6/ml_intro.html">http://docs.opencv.org/3.1.0/dc/dd6/ml_intro.html</a>.
 * </p>
 *
 * @author petr.stuchlik
 */
@Slf4j
public class Main {

    // Load OpenCV native libraries.
    static {
        OpenCV.loadShared();
        System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);
    }

    // Parametrization:
    public static final int SAMPLE_W = 80;
    public static final int SAMPLE_H = 60;
    public static final int NUM_NEURONS_INPUT = SAMPLE_W * SAMPLE_H;
    public static final int NUM_NEURONS_HIDDEN_LAYER = 3;
    public static final int NUM_NEURONS_OUTPUT = 1;
    public static final double PERCENT_TRN_SAMPLES = 0.85;

    private static SampleFactory sampleFactory = new CvSampleFactory();

    public static void main(String[] args) {
        int numIterations = 10;
        double tstErrRateSum = 0;
        for (int i = 0; i < numIterations; i++) {
            tstErrRateSum += trainAndTestMlp();
        }
        double meanTstErrRate = tstErrRateSum / numIterations;

        System.err.println("#######################################");
        System.err.println(String.format("Mean error rate on test data: %.2f", meanTstErrRate));
    }

    /**
     * Runs the training and testing algorithm and returns error rate on test data for a random split of samples.
     */
    @SuppressWarnings("unchecked")
    public static double trainAndTestMlp() {
        // Find input files.
        List<Path> inputFiles = IoUtils.findInputFiles(new File("input"));

        // Random split train and test data.
        int splitPos = (int) Math.ceil(inputFiles.size() * PERCENT_TRN_SAMPLES);
        Collections.shuffle(inputFiles);
        List<Path> trnInputFiles = inputFiles.subList(0, splitPos);
        List<Path> tstInputFiles = inputFiles.subList(splitPos, inputFiles.size());

        // Init MLP: NUM_NEURONS_INPUT x NUM_NEURONS_HIDDEN_LAYER x NUM_NEURONS_OUTPUT.
        ANN ann = new CvANN(NUM_NEURONS_INPUT, NUM_NEURONS_HIDDEN_LAYER, NUM_NEURONS_OUTPUT);

        // MLP training.
        System.err.println("Training in progress...");
        trnInputFiles.forEach(path -> {
            Sample sample = sampleFactory.createSample(path.toString(), SAMPLE_W, SAMPLE_H);
            ann.addTrainingSample(sample);
        });
        int iterationsCounter = ann.train();
        System.err.println(String.format("Done after %d iterations", iterationsCounter));

        // Compute MLP error rate on train and test data.
        double trnErrRate = computeErrorRate(ann, trnInputFiles);
        double tstErrRate = computeErrorRate(ann, tstInputFiles);

        System.err.println(String.format("Error rate on train data: %f", trnErrRate));
        System.err.println(String.format("Error rate on test data: %f", tstErrRate));

//        CvUtils.openAsImage(createClassificationRaster(ann, tstInputFiles));
//        System.exit(0);

        return tstErrRate;
    }

    /**
     * Computes mean error rate using a given {@code mlp} classifier on given {@code inputFiles}.
     */
    public static double computeErrorRate(ANN ann, List<Path> inputFiles) {
        int errCount = 0;
        for (Path path : inputFiles) {
            Sample sample = sampleFactory.createSample(path.toString(), SAMPLE_W, SAMPLE_H);
            Label predictedLabel = ann.predict(sample.getFeatures());

            int expClass = (int) ((Mat) sample.getLabel().getValue()).get(0, 0)[0];
            int predClass = (int) ((Mat) predictedLabel.getValue()).get(0, 0)[0];

            boolean err = expClass != predClass;
            if (err) {
                errCount++;
            }

            LOG.debug(String.format("Predicted: %d | Expected: %d | Err: %b", expClass, predClass, err));
        }
        return (double) errCount / inputFiles.size();
    }

    public static Mat createClassificationRaster(ANN ann, List<Path> inputFiles) {
        int rasterSideSize = (int) Math.ceil(Math.sqrt(inputFiles.size()));
        int rasterWidthPx = rasterSideSize * SAMPLE_W;
        int rasterHeightPx = rasterSideSize * SAMPLE_H;

        int xPos = 0;
        int yPos = 0;

        Mat raster = Mat.zeros(rasterHeightPx, rasterWidthPx, CvType.CV_32F);

        for (int i = 0; i < inputFiles.size(); i++) {
            Path path = inputFiles.get(i);
            Sample sample = sampleFactory.createSample(path.toString(), SAMPLE_W, SAMPLE_H);
            Label predictedLabel = ann.predict(sample.getFeatures());

            int expClass = (int) ((Mat) sample.getLabel().getValue()).get(0, 0)[0];
            int predClass = (int) ((Mat) predictedLabel.getValue()).get(0, 0)[0];

            Mat image = (Mat) sample.getFeatures().getValue();
            image = image.reshape(0, SAMPLE_H);

            if (expClass != predClass) {
                Mat.zeros(5, SAMPLE_W, CvType.CV_32F).copyTo(image.submat(0, 5, 0, SAMPLE_W));
            }

            image.copyTo(raster.submat(yPos, yPos + SAMPLE_H, xPos, xPos + SAMPLE_W));
            xPos += SAMPLE_W;
            if (xPos >= rasterWidthPx) {
                xPos = 0;
                yPos += SAMPLE_H;
            }
        }

        raster.convertTo(raster, CvType.CV_32F, 255.5);
        return raster;
    }
}

