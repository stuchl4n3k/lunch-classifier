package net.stuchl4n3k.lunchtime;

import java.io.File;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import lombok.extern.slf4j.Slf4j;
import nu.pattern.OpenCV;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.CvANN_MLP;

/**
 * LunchCam classifier using ANN (OpenCV implementation).
 * <p>
 *     Based on <a href="http://docs.opencv.org/2.4/modules/ml/doc/neural_networks.html">http://docs.opencv.org/2.4/modules/ml/doc/neural_networks.html</a>
 *     and <a href="http://docs.opencv.org/3.1.0/dc/dd6/ml_intro.html">http://docs.opencv.org/3.1.0/dc/dd6/ml_intro.html</a>.
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

    public static final Pattern SAMPLE_PATH_CLASS_PATTERN = Pattern.compile(".*_([012])\\.jpg$");

    // Parametrization:
    public static final int SAMPLE_W = 10;
    public static final int SAMPLE_H = 7;
    public static final int NUM_NEURONS_INPUT = SAMPLE_W * SAMPLE_H;
    public static final int NUM_NEURONS_HIDDEN_LAYER = NUM_NEURONS_INPUT / 2;
    public static final int NUM_NEURONS_OUTPUT = 1;
    public static final double PERCENT_TRN_SAMPLES = 0.85;

    public static void main(String[] args) {
        int numIterations = 20;
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
    public static double trainAndTestMlp() {
        // Find input files.
        List<Path> inputFiles = findInputFiles(new File("input"));

        // Random split train and test data.
        int splitPos = (int) Math.ceil(inputFiles.size() * PERCENT_TRN_SAMPLES);
        Collections.shuffle(inputFiles);
        List<Path> trnInputFiles = inputFiles.subList(0, splitPos);
        List<Path> tstInputFiles = inputFiles.subList(splitPos, inputFiles.size());

        // Init MLP: NUM_NEURONS_INPUT x NUM_NEURONS_HIDDEN_LAYER x NUM_NEURONS_OUTPUT.
        MatOfInt layerSizes = new MatOfInt();
        layerSizes.fromArray(NUM_NEURONS_INPUT, NUM_NEURONS_HIDDEN_LAYER, NUM_NEURONS_OUTPUT);

        CvANN_MLP mlp = new CvANN_MLP();
        mlp.create(layerSizes);

        // MLP training.
        System.err.println("Training in progress...");
        int iterationsCounter = train(mlp, trnInputFiles);
        System.err.println(String.format("Done after %d iterations", iterationsCounter));

        // Compute MLP error rate on train and test data.
        double trnErrRate = computeErrorRate(mlp, trnInputFiles);
        double tstErrRate = computeErrorRate(mlp, tstInputFiles);

        System.err.println(String.format("Error rate on train data: %f", trnErrRate));
        System.err.println(String.format("Error rate on test data: %f", tstErrRate));

        return tstErrRate;
    }

    /**
     * Lists all {@code *.jpg} files in a given {@code inputDir}.
     */
    public static List<Path> findInputFiles(File inputDir) {
        List<Path> result = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(inputDir.toPath(), "*.jpg")) {
            stream.forEach(result::add);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return result;
    }

    /**
     * Loads an image on a given {@code filePath} to a Matrix.
     * <p>
     * The image is loaded in grayscale and subsampled to SAMPLE_W x SAMPLE_H px.
     * The resulting Matrix is a row vector of features (normalized intensity values in [0-1]) of size 1x7500.
     * </p>
     */
    public static Mat loadImage(String filePath) {
        LOG.debug("Loading image '{}'.", filePath);

        // Load as grayscale image.
        Mat image = Highgui.imread(filePath, Highgui.CV_LOAD_IMAGE_GRAYSCALE);

        // Resize it.
        Imgproc.resize(image, image, new Size(SAMPLE_W, SAMPLE_H));

        // Reshape it to a row vector.
        Mat imageMat = new MatOfFloat();
        image.convertTo(imageMat, CvType.CV_32F, 1.0 / 255.5);
        imageMat = imageMat.reshape(0, 1);

        return imageMat;
    }

    /**
     * Gets the class of a given image {@code filePath} by inspecting the filename.
     * <p>
     * Each filename is expected to be in form {@code /option/path/to/file_[012].jpg}.
     * Where [012] is one of possible classes.
     * </p>
     */
    public static int getClassOf(String filePath) {
        Matcher matcher = SAMPLE_PATH_CLASS_PATTERN.matcher(filePath);
        if (matcher.matches()) {
            int imgClass = Integer.parseInt(matcher.group(1));
            switch (imgClass) {
                case 0:
                case 1:
                    return 0;
                case 2:
                    return 1;
            }
        }

        throw new IllegalArgumentException(String.format("Cannot deduce class of file '%s' - the path must end with _[012].jpg", filePath));
    }

    /**
     * Transforms a given {@code imgClass} to a (vector) value between [-1;1].
     */
    public static Mat toMat(int imgClass) {
        int mlpOutput;
        if (imgClass > 0) {
            mlpOutput = 1;
        } else {
            mlpOutput = -1;
        }

        Mat mlpOutputVector = Mat.zeros(1, 1, CvType.CV_32F);
        mlpOutputVector.put(0, 0, mlpOutput);
        return mlpOutputVector;
    }

    /**
     * Transforms a given vector {@code mlpOutputVector} (an output of MLP prediction) back to a class {0;1}.
     */
    public static int toClass(Mat mlpOutputVector) {
        double mlpOutput = mlpOutputVector.get(0, 0)[0];
        int imgClass;
        if (mlpOutput < 0) {
            imgClass = 0;
        } else {
            imgClass = 1;
        }
        return imgClass;
    }

    public static int train(CvANN_MLP mlp, List<Path> inputFiles) {
        Mat inputRowVectors = new Mat();
        Mat outputRowVectors = new Mat();
        Mat sampleWeightVectors = Mat.ones(inputFiles.size(), 1, CvType.CV_32FC1);

        for (int i = 0; i < inputFiles.size(); i++) {
            Path path = inputFiles.get(i);

            Mat imageVector = loadImage(path.toString());
            inputRowVectors.push_back(imageVector);

            int imgClass = getClassOf(path.toString());
            Mat imgClassVector = toMat(imgClass);
            outputRowVectors.push_back(imgClassVector);

            // Adjust weight of this sample.
//            double sampleWeight = imgClass == 1 ? 0.9 : 0.1;
//            sampleWeightVectors.put(i, 0, sampleWeight);
        }

        LOG.debug("inputRowVectors: \n{}", debug(inputRowVectors));
        LOG.debug("outputRowVectors: \n{}", debug(outputRowVectors));

        return mlp.train(inputRowVectors, outputRowVectors, sampleWeightVectors);
    }

    /**
     * Computes mean error rate using a given {@code mlp} classifier on given {@code inputFiles}.
     */
    public static double computeErrorRate(CvANN_MLP mlp, List<Path> inputFiles) {
        int errCount = 0;
        for (Path path : inputFiles) {
            Mat imageVector = loadImage(path.toString());

            Mat predictedImgVector = Mat.zeros(1, 1, CvType.CV_32F);
            mlp.predict(imageVector, predictedImgVector);

            int expImgClass = getClassOf(path.toString());
            int predImgClass = toClass(predictedImgVector);

            boolean err = expImgClass != predImgClass;
            if (err) {
                errCount++;
            }

            LOG.debug(String.format("Predicted: %d | Expected: %d | Err: %b", predImgClass, expImgClass, err));
        }
        return (double) errCount / inputFiles.size();
    }

    public static String debug(Mat mat) {
        return debug(mat, false);
    }

    public static String debug(Mat mat, boolean withContent) {
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

