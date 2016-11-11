package net.stuchl4n3k.lunchtime;

import java.io.File;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import lombok.extern.java.Log;
import lombok.extern.slf4j.Slf4j;
import nu.pattern.OpenCV;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.highgui.Highgui;
import org.opencv.ml.CvANN_MLP;

/**
 * Base on http://docs.opencv.org/2.4/modules/ml/doc/neural_networks.html#void%20CvANN_MLP::create(const%20Mat&%20layerSizes,%20int%20activateFunc,%20double%20fparam1,%20double%20fparam2)
 *
 * @author petr.stuchlik
 */
@Slf4j
public class Main {

    static {
        OpenCV.loadShared();
        System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        File inputDir = new File("input/trn");
        List<Path> inputFiles = findInputFiles(inputDir);

        Mat inputRowVectors = new Mat();

        for (Path path : inputFiles) {
            Mat imageVector = loadImage(path.toString());
            debug(imageVector);

            inputRowVectors.push_back(imageVector);
        }





        // MLP init
        CvANN_MLP mlp = new CvANN_MLP();
        MatOfInt layerSizes = new MatOfInt();
        layerSizes.fromArray(7500, 50, 1);
        mlp.create(layerSizes);

        // MLP training
//        mlp.train(inputRowVectors)

    }

    public static void debug(Mat mat) {
        debug(mat, false);
    }

    public static void debug(Mat mat, boolean withContent) {
        System.err.println("");
        System.err.println(mat.toString());
        if (withContent) {
            for (int i = 0; i < mat.rows(); i++) {
                for (int j = 0; j < mat.cols(); j++) {
                    System.err.print(String.format("%.2f ", mat.get(i, j)[0]));
                }
                System.err.println("");
            }
        }
    }

    public static List<Path> findInputFiles(File inputDir) {
        List<Path> result = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(inputDir.toPath(), "*.jpg")) {
            stream.forEach(result::add);
        } catch (IOException e) {
            throw new RuntimeException(e.getCause());
        }
        return result;
    }

    public static Mat loadImage(String filePath) {
        LOG.info("Loading image '{}'.", filePath);
        Mat image = Highgui.imread(filePath, Highgui.CV_LOAD_IMAGE_GRAYSCALE);

        Mat imageMat = new MatOfFloat();
        image.convertTo(imageMat, CvType.CV_32F, 1.0/255.5);
        imageMat = imageMat.reshape(0, 1);

        return imageMat;
    }

}

