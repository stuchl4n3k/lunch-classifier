package net.stuchl4n3k.lunchtime.classifier.util;

import java.awt.Desktop;
import java.io.File;
import java.io.IOException;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

/**
 * @author petr.stuchlik
 */
@Slf4j
public final class CvUtils {

    private static final double INTENSITY_NORMALIZATION_FACTOR = 1.0 / 255.5;

    private CvUtils() {
        // No instantiation.
    }

    /**
     * Loads an image on a given {@code filePath} to a Matrix.
     * <p>
     * The image is loaded in grayscale and subsampled to SAMPLE_W x SAMPLE_H px.
     * The resulting Matrix is a row vector of features (normalized intensity values in [0-1]) of size 1xN, where
     * <pre>
     * N = SAMPLE_W * SAMPLE_H
     * </pre>
     * </p>
     */
    public static Mat loadImage(String filePath, int width, int height) {
        return loadImage(filePath, width, height, false);
    }

    /**
     * Loads an image on a given {@code filePath} to a Matrix allowing you to increase contrast before the image
     * is subsampled.
     * <p>
     * The image is loaded in grayscale and subsampled to SAMPLE_W x SAMPLE_H px.
     * The resulting Matrix is a row vector of features (normalized intensity values in [0-1]) of size 1xN, where
     * <pre>
     * N = SAMPLE_W * SAMPLE_H
     * </pre>
     * </p>
     */
    public static Mat loadImage(String filePath, int width, int height, boolean increaseContrast) {
        LOG.debug("Loading image '{}'.", filePath);

        // Load as grayscale image.
        Mat image = Highgui.imread(filePath, Highgui.CV_LOAD_IMAGE_GRAYSCALE);

        // Increase its contrast.
        if (increaseContrast) {
            increaseContrast(image);
        }

        // Downsample it.
        Imgproc.resize(image, image, new Size(width, height), 0, 0, Imgproc.INTER_AREA);

        // Normalize intensities.
        Mat imageNorm = new MatOfFloat();
        image.convertTo(imageNorm, CvType.CV_32F, INTENSITY_NORMALIZATION_FACTOR);

        // Reshape it to a row vector.
        Mat imageRowVec = imageNorm.reshape(0, 1);


        return imageRowVec;
    }

    public static void openAsImage(Mat image) {
        try {
            File tempFile = File.createTempFile("temp-image-" + System.currentTimeMillis(), ".jpg");
            Highgui.imwrite(tempFile.getAbsolutePath(), image);
            Desktop.getDesktop().open(tempFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Applies {@code image = a*image + beta} to each pixel intensity {@code a}.
     */
    public static void increaseContrast(Mat image) {
        double alpha = 2.5;
        double beta = -100;
        image.convertTo(image, -1, alpha, beta);
    }

    /**
     * Transforms a given {@code value} to a vector.
     */
    public static Mat toMat(double value) {
        Mat mlpOutputVector = Mat.zeros(1, 1, CvType.CV_32F);
        mlpOutputVector.put(0, 0, value);
        return mlpOutputVector;
    }

}
