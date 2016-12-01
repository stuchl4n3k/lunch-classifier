package net.stuchl4n3k.lunchtime.service.impl;

import java.io.File;
import java.util.List;
import javax.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import net.stuchl4n3k.lunchtime.classifier.ANN;
import net.stuchl4n3k.lunchtime.classifier.Label;
import net.stuchl4n3k.lunchtime.classifier.Sample;
import net.stuchl4n3k.lunchtime.classifier.SampleFactory;
import net.stuchl4n3k.lunchtime.classifier.impl.opencv.CvANN;
import net.stuchl4n3k.lunchtime.classifier.impl.opencv.CvSampleFactory;
import net.stuchl4n3k.lunchtime.classifier.util.IoUtils;
import net.stuchl4n3k.lunchtime.domain.ClassificationResult;
import net.stuchl4n3k.lunchtime.service.ClassificationService;
import nu.pattern.OpenCV;
import org.opencv.core.Mat;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import static net.stuchl4n3k.lunchtime.Main.NUM_NEURONS_HIDDEN_LAYER;
import static net.stuchl4n3k.lunchtime.Main.NUM_NEURONS_INPUT;
import static net.stuchl4n3k.lunchtime.Main.NUM_NEURONS_OUTPUT;
import static net.stuchl4n3k.lunchtime.Main.SAMPLE_H;
import static net.stuchl4n3k.lunchtime.Main.SAMPLE_W;
import static net.stuchl4n3k.lunchtime.Main.computeErrorRate;

/**
 * @author petr.stuchlik
 */
@Service
@Slf4j
public class ClassificationServiceImpl implements ClassificationService {

    private final ANN ann = new CvANN(NUM_NEURONS_INPUT, NUM_NEURONS_HIDDEN_LAYER, NUM_NEURONS_OUTPUT);
    private final SampleFactory sampleFactory = new CvSampleFactory();

    @Value("#{'${LUNCHTIME_HOME:}' ?: '${user.home}/lunchtime'}")
    private String lunchtimeHomeDir;

    // Load OpenCV native libraries.
    static {
        OpenCV.loadShared();
        System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);
    }

    @PostConstruct
    public void trainAnn() {
        // Find input files.
        File trainingDatasetDir = new File(lunchtimeHomeDir, "training_dataset");
        if (!trainingDatasetDir.exists()) {
            LOG.info("Training dir '{}' does not exist. Defaulting to './training_dataset'.", trainingDatasetDir);
            trainingDatasetDir = new File("training_dataset");
        }

        List<String> inputFiles = IoUtils.findInputFiles(trainingDatasetDir);

        // MLP training.
        LOG.info("ANN training in progress...");
        inputFiles.forEach(path -> {
            Sample sample = sampleFactory.createLabeledSample(path, SAMPLE_W, SAMPLE_H);
            ann.addTrainingSample(sample);
        });
        int iterationsCounter = ann.train();
        LOG.info("Training done after {} iterations", iterationsCounter);

        // Compute MLP error rate on train data.
        double trnErrRate = computeErrorRate(ann, inputFiles);
        LOG.info("Error rate on train data: {}", trnErrRate);
    }

    public ClassificationResult classify(File file) {
        Sample sample = sampleFactory.createSample(file.getAbsolutePath(), SAMPLE_W, SAMPLE_H);
        Label prediction = ann.predict(sample.getFeatures());
        Mat predictionValue = (Mat) prediction.getValue();
        return ClassificationResult.getByNumericClass((int) predictionValue.get(0,0)[0]);
    }

}
