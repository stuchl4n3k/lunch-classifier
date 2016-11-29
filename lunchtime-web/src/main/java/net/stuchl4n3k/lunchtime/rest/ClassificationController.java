package net.stuchl4n3k.lunchtime.rest;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import lombok.extern.slf4j.Slf4j;
import net.stuchl4n3k.lunchtime.service.ClassificationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * @author petr.stuchlik
 */
@RestController
@Slf4j
public class ClassificationController {

    public static final String LUNCH_CAM_URL = "https://portal.ppf.cz/_layouts/IMAGES/KameraJidelna/Jidelna000M.jpg";

    @Autowired
    private ClassificationService classificationService;

    @RequestMapping("/")
    public String get() {
        File inputFile = null;
        try {
            Path inputFilePath = Files.createTempFile("lunchcam", null);
            try (InputStream in = new URL(LUNCH_CAM_URL).openStream()) {
                Files.copy(in, inputFilePath, StandardCopyOption.REPLACE_EXISTING);
            }
            inputFile = inputFilePath.toFile();
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
        LOG.info("Classifying file: {}", inputFile.getAbsolutePath());
        return classificationService.classify(inputFile).toString();
    }

}
