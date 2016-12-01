package net.stuchl4n3k.lunchtime.classifier.util;

import java.io.File;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import lombok.extern.slf4j.Slf4j;

/**
 * @author petr.stuchlik
 */
@Slf4j
public final class IoUtils {

    private IoUtils() {
        // No instantiation.
    }

    /**
     * Lists all {@code *.jpg} files in a given {@code inputDir}.
     */
    public static List<String> findInputFiles(File inputDir) {
        List<String> result = new ArrayList<>();
        if (inputDir.exists()) {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(inputDir.toPath(), "*.jpg")) {
                stream.forEach(path -> result.add(path.toString()));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        } else {
            throw new IllegalArgumentException("Input dir does not exist: " + inputDir);
        }
        return result;
    }
}
