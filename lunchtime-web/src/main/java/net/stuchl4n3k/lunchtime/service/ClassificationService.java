package net.stuchl4n3k.lunchtime.service;

import java.io.File;
import net.stuchl4n3k.lunchtime.domain.ClassificationResult;

/**
 * @author petr.stuchlik
 */
public interface ClassificationService {

    ClassificationResult classify(File file);

}
