package net.stuchl4n3k.lunchtime.classifier.impl;

import java.util.regex.Matcher;
import java.util.regex.Pattern;
import net.stuchl4n3k.lunchtime.classifier.Features;
import net.stuchl4n3k.lunchtime.classifier.Label;
import net.stuchl4n3k.lunchtime.classifier.Sample;
import net.stuchl4n3k.lunchtime.classifier.SampleFactory;

/**
 * @author petr.stuchlik
 */
public abstract class AbstractSampleFactory implements SampleFactory {

    public static final Pattern SAMPLE_PATH_CLASS_PATTERN = Pattern.compile(".*_([012])\\..+$");

    @Override
    public Sample createSample(String path, int width, int height) {
        return new Sample(createFeatures(path, width, height), createLabel(path));
    }

    protected abstract Features createFeatures(String path, int width, int height);

    protected abstract Label createLabel(String path);

    /**
     * Gets the class of a given image {@code filePath} by inspecting the filename.
     * <p>
     * Each filename is expected to be in form {@code /option/path/to/file_[012].jpg}.
     * Where [012] is one of possible classes.
     * </p>
     */
    public int getClassOf(String filePath) {
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

        throw new IllegalArgumentException(String.format("Cannot deduce class of file '%s' - the path must end with _[012].ext", filePath));
    }

}
