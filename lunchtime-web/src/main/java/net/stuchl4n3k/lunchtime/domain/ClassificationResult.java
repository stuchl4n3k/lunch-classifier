package net.stuchl4n3k.lunchtime.domain;

import java.util.Arrays;

/**
 * @author petr.stuchlik
 */
public enum ClassificationResult {

    EMPTY(-1),
    CROWDED(1);

    private int numericClass;

    ClassificationResult(int numericClass) {
        this.numericClass = numericClass;
    }

    public int getNumericClass() {
        return numericClass;
    }

    public static ClassificationResult getByNumericClass(int numericClass) {
        return Arrays.stream(ClassificationResult.values())
            .filter(classificationResult -> classificationResult.getNumericClass() == numericClass)
        .findFirst().get();
    }
}
