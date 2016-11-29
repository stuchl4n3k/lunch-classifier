package net.stuchl4n3k.lunchtime;

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
}
