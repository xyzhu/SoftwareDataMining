package main.java.eden;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.core.Instances;


public class MyEvaluation extends Evaluation{

	public MyEvaluation(Instances data) throws Exception {
		super(data);
	}

	public void crossValidateModel(Classifier classifier,
			Instances data, int numFolds, Random random,
			Object... forPredictionsPrinting) 
					throws Exception {

		// Make a copy of the data we can reorder
		data = new Instances(data);
		data.randomize(random);
		if (data.classAttribute().isNominal()) {
			data.stratify(numFolds);
		}

		// We assume that the first element is a 
		// weka.classifiers.evaluation.output.prediction.AbstractOutput object
		AbstractOutput classificationOutput = null;
		if (forPredictionsPrinting.length > 0) {
			// print the header first
			classificationOutput = (AbstractOutput) forPredictionsPrinting[0];
			classificationOutput.setHeader(data);
			classificationOutput.printHeader();
		}

		// Do the folds
		for (int i = 0; i < numFolds; i++) {
			Instances train = data.trainCV(numFolds, i, random);
			setPriors(train);
			Classifier copiedClassifier = Classifier.makeCopy(classifier);
			copiedClassifier.buildClassifier(train);
			Instances test = data.testCV(numFolds, i);
			evaluateModel(copiedClassifier, test, forPredictionsPrinting);
		}
		m_NumFolds = numFolds;

		if (classificationOutput != null)
			classificationOutput.printFooter();

	}
}
