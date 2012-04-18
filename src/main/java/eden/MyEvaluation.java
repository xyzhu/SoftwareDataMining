package main.java.eden;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;


public class MyEvaluation extends Evaluation{

	double tenfoldResult[] = new double[10];//xyzhu
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

		double predict[] = null;//xyzhu
		int num_inst = 0;
		// Do the folds
		for (int i = 0; i < numFolds; i++) {
			FastVector predictions = null;
			Instances train = data.trainCV(numFolds, i, random);
			setPriors(train);
			Classifier copiedClassifier = Classifier.makeCopy(classifier);
			copiedClassifier.buildClassifier(train);
			Instances test = data.testCV(numFolds, i);
			predict = evaluateModel(copiedClassifier, test, forPredictionsPrinting);
			tenfoldResult[i] = calculateAccuray(predict, test);
			predictions = predictions();
			num_inst+=test.numInstances();
			double auc = areaUnderROC(0,predictions);
			System.out.println(num_inst+","+predictions.size()+","+auc);
		}
		m_NumFolds = numFolds;

		if (classificationOutput != null)
			classificationOutput.printFooter();

	}

	/*
	 * xyzhu
	 * param predict--the predicted result for the test data set
	 * test--the test dataset
	 * return the accuray of the prediction of the test data set
	 */
	private double calculateAccuray(double[] predict, Instances test) {
		double acturalClass[] = test.attributeToDoubleArray(test.classIndex());
		int num_testInstances = test.numInstances();
		int correct = 0;
		for(int i=0;i<num_testInstances;i++){
			if(predict[i]==acturalClass[i]){
				correct++;
			}
		}
		return (double)correct/num_testInstances;
	}

	public double[] getTenfoldResult(){
		return tenfoldResult;
	}

	public double areaUnderROC(int classIndex, FastVector predictions) {

		// Check if any predictions have been collected
		if (predictions == null) {
			return Instance.missingValue();
		} else {
			ThresholdCurve tc = new ThresholdCurve();
			Instances result = tc.getCurve(predictions, classIndex);
			return ThresholdCurve.getROCArea(result);
		}
	}
}
