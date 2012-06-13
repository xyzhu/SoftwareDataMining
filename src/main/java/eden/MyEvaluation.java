package main.java.eden;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.ThresholdCurve;
//import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;


public class MyEvaluation extends Evaluation{

	FastVector crs = null;
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
/*		AbstractOutput classificationOutput = null;
		if (forPredictionsPrinting.length > 0) {
			// print the header first
			classificationOutput = (AbstractOutput) forPredictionsPrinting[0];
			classificationOutput.setHeader(data);
			classificationOutput.printHeader();
		}*/
		
		int num_inst = 0;
		double num_correct = 0;
		double num_tp1 = 0;
		double num_tp2 = 0;
		int numclass1 = 0;
		int numclass2 = 0;
		NominalPrediction np = null;
		FastVector predictions = null;
		FastVector cur_predictions = new FastVector();
		ClassificationResult cr = null;
		crs = new FastVector();
		// Do the folds
		for (int i = 0; i < numFolds; i++) {
			Instances train = data.trainCV(numFolds, i, random);
			setPriors(train);
			Classifier copiedClassifier = Classifier.makeCopy(classifier);
			copiedClassifier.buildClassifier(train);
			Instances test = data.testCV(numFolds, i);
			//evaluateModel(copiedClassifier, test, forPredictionsPrinting);
			evaluateModel(copiedClassifier, test);
			predictions = predictions();
			num_inst = test.numInstances();
			cr = new ClassificationResult();
			for(int n=predictions.size()-num_inst;n<predictions.size();n++){
				cur_predictions.addElement(predictions.elementAt(n));
			}
			cr.setAuc(areaUnderROC(0,cur_predictions));
			for(int n=0;n<cur_predictions.size();n++){
				np = (NominalPrediction)cur_predictions.elementAt(n);
				if(np.actual()==0){
					numclass1++;
				}
				else{
					numclass2++;
				}
				if(np.actual()==np.predicted()){
					num_correct++;
					if(np.actual()==0){
						num_tp1++;
					}
					else{
						num_tp2++;
					}
				}
			}
			cr.setAccuracy(num_correct/num_inst);
			cr.setRecall1(num_tp1/numclass1);
			cr.setRecall2(num_tp2/numclass2);
			crs.addElement(cr);
//			System.out.println(cr.accuracy+","+cr.auc+","+cr.recall1+","+cr.recall2);
			numclass1 = 0;
			numclass2 = 0;
			cur_predictions.removeAllElements();
			num_correct = 0;
			num_tp1 = 0;
			num_tp2 = 0;
		}
		m_NumFolds = numFolds;

/*		if (classificationOutput != null)
			classificationOutput.printFooter();*/

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
	
	public FastVector getCrossValidateResult(){
		return crs;
	}
}
