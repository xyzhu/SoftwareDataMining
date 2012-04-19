package main.java.eden;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instances;

public class Classification {

	public Evaluation StartClassification(Instances data) throws Exception {
		int randomSeed = 0;
		Bagging classifier = new Bagging();
		classifier.setClassifier(new J48());
		Evaluation evaluation = new Evaluation(data);
		Random rand = new Random(randomSeed);
	
		evaluation.crossValidateModel(classifier, data, 10, rand);
		return evaluation;
		
		}
	
	public void getSpecificClassficationResult(Instances data,String outputFilePath,
			String projectName,String deletemethod) throws Exception{
		int pass = 10;
		int randomSeed = 0;
		Bagging classifier = new Bagging();
		classifier.setClassifier(new J48());
		MyEvaluation evaluation = new MyEvaluation(data);
		FastVector result = new FastVector();
		FastVector finalresult = new FastVector();
		for (int i = 0; i < pass; i++) {
			randomSeed = i;
			Random rand = new Random(randomSeed);	
			evaluation.crossValidateModel(classifier, data, 10, rand);
			result = evaluation.getCrossValidateResult();
			for(int j=0;j<result.size();j++){
				finalresult.addElement(result.elementAt(j));
			}
		}
		
		String str_result = "accuracy,auc,recall1,recall2\n";
		int size = finalresult.size();
		ClassificationResult cr = new ClassificationResult();
		for(int m=0;m<size;m++){
			cr = (ClassificationResult)(finalresult.elementAt(m));
			str_result += doubleformat(cr.accuracy)+","+doubleformat(cr.auc)+
					","+doubleformat(cr.recall1)+","+doubleformat(cr.recall2)+"\n";
		}
		String attrSet = "";
		if(deletemethod.contains("rank,statement")){
			attrSet = "1";
		}
		else if(deletemethod.contains("rank")){
			attrSet = "2";
		}
		else if(deletemethod.contains("statement")){
			attrSet = "3";
		}
		else if(deletemethod.contains("oldchange,understand")){
			attrSet = "4";
;		}
		Util util = new Util();
		util.saveResult(str_result, outputFilePath+"AttributeValidation/"+projectName+attrSet+".csv");
	}
	public String doubleformat(double d){
		DecimalFormat format = (DecimalFormat) NumberFormat.getInstance();
		format.applyPattern("0.00");
		String s = String.valueOf(format.format(d));
		return s;
	}
}
