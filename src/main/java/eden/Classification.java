package main.java.eden;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class Classification {

	public Evaluation StartClassification(Instances data) throws Exception {
		int randomSeed = 1;
		Bagging classifier = new Bagging();
		classifier.setClassifier(new J48());
		Evaluation evaluation = new Evaluation(data);
		Random rand = new Random(randomSeed);
	
		evaluation.crossValidateModel(classifier, data, 10, rand);
		return evaluation;
		
		}
	
	public void getSpecificClassficationResult(Instances data,String outputFilePath,
			String projectName,String deletemethod) throws Exception{
		int pass = 1;
		int randomSeed = 0;
		Bagging classifier = new Bagging();
		classifier.setClassifier(new J48());
		MyEvaluation evaluation = new MyEvaluation(data);
		double tenfoldResult[] = new double[10];
		double specificResult[] = new double[pass*10];
		for (int i = 1; i <= pass; i++) {
			randomSeed = i;
			Random rand = new Random(randomSeed);	
			evaluation.crossValidateModel(classifier, data, 10, rand);
			tenfoldResult = evaluation.getTenfoldResult();
			for(int j=0;j<10;j++){
				specificResult[(i-1)*10+j] = tenfoldResult[j];
			}
		}
		
		DecimalFormat format = (DecimalFormat) NumberFormat.getInstance();
		format.applyPattern("0.00");
		String result = deletemethod+"\n";
		for(int m=0;m<10*pass;m++){
			result += String.valueOf(format.format(specificResult[m]))+"\n";
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
			
		Util util = new Util();
		util.saveResult(result, outputFilePath+"AttributeValidation/"+projectName+attrSet+".txt");
	}
}
