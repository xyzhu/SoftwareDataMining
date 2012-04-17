package main.java.eden;
import java.io.*;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.REPTree;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

public class StartMining {

	public static void main(String argv[]) throws Exception{
		String inputFilePath = argv[0];
		String history = argv[1];
		String projectName = argv[2];
		String type = argv[3];
		int seperator = Integer.parseInt(argv[4]);
		String outputFilePath = argv[5];
		String inputFile = inputFilePath+history+"/"+projectName+"_"+type+".csv";//"ant_line.csv";
		File file = new File(inputFile);
		CSVLoader csvLoader = new CSVLoader();
		csvLoader.setSource(file);
		Instances orig_data = csvLoader.getDataSet();
		Instances data = new Instances(orig_data);
		
		//set class value to be a or b
		DataReorganize organizer = new DataReorganize();
		organizer.resetClassValue(data, seperator);
		Util util = new Util();
//		util.outputCoefficient(data,outputFilePath,projectName);
		String deletemethod = "high related";
		data = organizer.deleteAttr(data,deletemethod);
//		util.outputSelectedAttributes(data);
		Evaluation eval = StartClassification(data);
		String outputFile = outputFilePath+"Classification/"+type+"_"+String.valueOf(seperator);
		util.saveClassificationResult(eval,outputFile,projectName,history,deletemethod);
	}
	
	public static Evaluation StartClassification(Instances data) throws Exception {
	
		int randomSeed = 0;
		Bagging classifier = new Bagging();
		classifier.setClassifier(new J48());
		Evaluation evaluation = new Evaluation(data);
		Random rand = new Random(randomSeed);
	
		evaluation.crossValidateModel(classifier, data, 10, rand);
		return evaluation;
		}
		
}
