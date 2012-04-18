package main.java.eden;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;

import weka.classifiers.Evaluation;
import weka.core.Instances;


public class Util {
	
	public void outputSelectedAttributes(Instances data, String outputFilePath,
			String projectName) throws IOException {
		int num_attr = data.numAttributes();
		String attr = projectName+",";
		for(int i=0;i<num_attr;i++)
		{
			attr +=data.attribute(i).name()+ ",";	
		}
		attr += "\n";
		appendResult(attr,outputFilePath+"AttributeSelection/sel_attr.txt");
	}

	public void saveClassificationResult(Evaluation eval,String outputFilePath,String type, int seperator,String projectName,
			String history,String deletemethod) throws IOException {
		String predictResult = "";
		if(projectName.equals("ant")){
			predictResult = "\nJ48-"+deletemethod+"-" +history+"\n";
		}
		
		DecimalFormat format = (DecimalFormat) NumberFormat.getInstance();
		format.applyPattern("0.0000");
		predictResult += projectName +", "+format.format(eval.pctCorrect())+", "
				+ format.format(eval.areaUnderROC(0))+", "
				+ format.format(eval.truePositiveRate(0))+", "
				+ format.format(eval.truePositiveRate(1))+"\n";
		String outputFile = outputFilePath+"Classification/"+type+"_"+String.valueOf(seperator)+".txt";
		appendResult(predictResult,outputFile);
		
	}

	public void outputCoefficient(Instances data, String outputFilePath, String projectName)
			throws Exception {
		String corrResult = "";
		KendallCorrelate kendall = new KendallCorrelate(data, outputFilePath);
		kendall.calculateCoefficientWithClass(data);
		String []attrName = kendall.getAttributeName();
		double []corrs = kendall.getCoefficient();
		double []pvalue = kendall.getPvalue();
		int m_attr = corrs.length;
		for(int i=0;i<m_attr;i++){
			corrResult += attrName[i]+","+String.valueOf(corrs[i])+","+pvalue[i]+"\n";
		}
		Util util = new Util();
		util.saveResult(corrResult,outputFilePath+"Correlation/"+projectName+".txt");
	}
	
	public void saveResult(String result, String file) throws IOException{
		FileWriter fw=new FileWriter(file, false);
		BufferedWriter bw = new BufferedWriter(fw);
		bw.write(result);
		bw.flush();
		bw.close();
	}
	
	public void appendResult(String result, String file) throws IOException{
		FileWriter fa=new FileWriter(file, true);
		BufferedWriter ba = new BufferedWriter(fa);
		ba.write(result);
		ba.flush();
		ba.close();
	}
}
