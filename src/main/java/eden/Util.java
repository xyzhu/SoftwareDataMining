package main.java.eden;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;

import weka.classifiers.Evaluation;
import weka.core.Instances;


public class Util {
	
	public void outputSelectedAttributes(Instances data) {
		int num_attr = data.numAttributes();
		String attr;
		for(int i=0;i<num_attr;i++)
		{
			attr = data.attribute(i).name();
			System.out.print(attr+",");			
		}
		System.out.println();
	}

	public void saveClassificationResult(Evaluation eval,String outputFile,String projectName,
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
		
		FileWriter fa=new FileWriter(outputFile+".txt",true);
		BufferedWriter ba = new BufferedWriter(fa);
		ba.write(predictResult);
		ba.flush();
		ba.close();
		
	}

	public void outputCoefficient(Instances data, String outputFilePath, String projectName)
			throws Exception {
		String corrResult = "";
		KendallCorrelate kendall = new KendallCorrelate(data);
		kendall.calculateCoefficientWithClass(data);
		String []attrName = kendall.getAttributeName();
		double []corrs = kendall.getCoefficient();
		double []pvalue = kendall.getPvalue();
		int m_attr = corrs.length;
		for(int i=0;i<m_attr;i++){
			corrResult += attrName+","+String.valueOf(corrs[i])+","+pvalue[i]+"\n";
		}
		Util util = new Util();
		util.saveResult(corrResult,outputFilePath+"Correlation/"+projectName);
	}
	
	public void saveResult(String result, String file) throws IOException{
		FileWriter fa=new FileWriter(file, false);
		BufferedWriter ba = new BufferedWriter(fa);
		ba.write(result);
		ba.flush();
		ba.close();
	}
}
