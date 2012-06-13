package main.java.eden;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
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

	public void outputCoefficient(Instances data, String outputFilePath, String projectName)
			throws Exception {
		String corrResult = "attr,corr,pvalue\n";
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
		util.saveResult(corrResult,outputFilePath+"Correlation/"+projectName+".csv");
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
	
	public void saveArffFile(Instances data,String filepath, String history,
			String project, String type, String deletemethod) throws IOException {
		
		String outputfile = filepath+history+"/"+project+"_"+type+".arff";
		saveResult(data.toString(), outputfile);
	}
}
