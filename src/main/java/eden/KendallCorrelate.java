package main.java.eden;
import java.io.BufferedInputStream;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Scanner;

import weka.core.Instances;

 /* @author Xiaoyan Zhu
 * 2012.4.9
 */
public class KendallCorrelate{
	int m_classIndex = 0;
	int num_attribute = 0;
	int num_instance = 0;
	double coefficient[] = null;
	double pvalue[] = null;
	String attrName[] = null;
	String filePath = "";
	
	public KendallCorrelate(Instances data, String filepath){
		m_classIndex = data.classIndex();
		num_attribute = data.numAttributes()-1;
		num_instance = data.numInstances();
		coefficient = new double[num_attribute];
		pvalue = new double[num_attribute];
		attrName = new String[num_attribute];
		this.filePath = filepath;
	}
	
	public void calculateCoefficientWithClass(Instances data) throws Exception{

		double x[] = null;
		double y[] = data.attributeToDoubleArray(m_classIndex);
		double corr[] = null;
		for(int i=0;i<num_attribute;i++){
			x = data.attributeToDoubleArray(i);
			corr = correlate(x,y);
			coefficient[i] = corr[0];
			pvalue[i] = corr[1];
			attrName[i] = data.attribute(i).name();
			
		}	
	}
	
	public int calculateCoefficientWithTLOC(Instances data) throws Exception{
		int tlocIndex = -1;
		for(int i=0;i<num_attribute;i++){
			if(data.attribute(i).name().equals("CountLine")){
				tlocIndex = i;
				break;
			}
		}
		double x[] = null;
		double y[] = data.attributeToDoubleArray(tlocIndex);
		double corr[] = null;
		for(int i=0;i<num_attribute;i++){
			x = data.attributeToDoubleArray(i);
			corr = correlate(x,y);
			coefficient[i] = corr[0];		
		}
		return tlocIndex;
	}
  
  public double[] correlate (double[] x, double[] y) throws Exception {
	  double corr[] = new double[2]; 
	  String str = "a,b\n";
	  for(int i=0;i<x.length-1;i++){
		  str+=x[i]+","+y[i]+"\n";
	  }
	  str+=x[x.length-1]+","+y[x.length-1];
	  FileWriter fa=new FileWriter(filePath+"Correlation/corr.csv",false);
	  BufferedWriter ba = new BufferedWriter(fa);
	  ba.write(str);
	  ba.flush();
	  ba.close();
	  Process p = Runtime.getRuntime().exec("Rscript "+filePath+"Correlation/test.R "+filePath+"Correlation/corr.csv a b");
	  
      BufferedInputStream buf=new BufferedInputStream(p.getInputStream());
      Scanner s=new Scanner(buf);  
      s.nextLine();
      corr[0] =  Double.valueOf(s.nextLine());
      corr[1] = Double.valueOf(s.nextLine().substring(4));
	  return corr;
  }
  
  public double[] getCoefficient(){
	  return coefficient;
  }
  
  public double[] getPvalue(){
	  return pvalue;
  }
  
  public String[] getAttributeName(){
	  return attrName;
  }
}
