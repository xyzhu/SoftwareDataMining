package main.java.eden;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.CSVLoader;


public class DataReorganize {
	Instances data;

	/*
	 * change class value from number to a or b by seperator
	 */
	public void resetClassValue(int seperator){

		data.deleteAttributeAt(0);
		data.deleteAttributeAt(0);
		data.setClassIndex(-1);		
		double classValue[] = data.attributeToDoubleArray(0);
		data.deleteAttributeAt(0);
		int num_attributes = data.numAttributes();
		int num_instances = data.numInstances();
		FastVector a = new FastVector(2);
		a.addElement("a");
		a.addElement("b");
		Attribute attr = new Attribute("Change",a);
		data.insertAttributeAt(attr, num_attributes);
		for(int i=0;i<num_instances;i++){
			if(classValue[i]<seperator){
				data.instance(i).setValue(num_attributes, "a");
			}
			else{
				data.instance(i).setValue(num_attributes,"b");
			}
		}
		data.setClassIndex(num_attributes);
	}
	/*
	 * change class value from number to a or b by percentage
	 */
	public void resetClassValue() {
		data.deleteAttributeAt(0);
		data.deleteAttributeAt(0);
		data.setClassIndex(-1);	
		int num_instances = data.numInstances();	
		double classValue[] = data.attributeToDoubleArray(0);
		Arrays.sort(classValue);
		double seperator = classValue[(int) (num_instances*0.8)];
		data.deleteAttributeAt(0);
		int num_attributes = data.numAttributes();
		FastVector a = new FastVector(2);
		a.addElement("a");
		a.addElement("b");
		Attribute attr = new Attribute("Change",a);
		data.insertAttributeAt(attr, num_attributes);
		for(int i=0;i<num_instances;i++){
			if(classValue[i]<seperator){
				data.instance(i).setValue(num_attributes, "a");
			}
			else{
				data.instance(i).setValue(num_attributes,"b");
			}
		}
		data.setClassIndex(num_attributes);
	}

	public void deleteAttributeTypes(String method) throws Exception {
		if(method.contains("understand")){
			for(int i=48;i>=39;i--){
				data.deleteAttributeAt(i);
			}
		}
		if(method.contains("statement")){
			for(int i=38;i>=15;i--){
				data.deleteAttributeAt(i);
			}
		}
		if(method.contains("rank")){
			for(int i=14;i>=7;i--){
				data.deleteAttributeAt(i);
			}
		}
		if(method.contains("oldchange")){
			for(int i=6;i>=0;i--){
				data.deleteAttributeAt(i);
			}
		}
	}
	private void selectAttribute(String method) throws Exception{
		if(method.contains("cfs best first")){
			attributeSelectCFS();
		}
		if(method.contains("wrapper best first")){
//			attributeSelectWrapper();
		}		
	}

/*	private void attributeSelectWrapper() throws Exception {
		AttributeSelection fs = new AttributeSelection();
		WrapperSubsetEval evaluator = new WrapperSubsetEval();
		evaluator.setEvaluationMeasure(new SelectedTag(5,WrapperSubsetEval.TAGS_EVALUATION));
		evaluator.setClassifier(new J48());
		fs.setEvaluator(evaluator);
		BestFirst search = new BestFirst();
		fs.setSearch(search);
		fs.setSeed(2);
		fs.SelectAttributes(data);
		data = fs.reduceDimensionality(data);
	}*/

	private void attributeSelectCFS() throws Exception {
		AttributeSelection fs = new AttributeSelection();
		CfsSubsetEval evaluator = new CfsSubsetEval();
		fs.setEvaluator(evaluator);
		BestFirst search = new BestFirst();
		fs.setSearch(search);
		fs.setSeed(2);
		fs.SelectAttributes(data);
		data = fs.reduceDimensionality(data);
	}

	private void deleteHighRelated(String rfilePath) throws Exception {
		KendallCorrelate kendall = new KendallCorrelate(data, rfilePath);
		int tlocIndex = kendall.calculateCoefficientWithTLOC(data);
		double []corrs = kendall.getCoefficient();
		int m_attr = corrs.length;
		for(int i=m_attr-1;i>=0;i--){
			if(corrs[i]>0.9 && i!=tlocIndex){
				data.deleteAttributeAt(i);
			}
		}
	}

	public Instances deleteAttributes(String filepath, String history,
			String project, String type, Integer seperator, String deletemethod) throws Exception {
		String inputfile = filepath+history+"/"+project+"_"+type+".csv";
		data = readCsvFile(inputfile);
		resetClassValue(seperator);
		deleteAttributeTypes(deletemethod);
		if(deletemethod.contains("high-related")&&!deletemethod.contains("understand")){
			deleteHighRelated(filepath);
		}
		selectAttribute(deletemethod);
		return data;
	}

	private Instances readCsvFile(String inputfile) throws IOException {

		File file = new File(inputfile);
		CSVLoader csvLoader = new CSVLoader();
		csvLoader.setSource(file);
		Instances data = csvLoader.getDataSet();
		return data;
	}

}
