package main.java.eden;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;


public class DataReorganize {

	public void resetClassValue(Instances data, int seperator){

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

	public Instances deleteAttr(Instances data,String method) throws Exception {
		if(method.contains("understand")){
			deleteUnderstand(data);
		}
		if(method.contains("statement")){
			deleteStatement(data);
		}
		if(method.contains("rank")){
			deleteRank(data);
		}
		if(method.contains("preciouschange")){
			deletePreviouschange(data);
		}
		if(method.contains("high related")){
			deleteHighRelated(data);
		}
		
		if(method.contains("cfs best first")){
			data = attributeSelectCFS(data);
		}
		if(method.contains("wrapper best first")){
			data = attributeSelectWrapper(data);
		}
		return data;
		
	}

	private void deleteUnderstand(Instances data) {
		for(int i=80;i>=44;i--){
			data.deleteAttributeAt(i);
		}		
	}

	private void deleteStatement(Instances data) {
		for(int i=43;i>=17;i--){
			data.deleteAttributeAt(i);
		}
	}

	private void deleteRank(Instances data) {
		for(int i=16;i>=7;i--){
			data.deleteAttributeAt(i);
		}
	}

	private void deletePreviouschange(Instances data) {
		for(int i=6;i>=0;i--){
			data.deleteAttributeAt(i);
		}
	}

	private Instances attributeSelectWrapper(Instances data) throws Exception {
		AttributeSelection fs = new AttributeSelection();
		WrapperSubsetEval evaluator = new WrapperSubsetEval();
//		evaluator.setEvaluationMeasure(new SelectedTag(5,WrapperSubsetEval.TAGS_EVALUATION));
		evaluator.setClassifier(new J48());
		fs.setEvaluator(evaluator);
		BestFirst search = new BestFirst();
		fs.setSearch(search);
		fs.setSeed(2);
		fs.SelectAttributes(data);
		data = fs.reduceDimensionality(data);
		return data;
	}
	
	private Instances attributeSelectCFS(Instances data) throws Exception {
		AttributeSelection fs = new AttributeSelection();
		CfsSubsetEval evaluator = new CfsSubsetEval();
		fs.setEvaluator(evaluator);
		BestFirst search = new BestFirst();
		fs.setSearch(search);
		fs.setSeed(2);
		fs.SelectAttributes(data);
		data = fs.reduceDimensionality(data);
		return data;
	}

	private void deleteHighRelated(Instances data) throws Exception {
		KendallCorrelate kendall = new KendallCorrelate(data);
		kendall.calculateCoefficientWithTLOC(data);
		double []corrs = kendall.getCoefficient();
		int m_attr = corrs.length;
		for(int i=m_attr-1;i>=0;i--){
			if(corrs[i]>0.9){
				data.deleteAttributeAt(i);
			}
		}
	}

}
