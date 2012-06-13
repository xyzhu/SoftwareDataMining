package main.java.eden;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

public class StartMining {

	public static void main(String argv[]) throws Exception{
		/**
		 * Command line parsing
		 */
		CmdLineParser cmdparser = new CmdLineParser();
		CmdLineParser.Option file_opt = cmdparser.addStringOption('f', "filepath");
		CmdLineParser.Option history_opt = cmdparser.addStringOption('h', "history");
		CmdLineParser.Option project_opt = cmdparser.addStringOption('p', "project");
		CmdLineParser.Option type_opt = cmdparser.addStringOption('t', "type");
		CmdLineParser.Option seperator_opt = cmdparser.addIntegerOption('s', "seperator");
		CmdLineParser.Option delete_opt = cmdparser.addStringOption('d',"deletemethod");
		try {
			cmdparser.parse(argv);
		} catch (CmdLineParser.OptionException e) {
			System.err.println(e.getMessage());
			//            printUsage();
			System.exit(2);
		}

		String filepath = (String) cmdparser.getOptionValue(file_opt, "/home/xyzhu/change-prediction/predict/");
		String history = (String) cmdparser.getOptionValue(history_opt, "long-history");
		String project = (String) cmdparser.getOptionValue(project_opt, "ant");
		String type = (String) cmdparser.getOptionValue(type_opt, "line");
		Integer seperator = (Integer) cmdparser.getOptionValue(seperator_opt, 5);
		String deletemethod = (String) cmdparser.getOptionValue(delete_opt, "high-related");
		
		DataReorganize organizer = new DataReorganize();
		Instances data = null;
		data = organizer.deleteAttributes(filepath, history, project, type, seperator,deletemethod);
		Filter filter = new Discretize();
		filter.setInputFormat(data);
		data = Filter.useFilter(data, filter);
//		Util util = new Util();
//		util.saveArffFile(data, filepath, history, project, type, deletemethod);
//		util.outputCoefficient(data,filepath,project);
//		util.outputSelectedAttributes(data, filepath, project);
		Classification classification = new Classification();
//		classification.getSpecificClassficationResult(data,filepath,project,deletemethod);
		classification.getClassificationResult(data,filepath,type,seperator,project,history,deletemethod);
	}
		
}
