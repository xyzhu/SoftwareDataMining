package main.java.eden;

import weka.classifiers.Evaluation;
import weka.core.Instances;

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
		CmdLineParser.Option csv_opt = cmdparser.addBooleanOption('c', "readcsv");
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
		Boolean readcsv = (Boolean) cmdparser.getOptionValue(csv_opt, true);
		
		DataReorganize organizer = new DataReorganize();
		Instances data = null;
		String deletemethod = "high-related";
		if(readcsv){
			data = organizer.generateArffFile(filepath, history, project, type, seperator,deletemethod);
		}
		else{
			data = organizer.readArffFile(filepath,history,project,type,seperator);
		}
		Util util = new Util();
//		util.outputCoefficient(data,outputFilePath,projectName);
//		util.outputSelectedAttributes(data, outputFilePath, projectName);
		Classification classification = new Classification();
//		classification.getSpecificClassficationResult(data,outputFilePath,projectName,deletemethod);
		Evaluation eval = classification.StartClassification(data);
		util.saveClassificationResult(eval,filepath,type,seperator,project,history,deletemethod);
	}
		
}
