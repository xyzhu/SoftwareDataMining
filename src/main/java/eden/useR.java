package main.java.eden;

import java.io.BufferedInputStream;
import java.util.Scanner;

public class useR {
	public static void main(String argv[]) throws Exception{
		System.out.println("good");
		Process p = Runtime.getRuntime().exec("Rscript /home/xyzhu/change-prediction/predict/Correlation/test.R /home/xyzhu/change-prediction/predict/Correlation/test.csv a b");
		  
        BufferedInputStream buf=new BufferedInputStream(p.getInputStream());
        Scanner s=new Scanner(buf);
         
        while(s.hasNextLine()){
          System.out.println(s.nextLine());
        }
	}

}
