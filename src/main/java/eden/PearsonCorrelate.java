package main.java.eden;
import weka.core.Attribute;
import weka.core.ContingencyTables;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

 /* @author Xiaoyan Zhu
 * 2012.4.9
 */
public class PearsonCorrelate{
	
	//is the class numric
	public boolean m_isNumeric;
	public int m_numInstances;
	public Instances m_trainInstances;
	public int m_classIndex;
	private boolean m_missingSeperate;
	
	public PearsonCorrelate(Instances data){
		m_isNumeric = true;
		m_numInstances = data.numInstances();
		m_trainInstances = data;
		m_classIndex = data.classIndex();
		m_missingSeperate = false;
	}
	
	public float[] calculate(int locindex){
		int m_numAttr = m_trainInstances.numAttributes() - 1;
		float corr [] = new float[m_numAttr];
		for(int i=0;i<m_numAttr;i++){
			corr[i] = correlate(i,locindex);
		}
		return corr;
	}
  
  public float correlate (int att1, int att2) {
    if (!m_isNumeric) {
      return  (float) symmUncertCorr(att1, att2);
    }

    boolean att1_is_num = (m_trainInstances.attribute(att1).isNumeric());
    boolean att2_is_num = (m_trainInstances.attribute(att2).isNumeric());

    if (att1_is_num && att2_is_num) {
      return  (float) num_num(att1, att2);
    }
    else {if (att2_is_num) {
      return  (float) num_nom2(att1, att2);
    }
    else {if (att1_is_num) {
      return  (float) num_nom2(att2, att1);
    }
    }
    }

    return (float) nom_nom(att1, att2);
  }

  private double num_num (int att1, int att2) {
    int i;
    Instance inst;
    double r, diff1, diff2, num = 0.0, sx = 0.0, sy = 0.0;
    double mx = m_trainInstances.meanOrMode(m_trainInstances.attribute(att1));
    double my = m_trainInstances.meanOrMode(m_trainInstances.attribute(att2));

    for (i = 0; i < m_numInstances; i++) {
      inst = m_trainInstances.instance(i);
      diff1 = (inst.isMissing(att1))? 0.0 : (inst.value(att1) - mx);
      diff2 = (inst.isMissing(att2))? 0.0 : (inst.value(att2) - my);
      num += (diff1*diff2);
      sx += (diff1*diff1);
      sy += (diff2*diff2);
    }
/*
    if (sx != 0.0) {
      if (m_std_devs[att1] == 1.0) {
        m_std_devs[att1] = Math.sqrt((sx/m_numInstances));
      }
    }

    if (sy != 0.0) {
      if (m_std_devs[att2] == 1.0) {
        m_std_devs[att2] = Math.sqrt((sy/m_numInstances));
      }
    }
*/
    if ((sx*sy) > 0.0) {
      r = (num/(Math.sqrt(sx*sy)));
      return  ((r < 0.0)? -r : r);
    }
    else {
      if (att1 != m_classIndex && att2 != m_classIndex) {
        return  1.0;
      }
      else {
        return  0.0;
      }
    }
  }


  private double num_nom2 (int att1, int att2) {
    int i, ii, k;
    double temp;
    Instance inst;
    int mx = (int)m_trainInstances.
      meanOrMode(m_trainInstances.attribute(att1));
    double my = m_trainInstances.
      meanOrMode(m_trainInstances.attribute(att2));
    double stdv_num = 0.0;
    double diff1, diff2;
    double r = 0.0, rr;
    int nx = (!m_missingSeperate) 
      ? m_trainInstances.attribute(att1).numValues() 
      : m_trainInstances.attribute(att1).numValues() + 1;

    double[] prior_nom = new double[nx];
    double[] stdvs_nom = new double[nx];
    double[] covs = new double[nx];

    for (i = 0; i < nx; i++) {
      stdvs_nom[i] = covs[i] = prior_nom[i] = 0.0;
    }

    // calculate frequencies (and means) of the values of the nominal 
    // attribute
    for (i = 0; i < m_numInstances; i++) {
      inst = m_trainInstances.instance(i);

      if (inst.isMissing(att1)) {
        if (!m_missingSeperate) {
          ii = mx;
        }
        else {
          ii = nx - 1;
        }
      }
      else {
        ii = (int)inst.value(att1);
      }

      // increment freq for nominal
      prior_nom[ii]++;
    }

    for (k = 0; k < m_numInstances; k++) {
      inst = m_trainInstances.instance(k);
      // std dev of numeric attribute
      diff2 = (inst.isMissing(att2))? 0.0 : (inst.value(att2) - my);
      stdv_num += (diff2*diff2);

      // 
      for (i = 0; i < nx; i++) {
        if (inst.isMissing(att1)) {
          if (!m_missingSeperate) {
            temp = (i == mx)? 1.0 : 0.0;
          }
          else {
            temp = (i == (nx - 1))? 1.0 : 0.0;
          }
        }
        else {
          temp = (i == inst.value(att1))? 1.0 : 0.0;
        }

        diff1 = (temp - (prior_nom[i]/m_numInstances));
        stdvs_nom[i] += (diff1*diff1);
        covs[i] += (diff1*diff2);
      }
    }

    // calculate weighted correlation
    for (i = 0, temp = 0.0; i < nx; i++) {
      // calculate the weighted variance of the nominal
      temp += ((prior_nom[i]/m_numInstances)*(stdvs_nom[i]/m_numInstances));

      if ((stdvs_nom[i]*stdv_num) > 0.0) {
        //System.out.println("Stdv :"+stdvs_nom[i]);
        rr = (covs[i]/(Math.sqrt(stdvs_nom[i]*stdv_num)));

        if (rr < 0.0) {
          rr = -rr;
        }

        r += ((prior_nom[i]/m_numInstances)*rr);
      }
      /* if there is zero variance for the numeric att at a specific 
         level of the catergorical att then if neither is the class then 
         make this correlation at this level maximally bad i.e. 1.0. 
         If either is the class then maximally bad correlation is 0.0 */
      else {if (att1 != m_classIndex && att2 != m_classIndex) {
        r += ((prior_nom[i]/m_numInstances)*1.0);
      }
      }
    }

    /*
    // set the standard deviations for these attributes if necessary
    // if ((att1 != classIndex) && (att2 != classIndex)) // =============
    if (temp != 0.0) {
      if (m_std_devs[att1] == 1.0) {
        m_std_devs[att1] = Math.sqrt(temp);
      }
    }

    if (stdv_num != 0.0) {
      if (m_std_devs[att2] == 1.0) {
        m_std_devs[att2] = Math.sqrt((stdv_num/m_numInstances));
      }
    }
*/
    if (r == 0.0) {
      if (att1 != m_classIndex && att2 != m_classIndex) {
        r = 1.0;
      }
    }

    return  r;
  }


  private double nom_nom (int att1, int att2) {
    int i, j, ii, jj, z;
    double temp1, temp2;
    Instance inst;
    int mx = (int)m_trainInstances.
      meanOrMode(m_trainInstances.attribute(att1));
    int my = (int)m_trainInstances.
      meanOrMode(m_trainInstances.attribute(att2));
    double diff1, diff2;
    double r = 0.0, rr;
    int nx = (!m_missingSeperate) 
      ? m_trainInstances.attribute(att1).numValues() 
      : m_trainInstances.attribute(att1).numValues() + 1;

    int ny = (!m_missingSeperate)
      ? m_trainInstances.attribute(att2).numValues() 
      : m_trainInstances.attribute(att2).numValues() + 1;

    double[][] prior_nom = new double[nx][ny];
    double[] sumx = new double[nx];
    double[] sumy = new double[ny];
    double[] stdvsx = new double[nx];
    double[] stdvsy = new double[ny];
    double[][] covs = new double[nx][ny];

    for (i = 0; i < nx; i++) {
      sumx[i] = stdvsx[i] = 0.0;
    }

    for (j = 0; j < ny; j++) {
      sumy[j] = stdvsy[j] = 0.0;
    }

    for (i = 0; i < nx; i++) {
      for (j = 0; j < ny; j++) {
        covs[i][j] = prior_nom[i][j] = 0.0;
      }
    }

    // calculate frequencies (and means) of the values of the nominal 
    // attribute
    for (i = 0; i < m_numInstances; i++) {
      inst = m_trainInstances.instance(i);

      if (inst.isMissing(att1)) {
        if (!m_missingSeperate) {
          ii = mx;
        }
        else {
          ii = nx - 1;
        }
      }
      else {
        ii = (int)inst.value(att1);
      }

      if (inst.isMissing(att2)) {
        if (!m_missingSeperate) {
          jj = my;
        }
        else {
          jj = ny - 1;
        }
      }
      else {
        jj = (int)inst.value(att2);
      }

      // increment freq for nominal
      prior_nom[ii][jj]++;
      sumx[ii]++;
      sumy[jj]++;
    }

    for (z = 0; z < m_numInstances; z++) {
      inst = m_trainInstances.instance(z);

      for (j = 0; j < ny; j++) {
        if (inst.isMissing(att2)) {
          if (!m_missingSeperate) {
            temp2 = (j == my)? 1.0 : 0.0;
          }
          else {
            temp2 = (j == (ny - 1))? 1.0 : 0.0;
          }
        }
        else {
          temp2 = (j == inst.value(att2))? 1.0 : 0.0;
        }

        diff2 = (temp2 - (sumy[j]/m_numInstances));
        stdvsy[j] += (diff2*diff2);
      }

      // 
      for (i = 0; i < nx; i++) {
        if (inst.isMissing(att1)) {
          if (!m_missingSeperate) {
            temp1 = (i == mx)? 1.0 : 0.0;
          }
          else {
            temp1 = (i == (nx - 1))? 1.0 : 0.0;
          }
        }
        else {
          temp1 = (i == inst.value(att1))? 1.0 : 0.0;
        }

        diff1 = (temp1 - (sumx[i]/m_numInstances));
        stdvsx[i] += (diff1*diff1);

        for (j = 0; j < ny; j++) {
          if (inst.isMissing(att2)) {
            if (!m_missingSeperate) {
              temp2 = (j == my)? 1.0 : 0.0;
            }
            else {
              temp2 = (j == (ny - 1))? 1.0 : 0.0;
            }
          }
          else {
            temp2 = (j == inst.value(att2))? 1.0 : 0.0;
          }

          diff2 = (temp2 - (sumy[j]/m_numInstances));
          covs[i][j] += (diff1*diff2);
        }
      }
    }

    // calculate weighted correlation
    for (i = 0; i < nx; i++) {
      for (j = 0; j < ny; j++) {
        if ((stdvsx[i]*stdvsy[j]) > 0.0) {
          //System.out.println("Stdv :"+stdvs_nom[i]);
          rr = (covs[i][j]/(Math.sqrt(stdvsx[i]*stdvsy[j])));

          if (rr < 0.0) {
            rr = -rr;
          }

          r += ((prior_nom[i][j]/m_numInstances)*rr);
        }
        // if there is zero variance for either of the categorical atts then if
        // neither is the class then make this
        // correlation at this level maximally bad i.e. 1.0. If either is 
        // the class then maximally bad correlation is 0.0
        else {if (att1 != m_classIndex && att2 != m_classIndex) {
          r += ((prior_nom[i][j]/m_numInstances)*1.0);
        }
        }
      }
    }
/*
    // calculate weighted standard deviations for these attributes
    // (if necessary)
    for (i = 0, temp1 = 0.0; i < nx; i++) {
      temp1 += ((sumx[i]/m_numInstances)*(stdvsx[i]/m_numInstances));
    }

    if (temp1 != 0.0) {
      if (m_std_devs[att1] == 1.0) {
        m_std_devs[att1] = Math.sqrt(temp1);
      }
    }

    for (j = 0, temp2 = 0.0; j < ny; j++) {
      temp2 += ((sumy[j]/m_numInstances)*(stdvsy[j]/m_numInstances));
    }

    if (temp2 != 0.0) {
      if (m_std_devs[att2] == 1.0) {
        m_std_devs[att2] = Math.sqrt(temp2);
      }
    }
*/
    
    if (r == 0.0) {
      if (att1 != m_classIndex && att2 != m_classIndex) {
        r = 1.0;
      }
    }

    return  r;
  }

  private double symmUncertCorr (int att1, int att2) {
	    int i, j, k, ii, jj;
	    int ni, nj;
	    double sum = 0.0;
	    double sumi[], sumj[];
	    double counts[][];
	    Instance inst;
	    double corr_measure;
	    boolean flag = false;
	    double temp = 0.0;

	    if (att1 == m_classIndex || att2 == m_classIndex) {
	      flag = true;
	    }

	    ni = m_trainInstances.attribute(att1).numValues() + 1;
	    nj = m_trainInstances.attribute(att2).numValues() + 1;
	    counts = new double[ni][nj];
	    sumi = new double[ni];
	    sumj = new double[nj];

	    for (i = 0; i < ni; i++) {
	      sumi[i] = 0.0;

	      for (j = 0; j < nj; j++) {
	        sumj[j] = 0.0;
	        counts[i][j] = 0.0;
	      }
	    }

	    // Fill the contingency table
	    for (i = 0; i < m_numInstances; i++) {
	      inst = m_trainInstances.instance(i);

	      if (inst.isMissing(att1)) {
	        ii = ni - 1;
	      }
	      else {
	        ii = (int)inst.value(att1);
	      }

	      if (inst.isMissing(att2)) {
	        jj = nj - 1;
	      }
	      else {
	        jj = (int)inst.value(att2);
	      }

	      counts[ii][jj]++;
	    }

	    // get the row totals
	    for (i = 0; i < ni; i++) {
	      sumi[i] = 0.0;

	      for (j = 0; j < nj; j++) {
	        sumi[i] += counts[i][j];
	        sum += counts[i][j];
	      }
	    }

	    // get the column totals
	    for (j = 0; j < nj; j++) {
	      sumj[j] = 0.0;

	      for (i = 0; i < ni; i++) {
	        sumj[j] += counts[i][j];
	      }
	    }

	    // distribute missing counts
	    if (!m_missingSeperate && 
	        (sumi[ni-1] < m_numInstances) && 
	        (sumj[nj-1] < m_numInstances)) {
	      double[] i_copy = new double[sumi.length];
	      double[] j_copy = new double[sumj.length];
	      double[][] counts_copy = new double[sumi.length][sumj.length];

	      for (i = 0; i < ni; i++) {
	        System.arraycopy(counts[i], 0, counts_copy[i], 0, sumj.length);
	      }

	      System.arraycopy(sumi, 0, i_copy, 0, sumi.length);
	      System.arraycopy(sumj, 0, j_copy, 0, sumj.length);
	      double total_missing = 
	        (sumi[ni - 1] + sumj[nj - 1] - counts[ni - 1][nj - 1]);

	      // do the missing i's
	      if (sumi[ni - 1] > 0.0) {
	        for (j = 0; j < nj - 1; j++) {
	          if (counts[ni - 1][j] > 0.0) {
	            for (i = 0; i < ni - 1; i++) {
	              temp = ((i_copy[i]/(sum - i_copy[ni - 1]))*counts[ni - 1][j]);
	              counts[i][j] += temp;
	              sumi[i] += temp;
	            }

	            counts[ni - 1][j] = 0.0;
	          }
	        }
	      }

	      sumi[ni - 1] = 0.0;

	      // do the missing j's
	      if (sumj[nj - 1] > 0.0) {
	        for (i = 0; i < ni - 1; i++) {
	          if (counts[i][nj - 1] > 0.0) {
	            for (j = 0; j < nj - 1; j++) {
	              temp = ((j_copy[j]/(sum - j_copy[nj - 1]))*counts[i][nj - 1]);
	              counts[i][j] += temp;
	              sumj[j] += temp;
	            }

	            counts[i][nj - 1] = 0.0;
	          }
	        }
	      }

	      sumj[nj - 1] = 0.0;

	      // do the both missing
	      if (counts[ni - 1][nj - 1] > 0.0 && total_missing != sum) {
	        for (i = 0; i < ni - 1; i++) {
	          for (j = 0; j < nj - 1; j++) {
	            temp = (counts_copy[i][j]/(sum - total_missing)) * 
	              counts_copy[ni - 1][nj - 1];
	            
	            counts[i][j] += temp;
	            sumi[i] += temp;
	            sumj[j] += temp;
	          }
	        }

	        counts[ni - 1][nj - 1] = 0.0;
	      }
	    }

	    corr_measure = ContingencyTables.symmetricalUncertainty(counts);

	    if (Utils.eq(corr_measure, 0.0)) {
	      if (flag == true) {
	        return  (0.0);
	      }
	      else {
	        return  (1.0);
	      }
	    }
	    else {
	      return  (corr_measure);
	    }
	  }

}
