package assignment1;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.net.URL;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class ChessSet {

	public Instances returnTrainingSet() throws Exception {
		//String dir = System.getProperty("user.dir") + "chess_king_rook_king_pawn_training.arff";
		URL path = ChessSet.class.getResource("chess_king_rook_king_pawn_training.arff");
		File f = new File(path.getFile());
		BufferedReader reader = new BufferedReader(new FileReader(f));
		Instances data = new Instances(reader);
		reader.close();
		// setting class attribute
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public Instances returnTestSet() throws Exception {
		//String dir = System.getProperty("user.dir") + "chess_king_rook_king_pawn_test.arff";
		URL path = ChessSet.class.getResource("chess_king_rook_king_pawn_test.arff");
		File f = new File(path.getFile());
		BufferedReader reader = new BufferedReader(new FileReader(f));
		Instances data = new Instances(reader);
		reader.close();
		// setting class attribute
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
	public Evaluation returnCrossVal(Classifier cls) throws Exception{
		Instances train = returnTrainingSet();
		cls.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		eval.crossValidateModel(cls, train, 10, new Random(1));
		//PrintWriter out = new PrintWriter("Comp" + "DecTreeTrainingFPR.dat");
		//return eval.falsePositiveRate(0);
		return eval;
	}
	
	public void DecisionTreeTraining() throws Exception {
		Instances train = returnTrainingSet();
		Instances test = returnTestSet();
		// train classifier
		Classifier cls = new J48();
		PrintWriter out = new PrintWriter("DecTreeTrainingFPR.dat");
		for (int i=10; i>-1; i--){
			int index = 10-i;
			Instances newData = setPercent(train, i);
			cls.buildClassifier(newData);
			Evaluation eval = new Evaluation(newData);
			eval.evaluateModel(cls, newData);
			out.println(index*10 + "\t" + eval.falsePositiveRate(0));
		}
		out.close();
		
		out = new PrintWriter("DecTreeTestFPR.dat");
		for (int i=10; i>-1; i--){
			int index = 10-i;
			Instances newData = setPercent(train, i);
			cls.buildClassifier(newData);
			Evaluation eval = new Evaluation(newData);
			eval.evaluateModel(cls, test);
			out.println(index*10 + "\t" + eval.falsePositiveRate(0));
		}
		out.close();

		out = new PrintWriter("DecTreeTrainingFNR.dat");
		for (int i=10; i>-1; i--){
			int index = 10-i;
			Instances newData = setPercent(train, i);
			cls.buildClassifier(newData);
			Evaluation eval = new Evaluation(newData);
			eval.evaluateModel(cls, newData);
			out.println(index*10 + "\t" + eval.falseNegativeRate(0));
		}
		out.close();

		out = new PrintWriter("DecTreeTestFNR.dat");
		for (int i=10; i>-1; i--){
			int index = 10-i;
			Instances newData = setPercent(train, i);
			cls.buildClassifier(newData);
			Evaluation eval = new Evaluation(newData);
			eval.evaluateModel(cls, test);
			out.println(index*10 + "\t" + eval.falseNegativeRate(0));
		}
		out.close();	
	}
	
	public void ANNTraining() throws Exception{
		Instances train = returnTrainingSet();
		Instances test = returnTestSet();
		MultilayerPerceptron mlp = new MultilayerPerceptron(); 
		mlp.setOptions(Utils.splitOptions("-L 0.3 -M 0.2 -N 100 -V 0 -S 0 -E 20 -H 2"));
		PrintWriter out = new PrintWriter("ANNTrainingFPR.dat");
		for (int i=9; i>-1; i--){
			int index = 10-i;
			Instances newData = setPercent(train, i);
			mlp.buildClassifier(newData); 
			Evaluation eval = new Evaluation(newData);
			eval.evaluateModel(mlp, newData);
			out.println(index*10 + "\t" + eval.falsePositiveRate(0));
		}
		out.close();
		
		out = new PrintWriter("ANNTestFPR.dat");
		for (int i=9; i>-1; i--){
			int index = 10-i;
			Instances newData = setPercent(train, i);
			mlp.buildClassifier(newData); 
			Evaluation eval = new Evaluation(newData);
			eval.evaluateModel(mlp, test);
			out.println(index*10 + "\t" + eval.falsePositiveRate(0));
		}
		out.close();
	}
	
	public void IBKTraining() throws Exception{
		Instances train = returnTrainingSet();
		Instances test = returnTestSet();
		IBk ibk = new IBk(3);
		PrintWriter out = new PrintWriter("IBKTrainingFPR.dat");
		for (int i=10; i>-1; i--){
			int index = 10-i;
			Instances newData = setPercent(train, i);
			ibk.buildClassifier(newData); 
			Evaluation eval = new Evaluation(newData);
			eval.evaluateModel(ibk, newData);
			out.println(index*10 + "\t" + eval.falsePositiveRate(0));
		}
		out.close();
		out = new PrintWriter("IBKTestFPR.dat");
		for (int i=10; i>-1; i--){
			int index = 10-i;
			Instances newData = setPercent(train, i);
			ibk.buildClassifier(newData); 
			Evaluation eval = new Evaluation(newData);
			eval.evaluateModel(ibk, test);
			out.println(index*10 + "\t" + eval.falsePositiveRate(0));
		}
		out.close();
	}
	
	public void SMOTrainingPolyKernel() throws Exception{
		Instances train = returnTrainingSet();
		Instances test = returnTestSet();
		SMO smo = new SMO();
		PolyKernel poly = new PolyKernel();
		String[] options = new String[2];
		options[0] = "-E";
		options[1] = "5";
		poly.setOptions(options);
		smo.setKernel(poly);
		PrintWriter out = new PrintWriter("SMOTrainingFPR.dat");
		for (int i=9; i>-1; i--){
			int index = 10-i;
			Instances newData = setPercent(train, i);
			smo.buildClassifier(newData); 
			Evaluation eval = new Evaluation(newData);
			eval.evaluateModel(smo, newData);
			out.println(index*10 + "\t" + eval.falsePositiveRate(0));
		}
		out.close();
		out = new PrintWriter("SMOTestFPR.dat");
		for (int i=9; i>-1; i--){
			int index = 10-i;
			Instances newData = setPercent(train, i);
			smo.buildClassifier(newData); 
			Evaluation eval = new Evaluation(newData);
			eval.evaluateModel(smo, test);
			out.println(index*10 + "\t" + eval.falsePositiveRate(0));
		}
		out.close();
	}
	
	public void SMOTrainingRBFKernel() throws Exception{
		Instances train = returnTrainingSet();
		Instances test = returnTestSet();
		SMO smo = new SMO();
		RBFKernel rbf = new RBFKernel();
		String[] options = new String[2];
		options[0] = "-G";
		options[1] = "0.5";
		rbf.setOptions(options);
		smo.setKernel(rbf);	
		PrintWriter out = new PrintWriter("SMORBFTrainingFPR.dat");
		for (int i=9; i>-1; i--){
			int index = 10-i;
			Instances newData = setPercent(train, i);
			smo.buildClassifier(newData); 
			Evaluation eval = new Evaluation(newData);
			eval.evaluateModel(smo, newData);
			out.println(index*10 + "\t" + eval.falsePositiveRate(0));
		}
		out.close();
		out = new PrintWriter("SMORBFTestFPR.dat");
		for (int i=9; i>-1; i--){
			int index = 10-i;
			Instances newData = setPercent(train, i);
			smo.buildClassifier(newData); 
			Evaluation eval = new Evaluation(newData);
			eval.evaluateModel(smo, test);
			out.println(index*10 + "\t" + eval.falsePositiveRate(0));
		}
		out.close();
	}
	
	public void boosting() throws Exception{
		Instances train = returnTrainingSet();
		Instances test = returnTestSet();
		AdaBoostM1 ada = new AdaBoostM1();
		String[] options = new String[2];
		options[0] = "-W";
		options[1] = "weka.classifiers.trees.J48";
		ada.setOptions(options);
		PrintWriter out = new PrintWriter("ADATrainingFPR.dat");
		for (int i=9; i>-1; i--){
			int index = 10-i;
			Instances newData = setPercent(train, i);
			ada.buildClassifier(newData); 
			Evaluation eval = new Evaluation(newData);
			eval.evaluateModel(ada, newData);
			out.println(index*10 + "\t" + eval.falsePositiveRate(0));
		}
		out.close();
		out = new PrintWriter("ADATestFPR.dat");
		for (int i=9; i>-1; i--){
			int index = 10-i;
			Instances newData = setPercent(train, i);
			ada.buildClassifier(newData); 
			Evaluation eval = new Evaluation(newData);
			eval.evaluateModel(ada, test);
			out.println(index*10 + "\t" + eval.falsePositiveRate(0));
		}
		out.close();
	}
	
	public Instances setPercent(Instances train, int removeAmount) throws Exception {
		// Evaluation eval = decisionTree();
		RemovePercentage remove = new RemovePercentage();
		// System.out.println("Training FPR");
		remove.setInputFormat(train);
		remove.setPercentage(removeAmount * 10);
		Instances newData = Filter.useFilter(train, remove);
		return newData;

	}
	
	public void decisionTreeNodeChanger() throws Exception{
		//PrintWriter out = new PrintWriter("SMORBFTrainingFPR.dat");
		double[] treeSize = new double[10];
		double prevTreeSize = 0;
		int count = 0;
		double[] FPR = new double[10];
		double[] FPRCrossEval = new double[10];
		boolean done = false;
		Integer j = new Integer(0);
		Instances train = returnTrainingSet();
		J48 cls = new J48();
		String[] options = new String[2];
		
		while(count<10){
			j=j+4;
			options[0] = "-M";
			options[1] = j.toString();
			System.out.println(options[1]);
			cls.setOptions(options);
			cls.buildClassifier(train); 
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(cls, train);
			Evaluation crossEval = returnCrossVal(cls);
			crossEval.evaluateModel(cls, train);
	
			if (prevTreeSize != cls.measureTreeSize()){
				treeSize[count] = cls.measureTreeSize();
				FPR[count] = eval.falsePositiveRate(0);
				FPRCrossEval[count] = crossEval.falsePositiveRate(0);
				prevTreeSize = cls.measureTreeSize();
				count++;
			}
			
			if (count == 9 && !done){
				PrintWriter out = new PrintWriter("CompDecTreeTrainingFPR.dat");
				for (int i=9; i>-1; i--){	
					out.println(treeSize[i] + "\t" + FPR[i]);
				}
				out.close();
				out = new PrintWriter("CompDecTreeTestFPR.dat");
				for (int i=9; i>-1; i--){	
					out.println(treeSize[i] + "\t" + FPRCrossEval[i]);
				}
				out.close();
				done = true;
			}	
		}			
	}

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		ChessSet cs = new ChessSet();
		//cs.DecisionTreeTraining();
		//cs.ANNTraining();
		//cs.IBKTraining();
		//cs.SMOTrainingPolyKernel();
		//cs.SMOTrainingRBFKernel();
		//cs.boosting();
		cs.decisionTreeNodeChanger();

	}

}
