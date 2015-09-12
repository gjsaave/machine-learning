package assignment1;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.net.URL;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
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
	
	public void FPRDecisionTreeTraining() throws Exception {
		Instances train = returnTrainingSet();
		Instances test = returnTestSet();
		// train classifier
		Classifier cls = new J48();
		PrintWriter out = new PrintWriter("TrainingFPR.dat");
		for (int i=10; i>-1; i--){
			int index = 10-i;
			Instances newData = setPercentDecisionTree(train, i);
			cls.buildClassifier(newData);
			Evaluation eval = new Evaluation(newData);
			eval.evaluateModel(cls, newData);
			out.println(index + "\t" + eval.falsePositiveRate(0));
		}
		out.close();
		
		out = new PrintWriter("TestFPR.dat");
		for (int i=10; i>-1; i--){
			int index = 10-i;
			Instances newData = setPercentDecisionTree(train, i);
			cls.buildClassifier(newData);
			Evaluation eval = new Evaluation(newData);
			eval.evaluateModel(cls, test);
			out.println(index + "\t" + eval.falsePositiveRate(0));
		}
		out.close();

		out = new PrintWriter("TrainingFNR.dat");
		for (int i=10; i>-1; i--){
			int index = 10-i;
			Instances newData = setPercentDecisionTree(train, i);
			cls.buildClassifier(newData);
			Evaluation eval = new Evaluation(newData);
			eval.evaluateModel(cls, newData);
			out.println(index + "\t" + eval.falseNegativeRate(0));
		}
		out.close();

		out = new PrintWriter("TestFNR.dat");
		for (int i=10; i>-1; i--){
			int index = 10-i;
			Instances newData = setPercentDecisionTree(train, i);
			cls.buildClassifier(newData);
			Evaluation eval = new Evaluation(newData);
			eval.evaluateModel(cls, test);
			out.println(index + "\t" + eval.falseNegativeRate(0));
		}
		out.close();	
	}
	
	
	public Instances setPercentDecisionTree(Instances train, int removeAmount) throws Exception {
		// Evaluation eval = decisionTree();
		RemovePercentage remove = new RemovePercentage();
		// System.out.println("Training FPR");
		remove.setInputFormat(train);
		remove.setPercentage(removeAmount * 10);
		Instances newData = Filter.useFilter(train, remove);
		return newData;

	}

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		ChessSet cs = new ChessSet();
		cs.FPRDecisionTreeTraining();

	}

}
