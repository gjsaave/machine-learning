package assignment1;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
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

	public void FPRDecisionTreeTraining(Instances newData) throws Exception {
		Instances train = newData;
		//Instances test = returnTestSet();
		// train classifier
		Classifier cls = new J48();
		cls.buildClassifier(train);
		// evaluate classifier and print some statistics
		Evaluation eval = new Evaluation(train);
		//Evaluation eval_train = new Evaluation(train);
		eval.evaluateModel(cls, train);
		//eval_train.evaluateModel(cls, train);
		//System.out.println("Training FPR");
		//System.out.println(eval_train.falsePositiveRate(0));
		//System.out.println("Test FPR");
		System.out.println(eval.falsePositiveRate(0));

	}
	
	public void FPRDecisionTreeTest(Instances newData) throws Exception {
		Instances train = newData;
		Instances test = returnTestSet();
		// train classifier
		Classifier cls = new J48();
		cls.buildClassifier(train);
		// evaluate classifier and print some statistics
		Evaluation eval = new Evaluation(train);
		Evaluation eval_train = new Evaluation(train);
		eval.evaluateModel(cls, test);
		eval_train.evaluateModel(cls, train);
		//System.out.println("Training FPR");
		//System.out.println(eval_train.falsePositiveRate(0));
		//System.out.println("Test FPR");
		System.out.println(eval.falsePositiveRate(0));

	}
	
	public void FNRDecisionTreeTraining(Instances newData) throws Exception {
		Instances train = newData;
		//Instances test = returnTestSet();
		// train classifier
		Classifier cls = new J48();
		cls.buildClassifier(train);
		// evaluate classifier and print some statistics
		Evaluation eval = new Evaluation(train);
		//Evaluation eval_train = new Evaluation(train);
		eval.evaluateModel(cls, train);
		//eval_train.evaluateModel(cls, train);
		//System.out.println("Training FPR");
		//System.out.println(eval_train.falsePositiveRate(0));
		//System.out.println("Test FPR");
		System.out.println(eval.falsePositiveRate(0));

	}
	
	public void FNRDecisionTreeTest(Instances newData) throws Exception {
		Instances train = newData;
		Instances test = returnTestSet();
		// train classifier
		Classifier cls = new J48();
		cls.buildClassifier(train);
		// evaluate classifier and print some statistics
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(cls, test);
		//System.out.println("Test FNR");
		System.out.println(eval.falseNegativeRate(0));

	}
	
	public void setPercentDecisionTree() throws Exception{
		//Evaluation eval = decisionTree();
		Instances train = returnTrainingSet();
		RemovePercentage remove = new RemovePercentage();
		System.out.println("Training FPR");
		for (int i=10; i>-1; i--){
			remove.setInputFormat(train);
			remove.setPercentage(i*10);
			Instances newData = Filter.useFilter(train, remove);
			FPRDecisionTreeTraining(newData);	
		}
		
		System.out.println("Test FPR");
		for (int i=10; i>-1; i--){
			remove.setInputFormat(train);
			remove.setPercentage(i*10);
			Instances newData = Filter.useFilter(train, remove);
			FPRDecisionTreeTest(newData);	
		}
		
		System.out.println("Training FNR");
		for (int i=10; i>-1; i--){
			remove.setInputFormat(train);
			remove.setPercentage(i*10);
			Instances newData = Filter.useFilter(train, remove);
			FNRDecisionTreeTraining(newData);	
		}
		
		System.out.println("Test FNR");
		for (int i=10; i>-1; i--){
			remove.setInputFormat(train);
			remove.setPercentage(i*10);
			Instances newData = Filter.useFilter(train, remove);
			FNRDecisionTreeTest(newData);	
		}
	}

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		ChessSet cs = new ChessSet();
		cs.setPercentDecisionTree();

	}

}
