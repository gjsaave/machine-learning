1.  Setup IDE to allow weka imports

2.  Code is included that breaks down training data into percentage increments, builds the classifier, and applies the classifier to a separate test set.  The results are output.

3.  You will need GNUPlot to plot the data.  It can be found at http://sourceforge.net/projects/gnuplot/?source=typ_redirect

4.  Run the code and it will generate .dat files for each algorithms training and test set for False negative rate and false positive rate.  These .dat files can be plugged into GNUPlot.    

5.  In the GNUPlot cmd line cd to the directory containing the .dat files.   Enter the following command:

	plot 'TrainingFPR.dat' u 1:2 smooth bezier title "Training Set",'TestFPR.dat' u 1:2 smooth bezier title "Test Set"
	
	where TrainingFPR.dat and TestFPR.dat are the names of your .dat files containing the training and test sets
	
6.	