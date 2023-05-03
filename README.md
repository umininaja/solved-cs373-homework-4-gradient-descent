Download Link: https://assignmentchef.com/product/solved-cs373-homework-4-gradient-descent
<br>
In this homework, you will be working with the same ‘Review Polarity’ dataset that was introduced for the previous homework. You are given movie reviews and the task is to classify them as positive or negative. The entire dataset has 1000 movie reviews, out of which 500 are labeled positive and the other 500 are labeled as negative. The dataset is split into training (train.tsv) and test (test.tsv) sets. Training set consists of 699 instances and the test set has 301 instances. Look into the training set file and try to gauge the nature of the data and the task.

Your submission will be tested on a hidden dataset which is different from given dataset. The hidden set would have similar number of instances and data splits.

<strong>Input: </strong>Movie review consisting of a few sentences of text.

<strong>Label: </strong>Positive (+1)/Negative (−1)

The features you will use for the classification task are bag-of-words. Provided skeleton code has 2 feature extractors already implemented. The first feature extractor extracts binary word presence/absence and the second feature extractor extracts word frequency. You are free to implement and add any features that you believe would help you get a better performance on the task, but you can get full credit without using any extra features. Top-10% submissions according to the performance on the hidden test set will be given extra credit. You might need to use additional features to get extra credit. You are allowed to use libraries such as NLTK or Spacy to extract those features.

<h2>1.2      Skeleton Code</h2>

You are provided a skeleton code with this homework. The code has the following folder structure:

cs373-hw4/ handout.pdf data/ given/ train.tsv test.tsv

src/ init .py main sgd.py main bgd.py classifier.py utils.py config.py bgd.py sgd.py report/ report.pdf

<strong>Do not </strong>modify the given folder structure. You should not need to, but you may, add any extra files of code in cs373-hw4/src/ folder. You should place your report (report.pdf) in the cs373-hw4/report/ folder. You <strong>must not </strong>modify init .py, main sgd.py, main bgd.py or classifier.py. They may be replaced while grading. All your coding work for this homework must be done inside cs373-hw4/src/ folder. You are not allowed to use any external libraries for classifier implementations such as scikit-learn etc. Make sure that you understand the given code fully, before you start coding.

Your code will be tested using the command python3 main bgd.py and python3 main sgd.py from inside the cs373-hw4/src/ folder for SGD and BGD evaluation. Make sure that you test your code on data.cs.purdue.edu before submitting.

<h2>1.3      Log Loss</h2>

The log loss function is defined as,

<em>L</em>(<em>x<sub>i</sub>,y<sub>i</sub></em>;<em>w</em>) = <sup>−X</sup><em>y<sub>i</sub>log</em>(<em>g</em>(<em>z<sub>i</sub></em>)) + (1 − <em>y<sub>i</sub></em>)<em>log</em>(1 − <em>g</em>(<em>z<sub>i</sub></em>))

<em>i</em>

<em>g</em>(<em>z<sub>i</sub></em>) = 1<em>/</em>(1 + <em>e</em><sup>−<em>z</em></sup><em><sup>i</sup></em>)<em>,           z<sub>i </sub></em>= <em>w<sup>T</sup>x<sub>i </sub></em>+ <em>b</em>

Note that here y can take 1 or 0

<h2>1.4      Hinge Loss</h2>

The hinge loss function is defined as,

<em>L</em>(<em>x<sub>i</sub>,y<sub>i</sub></em>;<em>w</em>) = <em>max</em>(0<em>,</em>1 − <em>y<sub>i</sub>f</em>(<em>x<sub>i</sub></em>)) <em>f</em>(<em>x<sub>i</sub></em>) = <em>w<sup>T</sup>x<sub>i </sub></em>+ <em>b</em>

Note that here y can take +1 or -1

<h1>2       Theory</h1>

<ol>

 <li>Briefly explain the difference between batch gradient descent and stochastic gradient descent. When would you prefer one over the other?</li>

 <li>In gradient descent, you need to stop training when the model converges. How will you know that your model has converged? State the stopping criteria and apply the criteria throughout your implementation.</li>

 <li>What will be the effect of bias term in BGD/SGD learning?</li>

 <li>True/False. Stochastic gradient descent performs less computation per update than batch gradient descent. Give brief reasoning?</li>

 <li>Why do we randomly shuffle the training examples before using SGD optimization?</li>

 <li>Hinge Loss which was introduced in class is not differentiable everywhere unlike log loss. Then show how we can use hinge loss function in the gradient descent optimization algorithm. What will be the gradient of hinge loss function without regularization term?</li>

 <li>What will be the gradient of the log and hinge loss function if you add the <em>L</em><sub>2 </sub>regularization term (1<em>/</em>2)<em>λ</em>||<em>w</em><sup>2</sup>||?</li>

 <li>Why is regularization used? What’s a potential issue with <em>L</em><sub>2 </sub>regularization, if the <em>λ </em>hyperparameter can take negative values?</li>

</ol>

<h1>3         Batch Gradient Descent</h1>

<strong>3.1      Algorithm </strong>

Write down the batch gradient descent algorithm for <strong>log loss </strong>in appropriate algorithmic format.

<h2>3.2      Implementation</h2>

<ul>

 <li>Implement batch gradient descent algorithm for ‘log’ and ‘hinge’ loss functions.</li>

 <li>Apply <em>L</em><sub>2 </sub>regularization for both loss functions. Note that a hyperparameter <em>λ </em>is used which controls the amount of regularization. Also, tune the value of <em>λ </em>when you are using regularization.</li>

 <li>Both implementations will have the ‘bias term’.</li>

</ul>

This part could be completed by editing only bgd.py, unless you plan to extract any new features for the task. You need to initialize the parameters ( init () method), learn them from training data (fit() method) and use the learned parameters to classify new instances (predict() method) for each of the loss functions. Take note that init .py, main.py and classifier.py may be replaced while grading. Do not make any modifications to these files. Report the results obtained on the given test set for both the loss functions in your report. You should submit your code with the hyperparameter settings (config.py) that produce the best performance.

<strong>Output format </strong>is given below. Your final numbers might differ. <strong>The given numbers are just a guideline, meant for sanity checking</strong>.

$ python3 main_bgd.py

BGD Log Loss (No Regularization) Results:

Accuracy: 74.09, Precision: 80.17, Recall: 62.83, F1: 70.45

BGD Log Loss (With Regularization) Results:

Accuracy: 76.41, Precision: 80.31, Recall: 68.91, F1: 74.18

BGD Hinge Loss (No Regularization) Results:

Accuracy: 77.41, Precision: 79.84, Recall: 72.29, F1: 75.88

BGD Hinge Loss (With Regularization) Results:

Accuracy: 78.07, Precision: 79.70, Recall: 74.32, F1: 76.92

<h2>3.3       BGD Analysis</h2>

You need to perform additional experiments to answer the following questions. You don’t need to submit your code for this part. You only need to include your plots and discussions in your report. Make sure that the code you submit doesn’t include any changes you don’t want to be included, as that might affect your chances of getting extra credit.

You need to record the training and test accuracy after every weight update. Let’s define this as one epoch.

<ol>

 <li>Plot training and test accuracy vs epoch without regularization in a graph. We will have one graph for each type of loss function. A sample graph attached in the end of this handout. For this question 2 graphs are expected. <strong>(5 points)</strong></li>

 <li>Run the same experiments with regularization and plot the training and test accuracy for each epoch again for each type of loss function. For this question 2 graphs are expected. <strong>(5 points)</strong></li>

</ol>

<h1>4          Stochastic Gradient Descent Implementation</h1>

<strong>4.1      Algorithm (5 points)</strong>

Write down the stochastic gradient descent algorithm for <strong>hinge loss </strong>in appropriate algorithmic format.

<h2>4.2      Implementation</h2>

<ul>

 <li>Implement stochastic gradient descent algorithm for ’log’ and ’hinge’ loss functions.</li>

 <li>Apply <em>L</em><sub>2 </sub>regularization for both loss functions. Note that a hyperparameter <em>λ </em>is used which controls the amount of regularization. Also, tune the value of <em>λ </em>when you are using regularization.</li>

 <li>Both implementations will have the ’bias term’.</li>

</ul>

This part could be completed by editing only sgd.py, unless you plan to extract any new features for the task. You need to initialize the parameters ( init () method), learn them from training data (fit() method) and use the learned parameters to classify new instances (predict() method) for each of the loss functions. Take note that init .py, main.py and classifier.py may be replaced while grading. Do not make any modifications to these files. Report the results obtained on the given test set for both the loss functions in your report. You should submit your code with the hyperparameter settings (config.py) that produce the best performance.

<strong>Output format </strong>is given below. Your final numbers might differ. <strong>The given numbers are just a guideline, meant for sanity checking</strong>.

$ python3 main_sgd.py

SGD Log Loss (No Regularization) Results:

Accuracy: 75.42, Precision: 80.83, Recall: 65.54, F1: 72.38

SGD Log Loss (With Regularization) Results:

Accuracy: 76.41, Precision: 80.31, Recall: 68.91, F1: 74.18

SGD Hinge Loss (No Regularization) Results:

Accuracy: 77.41, Precision: 79.84, Recall: 72.29, F1: 75.88

SGD Hinge Loss (With Regularization) Results:

Accuracy: 78.07, Precision: 79.70, Recall: 74.32, F1: 76.92

<h2>4.3       SGD Analysis</h2>

You need to perform additional experiments to answer the following questions. You don’t need to submit your code for this part. You only need to include your plots and discussions in your report. Make sure that the code you submit doesn’t include any changes you don’t want to be included, as that might affect your chances of getting extra credit.

You need to record the training and test accuracy after every weight update. Let’s define this as one epoch.

<ol>

 <li>Plot training and test accuracy vs epoch without regularization in a graph. We will have one graph for each type of loss function. A sample graph attached in the end of this handout. For this question 2 graphs are expected. <strong>(5 points)</strong></li>

 <li>Run the same experiments with regularization and plot the training and test accuracy for each epoch again for each type of loss function. For this question 2 graphs are expected. <strong>(5 points)</strong></li>

</ol>

<h1>5       Time Limit</h1>

Your code must terminate with in 5 − 10 minutes for all models combined together. If it doesn’t terminate with in 5 − 10 minutes, you will be graded for the output your code generates at the end of the 10 minute time limit.

<h1>6       Important Notes</h1>

<ul>

 <li>For all hyper-parameter tuning Accuracy will be the performance measure in this assignment.</li>

 <li>Place all the graphs in your report.</li>

 <li>Initialize all the weights randomly.</li>

 <li>Use gnuplot / matplotlib for generating the graphs.</li>

 <li>Your graphs should look like similar to Figure 1</li>

</ul>

<h1>7        Extra Credit</h1>

<h2>7.1       Top – 10% on Leaderboard</h2>

Your submission will be evaluated on a hidden dataset of similar nature. The hidden dataset is of similar size and splits. Top-10% (17 in number) of the students in the class would be awarded 5 extra credit points. You are not allowed to use any optimization libraries but you are allowed to use feature extraction software. If in doubt, ask a question on piazza and the instructors would respond. Remember that the extra credit only depends on the results on the hidden dataset. Overfitting the given dataset might prove counter-productive.




Figure 1: Sample graph. Curves are randomly drawn.

<h2>7.2        K-Fold Cross Validation</h2>

Perform K-fold cross validation (with K = 5) using the train set for evaluating any one of the models (GD or SGD). Any loss function can be used. Include a file called cv.py which has the implementation and report results in the homework report.

<ol>

 <li>Randomly split your entire dataset into K – ‘folds’</li>

 <li>For each K – fold in your dataset, build your model on K – 1 folds of the dataset. Then, test the model to check the effectiveness for K<em><sup>th </sup></em>fold</li>

 <li>Record the accuracy you see on each of the predictions</li>

 <li>Repeat this until each of the K – folds has served as the test set</li>

 <li>The average of your K recorded accuracy is called the cross-validation accuracy and will serve as your performance metric for the model.</li>

</ol>