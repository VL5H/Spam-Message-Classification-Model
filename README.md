# Spam-Message-Classification-Model
A Naive Bayes Classifier optimized to classify SMS messages as either spam or not spam.

Uses a publicly available, tab-separated, dataset of 5000+ spam and non-spam messages (labelled as "spam" and "ham") for model training.
Utilizes Python's NLTK library for data preprocessing.
Naive Bayes logic is implemented via the Collections library following the standard Bayesian inference equation:
```math
P(\theta |x)=\frac{P(x|\theta )P(\theta )}{P(x)}
```

The model functions like this:
1. Dataset is loaded from "SMSSpamCollection.txt". Messages are stripped, tokenized, labelled, and randomly split 80-20 between training and testing data via the NLTK library. (Data_Loader.py)
2. Training (Naive_Bayes.py):
    - The model first calculates the prior probabilities of P(ham) and P(spam) from the training data.
    - It then computes the conditional probability of each word appearing in its respective class P(word|spam) and P(word|ham) via Laplace Smoothing: $P(x_{i}|y)=\frac{\text{count}(x_{i},y)+\alpha }{\text{count}(y)+\alpha N}$  and stored in a Collections "defaultdict" subclass.
    - Word frequencies within spam and ham messages are also tracked separately via the Collections "defaultdict" subclass.
3. Prediction (Naive_Bayes.py):
    - Bayes theorem (see above) is applied to each message in the testing set.
    - Log space calculations are used to handle 0 cases.
    - Posterior probability of Ham and Spam are calculated and compared with the greater probability becoming the final prediction.
    - Assumes all words are conditionally independent (Naive Assumption)
4. Statistical Analysis (EvaluationMetrics.py):
    - The following statistics are computed during training and testing and outputted to an auto-created log file called "results.log":
      -  Accuracy
      -  Precision
      -  Recall
      -  F1
      -  True Positives (TP)
      -  True Negatives (TN)
      -  False Positives (FP)
      -  False Negatives (FN)

    *Note: As this model is an initial proof of concept, I have not included a UI at this time. I plan to create a upgraded, more intelligent version of this model featuring better classification logic and a GUI.

Installation/SetUp:
1. Download the following files:
    - Data_Loader.py
    - Naive_Bayes.py
    - EvaluationMetrics.py
    - Main.py
    - SMSSpamCollection.txt
    - requirements.txt
2. Ensure all files are in the same directory.

Running/Usage Instructions:
1. Create/activate your virtual environment (you will need Python 3.7 or higher) and run the command: ```pip install -r requirements.txt```
2. Open the "Main.py" file and run it.
3. View the model's results and performance in the "results.log" file. (Should be auto-created upon first run).
4. The model should average an accuracy of around 92% in training and 85% in testing.
