from Data_Loader import DataLoader
from Naive_Bayes import NaiveBayes
from EvaluationMetrics import EvaluationMetrics

def log(filename, data):
    with open(filename, 'a') as logbook:
        logbook.write(data)
        logbook.write("\n")

def main():
    filename = "results.log"

    labels, messages = DataLoader.load_data("SMSSpamCollection.txt")

    traininglabels, testinglabels, trainingmessages, testingmessages = DataLoader.split_data(labels, messages)

    trainingsamples = str(len(traininglabels))

    testingsamples = str(len(testinglabels))

    log(filename, " START LOG")

    log(filename, "\n# of training samples: ")
    log(filename, trainingsamples)

    log(filename, "\n# of testing samples: ")
    log(filename, testingsamples)

    mynb = NaiveBayes()

    mynb.train(traininglabels, trainingmessages)

    trainingpredictions = []
    for message in trainingmessages:
        trainingpredictions.append(mynb.prediction(message))

    testingpredictions = []
    for message in testingmessages:
        testingpredictions.append(mynb.prediction(message))

    trainmetrics = EvaluationMetrics()

    testmetrics = EvaluationMetrics()

    trainaccuracy, trainprecision, trainrecall, trainf1 = trainmetrics.compute_metrics(traininglabels, trainingpredictions)

    testaccuracy, testprecision, testrecall, testf1 = testmetrics.compute_metrics(testinglabels, testingpredictions)

    log(filename, "\nTraining Metrics:")
    log(filename, "Training Accuracy = ")
    log(filename, str(trainaccuracy))
    log(filename, "Training Precision = ")
    log(filename, str(trainprecision))
    log(filename, "Training Recall = ")
    log(filename, str(trainrecall))
    log(filename, "Training F1 = ")
    log(filename, str(trainf1))

    log(filename, "TP: ")
    log(filename, str(trainmetrics.TP))
    log(filename, "TN: ")
    log(filename, str(trainmetrics.TN))
    log(filename, "FP: ")
    log(filename, str(trainmetrics.FP))
    log(filename, "FN: ")
    log(filename, str(trainmetrics.FN))

    log(filename, "\nTesting Metrics:")
    log(filename, "Testing Accuracy = ")
    log(filename, str(testaccuracy))
    log(filename, "Testing Precision = ")
    log(filename, str(testprecision))
    log(filename, "Testing Recall = ")
    log(filename, str(testrecall))
    log(filename, "Testing F1 = ")
    log(filename, str(testprecision))

    log(filename, "TP: ")
    log(filename, str(testmetrics.TP))
    log(filename, "TN: ")
    log(filename, str(testmetrics.TN))
    log(filename, "FP: ")
    log(filename, str(testmetrics.FP))
    log(filename, "FN: ")
    log(filename, str(testmetrics.FN))

    log(filename, "\n END LOG")

if __name__ == "__main__":
    main()

