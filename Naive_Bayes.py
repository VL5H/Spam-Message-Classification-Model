import math
from collections import defaultdict
from collections import Counter

class NaiveBayes:

    def __init__(self):
        self.hamwords = defaultdict(int)
        self.spamwords = defaultdict(int)
        self.totalham = 0
        self.totalspam = 0
        self.probabilityofwords = defaultdict(lambda: defaultdict(float))
        self.uniquewords = 0
        self.probofham = 0
        self.probofspam = 0

    def train(self, traininglabels, trainingmessages):
        labelcount = Counter(traininglabels)

        self.totalham = labelcount[0]
        self.totalspam = labelcount[1]

        trainingset = zip(trainingmessages, traininglabels)

        for messages, labels in trainingset:
            words = Counter(messages)

            if labels == 0:
                self.hamwords.update(words)
            elif labels == 1:
                self.spamwords.update(words)

        self.probofham = self.totalham / (self.totalham + self.totalspam)
        self.probofspam = self.totalspam / (self.totalham + self.totalspam)

        self.uniquewords = len(set(self.hamwords.keys()).union(set(self.spamwords.keys())))

        for i, c in self.hamwords.items():
            self.probabilityofwords["ham"][i] = (c+1) / (self.totalham + self.uniquewords)

        for i, c in self.spamwords.items():
            self.probabilityofwords["spam"][i] = (c+1) / (self.totalspam + self.uniquewords)

    def prediction(self, messages):
        logprobofham = math.log(self.probofham)
        logprobofspam = math.log(self.probofspam)

        for i in messages:
            ifwordisham = self.probabilityofwords["ham"].get(i, (1 / (self.totalham + self.uniquewords)))
            logprobofham += math.log(ifwordisham)

            ifwordisspam = self.probabilityofwords["spam"].get(i, (1 / (self.totalspam + self.uniquewords)))
            logprobofspam += math.log(ifwordisspam)

        if logprobofspam > logprobofham:
            return 1
        elif logprobofspam < logprobofham:
            return 0
        else:
            return 0






