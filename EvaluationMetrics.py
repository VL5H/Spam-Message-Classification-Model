class EvaluationMetrics:

    def __init__(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

    def compute_metrics(self, actual, predicted):
        computeset = zip(actual, predicted)

        for act, pred in computeset:
            if act == 1 and pred == 1:
                self.TP += 1
            elif act == 0 and pred == 0:
                self.TN += 1
            elif act == 0 and pred == 1:
                self.FP += 1
            elif act == 1 and pred == 0:
                self.FN += 1

        accuracy = float((self.TP + self.TN) / (self.TP + self.TN + self.FN + self.FP))

        precision = float(self.TP / (self.TP + self.FP))

        recall = float(self.TP / (self.TP + self.FN))

        f1 = float(2*((precision * recall) / (precision + recall)))

        return accuracy, precision, recall, f1
