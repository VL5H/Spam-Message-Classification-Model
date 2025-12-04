import random

import nltk
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

class DataLoader:

    @staticmethod
    def preprocess(messageholder):
        messageholder = messageholder.lower()
        #justwords = re.sub(r'[^a-zA-Z0-9]', '', messageholder)  it breaks if i use this for some reason?
        tokenized = word_tokenize(messageholder)

        processed = []

        for i in tokenized:
            if i.lower() not in stop_words:
                stemmed = stemmer.stem(i)
                processed.append(i)

        return processed

    @staticmethod
    def load_data(filepath):
        labels = []
        messages = []

        with open(filepath, 'r', encoding="utf-8") as file:

            for line in file:
                line = line.strip()
                split = line.split("\t", 1)

                if len(split) < 2:
                    continue
                else:
                    labelholder, messageholder = split

                if labelholder.lower() == "spam":
                    labels.append(1)
                elif labelholder.lower() == "ham":
                    labels.append(0)
                else:
                    continue

                messageholder = DataLoader.preprocess(messageholder)
                messages.append(messageholder)

        return labels, messages

    @staticmethod
    def split_data(labels, messages, test_ratio = 0.2):
        combined = []
        combined = list(zip(labels, messages))

        random.shuffle(combined)

        length = len(combined)
        length = int(length*(1-test_ratio))

        training = combined[:length]
        testing = combined[length:]

        traininglabels = []
        testinglabels = []

        trainingmessages = []
        testingmessages = []

        for labels, messages in training:
            traininglabels.append(labels)
            trainingmessages.append(messages)

        for labels, messages in testing:
            testinglabels.append(labels)
            testingmessages.append(messages)

        return traininglabels, testinglabels, trainingmessages, testingmessages





