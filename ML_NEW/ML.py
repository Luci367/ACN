from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

class MachineLearning:
    def __init__(self):
        print("Loading dataset ...")
        self.counter = 0
        self.flow_dataset = pd.read_csv('cicddos2019_dataset.csv')

        # # Perform any required data cleaning or preprocessing here
        # self.flow_dataset.iloc[:, 2] = self.flow_dataset.iloc[:, 2].str.replace('.', '')
        # self.flow_dataset.iloc[:, 3] = self.flow_dataset.iloc[:, 3].str.replace('.', '')
        # self.flow_dataset.iloc[:, 5] = self.flow_dataset.iloc[:, 5].str.replace('.', '')

        self.X_flow = self.flow_dataset.iloc[:, :-2].values.astype('float64')
        self.y_flow = self.flow_dataset.iloc[:, -2].values
        self.X_flow_train, self.X_flow_test, self.y_flow_train, self.y_flow_test = train_test_split(self.X_flow, self.y_flow, test_size=0.25, random_state=0)

    def run_algorithm(self, classifier, label, color):
        print("------------------------------------------------------------------------------")
        print(label)

        self.classifier = classifier
        self.Confusion_matrix(label, color)

    def Confusion_matrix(self, label, color):
        self.counter += 1
        self.flow_model = self.classifier.fit(self.X_flow_train, self.y_flow_train)
        self.y_flow_pred = self.flow_model.predict(self.X_flow_test)

        print("------------------------------------------------------------------------------")
        cm = confusion_matrix(self.y_flow_test, self.y_flow_pred)
        print("confusion matrix", label)
        print(cm)

        acc = accuracy_score(self.y_flow_test, self.y_flow_pred)
        print(f"success accuracy {label} = {acc*100:.2f} %")
        print(f"failure accuracy {label} = {(1.0 - acc) * 100:.2f} %")

        x = ['TP', 'FP', 'FN', 'TN']
        x_indexes = np.arange(len(x))
        width = 0.10
        plt.xticks(ticks=x_indexes, labels=x)
        plt.title("Résultats des algorithmes")
        plt.xlabel('Classe predite')
        plt.ylabel('Nombre de flux')
        plt.tight_layout()
        plt.style.use("seaborn-darkgrid")

        y = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
        plt.bar(x_indexes + (self.counter - 3) * width, y, width=width, color=color, label=label)
        plt.legend()

        if self.counter == 5:
            plt.show()


def main():
    start_script = datetime.now()
    ml = MachineLearning()

    # start = datetime.now()
    # ml.run_algorithm(LogisticRegression(solver='liblinear', random_state=0), "LR", "#1b7021")
    # end = datetime.now()
    # print("LEARNING and PREDICTING Time: ", (end - start))

    start = datetime.now()
    ml.run_algorithm(KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2), "KNN", "#e46e6e")
    end = datetime.now()
    print("LEARNING and PREDICTING Time: ", (end - start))

    start = datetime.now()
    ml.run_algorithm(GaussianNB(), "NB", "#0000ff")
    end = datetime.now()
    print("LEARNING and PREDICTING Time: ", (end - start))

    start = datetime.now()
    ml.run_algorithm(DecisionTreeClassifier(criterion='entropy', random_state=0), "DT", "#e0d692")
    end = datetime.now()
    print("LEARNING and PREDICTING Time: ", (end - start))

    start = datetime.now()
    ml.run_algorithm(LogisticRegression(solver='liblinear', random_state=0), "LR", "#e0d692")
    end = datetime.now()
    print("LEARNING and PREDICTING Time: ", (end - start))

    
    

    end_script = datetime.now()
    print("Script Time: ", (end_script - start_script))


if __name__ == "__main__":
    main()
