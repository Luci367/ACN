from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class MachineLearning:
    def __init__(self):
        print("Loading dataset ...")
        self.flow_dataset = pd.read_csv('./cicddos2019_dataset.csv')

    def flow_training(self):
        print("Flow Training ...")
        X_flow = self.flow_dataset.iloc[:, :-2].values
        X_flow = X_flow.astype('float64')

        y_flow = self.flow_dataset.iloc[:, -2].values

        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.20, random_state=0)

        classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
        flow_model = classifier.fit(X_flow_train, y_flow_train)

        y_flow_pred = flow_model.predict(X_flow_test)

        print(y_flow_pred)

        print("------------------------------------------------------------------------------")

        print("confusion matrix")
        cm = confusion_matrix(y_flow_test, y_flow_pred)
        print(cm)

        acc = accuracy_score(y_flow_test, y_flow_pred)

        print("success accuracy = {0:.2f} %".format(acc * 100))
        fail = 1.0 - acc
        print("fail accuracy = {0:.2f} %".format(fail * 100))
        print("------------------------------------------------------------------------------")

        self.visualize_confusion_matrix(cm)

    def visualize_confusion_matrix(self, cm):
        plt.title("Random Forest")
        plt.xlabel('Classe predite')
        plt.ylabel('Nombre de flux')
        plt.tight_layout()
        plt.style.use("seaborn-darkgrid")

        num_classes = cm.shape[0]  # Get the number of classes

        x = [f'Class {i}' for i in range(num_classes)]  # Labeling classes from 0 to num_classes - 1
        y = []

        for i in range(num_classes):
            row_sum = sum(cm[i])  # Sum elements in each row
            y.append(row_sum)

        plt.bar(x, y, color="#000000", label='RF')
        plt.legend()
        plt.show()


start = datetime.now()

ml = MachineLearning()
ml.flow_training()

end = datetime.now()
print("Training time: ", (end - start))
