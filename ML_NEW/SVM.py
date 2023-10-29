from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

class MachineLearning:
    def __init__(self):
        print("Loading dataset ...")
        self.flow_dataset = pd.read_csv('cicddos2019_dataset.csv')

        # # Perform any required data cleaning or preprocessing here
        # self.flow_dataset.iloc[:, 2] = self.flow_dataset.iloc[:, 2].str.replace('.', '')
        # self.flow_dataset.iloc[:, 3] = self.flow_dataset.iloc[:, 3].str.replace('.', '')
        # self.flow_dataset.iloc[:, 5] = self.flow_dataset.iloc[:, 5].str.replace('.', '')

    def flow_training(self):
        print("Flow Training ...")
        
        X_flow = self.flow_dataset.iloc[:, :-2].values
        X_flow = X_flow.astype('float64')

        y_flow = self.flow_dataset.iloc[:, -2].values

        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)

        classifier = SVC(kernel='rbf', random_state=0)
        flow_model = classifier.fit(X_flow_train, y_flow_train)

        y_flow_pred = flow_model.predict(X_flow_test)

        print("------------------------------------------------------------------------------")

        print("confusion matrix")
        cm = confusion_matrix(y_flow_test, y_flow_pred)
        print(cm)

        acc = accuracy_score(y_flow_test, y_flow_pred)

        print("success accuracy = {0:.2f} %".format(acc * 100))
        fail = 1.0 - acc
        print("failure accuracy = {0:.2f} %".format(fail * 100))
        print("------------------------------------------------------------------------------")

        # # Visualizing Confusion Matrix
        # plt.figure(figsize=(6, 6))
        # plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
        # plt.title("Confusion Matrix")
        # plt.colorbar()

        # num_classes = len(cm)
        # plt.xticks(range(num_classes))
        # plt.yticks(range(num_classes))

        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        
        # for i in range(num_classes):
        #     for j in range(num_classes):
        #         plt.text(j, i, str(cm[i, j]), horizontalalignment='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')

        # plt.show()

def main():
    start = datetime.now()
    ml = MachineLearning()
    ml.flow_training()
    end = datetime.now()
    print("Training time: ", (end - start)) 

if __name__ == "__main__":
    main()
