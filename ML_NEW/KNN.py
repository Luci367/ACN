from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class MachineLearning:

    def __init__(self):
        print("Loading dataset ...")
        self.flow_dataset = pd.read_csv('cicddos2019_dataset.csv')

         

    def flow_training(self):
        print("Flow Training ...")
        
        X_flow = self.flow_dataset.iloc[:, :-2].values
        X_flow = X_flow.astype('float64')

        y_flow = self.flow_dataset.iloc[:, -2].values

        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)

        classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
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
        
        x = ['TP', 'FP', 'FN', 'TN']
        plt.title("KNN")
        plt.xlabel('Classe predite')
        plt.ylabel('Nombre de flux')
        plt.tight_layout()
        plt.style.use("seaborn-darkgrid")

        if cm.shape[0] > 1 and cm.shape[1] > 1:
            y = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
        else:
            y = [cm[0]]

        plt.bar(x, y, color="#e46e6e", label='KNN')
        plt.legend()
        plt.show()

def main():
    start = datetime.now()
    
    ml = MachineLearning()
    ml.flow_training()

    end = datetime.now()
    print("Training time: ", (end - start)) 

if __name__ == "__main__":
    main()
