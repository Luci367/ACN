from xgboost import XGBClassifier  # Import XGBoost classifier
from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, accuracy_score

class MachineLearning:
    def __init__(self):
        print("Loading dataset ...")
        self.flow_dataset = pd.read_csv('cicddos2019_dataset.csv')

        # # Perform data preprocessing
        # self.flow_dataset.iloc[:, 2] = self.flow_dataset.iloc[:, 2].str.replace('.', '')
        # self.flow_dataset.iloc[:, 3] = self.flow_dataset.iloc[:, 3].str.replace('.', '')
        # self.flow_dataset.iloc[:, 5] = self.flow_dataset.iloc[:, 5].str.replace('.', '')

    def flow_training(self):
        print("Flow Training ...")

        X_flow = self.flow_dataset.iloc[:, :-2].values.astype('float64')
        label_encoder = LabelEncoder()
        y_flow = label_encoder.fit_transform(self.flow_dataset.iloc[:, -2])

        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)

        # Use XGBoost Classifier
        classifier = XGBClassifier(random_state=0)  # XGBoost classifier
        flow_model = classifier.fit(X_flow_train, y_flow_train)

        y_flow_pred = flow_model.predict(X_flow_test)

        print("------------------------------------------------------------------------------")

        print("Confusion Matrix:")
        cm = confusion_matrix(y_flow_test, y_flow_pred)
        print(cm)

        acc = accuracy_score(y_flow_test, y_flow_pred)
        print(f"Success accuracy: {acc*100:.2f} %")
        print(f"Failure accuracy: {(1.0 - acc) * 100:.2f} %")
        print("------------------------------------------------------------------------------")

        # # Visualizing Confusion Matrix
        # x = ['TP', 'FP', 'FN', 'TN']
        # x_indexes = range(len(x))
        # plt.title("XGBoost")
        # plt.xlabel('Predicted Class')
        # plt.ylabel('Number of Instances')
        # plt.bar(x_indexes, cm.flatten(), color="#e0d692", label='XGBoost')
        # plt.xticks(x_indexes, x)
        # plt.legend()
        # plt.show()

def main():
    start = datetime.now()
    ml = MachineLearning()
    ml.flow_training()
    end = datetime.now()
    print("Training time:", (end - start))

if __name__ == "__main__":
    main()
