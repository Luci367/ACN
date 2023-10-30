import pandas as pd
import numpy as np
import sys
import sklearn
import io
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics

#col_names =['Unnamed: 0', 'Protocol', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Fwd Packets Length Total', 'Bwd Packets Length Total', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min', 'Packet Length Max', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Avg Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init Fwd Win Bytes', 'Init Bwd Win Bytes', 'Fwd Act Data Packets', 'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label', 'Class']
  
    # train_data = df.sample(frac=0.8, random_state=1)  # 80% of data for training
    # test_data = df.drop(train_data.index)  # Remaining 20% for testing

    # # Save the split data into separate CSV files
    # train_data.to_csv('train_data.csv', index=False)
    # test_data.to_csv('test_data.csv', index=False)


def preprocess(train_data_path,test_data_path):
    print("Reading Data")
    df_u = pd.read_csv(train_data_path )
    df_test_u = pd.read_csv(test_data_path)
    df = df_u.iloc[:,:-1]
    df_test = df_test_u.iloc[:,:-1]

    print('Dimensions of the Training set:',df.shape)
    print('Dimensions of the Test set:',df_test.shape)

    print('Label distribution Training set:')
    print(df['Label'].value_counts())
    print()

    # print('Class distribution Training set:')
    # print(df['Class'].value_counts())
    # print()


    print('Label distribution Test set:')
    print(df_test['Label'].value_counts())
    print()

    # print('Class distribution Test set:')
    # print(df_test['Class'].value_counts())


    print('Training set:')
    for col_name in df.columns:
        if df[col_name].dtypes == 'object' :
            unique_cat = len(df[col_name].unique())
            print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

    print()

    # Test set
    print('Test set:')
    for col_name in df_test.columns:
        if df_test[col_name].dtypes == 'object' :
            unique_cat = len(df_test[col_name].unique())
            print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

    print()
    print('Distribution of categories in Label:')
    print(df['Label'].value_counts().sort_values(ascending=False).head())

    # print('Test set:')
    # for col_name in df_test.columns:
    #     if df_test[col_name].dtypes == 'object' :
    #         unique_cat = len(df_test[col_name].unique())
    #         print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))


    categorical_columns=['Label']

    df_categorical_values = df[categorical_columns]
    testdf_categorical_values = df_test[categorical_columns]

    print(df_categorical_values.head())


    # Label type
    unique_Label=sorted(df.Label.unique())
    string1 = 'Label_type_'
    unique_Label2=[string1 + x for x in unique_Label]
    print(unique_Label2)

    # # Class type
    # unique_Class=sorted(df.Class.unique())
    # string1_Class = 'Class_type_'
    # unique_Class2=[string1_Class + x for x in unique_Class]
    # print(unique_Class2)

    dumcols=unique_Label2 
    #removed uniqueclass from here nd test set below

    #TEST SET

    # Label type
    unique_Label_test=sorted(df_test.Label.unique())
    string1_test = 'Label_type_'
    unique_Label2_test=[string1_test + x for x in unique_Label_test]
    print(unique_Label2_test)

    # # Class type
    # unique_Class_test=sorted(df_test.Class.unique())
    # string1_Class_test = 'Class_type_'
    # unique_Class2_test=[string1_Class_test + x for x in unique_Class_test]
    # print(unique_Class2_test)

    testdumcols=unique_Label2_test


    df_categorical_values_enc=df_categorical_values.apply(LabelEncoder().fit_transform)

    print(df_categorical_values.head())
    print('--------------------')
    print(df_categorical_values_enc.head())

    # test set
    testdf_categorical_values_enc=testdf_categorical_values.apply(LabelEncoder().fit_transform)

    enc = OneHotEncoder(categories='auto')
    df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
    df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(),columns=dumcols)


    # test set
    testdf_categorical_values_encenc = enc.fit_transform(testdf_categorical_values_enc)
    testdf_cat_data = pd.DataFrame(testdf_categorical_values_encenc.toarray(),columns=testdumcols)

    print(df_cat_data.head())


    ##Joining the new data

    newdf=df.join(df_cat_data)
    newdf_test=df_test.join(testdf_cat_data)


    labeldf=df['Label'].unique()
    print(labeldf)

    print(newdf.shape)
    print(newdf_test.shape)

    labeldf=newdf['Label']
    labeldf_test=newdf_test['Label']

    newlabeldf=labeldf.replace({
        'Benign': 0,
        'DrDoS_NTP': 1,
        'DrDoS_UDP': 1,
        'DrDoS_MSSQL': 1,
        'DrDoS_SNMP': 1,
        'DrDoS_DNS': 1,
        'DrDoS_LDAP': 1,
        'DrDoS_NetBIOS': 1,
        'WebDDoS': 1,
        'TFTP': 2,
        'LDAP': 2,
        'UDP': 2,
        'Syn': 2,
        'MSSQL': 2,
        'UDP-lag': 2,
        'Portmap': 2,
        'NetBIOS': 2,
        'UDPLag': 2
    })

    newlabeldf_test=labeldf_test.replace({
        'Benign': 0,
        'DrDoS_NTP': 1,
        'DrDoS_UDP': 1,
        'DrDoS_MSSQL': 1,
        'DrDoS_SNMP': 1,
        'DrDoS_DNS': 1,
        'DrDoS_LDAP': 1,
        'DrDoS_NetBIOS': 1,
        'WebDDoS': 1,
        'TFTP': 2,
        'LDAP': 2,
        'UDP': 2,
        'Syn': 2,
        'MSSQL': 2,
        'UDP-lag': 2,
        'Portmap': 2,
        'NetBIOS': 2,
        'UDPLag': 2
    })

    # classdf = newdf['Class']
    # classdf_test = newdf_test['Class']

    # newclassdf = classdf.replace({
    #     'Benign': 0,
    #     'Attack': 1
    # })

    # newclassdf_test = classdf_test.replace({
    #     'Benign': 0,
    #     'Attack': 1
    # })

    newdf['Label'] = newlabeldf
    newdf_test['Label'] = newlabeldf_test

    # newdf['Class'] = newclassdf
    # newdf_test['Class'] = newclassdf_test
    newdf.to_csv('preprocessed_train_data.csv', index=False)
    newdf_test.to_csv('preprocessed_test_data.csv', index=False)

    
# preprocess('./train_data.csv','test_data.csv')

newdf = pd.read_csv('preprocessed_train_data.csv')
newdf_test = pd.read_csv('preprocessed_test_data.csv')


to_drop_DDoS = [0, 1]  # Labels for DDoS
to_drop_NonDDoS = [0, 2]  # Labels for Non-DDoS

DDoS_df = newdf[newdf['Label'].isin(to_drop_DDoS)]
NonDDoS_df = newdf[newdf['Label'].isin(to_drop_NonDDoS)]

DDoS_df_test = newdf_test[newdf_test['Label'].isin(to_drop_DDoS)]
NonDDoS_df_test = newdf_test[newdf_test['Label'].isin(to_drop_NonDDoS)]

print('Train:')
print('Dimensions of DDoS:', DDoS_df.shape)
print('Dimensions of NonDDoS:', NonDDoS_df.shape)
print()
print('Test:')
print('Dimensions of DDoS:', DDoS_df_test.shape)
print('Dimensions of NonDDoS:', NonDDoS_df_test.shape)

#removing label_type columns as they are redundant

label_type_cols_to_drop = [
    'Label_type_Benign', 'Label_type_DrDoS_NTP', 'Label_type_DrDoS_NetBIOS', 'Label_type_DrDoS_SNMP',
    'Label_type_DrDoS_UDP', 'Label_type_NetBIOS', 'Label_type_Portmap', 'Label_type_Syn',
    'Label_type_TFTP', 'Label_type_UDP', 'Label_type_UDP-lag', 'Label_type_UDPLag', 'Label_type_WebDDoS', 'Unnamed: 0',
    'Label_type_DrDoS_DNS', 'Label_type_DrDoS_LDAP','Label_type_DrDoS_MSSQL', 'Label_type_LDAP', 'Label_type_MSSQL'
]

# For the DDoS dataset
DDoS_filtered = DDoS_df.drop(columns=label_type_cols_to_drop, axis=1)
DDoS_test_filtered = DDoS_df_test.drop(columns=label_type_cols_to_drop, axis=1)

# For the Non-DDoS dataset
NonDDoS_filtered = NonDDoS_df.drop(columns=label_type_cols_to_drop, axis=1)
NonDDoS_test_filtered = NonDDoS_df_test.drop(columns=label_type_cols_to_drop, axis=1)

X_DDoS = DDoS_filtered.drop(['Label'], axis=1)
Y_DDoS = DDoS_filtered[['Label']]

X_NonDDoS = NonDDoS_filtered.drop(['Label'], axis=1)
Y_NonDDoS = NonDDoS_filtered[['Label']]

X_DDoS_test = DDoS_test_filtered.drop(['Label'], axis=1)
Y_DDoS_test = DDoS_test_filtered[['Label']]

X_NonDDoS_test = NonDDoS_test_filtered.drop(['Label'], axis=1)
Y_NonDDoS_test = NonDDoS_test_filtered[['Label']]
colNames = list(X_DDoS)
colNames_test = list(X_DDoS_test)


scaler1 = preprocessing.StandardScaler().fit(X_DDoS)
X_DDoS = scaler1.transform(X_DDoS) 

scaler2 = preprocessing.StandardScaler().fit(X_NonDDoS)
X_NonDDoS = scaler2.transform(X_NonDDoS)

scaler3 = preprocessing.StandardScaler().fit(X_DDoS_test)
X_DDoS_test = scaler3.transform(X_DDoS_test)

scaler4 = preprocessing.StandardScaler().fit(X_NonDDoS_test)
X_NonDDoS_test = scaler4.transform(X_NonDDoS_test)

#converting Y values to 1D array format
Y_DDoS_test = Y_DDoS_test.values.ravel()
Y_NonDDoS_test = Y_NonDDoS_test.values.ravel()

Y_DDoS = Y_DDoS.values.ravel()
Y_NonDDoS = Y_NonDDoS.values.ravel()
#Feature Selection using RFE

clf = RandomForestClassifier(n_estimators=10,n_jobs=2)
rfe = RFE(estimator=clf, n_features_to_select=13, step=1)
#threshold can be changed here

rfe.fit(X_DDoS, Y_DDoS.astype(int))
X_rfeDDoS = rfe.transform(X_DDoS)
true = rfe.support_
rfecolindex_DDoS = [i for i, x in enumerate(true) if x]
rfecolname_DDoS = list(colNames[i] for i in rfecolindex_DDoS)

rfe.fit(X_NonDDoS, Y_NonDDoS.astype(int))
X_rfeNonDDoS = rfe.transform(X_NonDDoS)
true = rfe.support_
rfecolindex_NonDDoS = [i for i, x in enumerate(true) if x]
rfecolname_NonDDoS = list(colNames[i] for i in rfecolindex_NonDDoS)

print('Features selected for DDoS:', rfecolname_DDoS)
print()
print('Features selected for NonDDoS:', rfecolname_NonDDoS)
print()




print(X_rfeDDoS.shape)
print(X_rfeNonDDoS.shape)

clf_DDoS = RandomForestClassifier(n_estimators=10, n_jobs=2)
clf_NonDDoS = RandomForestClassifier(n_estimators=10, n_jobs=2)
clf_DDoS.fit(X_DDoS, Y_DDoS.astype(int))
clf_NonDDoS.fit(X_NonDDoS, Y_NonDDoS.astype(int))

clf_rfeDDoS = RandomForestClassifier(n_estimators=10, n_jobs=2)
clf_rfeNonDDoS = RandomForestClassifier(n_estimators=10, n_jobs=2)
clf_rfeDDoS.fit(X_rfeDDoS, Y_DDoS.astype(int))
clf_rfeNonDDoS.fit(X_rfeNonDDoS, Y_NonDDoS.astype(int))

clf_DDoS.predict(X_DDoS_test)
clf_DDoS.predict_proba(X_DDoS_test)[0:10]
Y_DDoS_pred = clf_DDoS.predict(X_DDoS_test)


# def evaluate(Y_DDoS_test,Y_DDoS_pred,name,clf_DDoS,X_DDoS_test):
#     pd.crosstab(Y_DDoS_test, Y_DDoS_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])
#     print(name)
#     accuracy = cross_val_score(clf_DDoS, X_DDoS_test, Y_DDoS_test, cv=10, scoring='accuracy')
#     print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
#     precision= cross_val_score(clf_DDoS, X_DDoS_test, Y_DDoS_test, cv=10, scoring='precision')
#     print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
#     recall = cross_val_score(clf_DDoS, X_DDoS_test, Y_DDoS_test, cv=10, scoring='recall')
#     print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
#     # f = cross_val_score(clf_DDoS, X_DDoS_test, Y_DDoS_test, cv=10, scoring='f1')
#     # print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))
#     print()

# print()
# evaluate(Y_DDoS_test,Y_DDoS_pred,"DDoS",clf_DDoS,X_DDoS_test)
# Y_NonDDoS_pred = clf_NonDDoS.predict(X_NonDDoS_test)
# evaluate(Y_NonDDoS_test,Y_NonDDoS_pred,"NonDDoS",clf_NonDDoS,X_NonDDoS_test)
# evaluate(Y_DDoS_test,Y_DDoS_pred,"DDoS",clf_DDoS,X_DDoS_test)
# evaluate(Y_DDoS_test,Y_DDoS_pred,"DDoS",clf_DDoS,X_DDoS_test)

# Create confusion matrix

pd.crosstab(Y_DDoS_test, Y_DDoS_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])

Y_NonDDoS_pred = clf_NonDDoS.predict(X_NonDDoS_test)
# Create confusion matrix
pd.crosstab(Y_NonDDoS_test, Y_NonDDoS_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])

accuracy_DDoS = cross_val_score(clf_DDoS, X_DDoS_test, Y_DDoS_test, cv=10, scoring='accuracy')
print("Accuracy for DDoS: %0.5f (+/- %0.5f)" % (accuracy_DDoS.mean(), accuracy_DDoS.std() * 2))
precision_DDoS = cross_val_score(clf_DDoS, X_DDoS_test, Y_DDoS_test, cv=10, scoring='precision')
print("Precision for DDoS: %0.5f (+/- %0.5f)" % (precision_DDoS.mean(), precision_DDoS.std() * 2))
recall_DDoS = cross_val_score(clf_DDoS, X_DDoS_test, Y_DDoS_test, cv=10, scoring='recall')
print("Recall for DDoS: %0.5f (+/- %0.5f)" % (recall_DDoS.mean(), recall_DDoS.std() * 2))
f_DDoS = cross_val_score(clf_DDoS, X_DDoS_test, Y_DDoS_test, cv=10, scoring='f1')
print("F-measure for DDoS: %0.5f (+/- %0.5f)" % (f_DDoS.mean(), f_DDoS.std() * 2))
print()

accuracy_NonDDoS = cross_val_score(clf_NonDDoS, X_NonDDoS_test, Y_NonDDoS_test, cv=10, scoring='accuracy')
print("Accuracy for NonDDoS: %0.5f (+/- %0.5f)" % (accuracy_NonDDoS.mean(), accuracy_NonDDoS.std() * 2))
precision_NonDDoS = cross_val_score(clf_NonDDoS, X_NonDDoS_test, Y_NonDDoS_test, cv=10, scoring='precision_macro')
print("Precision for NonDDoS: %0.5f (+/- %0.5f)" % (precision_NonDDoS.mean(), precision_NonDDoS.std() * 2))
recall_NonDDoS = cross_val_score(clf_NonDDoS, X_NonDDoS_test, Y_NonDDoS_test, cv=10, scoring='recall_macro')
print("Recall for NonDDoS: %0.5f (+/- %0.5f)" % (recall_NonDDoS.mean(), recall_NonDDoS.std() * 2))
f_NonDDoS = cross_val_score(clf_NonDDoS, X_NonDDoS_test, Y_NonDDoS_test, cv=10, scoring='f1_macro')
print("F-measure for NonDDoS: %0.5f (+/- %0.5f)" % (f_NonDDoS.mean(), f_NonDDoS.std() * 2))
print()


X_DDoS_test2 = X_DDoS_test[:, rfecolindex_DDoS]
X_NonDDoS_test2 = X_NonDDoS_test[:, rfecolindex_NonDDoS]


Y_DDoS_pred2 = clf_rfeDDoS.predict(X_DDoS_test2)
# Create confusion matrix
pd.crosstab(Y_DDoS_test, Y_DDoS_pred2, rownames=['Actual attacks'], colnames=['Predicted attacks'])

Y_NonDDoS_pred2 = clf_rfeNonDDoS.predict(X_NonDDoS_test2)
# Create confusion matrix
pd.crosstab(Y_NonDDoS_test, Y_NonDDoS_pred2, rownames=['Actual attacks'], colnames=['Predicted attacks'])

accuracy_rfeDDoS = cross_val_score(clf_rfeDDoS, X_DDoS_test2, Y_DDoS_test, cv=10, scoring='accuracy')
print("Accuracy for rfeDDoS: %0.5f (+/- %0.5f)" % (accuracy_rfeDDoS.mean(), accuracy_rfeDDoS.std() * 2))
precision_rfeDDoS = cross_val_score(clf_rfeDDoS, X_DDoS_test2, Y_DDoS_test, cv=10, scoring='precision')
print("Precision for rfeDDoS: %0.5f (+/- %0.5f)" % (precision_rfeDDoS.mean(), precision_rfeDDoS.std() * 2))
recall_rfeDDoS = cross_val_score(clf_rfeDDoS, X_DDoS_test2, Y_DDoS_test, cv=10, scoring='recall')
print("Recall for rfeDDoS: %0.5f (+/- %0.5f)" % (recall_rfeDDoS.mean(), recall_rfeDDoS.std() * 2))
f_rfeDDoS = cross_val_score(clf_rfeDDoS, X_DDoS_test2, Y_DDoS_test, cv=10, scoring='f1')
print("F-measure for rfeDDoS: %0.5f (+/- %0.5f)" % (f_rfeDDoS.mean(), f_rfeDDoS.std() * 2))
print()

accuracy_rfeNonDDoS = cross_val_score(clf_rfeNonDDoS, X_NonDDoS_test2, Y_NonDDoS_test, cv=10, scoring='accuracy')
print("Accuracy for rfeNonDDoS: %0.5f (+/- %0.5f)" % (accuracy_rfeNonDDoS.mean(), accuracy_rfeNonDDoS.std() * 2))
precision_rfeNonDDoS = cross_val_score(clf_rfeNonDDoS, X_NonDDoS_test2, Y_NonDDoS_test, cv=10, scoring='precision_macro')
print("Precision for rfeNonDDoS: %0.5f (+/- %0.5f)" % (precision_rfeNonDDoS.mean(), precision_rfeNonDDoS.std() * 2))
recall_rfeNonDDoS = cross_val_score(clf_rfeNonDDoS, X_NonDDoS_test2, Y_NonDDoS_test, cv=10, scoring='recall_macro')
print("Recall for rfeNonDDoS: %0.5f (+/- %0.5f)" % (recall_rfeNonDDoS.mean(), recall_rfeNonDDoS.std() * 2))
f_rfeNonDDoS = cross_val_score(clf_rfeNonDDoS, X_NonDDoS_test2, Y_NonDDoS_test, cv=10, scoring='f1_macro')
print("F-measure for rfeNonDDoS: %0.5f (+/- %0.5f)" % (f_rfeNonDDoS.mean(), f_rfeNonDDoS.std() * 2))
print()