import pandas as pd
import numpy as np


def get_features_labels():
    # df = pd.read_csv("spam_ham_dataset.csv")
    df = pd.read_csv('winequality\winequality-red.csv')
    keys = df.columns
    for i in range(len(keys)):
        print(f"{i} - {keys[i]}")
    label = input("Enter column number which represents label : ")
    feature = input("Enter column numbers which represents features seperated by comma")
    feature_list = feature.split(',')
    
    label_column = keys[int(label)]  
    feature_columns = list()
    for i in feature_list:
        feature_columns.append(keys[int(i)])
    return label_column,feature_columns,df

def train_test_split():
    label_column,feature_columns,df = get_features_labels()
    Y = pd.DataFrame(df[label_column])
    X = df[feature_columns]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train,X_test,Y_train,Y_test
