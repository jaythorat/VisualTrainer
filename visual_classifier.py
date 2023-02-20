# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import math



def get_features_labels(filename):
    df = pd.read_csv(filename)
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

def train_test_split(filename):
    label_column,feature_columns,df = get_features_labels(filename)
    Y = pd.DataFrame(df[label_column])
    X = df[feature_columns]
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    Y_train = Y_train.values.flatten()
    Y_test = Y_test.values.flatten()
    return X_train,X_test,Y_train,Y_test, X, Y


def knn_classifier(X_train,X_test,Y_train,Y_test):
    from sklearn.neighbors import KNeighborsClassifier
    neighbors = math.floor(math.sqrt(len(X_train)))
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(X_train,Y_train)
    KNN_predictions = knn.predict(X_test)
    knn_accuracy = accuracy_score(Y_test,KNN_predictions)
    return knn,knn_accuracy
    
def naive_bayes_gaussian(X_train,X_test,Y_train,Y_test):
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train,Y_train)
    gnb_predictions = gnb.predict(X_test)
    gnb_accuracy = accuracy_score(Y_test,gnb_predictions)
    return gnb,gnb_accuracy

def naive_bayes_multinomial(X_train,X_test,Y_train,Y_test):
    from sklearn.naive_bayes import MultinomialNB
    mnb = MultinomialNB()
    mnb.fit(X_train,Y_train)
    mnb_predictions = mnb.predict(X_test)
    mnb_accuracy = accuracy_score(Y_test,mnb_predictions)
    return mnb,mnb_accuracy
    
def random_forest_classifier(X_train,X_test,Y_train,Y_test):
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(random_state=0)
    rfc.fit(X_train,Y_train)
    rfc_predictions = rfc.predict(X_test)
    rfc_accuracy = accuracy_score(Y_test,rfc_predictions)
    return rfc,rfc_accuracy

def decision_tree_classifier(X_train,X_test,Y_train,Y_test):
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(random_state=0)
    dtc.fit(X_train,Y_train)
    dtc_predictions = dtc.predict(X_test)
    dtc_accuracy = accuracy_score(Y_test,dtc_predictions)
    return dtc,dtc_accuracy

def gradient_boosting_classifier(X_train,X_test,Y_train,Y_test):
    from sklearn.ensemble import GradientBoostingClassifier
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    gbc.fit(X_train,Y_train)
    gbc_predictions = gbc.predict(X_test)
    gbc_accuracy = accuracy_score(Y_test,gbc_predictions)
    return gbc,gbc_accuracy


def save_model(model,name):
    import pickle
    pickle.dump(model,open(f'{name}.pkl','wb'))

def main(filename):
    X_train,X_test,Y_train,Y_test, X, Y = train_test_split(filename)
    classifiers_list = {'0':"All classifiers",'1':'KNN_classifier','2':'Gaussian Naive Bayes Classifier','3':'Multinomial Naive Bayes Classifier','4':'Random Forest classifier','5':'Decision Tree Classifier','6':'Gradient Boosting classifier'}
    print("Select any classifier from below")
    for i in classifiers_list:
        print(f"{i} - {classifiers_list[i]}")
    classifier = input("Enter classifier number : ")
    if classifier == '1':
        model_knn,accuracy = knn_classifier(X_train,X_test,Y_train,Y_test)
        print(f"KNN classifier accuracy : {accuracy}")
        knn_save = input("Do you want to save this model ? (y/n) : ")
        if knn_save == 'y':
            save_model(model_knn,'KNN_model')
            print("Model saved successfully")

    elif classifier == '2':
        model_gnb,accuracy = naive_bayes_gaussian(X_train,X_test,Y_train,Y_test)
        print(f"Gaussian Naive Bayes classifier accuracy : {accuracy}")
        gnb_save = input("Do you want to save this model ? (y/n) : ")
        if gnb_save == 'y':
            save_model(model_gnb,'GNB_model')
            print("Model saved successfully")

    elif classifier == '3':
        model_mnb,accuracy = naive_bayes_multinomial(X_train,X_test,Y_train,Y_test)
        print(f"Multinomial Naive Bayes classifier accuracy : {accuracy}")
        mnb_save = input("Do you want to save this model ? (y/n) : ")
        if mnb_save == 'y':
            save_model(model_mnb,'MNB_model')
            print("Model saved successfully")

    elif classifier == '4':
        model_rfc,accuracy = random_forest_classifier(X_train,X_test,Y_train,Y_test)
        print(f"Random Forest classifier accuracy : {accuracy}")
        rfc_save = input("Do you want to save this model ? (y/n) : ")
        if rfc_save == 'y':
            save_model(model_rfc,'RFC_model')
            print("Model saved successfully")

    elif classifier == '5':
        model_dtc,accuracy = decision_tree_classifier(X_train,X_test,Y_train,Y_test)
        print(f"Decision Tree classifier accuracy : {accuracy}")
        dtc_save = input("Do you want to save this model ? (y/n) : ")
        if dtc_save == 'y':
            save_model(model_dtc,'DTC_model')
            print("Model saved successfully")

    elif classifier == '6':
        model_gbc,accuracy = gradient_boosting_classifier(X_train,X_test,Y_train,Y_test)
        print(f"Gradient Boosting classifier accuracy : {accuracy}")
        gbc_save = input("Do you want to save this model ? (y/n) : ")
        if gbc_save == 'y':
            save_model(model_gbc,'GBC_model')
            print("Model saved successfully")
    elif classifier == '0':
        model_knn,accuracy_knn = knn_classifier(X_train,X_test,Y_train,Y_test)
        model_gnb,accuracy_gnb = naive_bayes_gaussian(X_train,X_test,Y_train,Y_test)
        model_mnb,accuracy_mnb = naive_bayes_multinomial(X_train,X_test,Y_train,Y_test)
        model_rfc,accuracy_rfc = random_forest_classifier(X_train,X_test,Y_train,Y_test)
        model_dtc,accuracy_dtc = decision_tree_classifier(X_train,X_test,Y_train,Y_test)
        model_gbc,accuracy_gbc = gradient_boosting_classifier(X_train,X_test,Y_train,Y_test)
        print(f"KNN classifier accuracy : {accuracy_knn}")
        print(f"Gaussian Naive Bayes classifier accuracy : {accuracy_gnb}")
        print(f"Multinomial Naive Bayes classifier accuracy : {accuracy_mnb}")
        print(f"Random Forest classifier accuracy : {accuracy_rfc}")
        print(f"Decision Tree classifier accuracy : {accuracy_dtc}")
        print(f"Gradient Boosting classifier accuracy : {accuracy_gbc}")
        accuracy = [accuracy_knn,accuracy_gnb,accuracy_mnb,accuracy_rfc,accuracy_dtc,accuracy_gbc]
        max_accuracy = max(accuracy)
        if max_accuracy == accuracy_knn:
            save_model(model_knn,'KNN_model')
        elif max_accuracy == accuracy_gnb:
            save_model(model_gnb,'GNB_model')
        elif max_accuracy == accuracy_mnb:
            save_model(model_mnb,'MNB_model')
        elif max_accuracy == accuracy_rfc:
            save_model(model_rfc,'RFC_model')
        elif max_accuracy == accuracy_dtc:
            save_model(model_dtc,'DTC_model')
        elif max_accuracy == accuracy_gbc:
            save_model(model_gbc,'GBC_model')
        print("Model saved successfully")
    else:
        print("Invalid classifier")
        return

if __name__ == '__main__':
    main('iris.csv')

