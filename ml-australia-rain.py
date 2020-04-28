# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 12:12:40 2020

@author: Maria
"""
#Bibliotheken importieren
import pandas as pd
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble.forest import RandomForestClassifier
import warnings


warnings.filterwarnings("ignore")

#so werden pandas dataframe auf der Konsole gedruckt werden
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)



def preprocessDataset(filename):
    """ die Funktion lädt die CSV-Datei mit den Daten und verarbeitet den Datensatz
    Bei der Vorverarbeitung behalten wir nur die Monate 10,11,12,1,2  und verwendet nur den Monat statt das ganzes Datum.
    

    
    Parameters
    ---------- 
       filename (str): Dateiname/der Zielpfad der Datei 

    Returns
    ----------
         datasetColumnsWeWant: DataFrame
    """
    
    parser=lambda x: pd.datetime.strptime(x, "%Y-%m-%d")
    dataset = pd.read_csv(filename, header = 0,parse_dates=["Date"],date_parser=parser)
    dataset = dataset[( (pd.to_datetime(dataset["Date"]).dt.month == 10) | (pd.to_datetime(dataset["Date"]).dt.month == 11) | (pd.to_datetime(dataset["Date"]).dt.month == 12) | (pd.to_datetime(dataset["Date"]).dt.month == 1) | (pd.to_datetime(dataset["Date"]).dt.month == 2)  )]
    dataset["Month"] = pd.to_datetime(dataset["Date"]).dt.month
    
    datasetColumnsWeWant = dataset[["Date","Month","Location","MinTemp","MaxTemp","Rainfall","WindSpeed9am","WindSpeed3pm","Humidity9am","Humidity3pm","Pressure9am","Pressure3pm","Temp9am","Temp3pm","RainToday","RainTomorrow"]]
    # ich lösche die instances, die leere Werte(nan) enthalten
    datasetColumnsWeWant.dropna(inplace=True)
    datasetColumnsWeWant.reset_index(drop=True,inplace=True)  
   # Sortierung nach Datum 
    datasetColumnsWeWant.sort_values(by=["Date"],inplace=True)
    datasetColumnsWeWant.reset_index(drop=True,inplace=True)
   
    
    
  # Erzeugung eines LabelEncoder, das le bennenen wird und es ist Objekt der Klasse LabelEncoder aus der Bibliothek preprocessing
    le = preprocessing.LabelEncoder()
    # ich wandle die Strings in Nummern für die Attribute RainToday und Location um
    raintoday_encoded=le.fit_transform(datasetColumnsWeWant[["RainToday"]]) 
    location_encoded = le.fit_transform(datasetColumnsWeWant[["Location"]]) 
    
  # ich füge die neuen modifizierten Spalten in Dataset ein, welche nun Zahlen enthalten
    datasetColumnsWeWant["RainToday"] = raintoday_encoded
    datasetColumnsWeWant["Location"] = location_encoded
    datasetColumnsWeWant = datasetColumnsWeWant[["Month","Location","MinTemp","MaxTemp","Rainfall","WindSpeed9am","WindSpeed3pm","Humidity9am","Humidity3pm","Pressure9am","Pressure3pm","Temp9am","Temp3pm","RainToday","RainTomorrow"]]

    
    return datasetColumnsWeWant

def printDescription(dataset):
    
    """
    die Funktion nimmt ein DataFrame, welches das Dataset ist und druckt in der Konsole die Beschreibende
    Eigenschaften des Attributes und der Ausgabe aus
    
   
    
    Parameters
    ---------- 
       dataset (DataFrame): Der Datensatz, der alle instances enthält
    """
   
#   Erscheinung der beschreibenden Eigenschaften des Datensatzes und einer grafischen Darstellung
    print(dataset.head(5)) # print für die erste 5 Zeilen des Datensatzes
    print(dataset.tail(5)) # print für die letzte 5 Zeilen des Datensatzes
    datasetColumnsWeWant = dataset[["MinTemp","MaxTemp","Rainfall","WindSpeed9am","WindSpeed3pm","Humidity9am","Humidity3pm","Pressure9am","Pressure3pm","Temp9am","Temp3pm"]]
    description = datasetColumnsWeWant.describe(percentiles=[])
    my_colors = ["blue", "red"]
    dataset["RainTomorrow"].value_counts().plot(kind="bar",colors=my_colors)
    
    print(description)
    class_counts = dataset.groupby("RainTomorrow").size()
    print(class_counts)
    
    percentages = [class_counts[0]/sum(class_counts),class_counts[1]/sum(class_counts)]
    print(percentages)

    
    
def splitDataset(dataset,trainSplit):
    """ 
    Die Funktion nimmt ein DataFrame, welches das Datensatz ist und gibt das training- und testset zurück.
    Wird also das Datensatz in training- und testset unterteilt. Ich wähle aus, dass das Trainingsset 70% der instances des Datensatzes enthält und die restlichen 30% befinden sich im Testset.
    len gibt zürück, wie viele Zeile der Datensatz hat.
    Wird das 70% der Zeilen des trainingset berechnet und machen wir es floor, um Dezimalzahlen zu vermeiden 
    
    
    
    Parameters
    ---------- 
       dataset (DataFrame):  Der Datensatz, der alle instances enthält
       trainSplit (float): der Prozentsatz des trainingset(z.B. 0.7) in Bezug auf das gesamtes Datensatzes 
    Returns
    ----------
        trainingset: DataFrame
        testset: DataFrame
    """
    lines=len(dataset)
    trainingsetNumberOfInstances = math.ceil(lines*trainSplit)
    #print(trainingsetNumberOfInstances)
    #testsetNumberOfInstances = len(dataset)-trainingsetNumberOfInstances
    #print(testsetNumberOfInstances)
    trainingset = dataset.loc[0:trainingsetNumberOfInstances-1,:]
    testset = dataset.loc[trainingsetNumberOfInstances:lines-1,:]
    #print("trainingset",trainingset)
    #print("testset",testset)  
    return trainingset,testset

def calculateProfit(cm):
    """ 
    die Funktion nimmt ein Confusion Matrix und nach Berechnung gibt das Gewinn/Verlust zurück
    
    
    Parameters
    ---------- 
       cm (array): confusion metrix

    Returns
    ----------
        profit: float 
    """
    TP=cm[0,0]
    FP=cm[1,0]
    TN=cm[1,1]
    FN=cm[0,1]
    
    profit = (TN * 30000) - (FN * 200000 + TP * 10000 + FP * 10000)
    
    return profit

def trainML(trainingset,mlAlgo):
    """ 
    die Funktion nimmt das trainingset und den Klassifikator/Algorithmus.
    Das Training  wird anhand des trainingset und des Algorithmus.
    Die Funktion gibt das resultierende Modell zurück
    
    
    Parameters
    ---------- 
       trainingset (DataFrame): trainingset
      
       mlAlgo (Classifier): ist der zu verwendende Algorithmus

    Returns
    ----------
        model: das Modell, das trainiert hat: Classifier
    """
    columnsWeWant = ["Month","Location","MinTemp","MaxTemp","Rainfall","WindSpeed9am","WindSpeed3pm","Humidity9am","Humidity3pm","Pressure9am","Pressure3pm","Temp9am","Temp3pm","RainToday"]
    
#    Teilung in X und Y
#    X sind die Attribute und Y ist die Ausgabe, d.h. Ja oder Nein
    trainX = trainingset[columnsWeWant].values
    trainY = trainingset[["RainTomorrow"]].values.ravel()
    
    # Training des Modells
    model = mlAlgo.fit(trainX, trainY)
    
    return model

def evaluationModel(model,testset):
    """
    die Funktion hat als Parametern das model und das Testset. Auf der Basis diesen Parametern werden die Vorhersagen gemacht
    und danach druckt die Funktion die Ergebnisse des Confusion Matrix und den Metriken aus. Und gibt das Confusion Matrix zurück.
    
    
    
    
    Parameters
    ---------- 
       model (Scikit- learn Classifier):ein Model, welches type Scikit- learn Classifier ist
       testset (DataFrame): testset
      

    Returns
    ----------
        cm o pinakas sygxisis : array
    """
    columnsWeWant = ["Month","Location","MinTemp","MaxTemp","Rainfall","WindSpeed9am","WindSpeed3pm","Humidity9am","Humidity3pm","Pressure9am","Pressure3pm","Temp9am","Temp3pm","RainToday"]
#   testX wird die obere Spalten, d.h. die Attribute enthalten
   
    testX = testset[columnsWeWant].values
    # werden Vorhersagen gemacht, die nur mit testX werden
    predictions=model.predict(testX)
   
    testY = testset[["RainTomorrow"]].values.ravel() 
#    Confusion Matrix wird berechnet und gedruckt 
    cm=confusion_matrix(testY, predictions,labels=["Yes","No"])
    print(cm)
    
#    Accuracy wird berechnet und gedruckt
    ac=accuracy_score(testY, predictions)
    print("accuracy=",ac)
    #Ausdrucken den Metriken: precision recall f1-score 
    print(classification_report(testY, predictions, labels=["Yes","No"], digits=4))
    
    return cm

    
#Main Definition
if __name__ == "__main__":
    
    dataset = preprocessDataset("weatherAUS.csv")
    
    trainingset,testset=splitDataset(dataset,0.7)
    answer=(input("Type 1 for displaying statistical characteristics of attributes and 2 for creating ml models and 3 for exit \n"))
    if answer.strip() == "1": 
        printDescription(dataset)
    elif answer.strip() == "2":
        choice = ""
        while choice != "n":
            algorithm=(input("Select one of the following algorithms:\n GB for GaussianNB,\n LR for Logistic Regression,\n KNN for KNeighbors,\n DT for Decision Trees,\n RF for RandomForest, \n MLP for Multilayer-Perzeptron \n"))
            algorithm = algorithm.strip()
            if algorithm.lower() == "gb":
                print("GaussianNB")
                classifier = GaussianNB()
                model=trainML(trainingset,classifier)
                cm=evaluationModel(model,testset)
                profit= calculateProfit(cm)
                print("GaussianNB profit/loss: ", profit)
            elif algorithm.lower() == "dt":   
                
                criterions =["gini","entropy"]
                for criterion in criterions:
                    for maxDepth in range(3,11):
                        print("DecisionTreeClassifier",criterion,maxDepth)
                        classifier = DecisionTreeClassifier(random_state=0,criterion=criterion, max_depth=maxDepth)
                        model=trainML(trainingset,classifier)
                        cm=evaluationModel(model,testset)
                        profit= calculateProfit(cm)
                        print("DecisionTreeClassifier profit/loss: ", profit)
                
            elif algorithm.lower() == "knn":   
                 for p in range(1,3):
                    for nn in range(1,22,2):
                        print("KNeighborsClassifier",nn,p)
                        classifier = KNeighborsClassifier(n_neighbors=nn,p=p)#panta einai perittos arithmos to n_neighbors
                        model=trainML(trainingset,classifier)
                        cm=evaluationModel(model,testset)
                        profit= calculateProfit(cm)
                        print("KNeighborsClassifier profit/loss: ", profit)
            elif algorithm.lower() == "lr":    
                print("LogisticRegression")
                classifier = LogisticRegression()
                model=trainML(trainingset,classifier)
                cm=evaluationModel(model,testset)
                profit= calculateProfit(cm)
                print("LogisticRegression profit/loss: ", profit)
                
            elif algorithm.lower() == "mlp":         
                layers = [(50),(100),(150),(50,50),(100,100),(150,150),(50,50,50),(100,100,100),(150,150,150)]
                for l in layers:
                    print("MLPClassifier ",l)
                    classifier = MLPClassifier(random_state=0,hidden_layer_sizes=l,activation="logistic")
                    model=trainML(trainingset,classifier)
                    cm=evaluationModel(model,testset) 
                    profit= calculateProfit(cm)
                    print("MLPClassifier profit/loss: ", profit)
            elif  algorithm.lower() == "rf":    
                criterions =["gini","entropy"]
                for criterion in criterions:
                    for maxDepth in range(3,11):
                        print("RandomForestClassifier",criterion,maxDepth)
                        classifier=RandomForestClassifier(random_state=0,criterion=criterion, max_depth=maxDepth)
                        model=trainML(trainingset,classifier)
                        cm=evaluationModel(model,testset)  
                        profit= calculateProfit(cm)
                        print("RandomForestClassifier profit/loss: ", profit)
            else:
                print("This algorithm does not exist")
            choice = input("Would you like to continue? (y/n):")
            choice = choice.strip()
            while choice != 'y' and choice != 'Y' and choice != 'n' and choice != 'N':
                print("Wrong choice \n")
                choice = input("Would you like to continue? (y/n):")
    elif answer.strip() == '3':
        print("Exit")
    else:
        print("Wrong Choice, Exit")