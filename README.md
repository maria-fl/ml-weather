# ml-weather
Der folgende Code wird im Rahmen der Bachelorarbeit «Die Nutzung und Anwendung des maschinellen Lernens mit dem Ziel, Entscheidungen zu treffen» im Studienfach Informatik und Wirtschaft geschrieben.
Mit diesem Programm werden Vorhersagemodelle auf der Basis der folgenden maschinelles Lernen-Algorithmen erstellt: GaussianNB, logistische Regression, KNN, Entscheidungsbäume und Multilayer-Perzeptron. Die Vorhersagen beziehen sich auf das Wetter, und zwar ob am nächsten Tag an verschiedenen bestimmten Orten in Australien regnen wird oder nicht. Die Datei heißt 
«weatherAUS.csv» (https://www.kaggle.com/jsphyg/weather-dataset-rattle-package#weatherAUS.csv)
Der Code besteht aus 6 Funktionen. 5 davon beziehen sich auf die Generierung der Modelle basierend auf 6 maschinelle Lernen-Algorithmen und die andere auf die Erscheinung statistischer Daten (Mittelwert, Standardabweichung, Minimum, Maximum, Median) der Attribute des Datensatzes und einer grafischen Darstellung. Die Funktion preprocessDataset liest und verarbeitet den Datensatz. Die Funktion splitDataset  trennt den Datensatz in Training- und Testset. Die Funktion calculateProfit berechnet den resultierenden Gewinn aus der Verwendung von Algorithmen. Die Funktion trainML trainiert das Modell anhand des Trainingset und des Algorithmus.  Die Funktion evaluationModel macht die Vorhersagen und lässt die Metriken erscheinen. Die printDescription bezieht sich auf die Erscheinung statistischer Daten und einer grafischen Darstellung.

Beispiel für die Codeausführung:
Wenn man den Code ausführt, wird das folgende auf der Konsole angezeigt:
Type 1 for displaying statistical characteristics of attributes and 2 for creating ml models and 3 for exit 

Wenn man auf 1 drückt, dann erscheinen die statistischen Daten.
Wenn man auf 2 drückt, kommt das folgende:
Select one of the following algorithms:
 GB for GaussianNB,
 LR for Logistic Regression,
 KNN for KNeighbors,
 DT for Decision Trees,
 RF for RandomForest, 
 MLP for Multilayer-Perzeptron

GB
GaussianNB
[[ 1312  1677]
 [ 1097 13365]]
accuracy= 0.8410406280442382
              precision    recall  f1-score   support

         Yes     0.5446    0.4389    0.4861      2989
          No     0.8885    0.9241    0.9060     14462

    accuracy                         0.8410     17451
   macro avg     0.7166    0.6815    0.6960     17451
weighted avg     0.8296    0.8410    0.8341     17451

GaussianNB profit/loss:  41460000

Und wenn man auf 3 drückt,  ist das Programm beendet.

