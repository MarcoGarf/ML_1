import pandas as pd 
from sklearn.model_selection import train_test_split    #Hints profesor told us 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


# Load training data
X_train = pd.read_csv('X_train_passengers.csv')
Y_train = pd.read_csv('Y_train_passengers.csv')

# Load testing data
X_test = pd.read_csv('X_test_passengers.csv')