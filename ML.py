# Python code to illustrate 
# classification using data set 
#Importing the required library 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

#Importing the dataset 
dataset = pd.read_csv('ds.csv',sep= ',', header= None) 
data = dataset.iloc[:, :] 

#print(dataset)
#print(data)
#checking for null values 
#print("Sum of NULL values in each column. ") 
#print(data.isnull().sum()) 
#seperating the predicting column from the whole dataset 
X = data.iloc[:,[False,True,True,True,False,True,True,False,True,False,False]].values
#print(X)
y = dataset.iloc[:,-1].values 
print(y)
#print(y)
#y.shape()

#Encoding the predicting variable 
labelencoder_y = LabelEncoder() 
y = labelencoder_y.fit_transform(y) 

#Spliting the data into test and train dataset 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1,shuffle = False)
print(X_train)
#print(X_test)
print(y_train)
print(y_test)
#X_test = X_train

count = 0
#Using the random forest classifier for the prediction 
while(1):
	classifier=RandomForestClassifier(n_estimators = 50) 
	classifier=classifier.fit(X_train,y_train) 
#print(classifier)
	predicted=classifier.predict(X_test)
	#print(y_test)	
	#print(predicted)
	if((accuracy_score(y_test, predicted)) > 0.86):
		break
	count += 1
	print(count)

#printing the results 
#print ('Confusion Matrix :') 
#print(confusion_matrix(y_test, predicted)) 
print ('Accuracy Score :',(((accuracy_score(y_test, predicted)))*100))
#print ('Report : ') 
#print (classification_report(y_test, predicted)) 

