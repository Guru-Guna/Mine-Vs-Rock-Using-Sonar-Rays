# MINE_VS_ROCK_PREDICTION USING MACHINE LEARNING    
 
## OVERVIEW   
 
This project is about predicting rocks against Mines by the SONAR technology with the help of Machine Learning. SONAR is an abbreviated form of Sound Navigation and Ranging. It uses sound waves to detect objects underwater. Machine learning-based tactics, and deep learning-based approaches have applications in detecting sonar signals and hence targets.The three stages of Machine Learning are taking some data as input, extracting features, and predicting new patterns. The most common ML 
algorithms in this field are Logistic Regression, support vector machine, principal component analysis, k-nearest neighbors (KNN), etc.  

## OBJECTIVE      
   
The main aim is to predict the rock or mine in the underwater(sea , oceans) using SONAR that uses sound propagation (usually underwater, as in submarine navigation) to navigate, measure distances (ranging), communicate with or detect objects on or under the surface of the water , which will help the sea divers , submarines to know whether the object is mine or rock . I am using machine learning algorithms to predict these by using the dataset.
 
## LIBRARIES USED    

A Python library is a collection of related modules. It contains bundles of code that can be used repeatedly in different programs. It makes Python Programming simpler and convenient for the programmer. As we don’t need to write the same code again and again for different programs. Python libraries play a very vital role in fields of Machine Learning, Data Science, Data Visualization, etc.Python libraries that are used in the project are:
• Pandas
• Pickle 
• Numpy   
• Matplotlib   
  
## MODULES DESCRIPTION   
 
### Data Acquisition and Data Preprocessing: 
• Here the dataset is downloaded from https://www.kaggle.com/datasets/reshmaduseja/rock-vs-mine-predictionmachine-learning
• The dataset has been collected from UCI Repository. It has come across 61 features which define and differentiate Rocks and Mines and comprises of 209 samples. This data is used for training and testing purpose. The Last column in this dataset indicates that, whether it's a mine or a rock, which is useful in prediction.The dataset is included in this repository.
• The dataset is now pre processed to get the summary statistics of the dataset to decide the optimal prediction..

### Feature Extraction:     
Feature extraction refers to the process of transforming raw data into numerical features that can be processed while preserving the information in the original data set. It yields better results than applying machine learning directly to the raw data. 
Feature extraction can be accomplished manually or automatically:
• Manual feature extraction requires identifying and describing the features that are relevant for a given problem and implementing a way to extract those features. 
• Automated feature extraction uses specialized algorithms or deep networks to extract features automatically from signals or images without the need for human intervention. This technique can be very useful when you want to move quickly from raw data to developing machine learning algorithms. 

### Training and testing of the Model: 
 Next we train the machine to recognize mine and rock and pick the appropriate intent and we use the Machine Learning Algorithm like the Logistic regression for the complete training of the model. Every time we make changes to the dataset we need to train the machine to include the changes that have been made.It will train about 60% of the dataset inorder to make the machine well trained for the inputs
• Training — Up to 75 percent of the total dataset is used for training. The model learns on the training set; in other words, the set is used to assign the weights 
and biases that go into the model.
• Validation — Between 15 and 20 percent of the data is used while the model is being trained, for evaluating initial accuracy, seeing how the model learns and fine-tuning hyperparameters. The model sees validation data but does not use it to learn weights and biases.
• Test — Between five and 10 percent of the data is used for final evaluation. Having never seen this dataset, the model is free of any of its bias..

### Implementing User Interface using Streamlit:
 Now we create the actual web application that makes use of all the processed and trained data that we have up till now to take in user queries and give proper responses. Here we use the streamlit python library for the user interface. Here we load the model created and using the saved models from the google colab using pickle library and using various functions to take the user input and predict the responses. Once we run this web application ,it will ask to enter 60 inputs that are nothing but the frequencies that are taken from 60 different angles it will predict whether the object is mine or rock.

### Flow Chart
![Flow Chart](https://github.com/Guru-Guna/Mine-Vs-Rock-Using-Sonar-Rays/assets/93427255/f45f7396-74f9-43ee-a930-715289531b24)

## Program
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('/content/sonar_data.csv',header=None)

#DataFrame is a 2-dimensional labeled data structure with columns of potentially different types.
df.head()

pip install gradio
pip install tiktoken
pip install cohere
pip install openai
pip install cohere
df.head(10)
df.describe()
print(df.columns)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df[60].value_counts()

X=df.drop(columns=60,axis=1)
Y=df[60]

print(X)
print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)


neighbors = np.arange(1,14)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

print(X)
for column in X_train.columns:
  print(column)

for i,k in enumerate(neighbors):
 knn = KNeighborsClassifier(n_neighbors=k)
 knn.fit(X_train, y_train)
 train_accuracy[i] = knn.score(X_train, y_train)
 test_accuracy[i] = knn.score(X_test, y_test)

plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

plt.title('Logistic Regression Accuracy Testing')
plt.plot(neighbors,test_accuracy,label='Testing Accuracy', marker=".", markersize=20, markeredgecolor="red", markerfacecolor="green")
plt.plot(neighbors, train_accuracy, label='Training Accuracy', marker=".", markersize=20,markeredgecolor="blue", markerfacecolor="green")
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')

knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

print(X.shape,X_train.shape,X_test.shape)

print(X_train)
print(y_train)
print(X_test)
print(y_test)

model=LogisticRegression()
model.fit(X_train,y_train)

print("The accuracy score for the KNN neighbours is :")
knn.score(X_test,y_test)

#The crosstab() function is used to compute a simple cross tabulation of two (or more) factors.
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

print("The accuracy score for the logistic regression is : ")
score=model.score(X_test,y_test)
print(score)

prediction=model.predict(X_test)


pd.crosstab(y_test, prediction, rownames=['True'], colnames=['Predicted'], margins=True)


#input_data is the data we should provide to test our system.
input_data=(0.0392,0.0108,0.0267,0.0257,0.0410,0.0491,0.1053,0.1690,0.2105,0.2471,0.2680,0.3049,0.2863,0.2294,0.1165,0.2127,0.2062,0.2222,0.3241,0.4330,0.5071,0.5944,0.7078,0.7641,0.8878,0.9711,0.9880,0.9812,0.9464,0.8542,0.6457,0.3397,0.3828,0.3204,0.1331,0.0440,0.1234,0.2030,0.1652,0.1043,0.1066,0.2110,0.2417,0.1631,0.0769,0.0723,0.0912,0.0812,0.0496,0.0101,0.0089,0.0083,0.0080,0.0026,0.0079,0.0042,0.0071,0.0044,0.0022,0.0014)
input_data_Array=np.array(input_data)
input_data_reshaped=input_data_Array.reshape(1,-1)
predict=model.predict(input_data_reshaped)
if predict[0]=='R':
    print('Safe,Its just a  Rock')
else:
     print('DANGER,its MINE')

input_data=(0.0453,0.0523,0.0843,0.0689,0.1183,0.2583,0.2156,0.3481,0.3337,0.2872,0.4918,0.6552,0.6919,0.7797,0.7464,0.94441,0.8874,0.8024,0.7818,0.5212,0.4052,0.3957,0.3914,0.325,0.32,0.3271,0.2767,0.4423,0.2028,0.3788,0.2947,0.1984,0.2341,0.1306,0.4182,0.3835,0.1057,0.184,0.197,0.1674,0.0583,0.1401,0.1628,0.0621,0.0203,0.053,0.0742,0.0409,0.0061,0.0125,0.0084,0.0089,0.0048,0.0094,0.0191,0.014,0.0049,0.0052,0.0044,0.03)
input_data_Array=np.array(input_data)
input_data_reshaped=input_data_Array.reshape(1,-1)
predict=model.predict(input_data_reshaped)
print(predict)
if predict[0]=='R':
    print('Safe,Its just a  Rock')
else:
    print('DANGER,its MINE')


input_data=(0.0392,0.0108,0.0267,0.0257,0.0410,0.0491,0.1053,0.1690,0.2105,0.2471,0.2680,0.3049,0.2863,0.2294,0.1165,0.2127,0.2062,0.2222,0.3241,0.4330,0.5071,0.5944,0.7078,0.7641,0.8878,0.9711,0.9880,0.9812,0.9464,0.8542,0.6457,0.3397,0.3828,0.3204,0.1331,0.0440,0.1234,0.2030,0.1652,0.1043,0.1066,0.2110,0.2417,0.1631,0.0769,0.0723,0.0912,0.0812,0.0496,0.0101,0.0089,0.0083,0.0080,0.0026,0.0079,0.0042,0.0071,0.0044,0.0022,0.0014)
input_data_Array=np.array(input_data)
input_data_reshaped=input_data_Array.reshape(1,-1)
predict=model.predict(input_data_reshaped)
if predict[0]=='R':
    print('Safe,Its just a  Rock')
else:
     print('DANGER,its MINE')

input_data=(0.027,0.0163,0.0341,0.0247,0.0822,0.1256,0.1323,0.1584,0.2017,0.2122,0.221,0.2399,0.2964,0.4061,0.5095,0.5512,0.6613,0.6804,0.652,0.6788,0.7811,0.8369,0.8969,0.9856,1,0.9395,0.8917,0.8105,0.6828,0.5572,0.4301,0.3339,0.2035,0.0798,0.0809,0.1525,0.2626,0.2456,0.198,0.2412,0.2409,0.1901,0.2077,0.1767,0.1119,0.0779,0.1344,0.096,0.0598,0.033,0.0197,0.0189,0.0204,0.0085,0.0043,0.0092,0.0138,0.0094,0.0105,0.0093)
input_data_Array=np.array(input_data)
input_data_reshaped=input_data_Array.reshape(1,-1)
predict=model.predict(input_data_reshaped)
if predict[0]=='R':
  print("Safe it's just a Rock")
else:
  print("DANGER,it's MINE")


"""ACCURACY FOR VARIOUS ALGORITHMS"""
def models(X_train,y_train):


    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state=0)
    log.fit(X_train,y_train)


    from sklearn.neighbors import KNeighborsClassifier
    knn =  KNeighborsClassifier(n_neighbors = 5 , metric = 'minkowski',p =2)
    knn.fit(X_train,y_train)

    from sklearn.svm import SVC
    svc_lin = SVC(kernel='linear',random_state= 0)
    svc_lin.fit(X_train,y_train)

    from sklearn.svm import SVC
    svc_rbf = SVC(kernel='rbf',random_state= 0)
    svc_rbf.fit(X_train,y_train)

    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(X_train,y_train)

    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy',random_state = 0)
    tree.fit(X_train,y_train)


    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators =10, criterion='entropy',random_state = 0)
    forest.fit(X_train,y_train)

    print("MACHINE LEARNING ALGORITHM APLLIED CHECK OUT ACCURACY")
    print("")
    print("[0] logistic regression Accuracy             :" ,log.score(X_train,y_train))
    print("")
    print("[1]  K N N algorithm  Accuracy               :" ,knn.score(X_train,y_train))
    print("")
    print("[2] Support Vector Machine(svc) Accuracy     :" ,svc_lin.score(X_train,y_train))
    print("")
    print("[3] Support Vector Machine(rbf)  algorithm   :" ,svc_rbf.score(X_train,y_train))
    print("")
    print("[4] Naive Bayes Accuracy Accuracy            :" ,gauss.score(X_train,y_train))
    print("")
    print("[5] Desion Tree Algorithm Accuracy           :" ,tree.score(X_train,y_train))
    print("")
    print("[6] Random Forest Accuracy                   :" ,forest.score(X_train,y_train))
    print("")

    return log,knn,svc_lin,svc_rbf,gauss,tree,forest

model = models(X_train,y_train)

"""SAVING THE TRAINED MODEL:

"""

import pickle

filename='minevsrock.sav'
pickle.dump(model,open(filename,'wb'))

loaded_model = pickle.load(open('minevsrock.sav', 'rb'))
```

## Output

### Varying Number of Neighbours
![image](https://github.com/Guru-Guna/Mine-Vs-Rock-Using-Sonar-Rays/assets/93427255/ddfc9f6e-f4f9-4206-a9c9-67ac6dceaeb2)
### Logistic Regression Accuracy Testing
![image](https://github.com/Guru-Guna/Mine-Vs-Rock-Using-Sonar-Rays/assets/93427255/1ffc1803-4616-464a-b3b8-34974e69a293)
### Simple Cross Tabulation
 ![Simple Crosstab](https://github.com/Guru-Guna/Mine-Vs-Rock-Using-Sonar-Rays/assets/93427255/85875f72-8e18-43cd-a0f8-979a3ce95bf2)
### Machine Learnig Algorithm Applied Check Out Accuracy
![Machine Learning Algorithm for Accuracy](https://github.com/Guru-Guna/Mine-Vs-Rock-Using-Sonar-Rays/assets/93427255/72ad7a27-bd14-4ddb-b29c-445fd1c6cd23)

## Result








