# -*- coding: utf-8 -*-
"""
@author: Suresh Reddy Nusi
"""
"""
Basic approach
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.linear_model import LinearRegression

#defining data to work with. The inputs (regressors, 𝑥) and output (predictor, 𝑦) should be array
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])
print(x)
print(y)


y=mx+c
5=0+c
#create a model and fit it.
model = LinearRegression()
model.fit(x, y)
#model = LinearRegression().fit(x, y)

#get results
y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')

#getting the intercept
print('intercept:', model.intercept_)
#slope
print('slope:', model.coef_)
print(model.coef_)
#formula for calculating
y_pred = model.intercept_ + model.coef_ * x
print('predicted response:', y_pred, sep='\n')

#visualizing the actual and predicted output
df1 = pd.DataFrame({'Actual': y.flatten(), 'Predicted': y_pred.flatten()})
df1.plot(kind='bar')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

#Accuracy calculation
print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, y_pred)))



"""
Example:2
"""


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#loading the data
data = pd.read_csv('datasets_21716_27925_50_Startups.csv')
#shaping and description
data.shape
data.describe()

#preparing the data into the model
X = data.iloc[:,:-1].values
y = data.iloc[:,4].values

#Encoding categorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#splitting the dataset into test and training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.2, random_state= 0)
#fitting multiple regression model to the training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#predicting the test set results
y_pred = regressor.predict(X_test)
#accuracy part
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#visualizing the data
df1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df1.plot(kind='bar')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


"""
Advertising the data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score

#Loading the data
df = pd.read_csv('Advertising.csv')
df.head()
df.describe()

#taking care of missing values
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values='NaN', strategy = 'sum', axis = 0)
#imputer = imputer.fit(X[:,1:3])
#X[:, 1:3] = imputer.transform(X[:,1:3])

#Filling missing data
#print(df.fillna(value=5))

#Visualizing the each variable to understand the data
sns.distplot(df['Sales'],kde = False)
plt.show()

sns.distplot(df['TV'],kde = False)
plt.show()

sns.distplot(df['Radio'],kde = False)
plt.show()

sns.distplot(df['Newspaper'],kde = False)
plt.show()


sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=7, aspect=0.7, kind='reg');
plt.show()

#correlation plot to understand each variable which is correlated with the output
df.TV.corr(df.Sales)
df.corr()

#heatmap gives the full description
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True )
plt.show()

#model data
X = df.iloc[:,:-1].values
y = df.iloc[:,3].values

#Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#fitting multiple regression model to the training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#predicting the test set results
y_pred = regressor.predict(X_test)
#Accuracy
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
"""
R-squared or R2 explains the degree to which your input variables explain the variation of your output / predicted variable. 
So, if R-square is 0.8, it means 80% of the variation in the output variable is explained by the input variables. So, in simple terms, higher the R squared, 
the more variation is explained by your input variables and hence better is your model.
"""
print('r2 score for perfect model is', r2_score(y_test, y_pred) ) 

#r2 score calculating manually
SS_Residual = sum((y_test-y_pred)**2)       
SS_Total = sum((y_test-np.mean(y_test))**2)     
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-X.shape[1]-1)
print(r_squared, adjusted_r_squared)

