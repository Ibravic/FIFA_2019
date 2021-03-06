# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 21:40:01 2021

@author: Ibrahim
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as stats
import seaborn as sns
from sklearn import linear_model


df_fifa=pd.read_excel("f:\\Personal\\Data Analysis\\Fifa2019Data.xlsx")

#Missing Data
freq=df_fifa["Crossing"].value_counts().idxmax()
df_fifa["Crossing"].replace(np.nan,freq , inplace=True)

df_fifa.dropna(axis=0,inplace=True)

#Data Formatting
real_madrid={"R.Madrid":"Real Madrid","Real M.":"Real Madrid","R.M":"Real Madrid"}

df_fifa["Club"].replace(real_madrid.keys(),real_madrid.values(),inplace=True)


df_fifa["Value"]=df_fifa["Value"]/1000000
df_fifa.rename(columns={"Value":"Value in Millions"},inplace=True)



#Categorize Data
bins=np.linspace(min(df_fifa["Value in Millions"]),max(df_fifa["Value in Millions"]),4)

group_names=["low","medium","high"]

df_fifa["value-group"]=pd.cut(df_fifa["Value in Millions"],bins,labels=group_names,include_lowest=True)

#Descriptive Statistics
value_group_count=df_fifa["value-group"].value_counts().to_frame()

value_group_count.rename(columns={"value-group":"category-counts"},inplace=True)

#using box plot
df_fifa["Age"].plot(kind="box")

df_fifa["Age"].plot(kind="box",figsize=(15,10),grid="on")

#Scatter plot
plt.figure(figsize=(15,10))
plt.scatter(df_fifa["Age"],df_fifa["Value in Millions"])
plt.grid()
plt.xlabel("Age")
plt.ylabel("Values in Millions")
plt.show()


plt.figure(figsize=(15,10))
plt.scatter(df_fifa["Overall"],df_fifa["Value in Millions"])
plt.grid()
plt.xlabel("Overall")
plt.ylabel("Values in Millions")
plt.show()


#group data
df_nation=df_fifa[["Nationality","Overall"]]

df_grp=df_nation.groupby(["Nationality"],as_index=False).mean()

df_grp.sort_values(by="Overall",ascending=False, inplace=True)

df_nation=df_fifa[["Nationality","Position","Overall"]]

df_grp=df_nation.groupby(["Nationality","Position"],as_index=False).mean()

df_grp.pivot(index="Nationality",columns="Position")

#Correlation
p_coef,p_value=stats.pearsonr(df_fifa["Overall"],df_fifa["Value in Millions"])

p_coef2,p_value2=stats.pearsonr(df_fifa["Age"],df_fifa["Value in Millions"])

#headmap
corr=df_fifa.corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)

df_corr=df_fifa[["Age","Overall","Potential","Finishing","SprintSpeed","Value in Millions"]]

corr2=df_corr.corr()

sns.heatmap(corr2, xticklabels=corr2.columns, yticklabels=corr2.columns, cmap="YlGnBu")


#Linear Regression
lm=linear_model.LinearRegression()


X=df_fifa[["Overall"]]
Y=df_fifa["Value in Millions"]

lm.fit(X,Y)

Yhat=lm.predict(X)

b0=lm.intercept_
b1=lm.coef_

results=pd.DataFrame({"Overall":df_fifa["Overall"], "Actual value":df_fifa["Value in Millions"],"Predicted value":Yhat})

new_player=pd.DataFrame({"Overall":[99,95,89]})
Yhat2=lm.predict(new_player[["Overall"]])

fig,ax=plt.subplots(figsize=(15,10))
sns.regplot(x="Overall",y="Value in Millions",data=df_fifa,ax=ax)
plt.grid()
plt.show()

#Multiple Linear Regression

Z=df_fifa[["Overall","Age"]]

lm.fit(Z,df_fifa["Value in Millions"])

Yhat3=lm.predict(Z)


results=pd.DataFrame({"Overall":df_fifa["Overall"],"Age":df_fifa["Age"], "Actual value":df_fifa["Value in Millions"],"Predicted value":Yhat3})

fig,ax=plt.subplots(figsize=(15,10))
sns.residplot(x="Age",y="Value in Millions",data=df_fifa,ax=ax)
plt.grid()
plt.show()

#polynomial Regression

from sklearn.preprocessing import PolynomialFeatures

pr=PolynomialFeatures(degree=2, include_bias=False)

x_poly=pr.fit_transform(df_fifa[["Overall","Age"]])

lin_reg2 = linear_model.LinearRegression()
lin_reg2.fit(x_poly,Y)
Yhat4=lin_reg2.predict(x_poly)


results=pd.DataFrame({"Overall":df_fifa["Overall"],"Age":df_fifa["Age"], "Actual value":df_fifa["Value in Millions"],"Linear value":Yhat3,"Polynomial value":Yhat4})

#pipeline
from sklearn.pipeline import Pipeline

Input=[(('polynomial'),PolynomialFeatures(degree=2)),("mode",linear_model.LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(df_fifa[["Overall","Age"]],df_fifa["Value in Millions"])
Yhat5=pipe.predict(df_fifa[["Overall","Age"]])
results_compare=pd.DataFrame({"Polynomial value":Yhat4,"Pipeline":Yhat5})

Input=[(('polynomial'),PolynomialFeatures(degree=3)),("mode",linear_model.LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(df_fifa[["Overall","Age"]],df_fifa["Value in Millions"])

Yhat6=pipe.predict(df_fifa[["Overall","Age"]])


results2=pd.DataFrame({"Overall":df_fifa["Overall"],"Age":df_fifa["Age"], "Actual value":df_fifa["Value in Millions"],"Linear value":Yhat3,"2nd-degree":Yhat5,"3rd-degree":Yhat6})

#Mean Sqaure Error
from sklearn.metrics import mean_squared_error

MSE_SL=mean_squared_error(df_fifa["Value in Millions"],Yhat)

MSE_ML=mean_squared_error(df_fifa["Value in Millions"],Yhat3)

MSE_2d=mean_squared_error(df_fifa["Value in Millions"],Yhat5)

MSE_3d=mean_squared_error(df_fifa["Value in Millions"],Yhat6)

#R-Squared
from sklearn.metrics import r2_score

r_SL=r2_score(df_fifa[["Value in Millions"]],Yhat)

r_ML=r2_score(df_fifa[["Value in Millions"]],Yhat3)

r_2d=r2_score(df_fifa[["Value in Millions"]],Yhat4)

r_3d=r2_score(df_fifa[["Value in Millions"]],Yhat6)

#Use for loop to get the best model
from sklearn.pipeline import Pipeline

r2=[]
MSE=[]
degree=[1,2,3,4,5,6,7,8,9,10]
for i in degree:
    Input=[(('polynomial'),PolynomialFeatures(degree=i)),("mode",linear_model.LinearRegression())]
    pipe=Pipeline(Input)
    pipe.fit(df_fifa[["Overall","Age"]],df_fifa["Value in Millions"])
    Yhat=pipe.predict(df_fifa[["Overall","Age"]])
    r2_value=r2_score(df_fifa[["Value in Millions"]],Yhat)
    r2.append(r2_value)
    
    MSE_value=mean_squared_error(df_fifa["Value in Millions"],Yhat)
    MSE.append(MSE_value)
    
    plt.figure(figsize=(15,8))
    plt.scatter(df_fifa.index,df_fifa["Value in Millions"])
    plt.plot(df_fifa.index,Yhat)
    plt.grid()

plt.figure(figsize=(15,10))    
plt.plot(degree,r2)
plt.grid()
plt.xlabel("degree")
plt.ylabel("r2")

#Choosen Model
Input=[(('polynomial'),PolynomialFeatures(degree=9)),("mode",linear_model.LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(df_fifa[["Overall","Age"]],df_fifa["Value in Millions"])
Yhat=pipe.predict(df_fifa[["Overall","Age"]])

result_final=pd.DataFrame({"Overall":df_fifa["Overall"],"Age":df_fifa["Age"], "Actual value":df_fifa["Value in Millions"],"Predicted value":Yhat})