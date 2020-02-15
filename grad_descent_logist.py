# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:15:37 2020

@author: Jerron
"""

#this script generalises gradient descent algorithm
#for linear regression

#initialization of data set

import numpy as np
import pandas as pd

#####################################################################

def sigmoid(x):
    return 1/(1+np.exp(-x))


def logist_costfunc(reglambda,theta,x,y):
    
    thetax = theta.dot(np.transpose(x))
    thetax = thetax[:,None]
    #np.transpose does not work with 1D array.
    #Result from dot product gives a 1D array h0.
    #Hence, use [:,None] to transpose instead
    h0 = sigmoid(thetax)
    
    sz = np.shape(x) 
    
    jval = -(1/sz[0])*sum(y*np.log(h0)+(1-y)*np.log(1-h0))+sum((reglambda/(2*sz[0]))*np.square(theta))  
    #jval: cost value  
    
    grad = np.ones(sz[1])
    #grad: gradient step for next iteration
    
    for i in range(0,sz[1],1):  
        if i == 0:
            grad[i]=(1/sz[0])*sum(h0-y);
        else:
            grad[i]=(1/sz[0])*sum((np.transpose(h0-y)).dot(x[:,i]))-(reglambda/sz[0])*theta[i]
    
    print(grad," ",jval)
                
    return(grad,jval)


def feature_scale(x):
    #feature scaling via mean normalization
    #ie. x1_norm = (x1-mean(x1))/SD(x1)
    #Basically converting the x(i) variable into "Z",
    #the standardised normalised variable
    
    sz = np.shape(x)
    
    for i in range(0,sz[1],1):
        x[:,i] = (x[:,i] - np.mean(x[:,i]))/np.std(x[:,i])
        
    #appending of constant term
    cons = np.ones(sz[0])
    cons = cons[:,None]
    x = np.insert(x,[0],cons,axis=1)
    return(x)


def logist_graddsc(x,y,theta,a,reglambda):
    
    init_theta = theta #to document initial theta guess
    
    #perform gradient descent
    
    numiter = 100000
    jvallist = np.ones(numiter)
    
    for i in range(0,numiter,1):
        gradstep = logist_costfunc(reglambda,theta,x,y)
        jvallist[i] = gradstep[1]   #gradstep[1] represents the cost or error value jval
        if i > 2:
            if jvallist[i]-jvallist[i-1] > 0 and jvallist[i-2]-jvallist[i-1] > 0: 
                print("Regularization parameter: ",reglambda)
                print("First guess of weights: ",init_theta)
                print("Converged at the following weights: ",theta)
                return(theta)
                break
            else:
                theta = theta - a * gradstep[0]
                #gradstep[0] represents the gradient of each respective feature at the current point
        else:
            theta = theta - a * gradstep[0]  
            
        if i == (numiter-1) :
            print("NOTE: Gradient Descent did not converge, try increasing number of iterations, changing learning rate, or reducing regularization parameter")


def modeleva_logistic(theta,df_train_x,df_train_y,df_test_x,df_test_y,df_crossvalid_x,df_crossvalid_y,log_threshold = 0.5):
    
    #% of correct predictions on train data
    sz = np.shape(df_train_y)
    
    thetax = theta.dot(np.transpose(df_train_x))
    thetax = thetax[:,None]
    h0_train = sigmoid(thetax)
    h0_train = log_round(h0_train, log_threshold)
    
    train_incorr_pred = sum(abs(h0_train-df_train_y))
    train_acc = 1 - (train_incorr_pred/sz[0])
    
    #% of correct predictions on test data
    sz = np.shape(df_test_y)
    
    thetax = theta.dot(np.transpose(df_test_x))
    thetax = thetax[:,None]
    h0_test = sigmoid(thetax)
    h0_test = log_round(h0_test, log_threshold)
    
    test_incorr_pred = sum(abs(h0_test-df_test_y))
    test_acc = 1 - (test_incorr_pred/sz[0])
    
    #% of correct predictions on cross validation data
    sz = np.shape(df_crossvalid_y)
    
    thetax = theta.dot(np.transpose(df_crossvalid_x))
    thetax = thetax[:,None]
    h0_crossvalid = sigmoid(thetax)
    h0_crossvalid = log_round(h0_crossvalid, log_threshold)
    
    crossvalid_incorr_pred = sum(abs(h0_crossvalid-df_crossvalid_y))
    crossvalid_acc = 1 - (crossvalid_incorr_pred/sz[0])
    
    print("Training set accuracy:", train_acc)
    print("Test set accuracy:", test_acc) 
    print("Cross validation set accuracy:", crossvalid_acc)
    return(crossvalid_acc)
    
def log_round(x,threshold):
    
    #function to round up / round down sigmoid value according to a given threshold
    #threshold default value: 0.5
    
    sz = np.shape(x)
    
    for i in range(0,sz[0],1):
        if x[i] < threshold:
            x[i] = 0
        if x[i] >= threshold:
            x[i] = 1
        
    return(x)


def reg_optimization(range1,range2,a1,a2,regtype):
    accuracy_data = pd.DataFrame(np.ones(shape = (20,2),dtype = float),columns=["reglambda","crossvalid_acc"])
    #initialize pandas dataframe to store accuracy against reglambda values    
    i = 0
    if regtype == "logistic":    
        for reglambda in range1:    
            theta = np.array([avg_y,0,0,0,0,0], dtype = float)
            fin_theta = logist_graddsc(df_train_x,df_train_y,theta,a1,reglambda)
            crossvalid_acc = modeleva_logistic(fin_theta,df_train_x,df_train_y,df_test_x,df_test_y,df_crossvalid_x,df_crossvalid_y)
            accuracy_data["reglambda"].iloc[i] = reglambda
            accuracy_data["crossvalid_acc"].iloc[i] = crossvalid_acc
            i = i+1            
        for reglambda in range2:
            theta = np.array([avg_y,0,0,0,0,0], dtype = float)
            fin_theta = logist_graddsc(df_train_x,df_train_y,theta,a2,reglambda)
            crossvalid_acc = modeleva_logistic(fin_theta,df_train_x,df_train_y,df_test_x,df_test_y,df_crossvalid_x,df_crossvalid_y)
            accuracy_data["reglambda"].iloc[i] = reglambda
            accuracy_data["crossvalid_acc"].iloc[i] = crossvalid_acc
            i = i+1  
    elif regtype == "linear":        
        for reglambda in range1:    
            theta = np.array([avg_y,0,0,0,0,0], dtype = float)
            fin_theta = linear_graddsc(df_train_x,df_train_y,theta,a1,reglambda)
            crossvalid_acc = modeleva_linear(fin_theta,df_train_x,df_train_y,df_test_x,df_test_y,df_crossvalid_x,df_crossvalid_y)            
            accuracy_data["reglambda"].iloc[i] = reglambda
            accuracy_data["crossvalid_acc"].iloc[i] = crossvalid_acc
            i = i+1            
        for reglambda in range2:
            theta = np.array([avg_y,0,0,0,0,0], dtype = float)
            fin_theta = linear_graddsc(df_train_x,df_train_y,theta,a2,reglambda)
            crossvalid_acc = modeleva_linear(fin_theta,df_train_x,df_train_y,df_test_x,df_test_y,df_crossvalid_x,df_crossvalid_y)            
            accuracy_data["reglambda"].iloc[i] = reglambda
            accuracy_data["crossvalid_acc"].iloc[i] = crossvalid_acc
    print(accuracy_data)

####################################################################


def cont_2_categ(series):
    if series < 650:
        return 0
    elif series >= 650:
        return 1

####################################################################

#initialize data

df = pd.read_csv(r"C:\Users\Jerron\Desktop\caschool.csv")
df["testscr"] = df["testscr"].apply(cont_2_categ)

df_train = df.sample(frac=0.6,random_state=250)
df = df.drop(df_train.index)
df_test = df.sample(frac=0.5,random_state=250)
df_crossvalid = df.drop(df_test.index)

sz_train = df_train.shape
sz_test = df_test.shape
sz_crossvalid = df_crossvalid.shape

df_train_y = (df_train["testscr"].to_numpy()).reshape(sz_train[0],1)
df_train_x = df_train[["calw_pct","el_pct","avginc","comp_stu","meal_pct"]].values
avg_y = np.average(df_train_y)

df_test_y = (df_test["testscr"].to_numpy()).reshape(sz_test[0],1)
df_test_x = df_test[["calw_pct","el_pct","avginc","comp_stu","meal_pct"]].values

df_crossvalid_y = (df_crossvalid["testscr"].to_numpy()).reshape(sz_crossvalid[0],1)
df_crossvalid_x = df_crossvalid[["calw_pct","el_pct","avginc","comp_stu","meal_pct"]].values

#perform feature scaling to x values

df_train_x = feature_scale(df_train_x)
df_test_x = feature_scale(df_test_x)
df_crossvalid_x = feature_scale(df_crossvalid_x)

#initialize theta as first guess
    
theta = np.array([avg_y,0,0,0,0,0], dtype = float)

#initialize learning rate a

a = 0.003

#initialize regularization parameter reglambda

reglambda = 100

#run gradient descent script
      
fin_theta = logist_graddsc(df_train_x,df_train_y,theta,a,reglambda)

#evaluation of model accuracy

modeleva_logistic(fin_theta,df_train_x,df_train_y,df_test_x,df_test_y,df_crossvalid_x,df_crossvalid_y)

#cross-validation to find optimum regularization parameter

range1 = range(0,10,1)
range2 = range(10,110,10)
a1 = 0.1
a2 = 0.01
regtype = "logistic"
reg_optimization(range1,range2,a1,a2,regtype)