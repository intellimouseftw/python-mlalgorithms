# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:31:47 2020

@author: Jerron
"""

#this script generalises gradient descent algorithm
#for linear regression

#initialization of data set

import numpy as np
import pandas as pd

#####################################################################

def linear_costfunc(reglambda,theta,x,y):
    
    h0 = theta.dot(np.transpose(x))
    h0 = h0[:,None]
    #np.transpose does not work with 1D array.
    #Result from dot product gives a 1D array h0.
    #Hence, use [:,None] to transpose instead
    
    sz = np.shape(x) 
    
    jval = (1/(2*sz[0]))*sum(np.square(h0-y))+sum((reglambda/(2*sz[0]))*np.square(theta))
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

def mean_normalisation(y):
    #apply mean normalisation to y values
    y = (y - np.mean(y))/np.std(y)
    return(y)
    
    

def linear_graddsc(x,y,theta,a,reglambda):
    
    init_theta = theta #to document initial theta guess
    
    #perform gradient descent
    
    numiter = 10000
    jvallist = np.ones(numiter)
    
    for i in range(0,numiter,1):
        gradstep = linear_costfunc(reglambda,theta,x,y)
        jvallist[i] = gradstep[1]   
        #gradstep[1] represents the cost or error value jval
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


def modeleva_linear(theta,df_train_x,df_train_y,df_test_x,df_test_y,df_crossvalid_x,df_crossvalid_y):
    
    #jval calculation for train data
    sz = np.shape(df_train_x)
    h0 = theta.dot(np.transpose(df_train_x))
    h0 = h0[:,None]
    train_err = sum(np.square(h0 - df_train_y))/sz[0]
    
    #jval calculation for test data
    sz = np.shape(df_test_x)
    h0 = theta.dot(np.transpose(df_test_x))
    h0 = h0[:,None]
    test_err = sum(np.square(h0-df_test_y))/sz[0]
    
    #jval calculation for cross validation data
    sz = np.shape(df_crossvalid_x)
    h0 = theta.dot(np.transpose(df_crossvalid_x))
    h0 = h0[:,None]
    crossvalid_err = sum(np.square(h0-df_crossvalid_y))/sz[0]
    
    print("Training set error:", train_err)
    print("Test set error:", test_err)
    print("Cross validation set error:", crossvalid_err)
    return(crossvalid_err)

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
            crossvalid_err = modeleva_linear(fin_theta,df_train_x,df_train_y,df_test_x,df_test_y,df_crossvalid_x,df_crossvalid_y)            
            accuracy_data["reglambda"].iloc[i] = reglambda
            accuracy_data["crossvalid_acc"].iloc[i] = crossvalid_err
            i = i+1            
        for reglambda in range2:
            theta = np.array([avg_y,0,0,0,0,0], dtype = float)
            fin_theta = linear_graddsc(df_train_x,df_train_y,theta,a2,reglambda)
            crossvalid_err = modeleva_linear(fin_theta,df_train_x,df_train_y,df_test_x,df_test_y,df_crossvalid_x,df_crossvalid_y)            
            accuracy_data["reglambda"].iloc[i] = reglambda
            accuracy_data["crossvalid_acc"].iloc[i] = crossvalid_err
            i = i+1
    print(accuracy_data)        
    

####################################################################

#initialize data
  
df = pd.read_csv(r"C:\Users\Jerron\Desktop\caschool.csv")
df_train = df.sample(frac=0.6,random_state=200)
df = df.drop(df_train.index)
df_test = df.sample(frac=0.5,random_state=200)
df_crossvalid = df.drop(df_test.index)

target = ["testscr"]
features = ["calw_pct","el_pct","avginc","comp_stu","meal_pct"]

df_train_y = df_train[target].values
df_test_y = df_test[target].values
df_crossvalid_y = df_crossvalid[target].values

avg_y = np.average(df_train_y)

df_train_x = df_train[features].values
df_test_x = df_test[features].values
df_crossvalid_x = df_crossvalid[features].values

#perform feature scaling to x values

df_train_x = feature_scale(df_train_x)
df_test_x = feature_scale(df_test_x)
df_crossvalid_x = feature_scale(df_crossvalid_x)

#initialize theta as first guess
    
theta = np.array([avg_y,0,0,0,0,0], dtype = float)

#initialize learning rate a

a = 0.3

#initialize regularization parameter reglambda

reglambda = 0

#run gradient descent script
      
fin_theta = linear_graddsc(df_train_x,df_train_y,theta,a,reglambda)

#evaluation of model accuracy

modeleva_linear(fin_theta,df_train_x,df_train_y,df_test_x,df_test_y,df_crossvalid_x,df_crossvalid_y)

#further model optimization with cross-validation comparisons

range1 = range(0,10,1)
range2 = range(10,110,10)
a1 = 0.03
a2 = 0.003
regtype = "linear"
reg_optimization(range1,range2,a1,a2,regtype)