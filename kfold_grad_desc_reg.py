# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:15:37 2020

@author: Jerron
"""

#GRADIENT DESCENT ALGORITHM WITH KFOLD

#import packages used
import numpy as np
import pandas as pd

#initialize user-defined functions

#####################################################################
#LINEAR REGRESSION FUNCTIONS
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


def modeleva_linear(theta,df_x,df_y):
    
    #jval calculation for train data
    sz = np.shape(df_y)
    h0 = theta.dot(np.transpose(df_x))
    h0 = h0[:,None]
    
    err = sum(np.square(h0 - df_y))/sz[0]
    
    return(err)

#####################################################################
#LOGISTIC REGRESSION FUNCTIONS
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
    
    #print(grad," ",jval)
                
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


def modeleva_logistic(theta,df_x,df_y,log_threshold = 0.5):
    
    #% of correct predictions
    sz = np.shape(df_y)
    
    thetax = theta.dot(np.transpose(df_x))
    thetax = thetax[:,None]
    h0 = sigmoid(thetax)
    h0 = log_round(h0, log_threshold)
    
    incorr_pred = sum(abs(h0-df_y))
    acc = 1 - (incorr_pred/sz[0])
    
    return(acc)
    
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

#####################################################################
#HYPERPARAMETER OPTIMIZATION FUNCTIONS (WORK IN PROGRESS)
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
#K-FOLD FUNCTIONS
def kfold(df,num_fold,rand_s):

    df_row = df.shape[0]
    df_col = df.shape[1]
    fold_frac = 1/num_fold
    
    data_sets = {0:np.ones([int(df_row/num_fold),df_col])}
    
    for i in range (0,num_fold,1):
        fold_data = df.sample(frac=fold_frac/(1-(i*fold_frac)),random_state = rand_s)
        df = df.drop(fold_data.index)
        data_sets[i] = fold_data
        
    return data_sets


def kfoldreg(df,ylabel,xlabel,regtype,reglambda,num_fold=10,rand_s=100,a=0.03):
    
    #num_fold specifies number of folds (such that df_rows is divisible by num_fold without remainder)
    #default value set at 10.
    
    #rand_s specifies random state used in df.sample method in kfold
    #default value set at 100.
    
    #execute kfold algorithm to split data into num_fold equal sets of data
    kfold_data = kfold(df,num_fold,rand_s)
    
    crossvalidarr = np.ones(num_fold)
    trainarr = np.ones(num_fold)
    
    for i in range(0,num_fold,1):
    #iteration within different train and test data sets for kfold validation
        
        df_crossvalid = kfold_data[i]
        df_train = pd.DataFrame()
        
        for j in range(0,num_fold,1):
            if j != i:
                df_train = pd.concat([df_train,kfold_data[j]],axis=0)
                
        #dataset is split into train and cross validation sets
        #proceed with data prep for regression
        
        df_train_y = df_train[ylabel].values
        df_train_x = df_train[xlabel].values
        avg_y = np.average(df_train_y)
        
        df_crossvalid_y = df_crossvalid[ylabel].values
        df_crossvalid_x = df_crossvalid[xlabel].values
    
        #perform feature scaling to x values
        
        df_train_x = feature_scale(df_train_x)
        df_crossvalid_x = feature_scale(df_crossvalid_x)
        
        #initialize theta as first guess
            
        theta = np.array([avg_y,0,0,0,0,0], dtype = float)
        
        #run gradient descent script and evaluate model accuracy
        
        if regtype == "logistic":
            fin_theta = logist_graddsc(df_train_x,df_train_y,theta,a,reglambda)        
            crossvalidarr[i] = modeleva_logistic(fin_theta,df_train_x,df_train_y)
            trainarr[i] = modeleva_logistic(fin_theta,df_crossvalid_x,df_crossvalid_y)
        
        elif regtype == "linear":
            fin_theta = linear_graddsc(df_train_x,df_train_y,theta,a,reglambda)        
            crossvalidarr[i] = modeleva_linear(fin_theta,df_train_x,df_train_y)
            trainarr[i] = modeleva_linear(fin_theta,df_crossvalid_x,df_crossvalid_y)     
    
    #averaging out of accuracies in each kfold
    avg_crossvalid_acc = np.average(crossvalidarr)
    avg_train_acc = np.average(trainarr)
    
    print("Overall train accuracy = ",avg_train_acc)
    print("Overall cross validation accuracy = ",avg_crossvalid_acc)


#####################################################################
#MISCELLANEOUS/ADDITIONAL FUNCTIONS
def cont_2_categ(series):
    if series < 650:
        return 0
    elif series >= 650:
        return 1


#####################################################################

#import data
df = pd.read_csv(r"C:\Users\Jerron\Desktop\caschool.csv")
df["testscr"] = df["testscr"].apply(cont_2_categ)

#specify x and y columns in dataframe
xlabel = ["calw_pct","el_pct","avginc","comp_stu","meal_pct"]
ylabel = ["testscr"]

#extract test data from data set 
#----------------------------------------------------------------------
#OPTIONAL IF TEST DATA IS STORED IN A SEPERATE FILE
df_test = df.sample(frac=30/420,random_state = 1)
df = df.drop(df_test.index)

sz_test = df_test.shape
df_test_y = df_test[ylabel].values
df_test_x = df_test[xlabel].values
df_test_x = feature_scale(df_test_x)
#----------------------------------------------------------------------

#specify type of regression
regtype = "logistic"

#specify regularization parameter
reglambda = 10

#use "kfoldreg" function to perform regression with k-fold validation
#NOTE: refer to kfoldreg function above for available input settings

kfoldreg(df,ylabel,xlabel,regtype,reglambda,rand_s = 50)

#########

#cross-validation to find optimum regularization parameter

#range1 = range(0,10,1)
#range2 = range(10,110,10)
#a1 = 0.3
#a2 = 0.03
#regtype = "logistic"
#reg_optimization(range1,range2,a1,a2,regtype)
