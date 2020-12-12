# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 21:47:46 2020

@author: David Wang
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#----------------------converting data into a (1372,5) array
array=[]
f= open('data_banknote_authentication.txt','r')
for line in f:
    values=line.split(',')
    floatvalues =[float(i) for i in values]
    array.append(floatvalues)
    
#print(array)
matrix=np.asmatrix(array)
#print(matrix.shape)

#------------splitting X and T
X=np.delete(matrix,4,1) #X is (1372,4)
#print(X)
t=matrix
for i in range (4):
    t=np.delete(t,0,1) # T is (1372,1)
#print(t)


#------------------------------ Ready to split the data

# first split to get training data
from sklearn.model_selection import train_test_split 
X_train, X_leftover, t_train, t_leftover = train_test_split(X, t, test_size = 1/2, random_state = 3796)

#second split to get test and validation

X_test, X_valid, t_test, t_valid = train_test_split(X_leftover, t_leftover, test_size = 1/2, random_state = 3796)

#----------------------------Normalize the data

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, :] = sc.fit_transform(X_train[:, :])
X_valid[:, :] = sc.transform(X_valid[:, :])
X_test[:, :] = sc.transform(X_test[:, :])

#print(t_train.shape,t_valid.shape,t_test.shape)

#----------------------------Naming #of inputs 
#Training
I2=(X_train[:,:2]) #2 inputs
I3=(X_train[:,:3]) #3 inputs
I4=(X_train[:,:4]) #4 inputs

#add dummy features
D2=np.insert(I2,0,1,1)
D3=np.insert(I3,0,1,1)
D4=np.insert(I4,0,1,1)

#Validation
J2=(X_valid[:,:2])
J3=(X_valid[:,:3])
J4=(X_valid[:,:4])

V2=np.insert(J2,0,1,1)
V3=np.insert(J3,0,1,1)
V4=np.insert(J4,0,1,1)

#TEST
H2=(X_test[:,:2])
H3=(X_test[:,:3])
H4=(X_test[:,:4])

T2=np.insert(H2,0,1,1)
T3=np.insert(H3,0,1,1)
T4=np.insert(H4,0,1,1)
#---------------------------Inital Weights for Hidden Layers
#FOR 1 NODE Current LAYER(output)  
weights2x1= np.full((1,3),0.5)            
weights3x1= np.full((1,4),0.5)      
weights4x1= np.full((1,5),0.5)   

#FOR 2 NODE Current LAYER
weights2x2= np.full((2,3),0.5)                #2 hidden nodes
weights3x2= np.full((2,4),0.5)             #3 hidden nodes
weights4x2= np.full((2,5),0.5)       #4 hidden nodes

#FOR 3 NODES Current LAYER
weights2x3= np.full((3,3),0.5)                  #2 hidden nodes
weights3x3= np.full((3,4),0.5)           #3 hidden nodes
weights4x3= np.full((3,5),0.5)       #4 hidden nodes

#FOR 4 NODES Current LAYER
weights2x4= np.full((4,3),0.5)             #2 hidden nodes
weights3x4= np.full((4,4),0.5)            #3 hidden nodes
weights4x4= np.full((4,5),0.5)   
#print(weights3x4)

#--------------------------Setting learning rate
lr=0.005

#-----------------------------ReLU functions
def ReLU(x):
    x[x<0]=0
    return x

def dReLU(x): #derivative function
    x[x<0]=0
    x[x>0]=1
    return x

def sigmoid(x):
    return 1/(1+np.exp(-x))

#-----------------------------Cross-entropy Loss Function
def loss(y,t):
    xd=-t*np.log(y)-(1-t)*np.log(1-y)
    return xd
    
    
    
#-----------------------------neural network code for 2x2x2
#FORWARDPASS



#-----------------------Neural Function

def Neural(features,target,vfeatures,vtarget,epochs,w1,w2,w3):
    #setup
    rows= len(features)
    weight1=w1
    weight2=w2
    weight3=w3
    N=len(vfeatures)
    
    #plot arrays
    j_array=[]
    t_loss=[]
    v_loss=[]
    t_misclassification=[]
    v_misclassification=[]
   
    for j in range(epochs): #go through this number of epochs
        total_terror=0 #reset total error for each epoch
        total_verror=0
        y = np.zeros(rows)#reset output matrix
        yv= np.zeros(N)
        u=np.zeros(rows)
        uv=np.zeros(N)
        
        
         #shuffle per epoch
        from sklearn.utils import shuffle
        from random import randrange
        X, t = shuffle(features, target, random_state=73796)#randrange(9999))
        
        for i in range (0,rows,1): #go through every training set
            # #calculating hidden layer 1
            h1_pre= np.dot(X[i],np.transpose(weight1)) #output row is node values, col is different inputs
            
            #using activation function on h1
            h1_out= ReLU(h1_pre)
            
            #calculating hidden layer 2
            h1_out=np.insert(h1_out,0,1,1) #add dummy
            h2_pre=np.dot(h1_out, np.transpose(weight2))
            # if(i<10):
            #     print(h2_pre)
            
            #activation function on h2
            h2_out=ReLU(h2_pre)
            
            #calculating output
            h2_out=np.insert(h2_out,0,1,1)
            output_pre=np.dot(h2_out, np.transpose(weight3))
            
            #activation for output
            output_out=sigmoid(output_pre) #y
            
            #------------------------- Error Calculation
            terror=loss(output_out,t[i])
            total_terror= total_terror+terror
            
            #------------------------- Misclassification Rate
            if (output_out>0.5):
                y[i]=1
                
            
            #------------------------- #BACKWARD PASS
            
            #weight for h2->out
            #J
            J_out= output_out-t[i]

            #W
            W_out=J_out*h2_out
            
            #weight for h1->h2
            #Z
           
            Z_h2=np.multiply(dReLU(h2_pre),J_out*weight3[:,1:])
            
            #W
            W_h2=np.dot(np.transpose(Z_h2),h1_out) #transpose to get
            
            #weight for input->h1
            #Z
            Z_h1=np.multiply(dReLU(h1_pre),(np.dot(Z_h2, weight2[:,1:])))
        
            #W
            W_h1=np.dot(np.transpose(Z_h1),X[i])
            
            #-----------------------------------------Update Weights
            
            weight1=weight1-lr*W_h1
            weight2=weight2-lr*W_h2
            weight3=weight3-lr*W_out
            
            #if (i==10):                             
                #print("UPDATED WEIGHTS:","weight1:",weight1,'weight2',weight2,'weight3',weight3)
            
        #--------------end of each epoch    
        print("UPDATED WEIGHTS at iteration:",j,"\n","weight1:",weight1,"\n",'weight2',weight2,"\n",'weight3',weight3)
        training_error=total_terror/rows
        print("training error of epoch:",j,"\n", total_terror/rows)
        #missrate matrix arithemtic 
        for i in range (0,rows,1):
            u[i]=y[i]-t[i]
        missrate=np.count_nonzero(u)/(rows)
        print("Training Misclassification rate:", missrate)
        
        #validation set at end of each epoch
        for i in range(0,N,1):
            
            h1_pre= np.dot(vfeatures[i],np.transpose(weight1)) #output row is node values, col is different inputs
            
            #using activation function on h1
            h1_out= ReLU(h1_pre)
            
            #calculating hidden layer 2
            h1_out=np.insert(h1_out,0,1,1) #add dummy
            h2_pre=np.dot(h1_out, np.transpose(weight2))
            # if(i<10):
            #     print(h2_pre)
            
            #activation function on h2
            h2_out=ReLU(h2_pre)
            
            #calculating output
            h2_out=np.insert(h2_out,0,1,1)
            output_pre=np.dot(h2_out, np.transpose(weight3))
            
            #activation for output
            output_out=sigmoid(output_pre) #y
            
            #------------------------- Error Calculation
            verror=loss(output_out,vtarget[i])
            total_verror= total_verror+verror
            
            #misclass
            if (output_out>0.5):
                yv[i]=1
        validation_error=total_verror/N
        print("validation error:", validation_error)         
        for i in range (0,N,1):
            uv[i]=yv[i]-vtarget[i]
        vmissrate=np.count_nonzero(uv)/(N)
        print("Validation Misclassification rate:", vmissrate,'\n')  
        
        #plot arrays
        j_array=np.append(j_array,j)
        t_loss=np.append(t_loss,training_error)
        v_loss=np.append(v_loss, validation_error)
        t_misclassification=np.append(t_misclassification,missrate)
        v_misclassification=np.append(v_misclassification,vmissrate)
        
    #plot after running through all epochs
    #comment if plots aren't necessary
    plt.figure(1)
    plt.plot(j_array,t_loss,label='training error')
    plt.plot(j_array,v_loss, label='validation error')
    plt.plot(j_array,t_misclassification,label='training misclassification')
    plt.plot(j_array,v_misclassification,label='validation misclassification')
    plt.xlabel('number of epochs')
    plt.ylabel('error') 
    plt.legend()
    
     
#--------------------------THE 27 MODELS----------------------------- 
#D=2 uncomment one at a time for fast running
Neural(D2,t_train,V2,t_valid,10,weights2x2,weights2x2,weights2x1)
# Neural(D2,t_train,V2,t_valid,10,weights2x2,weights2x3,weights3x1)
# Neural(D2,t_train,V2,t_valid,10,weights2x2,weights2x4,weights4x1)
# Neural(D2,t_train,V2,t_valid,10,weights2x3,weights3x2,weights2x1)
# Neural(D2,t_train,V2,t_valid,10,weights2x3,weights3x3,weights3x1)
# Neural(D2,t_train,V2,t_valid,10,weights2x3,weights3x4,weights4x1)
# Neural(D2,t_train,V2,t_valid,10,weights2x4,weights4x2,weights2x1)
# Neural(D2,t_train,V2,t_valid,10,weights2x4,weights4x3,weights3x1)
# Neural(D2,t_train,V2,t_valid,10,weights2x4,weights4x4,weights4x1)

# #D=3
# Neural(D3,t_train,V3,t_valid,10,weights3x2,weights2x2,weights2x1)
# Neural(D3,t_train,V3,t_valid,10,weights3x2,weights2x3,weights3x1)
# Neural(D3,t_train,V3,t_valid,10,weights3x2,weights2x4,weights4x1)
# Neural(D3,t_train,V3,t_valid,10,weights3x3,weights3x2,weights2x1)
# Neural(D3,t_train,V3,t_valid,10,weights3x3,weights3x3,weights3x1)
# Neural(D3,t_train,V3,t_valid,10,weights3x3,weights3x4,weights4x1)
# Neural(D3,t_train,V3,t_valid,10,weights3x4,weights4x2,weights2x1)
# Neural(D3,t_train,V3,t_valid,10,weights3x4,weights4x3,weights3x1)
# Neural(D3,t_train,V3,t_valid,10,weights3x4,weights4x4,weights4x1)

# #D=4
# Neural(D4,t_train,V4,t_valid,10,weights4x2,weights2x2,weights2x1)
# Neural(D4,t_train,V4,t_valid,10,weights4x2,weights2x3,weights3x1)
# Neural(D4,t_train,V4,t_valid,10,weights4x2,weights2x4,weights4x1)
# Neural(D4,t_train,V4,t_valid,10,weights4x3,weights3x2,weights2x1)
# Neural(D4,t_train,V4,t_valid,10,weights4x3,weights3x3,weights3x1)
# Neural(D4,t_train,V4,t_valid,10,weights4x3,weights3x4,weights4x1)
# Neural(D4,t_train,V4,t_valid,10,weights4x4,weights4x2,weights2x1)
# Neural(D4,t_train,V4,t_valid,10,weights4x4,weights4x3,weights3x1)
# Neural(D4,t_train,V4,t_valid,10,weights4x4,weights4x4,weights4x1)

#----------------------Test of best classifier

#Neural(D3,t_train,T3,t_test,10,weights3x2,weights2x2,weights2x1)

