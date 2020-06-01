import numpy as np 
import matplotlib.pyplot as plt
import h5py

def init_params(n_x,n_h,n_y):
    #initialize all the weights and bias
    W1 = np.random.rand(n_h,n_x)*np.sqrt(2/n_x)
    b1 = np.zeros((n_h,1))
    W2 = np.random.rand(n_y,n_h)*np.sqrt(2/n_h)
    b2 = np.zeros((n_y,1))
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2}
    
    return parameters

def forward_prop(X,params):
    #calculate forward propagation of the neural network model
    #retrieve parameters value
    W1 = params["W1"]
    W2 = params["W2"]
    b1 = params["b1"]
    b2 = params["b2"]
    #compute forward propagation
    Z1 = np.dot(W1,X)+b1     
    A1 = np.tanh(Z1)         #tanh activation function
    Z2 = np.dot(W2,A1)+b2
    A2 = 1/(1 + np.exp(-Z2)) #sigmoid activation function 
    #compute cache to be used in later gradient descent calculation
    cache = { "Z1" : Z1,"Z2" : Z2, "A1" : A1, "A2" : A2}
    return A2,cache

def compute_cost(A2,Y,params):
    #compute using cross entropy cost algorithm
    m = Y.shape[1]                  #number of examples
    logprobs = np.multiply(np.log(A2),Y) + np.multiply((1-Y),np.log(1-A2))
    cost = (-1/m)*np.sum(logprobs)
    
    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect. 
    assert(isinstance(cost, float)) # E.g., turns [[17]] into 17 
    
    return cost

def backward_prop(params,cache,X,Y):
    #compute the backward propagation
    m = X.shape[1]                #number of training examples
    #retrieve the parameters and cache value
    W1 = params["W1"]
    W2 = params["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    #calculate the backward propagation dw1,db1,dw2 and db2
    dZ2 = A2-Y
    dW2 = (1/m)*np.dot(dZ2,A1.transpose())
    db2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.multiply(W2.transpose()*dZ2,(1-np.power(A1,2)))
    dW1 = (1/m)*np.dot(dZ1,X.transpose())
    db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)
    #store in grads dictionary
    grads = {"dW1":dW1,"db1":db1,"dW2":dW2,"db2":db2}
    #return grads value
    return grads    

def update_params(params,grads,learning_rate):
    #update parameters
    #retrieve the parameters and grad value
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    #update for each parameters
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    #return parameters
    parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2}
    return parameters

def Neural_Network_model(X,Y,n_H,alpha,num_iteration):    
    np.random.seed(3)                  #seed for random generator
    n_x = X.shape[0]                   #number of input layer
    #n_h = 512                          #number of hidden layer
    n_y = Y.shape[0]                   #number of output layer
    params = init_params(n_x,n_H,n_y)  #initialize parameter
    
    for i in range(0,num_iteration):
        #Forward Propagation
        A2,cache = forward_prop(X,params)
        #Cost Function
        cost = compute_cost(A2,Y,params)
        #Backpropagation
        grads = backward_prop(params,cache,X,Y)
        #Gradient Descent update
        params = update_params(params,grads,alpha)

        if i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    return params
    
# helper function to compute the classification error rate
def CER(predictions, labels):
    return (np.sum(predictions != labels) / np.size(predictions))

#function to predict output from input and neural network parameters
def make_prediction(X,params):
    m = X.shape[1]
    p = np.zeros((1,m))
    A2,cache = forward_prop(X,params)
    
    for i in range(0,A2.shape[1]):
        if A2[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    return p