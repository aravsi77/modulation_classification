import numpy as np 
import matplotlib.pyplot as plt
import h5py

def sigmoid(Z):
    #sigmoid activation function 
    A = 1/(1+np.exp(-Z))
    cache = Z
    #return A and cache
    return A,cache

def relu(Z):
    #ReLu activation function
    A = np.maximum(0,Z)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache

def relu_backward(dA, cache):
    #backward propagation for relu activation function
    Z = cache
    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ

def sigmoid_backward(dA, cache):
    #backward propagation for sigmoid activation function
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ= dA * s * (1-s)
    assert (dZ.shape == Z.shape)

    return dZ

def initialize_parameters(n_x,n_h,n_y):
    #initialize parameters
    np.random.seed(1) #seed 1
    W1 = np.random.randn(n_h,n_x)*np.sqrt(2/n_x) #initialize with he initialization
    b1 = np.zeros((n_h,1))                       #zero initialization
    W2 = np.random.randn(n_y,n_h)*np.sqrt(2/n_h) #
    b2 = np.zeros((n_y,1))
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2,}

    return parameters

def initialize_parameters_deep(layer_dims):
    #initialize parameters for deep learning neural network
    #layer_dim is an array contain the dimension of each layer in network
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
    return parameters

def linear_forward(A,W,b):
    #the linear part of the layer propagation network
    Z = W.dot(A) + b
    assert(Z.shape == (W.shape[0],A.shape[1]))
    cache = (A,W,b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    #implement the forward propagation for the linear activation
    if activation =="sigmoid":
        Z, lin_cache = linear_forward(A_prev, W, b)
        A, act_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, lin_cache = linear_forward(A_prev, W, b)
        A, act_cache = relu(Z)

    assert (A.shape == (W.shape[0],A_prev.shape[1]))
    cache = (lin_cache, act_cache)

    return A, cache

def L_model_forward(X, parameters):
    #implement forward propagation for the linear relu and linear relu 

    caches = []
    A = X
    L = len(parameters)//2
    #loop over the neural network layers
    for l in range(1,L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev,parameters['W' + str(l)], parameters['b' + str(l)],
                                            activation = "relu")
        caches.append(cache)
    
    #implement linear sigmoid activation for the last layer
    AL, cache = linear_activation_forward(A,parameters['W' + str(L)], parameters['b' + str(L)], 
                                        activation = "sigmoid")
    caches.append(cache)

    assert(AL.shape == (1,X.shape[1]))

    return AL,caches

def compute_cost(AL,Y):
    #calculate cost function
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    cost = np.squeeze(cost)      
    assert(cost.shape == ())

    return cost

def linear_backward(dZ, cache):
    #implement a linear portion of backward prop for single layer
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    #implement the backward prop 
    lin_cache, act_cache = cache

    if activation == "relu":        
        dZ = relu_backward(dA, act_cache)
        dA_prev, dW, db = linear_backward(dZ, lin_cache)
    
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, act_cache)
        dA_prev, dW, db = linear_backward(dZ, lin_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    #implement a linear backward propagation for linear relu and linear sigmoid
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA"+str(L-1)],grads["dW"+str(L)],grads["db"+str(L)] = linear_activation_backward(dAL, current_cache, activation ="sigmoid")
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads



def update_parameters(parameters, grads, learning_rate):
    #update the parameters using gradient descent
    L = len(parameters) // 2
    #update rule for each parameter
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    
    return parameters


def deep_network_model(X,Y,layer_dims,learning_rate=0.01,num_iter=2500,print_cost=False):
    #implementation of multiple layer neural network

    np.random.seed(1)
    costs = []

    parameters = initialize_parameters_deep(layer_dims)

    #loop over to calculate gradient descent
    for i in range(0, num_iter):
        #forward propagation
        AL, caches = L_model_forward(X,parameters)

        #calculate the cost
        cost = compute_cost(AL,Y)

        #Backward Propagation
        grads = L_model_backward(AL,Y,caches)

        #update parameters
        parameters = update_parameters(parameters,grads,learning_rate)

        #print cost every 100th examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    
    #plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

# helper function to compute the classification error rate
def CER(predictions, labels):
    return (np.sum(predictions != labels) / np.size(predictions))

#function to predict output from input and neural network parameters
def make_prediction(X,params):
    m = X.shape[1]
    p = np.zeros((1,m))
    A2,cache = L_model_forward(X,params)
    
    for i in range(0,A2.shape[1]):
        if A2[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    return p

