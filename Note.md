## I. Neural networks and deep learning
### Introduction
- Image data: CNN - convolution neural network
- Sequence data: Audio, language
- Structured data (presented in table) and unstructured data (audio, image, text)
- [Sigmoid vs activation functions]: for sigmoid function, when the slope -> 0, gradient descent is very low.

### Neural networks basics
- When implementing neural network with logistic regression, it is easier to keep b (intercept) and w(s) as separate parameters.
- Loss function of LR model: L(yhat,y)=-(ylog(yhat) + (1-y)log(1-yhat))  (a *convex* function)
- Cost function: average of loss.
- Forward propagation: Computation Graph: [C1_W2.pdf - page 18]
- Backward progragation: [C1_W2.pdf - page 20]
    - One step backward = Derivative once
        - Calculating dJ/dv
    - Two step backward:
        - dJ/da = (dJ/dv).(dv/da)  (Chain rule)
- When you are writting codes to implement backpropagation, there will be a final output variable that you really care about or want to optimize.
    - Computing dFinalOutputVar/dvar (eg. dJ/da)
    - When coding, that quantity can just be named 'dvar' rather than 'dFinalOutputVar/dvar'
- Logistic Regression gradient descent: [C1_W2.pdf - page 24]
    - da = dL(a,y)/da = -y/a + (1-y)/(1-a)
    - dz = (dL/da).(da/dz) = da.(a(1-a)) = a - y
        - Explanation: https://community.deeplearning.ai/t/derivation-of-dl-dz/165
    - dw1 = x1dz  => w1 := w1 - αdw1
    - dw2 = x2dz  => w2 := w2 - αdw2
    - db = dz     => b := b - αdb
- Gradient descent on m examples of algorithm: [C1_W2.pdf - page 27, page 33, page 38]
    - In this example, 2 for loops are explicitly implemented -> inefficient -> consider *Vectorization*
- Python and vectorization:
    - [Python run time of for loop vs vectorization.jpg]
    - Parallelization instructions - single instruction multiple data (SIMD) instruction
        - If you use bult-in functions such as *np.function* that don't require you explicityly implementing a for loop, it enables Python numpy to take much better advantag of parallelizm to do your computations much faster.
- Whenever possible, avoid explicit for-loops.
- More vectorzation examples: np.exp(), np.log(), np.abs(), np.maximum(), v**2, 1/v
- Replace one for loop of logistic regression derivatives by vectorization: [C1_W2.pdf - page 33]
- Replace all the for loops by vectorization - *Broadcasting in Python*: [C1_W2.pdf - page 35]
    - Backward propagation with vectorization: [C1_W2.pdf - page 38]
- Broadcasting example in Python:
    - division: percentage = 100*A/(cal.reshape(1,4))
    - adding by a constant, adding by a duplicated vector, adding by a duplicated transposed vector: [C1_W2.pdf - page 41, page 42]
- Tips and tricks to avoid bugs related to broadcasting in Python: 
    + Not using rank 1 array (eg: a = np.random.randn(5,)). Instead, defining a matrix/eg. a column vector: a = np.random.randn(5,1) 
        - Or make sure its shape by assert(a.shape==(5,1)) OR a = a.reshape((5,1))
- Logistic regression cost function:
    - Explain lost function [C1_W2.pdf - page 45]
### Overview
- Notation: x(i) for individual observation, x[i] for the order of input layer, a1^[1] node 1 in layer 1.
- Hidden layer: in the training set, the true value for the nodes in the middle are not observed.
    - A network with 1 hidden layer: 2-layer NN (iput layer doesnt count)
- Activation function: 
    - When you have centralized data, may apply tanh instead of sigmoid function. Furthermore, tanh is more superior than sigmoid.
    - For output layer, may apply sigmoid because it is more meaningful when y take value 0, 1.
    - Downside of sigmoid and tanh is when z is very large/small, the gradient descent will be very slow -> may consider ReLU (a = max(0,z))- default, most common choice for activation function.
    - One variation of ReLU: Leaky ReLU (a = max(0.01,z)).
    - If you use linear activation function, no matter how many layer, all the network is doing is just computing a linear activation function without hidden layers.
        - If the problem is linear regression  (eg. predicting temperatures), the output layer can have linear activation function.
- Derivative of sigmoid function: a(1-a)
- Derivative of tanh function: 1-(tanh(z))^2
- Derivative of ReLU function: 0 with z<0, 1 with z>0 -> sub-gradient of the activation function g of z
- Derivative of leaky ReLU function: 0.01 with z<0, 1 with z>0 
- Backpropagation intuition: derivatives and chain rule [C1_W3.pdf - page 30]
- Random initialization: if choosing all the initial weights as zero, the activate functions will be the same.
    - Instead, initialize them randomly: w[1] = np.random.randn((2,2))*0.01 (multiply by a small number so that z will not fall to the tail of sigmoid function)
    - b doesn't have the symmetry breaking problem: b[1] = np.zeros((2,1))
* Learn more about dimension of matrices anc vectors in neural network: https://medium.com/from-the-scratch/deep-learning-deep-guide-for-all-your-matrix-dimensions-and-calculations-415012de1568

### Deep L-layer Neural Networks
- Notation: 
    - n^[l] is the number of unit in layer l 
    - n^[L] = 1 (the last layer - output layer)
    - n^[0] = nx 
    - a^[l]: activation in layer l
- It is okay to have a for loop that goes across layers.
- Dimensions of matrices: (m = # of samples)
    - Z[1] = W[1].X + b[1]
        - Z[1]: (n[1], m)
        - X: (n[0], m)
        - W[1]: (n[1], n[0])
        - b[1]: (n[1], 1)  (will be broadcasted when being added)
    - In backward propagation, dimension of dW should be the same as W: (n[l], n[l-1]), db should be the same as b.
- Circuit theory and deep learning:
    - Informally: there are functions you can compute with a "small" L-layer deep neural network that shallower networks require exponentially more hidden units (neutron?) to compute.
- Forward and backward functions:
    - Storing the cache of z, w, and b is convenient to calculate the derivates later in backward process.
- Forward and backward propagation:
    - Forward: input a[l-1], output a[l], cache z[l] (and w[l], b[l])
    - Backward: input da[l], output da[l-1], dW[l], db[l] (see detail at [C1_W4 pg 17, 18])