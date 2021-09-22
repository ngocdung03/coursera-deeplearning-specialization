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
- Logistic Regression gradient descent:
    - da = dL(a,y)/da = -y/a + (1-y)/(1-a)
    - dz = (dL/da).(da/dz) = da.(a(1-a)) = a - y
    - dw1 = x1dz  => w1 := w1 - αdw1
    - dw2 = d2dz  => w2 := w2 - αdw2
    - db = dz     => b := b - αdb
- Gradient descent on m examples 