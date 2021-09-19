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
-Loss function of LR model: L(yhat,y)=-(ylog(yhat) + (1-y)log(1-yhat))  (a *convex* functio