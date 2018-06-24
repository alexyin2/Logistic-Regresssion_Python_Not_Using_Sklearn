# Logistic-Regresssion
## Sigmoid Function vs. Bayesian
1. Basic Logistic Regression uses a function called sigmoid function.
2. We will proof that the sigmoid function is the same as the Bayesian method.
![image](https://github.com/alexyin2/Logistic-Regresssion_Python_Not_Using_Sklearn/blob/master/Image/Proof_LR_Bayesian.png)

## Cross-Entropy Error function
1. When calculating Linear Regression, we usually use Least square error as our cost function. But in Logistic Regression, we should use another way since our output value is always between 0 and 1. 
2. Cross-Entropy Error function is the way we use in Logistic Regression and we can also proof that it's same as the maximum likelihood method.
![image](https://github.com/alexyin2/Logistic-Regresssion_Python_Not_Using_Sklearn/blob/master/Image/Cross_Entropy_Maximum_Likelihood.png)

## Warning!!
1. The above ways so far didn't give an answer of how to update the weights to make the prediction better.
2. Only when we assume that our data is Gaussian distributed with equal covariance, we can use the Bayesian method to calculate the weights.
3. But what if we can't make the assumption? So next, I'm going to introduce another way of updating the weights, which is known as Gradient Discent.

## Gradient Descent in Logistic Regression
