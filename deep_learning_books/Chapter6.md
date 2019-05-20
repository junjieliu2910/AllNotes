# Chapter 6   Deep Feedforward Networks

## Overview 

* The goal of a feedforward network is to approximate some function 
* Called **feedforward** cause no output will be the feedback of the input. If there are feedback, it's called **recurrent neural networks**
* Called networks since it's a composition of multiple simpler functions
* Feedforward network is inspired by brain. But it should better be regarded as a function approximation machine rather than a model of a brain machine

**As an extension of linear models**:

Use a non-linear transform $\phi$ to transform input features to a new feature space $\phi(\mathbb{x})$ 

Three methods to choose the mapping $\phi$ 

1. Use very generic $\phi$, such as the infinite-dimensional $\phi$ used in kernel machine based on RBF kernel
2. Use manually engineer $\phi$ 
3. Learning a $\phi$ , from a given model families derived by human knowledge (Adopted by deep learning)





## 6.1 Example: Learning XOR

* Linear model cannot learn XOR function 
* Feedforward network $\implies$ Affine transformation + Non-linear activation functions

> 这一章用一个双层的MLP 学习XOR 为例子说明的了什么是feedforward networks, 主要就是介绍了MLP 的组成部分，不了解的可以看一下





## 6.2 Gradient-based Learning 

### Overview 

* Non-linearity of neural networks $\implies$ Non-convex cost functions 
* Non-convex optimization $\implies$ No guarantee to converge to global minimal 
* Not robust
* Sensitive to the initial choice of parameters 
* Stochastic gradient descent 

In order to use SGD for optimization, we first need to define the cost function and find its gradient.



### 6.2.1 Cost Functions 

