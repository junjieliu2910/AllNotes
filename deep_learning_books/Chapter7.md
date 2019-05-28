# Chapter 7. Regularization for Deep Learning 



**A central problem in machine learning is how to make an algorithm that will perform well not just on the training data, but also on new inputs. ** 



**Regularization**

Any modification we make to a learning algorithm that is intended to reduce its regularization error but not its training error.



**Two ways of imposing regularization**

* Encode specific kinds of **prior knowledge** 
* Express a **generic preference for a simpler model class** in order to promote generalization 



**Trading increased bias for reduce variance** 



**The Importance of regularization**

Controlling the complexity of the model is not intended to find the model of the right size with the right number of parameters. Instead, we might find and indeed in practical deep learning scenarios, we almost always do find  that the best fitting model is a large model that has been regularized properly



## 7.1 parameter Norm Penalties 

Regularization by adding a parameter norm penalty $\Omega(\theta)$ to the objective function $j(\theta)$ 
$$
\tilde{J}(\theta;X, y) = J(\theta; X, y) + \alpha\Omega(\theta)
$$

* For neural network, the parameter norm penalty **only penalize the weights of the affine transformation but not the bias** 
  * Bias do not impose too much variance (why ???)
  * Regularize bias will lead to underfitting



### 7.1.1 $L^2$ Parameter Regularization 

In $L^2$ norm penalty, the regularization term is 
$$
\Omega(\theta) = \frac{1}{2}||w||^2_2
$$
$L^2$ regularization is also called **ridge regression** or **Tikhonov regularization** 



![The effects of l2 regularization](C:\Users\eee\Documents\GitHub\AllNotes\deep_learning_books\images\7_1.PNG)



**The effects of $L^2$ regularization**

* Rescale $w$ along the axes define by the eigenvectors of $H$, where $H$ is the hessian matrix of $J(\theta)$ 





### 7.1.2 $L^1$ Regularization 

In $L^1$ regularization, the penalty term is 
$$
\Omega(\theta) = \alpha\sum_i|w_i|
$$
