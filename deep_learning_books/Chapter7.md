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
**Effects of $L^1$ regularization on Quadratic Cost function**

* Add $L^1$ regularization is equal to shift $w$ by $\frac{\alpha}{H_{ii}}$ 



**Properties**

* $L^1$ result in a solution that is more sparse 





## 7.2 Norm Penalties as Constrained Optimization 

Instead of using Norm penalties as an additional term to the cost function
$$
\tilde{J}(\theta; X, y) = J(\theta; X, y) + \alpha\Omega(\theta)
$$
We can add a inequality constrain to the optimization process. And we could construct a generalized Lagrange function 
$$
\mathcal{L}(\theta, \alpha; X, y) = J(\theta; X, y) + \alpha(\Omega(\theta) - k)
$$
The solution to the constrained problem is given by 
$$
\theta^{\star} = \arg\min_{\theta}\max_{\alpha, \alpha\geq0}\mathcal{L}(\theta,\alpha)
$$
**Reason to use constraints rather than norm penalty**

* Penalties can cause non-convex optimization procedures to get stuck in local minima corresponding to small $\theta$ 



## 7.3 Regularization and Under-Constrained Problems 

> 这里没看懂是什么意思， 没学过优化很吃亏啊





## 7.4 Dataset Augmentation 

> Overfitting 最根本的原因其实就是训练数据太少了，如果可以增加训练数据，那么就可以解决overfitting的问题， 也就等价于加了regularization 

**Main Idea**

* Add training dataset by creating fake examples 
* Useful for classification tasks 



**Used in image classification**

* Adding additional training examples by rotating or rescaling origin images 



**Used in speech recognition**

* Regularize neural network by adding noise to input
* By adding noise to hidden units 





## 7.5 Noise Robustness

**Key points**

* For some models, adding noise with infinitesimal variance at the input of the model is equivalent to imposing a norm penalty on the weights 
* Noise injection to hidden units is powerful regularization tool 
* Noise injection to weights is used primarily in the context of recurrent neural networks 





### 7.5.1 Injecting Noise at the Output Targets 

**Key points**

* Most datasets have training example with mislabeled $y$, which is harmful to the training algorithm 
* **Label smoothing** can regularize the model in this case 





## 7.6 Semi-Supervised Learning

**Key points**

* Adding unsupervised learning model help to reduce the overfitting problem and thus conduct regularization 



**Semi-supervised Learning**

* Bothe $P(x)$ and $P(x, y)$ are used to estimate $P(y|x)$ 



**Separate unsupervised and supervised learning**

1. Use unsupervised learning to learn a new **representation** $h=f(x)$ 
2. Use the transformed representation $h$ as the input to supervised learning model

Example: Use PCA first, and then use the transformed features at the input to the supervised learning model 



**Share parameter between unsupervised and supervised learning** 

> 这里的叙述没看懂





## 7.7 Multi-Task learning

**Key points** 

* A model shared by multi-task tends to be more general, which avoid overfitting and regularize 



**Disadvantage**

* Only useful when there exists several tasks and have some shared factors of variations





## 7.8 Early Stopping

**Key points**

* We tend to use large model with sufficient representational capacity to overfit the task 
* The relationship between train error and test error are U-shape as following image
* We can stop early to reduce the generalization error 
* Need a validation set to conduct early stopping

![U-shape training process](C:\Users\eee\Documents\GitHub\AllNotes\deep_learning_books\images\7_2.PNG)



**Why overfitting can regularize**

* Early stopping can be regarded as **controlling the effective capacity** of the model by determining how many steps it can take to fit the training set 

* The number of training step can be regarded as a hyperparameter. And we use a validate set to choose a proper number of training step



**Advantage of early stopping**

* Early stopping requires no change in the underlying training procedure, the objective function and the allowable parameter values 



**Disadvantage of early stopping**

* Require a validate set, thus reducing the number of training examples 



**Solution to the disadvantage**

* Initialize the model again and retrain the model on all of the data with the same number of training step as the early stopping procedure determined in the first pass 
  * Question about training the same number of epochs or the same number of steps
* Continue the training process with the all the data.
  * How to determine when to stop in this training process 



**How early stopping acts a a regularizer**

* In the case of a simpler linear model with a quadratic cost function and a simple gradient descent, early stopping is equivalent to $L^2$ regularization 

> 这里的推导就省略了，其实也挺好奇对于其他的cost function和task这个结论是不是也成立



## 7.9 Parameter Tying and Parameter Sharing

> 之前的Penalty-based 或者 constraint-based Regularization都是引入了一些prior knowledge about the suitable values of the model parameters。 比如$L^2$ regularization penalizes model parameters fro deviating from the fixed value of zero. 
>
> 另外一个思路是Add prior knowledge about the relationship between the model parameters

**Key point**

* CNN 里面其实广泛应用了Parameter sharing. 比如同一层里面，一个filter 其实被整个input features共用的





## 7.10 Sparse Representations

**Sparse parameterization**

* $L^1$ regularization will make the model parameter sparse 



**Sparse representation** 

* Apply similar norm penalties to representation 

$$
\tilde{J}(\theta;X, y) = J(\theta; X, y) + \alpha\Omega(h)
$$





## 7.11 bagging and Other Ensemble Methods 

**Key points**

* **Bagging(short for bootstrap aggregating)** is a technique for reducing generalization error by combining several methods

* Bagging is an example of more general strategy in machine learning called **model averaging**. 
* Techniques employing this strategy are know as **ensemble methods** 



