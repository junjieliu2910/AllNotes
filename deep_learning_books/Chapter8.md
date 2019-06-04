# Chapter 8. Optimization for Training Deep Models



## 8.1 How Learning Differs from Pure Optimization  

**Risk:**
$$
J^\star(\theta) = \mathbb{E}_{(x, y)\sim p_{data}}L(f(x;\theta), y) 
$$
Where $p_{data}$ is the true data generating distribution 



**Empirical Risk:**
$$
J(\theta)  = \mathbb{E}_{(x, y)\sim\hat{p}_{data}}L(f(x;\theta), y)
$$
Where $\hat{p}_{data}$ is the empirical distribution of the training dataset 



**The ultimate goal of machine learning is to reduce the risk, but risk is usually infeasible to calculate. So we minimize empirical risk and in most situation reduce empirical risk will also reduce the risk**. 



### 8.1.1 Empirical Risk Minimization

**When Machine learning turns to pure optimization**

* Then it will be a pure optimization problem. We assume a model family based on our priori knowledge and minimize the $J(\theta)$ to get the best fitting parameter to fit the $p_{data}$ 

* When $p_{data}$ is unknown, and we only have a training set. That's a machine learning problem 



**How to Solve Machine Learning Problem**

* Minimize the **empirical risk** 
  $$
  \mathbb{E}_{x, y\sim \hat{p}_{data}}[L(f(x;\theta), y)] = \frac{1}{m} \sum^m_{i=1} L(f(x^{(i)};\theta), y^{(i)})
  $$

* This is called **empirical risk minimization** 



**Two Problems of Empirical Risk Minimization**

* Prone to overfitting
* In many cases, empirical risk minimization is not really feasible
  * Loss function have no useful derivatives(zero or undefined derivative) , cannot use gradient-based optimization method 





### 8.1.2 Surrogate Loss Functions and Early Stopping

**0-1 Loss Function**
$$
L(i, j) =
\begin{cases}
0 & \text{if $i= j$} \\
1 & \text{if $i\neq j$} \\
\end{cases}
$$

* 0-1 Loss Function is non-convex and discontinuous.
* Use Negative log likelihood as a surrogate for the 0-1 loss

* NLL provide additional robustness compared with 0-1 loss function 



**Not find the minimum**

* Machine learning problem usually do not want to find the local minimum or the global minimum
* Just reach a point that the objective function $J(\theta)$ is small enought



> 关于0-l loss function 可以看这两个帖子 [Why is a 0-1 loss function intractable?](https://stats.stackexchange.com/questions/365444/why-is-a-0-1-loss-function-intractable)
>
> [0-1 Loss Function explanation](https://stats.stackexchange.com/questions/284028/0-1-loss-function-explanation)
>
> 不太懂为什么要把 NLL 当成0-1的一种替代，除了方便计算我没看出0-1 loss function有其他任何好处，那为什么最开始要考虑用0-1 loss function， 发现不行了才考虑替代



### 8.1.3 Batch and Minibatch Algorithm 

**Deterministic gradient methods**

* Optimization algorithm that use the entire training set to compute the gradient 



**Stochastic or online gradient method**

* Use only a single example at a time



**Minibatch algorithm for machine learning**

* Use a small batch of examples 



**Consideration about batch size**

* Larger batch size
  * better gradient estimation but with less than linear returns 
  * may underutilize multicore system (Due to too few batches）
  * Increase memory requirement 
* Smaller batch size
  * Offer regularizing effect
    * Similar to adding noise to the input 
* First order or second order optimization 
  * First order optimization with only gradient $g$ is robust for small batch size
  * Second order optimization with $H$ need much larger batch size to make it stable 



**Random sampling**

* If the minibatch is randomly sampled, then this is a unbiased estimation of the gradient 
* However, for very large dataset, randomly sample is expensive or even infeasible 
  * Shuffle the training dataset first

> shuffle 可以看作是 random sample的一种替代品，虽然不是完美的random sample. 但是从实践中发现好像没有明显的损害



**Another motivation**

* Minibatch SGD follows the gradient of the true generalization error so long as no examples are repeated. 
  * So on the first pass, each minibatch will compute a unbiased estimation of the gradient 
  * But for later passed, each minibatch will be biased since each minibatch have been used 
  * However, the benefit of decreasing the objective function offset the damage of unbiased estimation 

> 也就是说是由第一个 epoch 是unbiased estimation， 之后因为重复使用的原因， 都会变成biased estimation。 但是一个epoch的训练并不能让Loss function变得足够小， 所以后续的epoch是必要的，虽然有biased estimation带来的误差， 但是让Loss function下降的好处更大，所以一般来说都是多个epoch 



## 8.2 Challenges in Neural Network Optimization 

### 8.2.1 Ill-Conditioning 

Refer to section 4.3.1 for what is ill-conditioning 



**Damage of Ill-conditioning**

The second-order Taylor series expansion of the cost function predicts that a gradient descent step of $-\epsilon g $ will add
$$
\Delta =  \frac{1}{2}\epsilon^2g\top Hg - \epsilon g\top g
$$
To decrease the cost function $\Delta$ should be negative.

However, if the Hessian matrix $H$ at that point is ill-conditioning, $\Delta$ may be positive thus increasing the cost function 



### 8.2.2 Local Minima 

**Model Identifiability**

* Neural network have multiple local minima because of the model identifiability problem 
  * For a same local minima there are many corresponding parameter combinations 
  * Weight space symmetry



**Is local minima a problem?**

* Local minima can be problematic if they have high cost in comparison to the global minimum 

* However, most researchers believed that most local minima have a low cost value and usually we do not aim to find the local minima but a position with small cost.

  

**How to avoid local minima**

* Check the norm of gradient overtime 
  * If the norm converge to 0, then maybe that's a local minima 





## 8.2.3 Plateaus, Saddle Points and Other Flat Regions

 