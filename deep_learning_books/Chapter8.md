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

>  这一个section是平坦区域， 下一section是变化很快的区域



**Plateaus**

* Gradient is small or even tends to zero
* Slow learning progress



**Saddle points**

* More common for complex model than local minima 
* Gradient descent empirically seems to be able to escape saddle point in many case 
* However, the true effect of saddle point to first order learning algorithm is somehow unclear 
* Saddle point is harmful to second-order methods like Newton's method, which find the critical points 
  * Second-order methods remain difficult to scale to large neural networks 
  * Hessian matrix $H$ is quite computation espensive 



### 8.2.4 Cliffs and Exploding Gradients 

![Cliffs](C:\Users\eee\Documents\GitHub\AllNotes\deep_learning_books\images\8_1.PNG)

* Gradient only reveal the information about optimal descent direction within an infinitesimal region
* The effect of large jump caused by cliff is somehow unknown  





### 8.2.5 Long-Term Dependencies

* Models with shared parameters 

* Gradient explosion and vanish in RNN 



### 8.2.6 Inexact Gradients 

* Biased estimation of gradient 
  * Caused by SGD used in practice 
* In some cases, the objective function is actually intractable

> 书中也没有举例那些objective function 是intractable的



### 8.2.7 Poor Correspondence between Local and Global Structure 

![](C:\Users\eee\Documents\GitHub\AllNotes\deep_learning_books\images\8_2.PNG)

* Sensitive to initial value of parameter



> 目前没有解决办法，



### 8.2.8 Theoretical Limits of Optimization 

> 一句话， 理论研究什么都没研究出来





## 8.3 Basic Algorithms 

### 8.3.1 Stochastic Gradient Descent 

![SGD](C:\Users\eee\Documents\GitHub\AllNotes\deep_learning_books\images\8_3.PNG)



**Key points**

* Learn rate is a crucial parameter for SGD 
* Learn rate should decrease over time 
* How to choose a proper learning rate is an art than science 



> Learning rate的选择很重要，但是怎么选是个玄学



### 8.3.2 Momentum

![SGD with Momentum](C:\Users\eee\Documents\GitHub\AllNotes\deep_learning_books\images\8_4.PNG)

**Key points**

* Aims to solve two problems
  * Poor conditioning of Hessian matrix
  * Variance in stochastic gradient estimation



> * 如果gradient 一直保持一个方向， Momentum就是一个等比数列的和。 Step size就会比不用momentum大很多
> * 如果gradient不稳定一直在变换方向， 用Momentum的step size就比不用的小
>
> 用Momentum确实有帮助，但是也一定找到Momentum 效果比原来的SGD 更差的情况



### 8.3.3 Nesterov Momentum 

![](C:\Users\eee\Documents\GitHub\AllNotes\deep_learning_books\images\8_5.PNG)

**Key points**

* Almost the same to momentum
* Only difference is where the gradient is computed
* Faster than Momentum for poor condition



> 简单来说就是在计算Gradient 的时候提前走了 $\alpha v$ ,
>
> 这里有两个帖子写的比较好[比Momentum更快：揭开Nesterov Accelerated Gradient的真面目](<https://zhuanlan.zhihu.com/p/22810533>)
>
> [路遥知马力——Momentum](<https://zhuanlan.zhihu.com/p/21486826>)



## 8.4 Parameter Initialization Strategies 

**Key points**

* When training deep learning models, the choice of initial parameter can determine whether the model converges at all

* We known few about how the initialize the parameter 

  

**Break symmetry**

* If two hidden units has same activation and same inputs, then these units mush have different initial parameters 
  * May help to make sure no input patterns are lose in the null space of forward progapation and no gradient patterns are lost in the null space of back-propagation 

> 这里关于null space的叙述没看懂



**Typical initialization strategies**

* Random initialization

* Orthogonal initialization 

* Typically, the value of the parameter are drawn randomly from a Guassian or uniform distribution 

  

**Large or small initial value**

* Large initial value
  * Stronger symmetry breaking effect 
  * Avoid losing information in feedforward and back propagation process
  * Gradient exploding
  * Saturation of activation function 
  * Strong priori about interaction between units 





**Optimization and Generalization**

* Usually the final parameter value after the training is close to the initial value 
  * Thus choosing a initial parameter $\theta_0$ is similar to imposing a a Gaussian prior $p(\theta)$ with mean $\theta_0$ 



**Optimal Criteria for initialization**

* There are many so-called optimal criteria 
* These so-called optimal criteria often do not lead to optimal performance 



> 简单来说，对于如何initialize value 了解很少



## 8.5 Algorithms with Adaptive Learning Rates 

Learning rate is the most difficult hyperparameter to choose for machine learning algorithm cause learning rate greatly affect the final performance 



### 8.5.1 AdaGrad

![AdaGrad](C:\Users\eee\Documents\GitHub\AllNotes\deep_learning_books\images\8_6.PNG)



**Key points** 

* Weight the learning rate by the second norm of all historical gradients

* Designed for convex optimization 
* Can overcome cliff and plateaus problem 



### 8.5.2 RMSProp

![RMSProp](C:\Users\eee\Documents\GitHub\AllNotes\deep_learning_books\images\8_7.PNG)



**Key points**

* Use exponential decaying gradient norm compared with AdaGrad



**RMSProp with Nesterov momentum**

![](C:\Users\eee\Documents\GitHub\AllNotes\deep_learning_books\images\8_8.PNG)

**Key points**

* Adding Nesterov momentum to origin RMSProp





### 8.5.3 Adam 

![Adam](C:\Users\eee\Documents\GitHub\AllNotes\deep_learning_books\images\8_9.PNG)

**Key points**

* Can be seen as a variants on the combination of RMSProp and momentum 
* Machenism to correct biased estimation
  * How ??????? 

> 玄学， 怎么就能correct biased estimation了， 原理是什么？？？ 



### 8.5.4 Choosing the  Right Optimization Algorithm

> 同样是玄学



## 8.6 Approximate Second-Order Methods 

In This section we discuss the application of **second-order methods to the training of deep neural networks** 



## 8.6.1 Newton's Method

Use second-order Taylor series expansion to approximate $J(\theta)$ new some point $\theta_0$ 
$$
J(\theta) \approx J(\theta_0) + (\theta-\theta_0)^\top\nabla_{\theta}J(\theta_0) + \frac{1}{2}(\theta-\theta_0)^\top H(\theta-\theta_0)
$$
Solve the critical point of this function, we obtain the newton parameter update rule:
$$
\theta^{\star} = \theta_0 - H^{-1}\nabla_{\theta}J(\theta_0)
$$

* This update can be conducted iteratively as long as the Hessian matrix $H$ is positive definite 

* If the eigenvalue of $H$ are not all positive (Near a saddle point) 

  * Regularize the Hessian 
    $$
    \theta^{\star} = \theta_0 - [H + \alpha I]^{-1}\nabla_{\theta}J(\theta_0)
    $$

The regularization of Hessian only works well when the negative eigenvalues are close to zero, otherwise Newton's method will be slow to converge than gradient descent.

Meanwhile, the Hessian matrix and the inverse Hessian matrix is computational expensive 

> 简单来说， 在训练Neural network的时候基本用不上， Convex optimization 会比较好用



### 8.6.2 Conjugate Gradients 

> 没看懂，就不总结了





### 8.6.3 BFGS 

> 一样没看懂， 不总结了





## 8.7 Optimization Strategies and Meta-Algorithms 

### 8.7.1 Batch Normalization 

> Idea 很简单，但是为什么有好的效果书里面没有说的很清楚，看了几个[Post](<https://www.zhihu.com/question/38102762>) 也没有完全弄明白，可能要等之后看的时候在搞清楚了



### 8.7.2 Coordinate Descent 

* Optimize the parameter one coordinate at a time 



> 感觉在deep learning 里面没什么应用



### 8.7.3 Polyak Averaging

> 没懂为什么会有用



### 8.7.4 Supervised Pretraining

