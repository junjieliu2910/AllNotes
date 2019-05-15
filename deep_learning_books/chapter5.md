# Chapter 5. Machine Learning Basics 

## Overview

Machine learning is a form of **applied statistics** that with increased emphasis on the use of computer to statistically estimate complicated functions and a decreases emphasis on proving confidence interval around these functions.



## 5.1 Learning Algorithm 

### Definition of learning algorithm 

A computer program is said to learn from experience E with respect to some classes of T and performance P, if its performance at tasks in T, as measured by P, improves with experience E.



### 5.1.1 The Task

* Classification
* Classification with missing inputs
* Regression
* Transcription
* Machine Translation 
* Structured output 
* Anomaly detection 
* Synthesis and sampling 
* Imputation of missing values 
* Denoise
* Density estimation 



### 5.1.2 The Performance Measure 

Use **test set** to evaluate the performance 



### 5.1.3 The Experience

* Supervised 
* Unsupervised 



### 5.1.4 Example: Linear Regression

Refer to the book





## 5.2 Capacity, Overfitting and Underfitting

#### Train error and test error 

* The ability to perform well on previously unseen inputs is called **generalization** 
* What separates machine learning from optimization is that we also want the **generalization error** or the **test error** to be low  



**Problem**: How can we affect the algorithm's performance on test set when we only get to observe the train set ?



#### Basic assumption about data generating process

* If the train and test dataset are generated arbitrarily, the machine learning algorithm is hard to generalize. Under this situation, the datasets affect the performances more than the choice and design of machine learning algorithm 
* **Basic Assumption:** The examples in each dataset are independent from each other. The train and test dataset are identically distributed, drawn from the same underlying probability distribution called **data generating distribution $p_{data}$**
* With this basic assumption, for any untrained machine learning algorithm, the expected train error and test error are the same.
* The two facts that determine the goodness of a machine learning algorithm.
  * Make train error small
  * Make the gap between the train and test error small 

* Fail to make train error small => underfitting
* Fail to make the gap between train and test error small => overfitting


#### Capacity 
Capacity is the ability to fit more complex function. 
* Low capacity algorithm tends to underfitting
* High capacity tends to overfitting 
* Choose a proper capacity is critical

**Method to control the capacity of a learning algorithm**:
* Properly choose a hypothesis space


**Representationnal capacity and effective capacity**:
* By choosing the hypothesis space, we choose the representational capacity. But the training process cannot find the optimal function in the hypothesis space.
* The effective capacity may be less than the representational capacity 

**Qualifying model capacity**
* Vapnik-Chervonenkis dimension ([VC dimension](http://beader.me/mlnotebook/section2/vc-dimension-one.html)) 
* The most important results in statistical learning
theory show that the discrepancy between training error and generalization error
is bounded from above by a quantity that grows as the model capacity grows but
shrinks as the number of training examples increases
    * If we want complex model, we need more training and testing examples


**Bayes error**
* If we get a oracle that already know the true probability distribution that generate the data, this oracle will still incur some error. 
* The error incurred by a oracle making prediction from the true distribution $p(\mathbb{x}, y)$ is called **bayes error**. 
* Bayes error is the error caused by the intrinsic stochastic property of the underlying problem/



### 5.2.1 The No Free Launch Theorem 
No machine learning algorithm is universally better than any other machine learning algorithm. 

The Most complex model may perform worse than a random gauss or a linear regression.

Our goal is to **understand what kinds of distributions** are relevant to the “real world” that an AI agent experiences, and **what kinds of machine learning algorithms perform well on data** drawn from the kinds of data generating distributions we care about.
* Understand the data generating probability 
* Choose proper machine learning algorithm for that specific data generating probability


### 5.2.2 Regularization
Regularization is any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error


## 5.3 Hyperparameters and validation set
**Hyperparameters**: settings that we used to control the behavior of machine learning algorithms
* settings that are hard to optimize
* settings that are not appropriate to learn
  * For all settings that control the model capacity, the machine learning algorithm tends to choose the settings that maximize the model capacity 

**How to determine the hyperparameters**:
* The test set cannot be used to choose the hyperparameter 
* Split another validate set to choose the hyperparameter from the training set

### 5.3.1 Cross-validation
* **Used when the training data is not enough**
* K-fold cross-validation(typical method)


## 5.4 Estimator, Bias and Variance
Question: How can the machine learning algorithm generalize to unseen examples given the fact that the algorithm is only trained with training dataset.

### 5.4.1 Point Estimation

**Point estimation** is the attempt to provide the single "best" prediction of some quality of interesting.
* The estimated quality value $\hat{\theta}$ 
* The true value denotes $\theta$

**Function estimation**: A type of point estimation that estimate a "best" function in a function space.


> The definition of point estimation here is quite general. I think the author just want to present the basic concept and idea. So just remember these concepts.


### 5.4.2 Bias 
#### Definition:
The bias of an estimator is defines as:
$$
bias(\hat{\theta}_m) = \mathbb{E}[\hat{\theta}_m] - \theta 
$$
The expectation is over the data, $\theta$ is the true underlying value used to define the data generating probability. 
* Unbiased => $bias(\hat{\theta}_m) = 0$
* asymptotically unbiased => $\lim_{m\to \infin}\mathbb{E}[\hat{\theta}_m] = \theta$

> 1. What is m ? (m is the number of data examples)
> 2. over that data ? 
> 3. what is the meaning of $m\to\infty$ (Have infinite data example) 

#### Example: Estimator of mean of Gaussian distribution
Consider a set of $\{x^{(1)}, x^{(2)}, \dots, x^{(m)}\}$ that are independently and identically distributed according to a Guassian distribution $p(x^{(0)}) = \mathcal{N}(x^{(i)}; \mu ,\sigma^2)$, where $i\in \{1, 2, \dots , m\}$.

The gaussian distribution density function is:
$$
p(x^{(i)}; \mu ,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{1}{2}\frac{(x^{(i)}-\mu)^2}{\sigma^2})
$$

A common estimator for Gaussian mean parameter is **sample mean**, this means we use the sample mean to estimate the true underlying mean
$$
\hat{\mu} = \frac{1}{m}\sum_{i=1}^mx^{(i)} 
$$

The estimation bias of sample mean is 
$$
\begin{aligned}
bias(\hat{\mu}) &= \mathbb{E[\hat{\mu}]} - \mu \\
&= \mathbb{E}[\frac{1}{m}\sum_{i=1}^mx^{(i)}] - \mu \\
&= \frac{1}{m}\mathbb{E}[\sum_{i=1}^mx^{(i)}] - \mu \\
&= \frac{1}{m}\sum_{i=1}^m\mathbb{E}[x^{(i)}] - \mu \\
&= \mu - \mu = 0
\end{aligned}
$$
**So the sample mean is a unbiased estimator** 

#### Example: Estimator of variance of Gaussian distribution
Under the same data distribution and data set, we estimate the variance this time. The most intuitive variance estimator is.
$$
\hat{\sigma}_m = \frac{1}{m}\sum_{i=i}^m(x^{(i)} - \bar{x})^2
$$

then the expectation of estimator is:
$$
\begin{aligned}
\mathbb{E}[\hat{\sigma}_m] 
&= \mathbb{E}[\frac{1}{m}\sum_{i=i}^m(x^{(i)} - \bar{x})^2]  \\
&= \frac{1}{m}\mathbb{E}[\sum_{i=i}^m(x^{(i)} - \bar{x})^2] \\
&= \frac{1}{m}\mathbb{E}[\sum_{i=i}^m(x^{(i)} - \mu + \mu - \bar{x})^2] \\
&= \frac{1}{m}\mathbb{E}[\sum_{i=i}^m((x^{i}-\mu)+(\mu -\bar{x}))^2] \\
&= \frac{1}{m}\mathbb{E}[\sum_{i=i}^m((x^{(i)}-\mu)^2 + 2(x^{(i)}-\mu)(\mu-\bar{x}) + (\mu-\bar{x})^2)] \\
&= \frac{1}{m}\mathbb{E}[\sum_{i=i}^m(x^{(i)}-\mu)^2 + 2(\mu-\bar{x})\sum_{i=i}^m(x^{(i)}-\mu) + m(\mu-\bar{x})^2] \\
&= \frac{1}{m}\mathbb{E}[\sum_{i=i}^m(x^{(i)}-\mu)^2 + 2m(\mu-\bar{x})(\bar{x}-\mu) + m(\mu-\bar{x})^2] \\
&= \frac{1}{m}\mathbb{E}[\sum_{i=i}^m(x^{(i)}-\mu)^2 - 2m(\mu-\bar{x})^2 + m(\mu-\bar{x})^2] \\
&= \frac{1}{m}\mathbb{E}[\sum_{i=i}^m(x^{(i)}-\mu)^2 - m(\mu-\bar{x})^2] \\
&= \frac{1}{m}\mathbb{E}[\sum_{i=i}^m(x^{(i)}-\mu)^2] - \mathbb{E}[(\bar{x}-\mu)^2] \\
&= \sigma^2 - \mathbb{E}[(\bar{x}-\mu)^2] \\
&= \sigma^2 - \frac{1}{m}\sigma^2 
\end{aligned}
$$
The estimation bias is
$$
bias(\hat{\sigma}_m) = -\frac{1}{m}\sigma^2
$$

So, the most intuitive estimator derived from the definition of variance is a biased estimator of the true variance.

The **unbiased estimator** for Gaussian distribution variance is
$$
\hat{\sigma}_m^{unbiased} = \frac{1}{m-1}\sum_{i=i}^m(x^{(i)} - \bar{x})^2
$$
Which is usually referred as the **sample variance**


**Note**:
* While unbiased estimators are clearly desirable, they are not always the “best” estimators.

### 5.4.3 Variance and Standard Error 
**Definition:**
Variance of a estimator $\hat{\theta}$ is 
$$
Var(\hat{\theta}_m) = \mathbb{E}[(\hat{\theta}_m - \mathbb{E}[\hat{\theta}_m])^2]
$$

Variance provides a measure of how the estimated value change as we resample the dataset with the same data generating distribution.

### 5.4.4 Trading off Bias and Variance
Bias and variance of the estimator are the two sources of error as show in the following figure.
![Bias and variance](images/5_1.png)

**Mean square error of a estimator**
$$
MSE = \mathbb{E}[(\hat{\theta}_m-\theta)^2] = Bias(\hat{\theta}_m)^2 + Var(\hat{\theta}_m)
$$


**How to choose better model**
* What happens when we are given a choice between two estimators, one with
more bias and one with more variance
* The most common way to negotiate this trade-off is to use **cross-validation**


**Relationship between bias, variance, capacity, overfitting and underfitting**
![Relationship between these concepts](images/5_2.png)


### 5.4.5 Consistency 
The discussion about bias and variance fix the size $m$ of the dataset $\{x^{(1)}, x^{(2)}, \dots, x^{(m)}\}$. In other aspect, we want the estimation converge to the true value as the dataset size $m$ increase.

**Weak Consistency**
$$
plim_{m\to\infty}\hat{\theta}_m = \theta
$$
$plim$ is the probability limit

**Strong Consistency:** \
Consider the estimation as the dataset size increase $\{\theta_1, \theta_2, \dots, \theta_n\}$, $\theta_i$ indicates the estimation when the dataset have size $i$. $\theta_i$ is a random variable. The strong consistency is:
$$
P(\lim_{m\to\infty}\theta_m = \theta) = 1
$$

**Consistency and unbiasedness**
* a estimator is consistent $\Rightarrow$ a estimator is asymptotically unbiased 
* a estimator is asymptotically unbiased $\nRightarrow$ a estimator is consistency


## 5.5 Maximum Likelihood Estimation

Only need to understand that maximum likelihood estimation is to find the best-fit probability distribution given some examples set $\{x^{(1)}, x^{(2)}, \dots, x^{(n)}\}$

Read this [post](https://zhuanlan.zhihu.com/p/26614750)


>Ignore the mathematical derivation here, I think I need to learn a statistic course to better  understand these related materials. So for the first reading pass the book, I choose to jump directly to Part II of the book. Complete this part when I reading the book the second time. 


### 5.5.1 Conditional Log-likelihood and Mean Squared Error
This is the basis for most supervised learning algorithms. If $X$ is all the input and $Y$ is all the labels, then the conditional maximum likelihood estimator is:
$$
\theta_{ML} = \arg \max_{\theta}P(Y|X;\theta)
$$

If the examples are assumed to be i.i.d. Then this can be decomposed into
$$
\theta_{ML} =\arg\max_{\theta}\sum_{i=1}^mP(y^{(i)} | x^{(i)};\theta)
$$

### 5.5.2 Properties of Maximum Likelihood 

Maximum likelihood estimator is the best estimator asymptotically.

Under following conditions, the maximum likelihood estimator has the property of consistency
* The true distribution $p_{data}$ must lie within the model family $p_{model}(\cdot|\theta)$ 
* The true distribution $p_{data}$$ must corresponding to exactly one value of $\theta$



## 5.6 Bayes Statistics
> 5.5 and 5.6 illustrate the basic ideas and the main difference of frequentist and bayes statistics. Up to the first pass reading, I am still quite comfused about these materials. 



### 5.6.1 Maximum A Posteriori(MAP) Estimation 



## 5.7 Supervised Learning Algorithms

