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

###5.4.1 Point Estimation