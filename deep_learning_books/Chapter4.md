# Chapter 4. Numerical Computation

## 4.1 Underflow and Overflow 

Very small value will be stored as zero in computer



## 4.2 Poor condition 

* Conditioning refers to how rapidly a function changes with respect to small changes in its inputs
* High condition number suffer more from underflow and overflow 



## 4.3 Gradient-Based Optimization 

**Gradient Descent:**

Reduce $f(x)$ by move $x$ in small step with opposite sign of its derivative 

For multivariable function $f(\mathbf{x})$ , there are many direction that can reduce $f(\mathbf{x})$ , but the steepest descent direction is  

$\mathbf{x}' = \mathbf{x} - \epsilon \nabla_x f(\mathbf{x})$

$\epsilon$ is the learning rate 



