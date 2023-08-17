---
title:
layout: default
permalink: /research/supervised_ml
published: true
---

# Table of contents

1. [Fundamentals](#fundamentals)

    1.1. [Definition](#definition)

    1.2. [Terminology and Notation](#term)

    1.3. [Cost Function](#costfunc)

    1.4. [Types of supervised learning](#types)
2. [Regression Model](#regression)

    2.1. [Linear Regression Model](#linearregression)

    2.2. [Polynomial Regression Model](#polyregression)
3. [Classification Model](#classification)

    3.1. [Logistic Regression](#logisticregress)

> It has been a while since I started to learn fundmentals of Machine Learning. In this post, I will re-study and take some note of useful concepts, formulas, and practical tips related to ***supervised learning*** to reduce the searching time in the future. 

## Supervised Learning

<div id='fundamentals'/>

### 1. Fundamentals

<div id='definition'/>

#### Definition 

Given the input $X$ (or *labeled data*) which associated with output labels $y$, our algorithms will learn from them to make an *answer* or prediction $\hat{y}$. 

![](https://static.javatpoint.com/tutorial/machine-learning/images/supervised-machine-learning.png)

`Figure 1: The workflow of supervised machine learning`

<div id='term'/>

#### Terminology and Notation

+ $x$: an input variable/feature

+ $X$: a matrix of multiple features

$$
X=
\begin{bmatrix}
1, x^{(1)}_1, x^{(1)}_2,...,x^{(1)}_n \\
1, x^{(2)}_1, x^{(2)}_2,...,x^{(2)}_n  \\
. \\
. \\
1, x^{(m)}_1, x^{(1)}_2,...,x^{(m)}_n 
\end{bmatrix}
$$

+ $m$: the total number of training example

+ $n$: the total number of features per one training example

+ $x^{(m)}_{n}$: the value of $n^{th}$ feature in the $m^{th}$ training example  

+ $y$: output variable/target variable/label

$$
y=
\begin{bmatrix}
y^{(1)}\\
y^{(2)}\\
. \\
. \\
y^{(m)}
\end{bmatrix}
$$

+ $W$: a matrix of weight

$$
W=
\begin{bmatrix}
w_0\\
w_1\\
. \\
. \\
w_n
\end{bmatrix}
$$

+ $f$: function/hypothesis

+ $\hat{y}$: prediction/estimated $y$


+ $b$: a matrix of bias

+ $\alpha$: learning rate 

+ $\mu$: sample mean

+ $\sigma$: standard deviation

+ $z$: activation function

+ $L$: loss function

<div id='costfunc'/>

#### Cost Function

*Cost function*, denoted as $J$, is to determine how well the model (or *the line* or the choice of adjusting $w$ and $b$) fits the training data. The example below is the *cost function* (i.e. mean squared error - MSE) of *linear regression model*.

$$
J(w,b)= \frac{1}{2m}\sum\limits_{i=1}^{m}{\underbrace{(\hat y^{(i)} - y^{(i)})}_\text{error}}^2=\frac{1}{2m}\sum\limits_{i=1}^{m}{( f_{w,b}{(x^{(i)})} - y^{(i)})}^2
$$

where $i$ represents the $i^{th}$ training example, and $b$ and $w$ are *training parameters* which will be adjusted during the training process.

![](https://res.cloudinary.com/practicaldev/image/fetch/s--OmQqkAcP--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://dev-to-uploads.s3.amazonaws.com/uploads/articles/c8xvp5hnpmg375il4p1x.png)

`Figure 2: The error between the predicted value and the actual label`


<div id='gd'/>

#### Gradient Descent

*Gradient descent* algorithm is to minimize the cost function $J$ by finding values of parameters $w$ and $b$. For example, given the cost function $J(W,b)$ and we want to minimize it. 

$$
{\min_{W,b}}{J(W,b)}
$$

<span style="color:red">*Gradient descent algorithm*</span>

+ Step 1: Choose an appropriate $\alpha$ and Randomly initialize $w$ and $b$ parameters
+ Step 2: Repeat until convergence {

    $$w=w-\alpha{d \over dw}J(w,b)$$ 

    $$b=b-\alpha{d \over db}J(w,b)$$ 
    }


*Types of gradient descent*

+ `Batch GD`: each training epoch of GD uses all the training example before updating parameters.

+ `Stochastic GD`: each training epoch of GD uses one training example.

+ `Mini-batch GD`: each training epoch of GD uses a small batch size of training example.

<span style="color:red">*Gradient descent tips*</span>

+ `Feature scaling`: is applied to features where their values have a large gap. For example, in the housing price prediciton problem, we have a feature *bedrooms* $X_1=[x_1, x_2,...,x_n]$ ($1 \le x_1, x_2,..., x_n \le 5$) and a feature *size* $X_2=[x_1, x_2,...,x_n]$ ($500 \le x_1, x_2,..., x_n \le 1000$).


    + *Mean normalization technique*

    $$
    X_i = \frac{X_i-\mu_{i}}{\max_{(X_i)} - \min_{(X_i)}}
    $$

    + Z-score normalization method

    $$
    X_i=\frac{X_i-\mu_{i}}{\sigma_{i}}
    $$

> If we do not apply **feature scaling**, features that have larger values than other ones will have a larger impact on the ***cost function***. This is because any small change in the *weight* value in the former leading to big impact on the value of cost function (these *weight* values are multiplied with big values of features).

+ Another important question raised is when do we use ***normalization*** and ***standardization***

    + `Normalization`:

        + is used when the data doesn't have Gaussian distribution

        + The value scale falls between [0, 1] or [-1, 1],...

    + `Standardization`:

        + is used when the data have Gaussian distribution

        + All features will have a mean $\mu$ of 0 and a standard deviation $\sigma$ of 1.

        + *eg*: Z-score normalization

> **NOTE**: It is important to note that we need to *normalize/standardize our new data to the same scale as our model works before* making a prediction.

<span style="color:red">*Choose a proper learning rate*</span>

+ Values of $\alpha$ we can try at the firt glance: *0.001, 0.003, 0.01, 0.03, 0.1, 0.3*, and so on.

<span style="color:red">*Feature Engineering*</span>

It relates to the process of choosing, transforming and combining the most proper features.


<div id='types'/>

#### Types of supervised learning

+ <span style="color:blue">Regression</span>

    + The number of labels is *large* and the value of the *label* is a [*continuous variable*](https://en.wikipedia.org/wiki/Continuous_or_discrete_variable#Continuous_variable), integer,... For example, predicting the housing price,...

+ <span style="color:blue">Classification</span>

    + The number of labels is *small* and the value of the *label* is a [*dicrete variable*](https://en.wikipedia.org/wiki/Continuous_or_discrete_variable#Discrete_variable). For example, predicting the given animal is dog or cat,... 

We will delve into each of these types in the following.

<div id='regression'/>

### 2. Regression Model

<div id='linearregression'/>

#### 2.1. Linear Regression Model

We fit <u>*a straight line*</u> across data points where <u>*the straight line*</u> is formulated as follows: 

$$f_{w,b}(x)=f(x)=\hat y=w{x}+b$$

where $W$ and $b$ are parameters learned during the training process. Please refer to terminology [here](#term).

Here is the example of using [sklearn](https://scikit-learn.org/) to implement *linear regression*. [[code link]]()

<div id='polyregression'/>

#### 2.1. Polynomial Regression Model

We can fit <u>*a curve*</u> across data points. The formula of polynomial regression is denoted as follows:

$$f_{W,b}(X)=f(X)=\hat y=W{X}+b$$


![](https://serokell.io/files/ka/kawer8rc.5_(5).png)

`Figure 3: Linear regression vs. Polynomial regression`

<div id='classification'/>

### 3. Classification Model

<div id='logisticregress'/>

#### 3.1. Logistic Regression

<span style="color:blue">*Sigmoid function*</span>

![](https://machinelearningmastery.com/wp-content/uploads/2021/08/sigmoid.png)

`Figure 4: The sigmoid function and its derivative (The axis of x in the figure will be the axis of z for below formulation.)`

<span style="color:blue">*Formulation*</span>

+ We will consider the logistic regression output $g(z)$ is the *probability* that class is 1. Particularly, $g(z)$ is the probability that $y$ is 1, given input $X$, parameters $W$ and $b$.

$$
z = WX + b$$

and 

$$
f_{W,b}{(X)} = \hat y = g(z) = {1 \over 1 + e ^ {-(WX+b)}} = {1 \over 1 + e ^ {-z}}  = P(y=1 | X; W, b)$$

where 

$$ 0 \lt g(z) \lt 1 
$$


<span style="color:blue">*Decision boundary*</span>

It can be a *linear/curve* decision boundary. 

![](https://www.researchgate.net/publication/349186066/figure/fig1/AS:989978611953666@1613040702298/Example-of-overfitting-in-classification-a-Decision-boundary-that-best-fits-training.png)
`Figure 5: The decision boundary`

<span style="color:blue">*Cost function for logistic regression*</span>

> Why don't we use MSE in logistic regression?
[**Answer**](https://www.baeldung.com/cs/cost-function-logistic-regression-logarithmic-expr#2-the-problem-of-convexity): For the case of the linear model, MSE was guaranteed to be convex because it was a linear combination of a prediction function that was also composed. For the case of logistic regression, MSE isn’t guaranteed to be convex because *it’s a linear combination between scalars and a function*, the logistic function, that’s also not convex.

+ Logistic loss function: 

    $$
    L(f_{W,b}(X), y) = 
    $$