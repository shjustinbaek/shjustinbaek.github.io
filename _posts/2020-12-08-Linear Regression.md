---
layout: single
title:  "Linear Regression"
date:   2020-12-08
mathjax: true
---
# Linear Regression

​	This chapter is about *linear regression*, a very simple approach for supervised learning.  

Linear regression is useful for predicting a quantitative response. $$X+Y+TEST$$

![File:Logo-Test.png - Wikimedia Commons](https://upload.wikimedia.org/wikipedia/commons/8/85/Logo-Test.png)

![image](https://user-images.githubusercontent.com/46898478/101401388-a0373080-3915-11eb-8c17-25d71e6be8b0.png)


Linear regression is an old technique, and it may seem dull. However, it is still useful and widely used. 
* Many fancy statistical learning approaches are generalizations or extensions of linear regression. Therefore, it is important to understand linear regression.

---

##  3.1 Simple Linear Regression

​	*Simple linear regression* is a straightforward approach for predicting a quantitative response $$Y$$ on the basis of a single predictor variable $$X$$.

​	The model assumes that there is approximately a linear relationship between $$X$$ and $$Y$$ .

that is:	$$ Y \approx \beta_0 + \beta_1 X$$ . $$(3.1) $$

---

​	Note: (3.1) is often expression as regressing Y on X (Y onto X).

​	e.g if $$ X $$ represent TV advertising, and $$ Y $$ sales. By **regressing sales onto TV**, we can acquire the model: $$ sales \approx \beta_0 + \beta_1 TV $$.

---



$$ \beta_0, \beta_1 $$ are known as the model *coefficients* or *parameters.*

$$ \hat{\beta0},\hat{\beta1} $$ are the estimates of the model coefficients.

$$ \hat{y} $$ indicates a prediction of $$ Y $$ on the basis of $$ X=x $$. 

*Note: the $$ \hat{} $$ symbol denotes the estimated value for an unknown parameter* (coefficient) or *the predicted value of the response.*

### 3.1.1 Estimating the coefficient

​	In practice, $$ \beta_0 $$ and $$ \beta_1 $$ are unknown. Therefore, we must use data to estimate the coefficients.

​	Data: n observation pairs

​	$$ (x_1,y_1),(x_2,y_2),...,(x_n,y_n) $$ 

​	where $$ x_i $$ is a measurement of $$ X $$ , and $$ y_i $$ is a measurement of $$ Y $$. 

Our goal is to obtain coefficient estimates $$\hat{\beta0},\hat{\beta1}$$ such that the linear model $$Y \approx \beta_0 + \beta_1 X$$ fits the available data well. 

that is: $$y_i \approx \hat{\beta_0} + \hat{\beta_1} X$$ for $$ i= 1,2,...,n$$. 

In other words, find an intercept $\hat{\beta_0}$ and a slope $$\hat{\beta_1}$$ such that the resulting line is as *close* as possible to the $$n$$ data points.

There are many approaches to measure *closeness*. 

The most common method involves minimizing the *least squares* criterion. - focus of Ch 3. 

---



![1552817416650](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1552817416650.png)

Data: Advertising data. 

Blue line: the least squares fit for the regression. Grey line: error. 

In this case a linear fit captures the essence of the relationship. However, the relationship differs from left to right. 

---

Let $$\hat{y_i}=\hat{\beta_0}+\hat{\beta_1}x_i $$ be the prediction for $$Y$$ based on the $$i$$th value of $$X$$.

Then $$e_i=y_i-\hat{y_i}$$  is the $$i$$th *residual*, the difference between $$i$$th observed response and the 

$$i$$th response value that is predicted by our linear model.. 

We also define the *residual sum of squares* (**RSS**) as:

 $$RSS = e_1^2 + e_2^2 + ... + e_n^2=\sum_{i=1}^{n} e_i$$

​	$$RSS= (y1 - \hat{\beta_0}-\hat{\beta_1}x_1)^2+(y2 - \hat{\beta_0}-\hat{\beta_1}x_2)+...+(y_n - \hat{\beta_0}-\hat{\beta_1}x_n)  = \sum_{i=1}^{n} (y_i - \hat{\beta_0}-\hat{\beta_1}x_i)^2 $$



The least squares approach chooses $$\hat{\beta_0}$$ and $$\hat{\beta_1}$$ that yields the smallest **RSS**.

![1552819140869](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1552819140869.png) 

$$ \frac{\partial (SSE)}{\partial b_0} = -2 \sum_{i=1}^n(y_i-b_0-b_1x_i)= 0 $$, $$ \frac{\partial (SSE)}{\partial b_1} = -2 \sum_{i=1}^n(y_i-b_0-b_1x_i)x_i=0$$

$$\hat{\beta_0}$$  $$\hat{\beta_1}$$ that makes these 2 equations 0 yields minimum RSS, which are:

$$ \beta_1 = \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})} {\sum_{i=1}^n(x_i-\bar{x})^2}$$

$$\beta_0 = \bar{y}-\hat{\beta_1}\bar{x} $$                                          $$(3.4)$$

---

e.g $$\hat{\beta_0}$$ = 7.03, $$\hat{\beta_1}$$ = 0.0475 for the Advertising data. Where $$Y$$ is sales and $$X$$ is money spent on advertisement.

The simple linear model becomes:

$$y_i = 7.03 + 0.0475 X$$

$1000 increase in advertising would approximately lead to 47.5 increase in sales. 

---

### 3.1.2 Assessing the Accuracy of the Coefficient Estimates

​	Recall that we made the following assumption:

* The true relationship between X&Y is: $$Y=f(X)+\epsilon$$ , 

  f: unknown function; $$\epsilon$$ : error with $$E(\epsilon)=0$$

If f is approximated by a linear model, that is if $$Y=\beta_0 +\beta_1X +\epsilon$$ . 

that is if $$Y = \beta_0+\beta_1X+\epsilon$$:

The error term represents what we miss this simple model.

1. The true relationship is probably not linear
2. there may be other variables that cause variation in Y
3. There may be measurement error

Note: we typically assume $$\epsilon$$ is independent of $$X$$

---

Also, $$Y=\beta_0+\beta X +\epsilon$$ is the population regression line, *The best linear approximation to the true relationship between$$X$$ and $$Y$$.*

$$y_i = \hat{\beta_0} + \hat{\beta_1} X$$ is the least squares line.

![1552828724432](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1552828724432.png)

Left image: 

simulation result (100 simulated obs)

Generated with $$Y = 2 + 3X + \epsilon$$ where $$epsilon ~ N(0,\sigma^2)$$

red line: true relationship; blue line: least squares line

*In general, the true relationship of $$X$$ and $$Y$$ is not known. However, we can compute the least squares line.*

Right image: 

Generated 10 data set from $$Y = 2 + 3X + \epsilon$$. 

plotted the resulting least squares line from each of the data set. (the light blue lines)

Note: 

1. Each set resulted in different least squares line.
2. The population regression line does not change.

Important concept of these two lines:

a natural extension of the standard statistical approach: Using information from a sample to estimate a large population. 

*e.g* Use n observation to estimate $$\mu$$ .

​	$$\bar{y} = \hat{\mu}$$ is reasonable. 

​	Although, $$\bar{y} \neq \mu$$. This is a good estimate. 

In the same way, in linear regression we are estimating the unknown $$\beta$$ s.

---

$$\hat{\beta_0}$$ and $$\hat{\beta_1}$$ from OLS method are unbiased estimators. 

Unbiased estimators: does not systematically over- or under-estimate the true parameter. 

If we average the estimates over $$\infty$$ datasets, then:  

$$\hat{\beta_i} =  \beta_i$$ for all $$i$$ 

Note: the simulation result concurs with the analogy. the average of 10 regression line is almost the same as the population regression line. 

---

​	To answer the question of how close $$\hat{\beta_0}$$ and $$\hat{\beta_1}$$ are to the true values of $$\beta_0$$ and $$\beta_1$$, we need to compute the standard error (SE) of the parameter.

​	$$SE(\hat{\beta_0})^2=\sigma^2[\frac{1}{n}+\frac{\bar{x}^2}{\sum_{i=1}^{n}(x_i-\bar{x})^2}]$$

​	$$SE(\hat{\beta_1})^2=\frac{\sigma^2}{\sum_{i=i}^{n}(x_i-\bar{x})^2}$$						$$\sigma^2 = Var(\epsilon)$$

Caution!) condition for the equations to be valid: 

​	$$\epsilon$$ for each observation are uncorrelated with common variance $$\sigma^2$$.

​	Note that: 

* $$SE(\hat{\beta_1})$$ is smaller when $$x_i$$ are more spread out.
* $$SE(\hat{\beta_0})$$ is the same as $$SE(\hat{\mu})$$ if $$\bar{x}$$ were zero. 

We need $$\sigma^2$$ in order to find the standard error. We don't know that. However, we can estimate $$\sigma^2$$ from data. 

The estimate of $$\sigma$$ is the *residual standard error* (RSE):  $$RSE = \sqrt{\frac{RSS}{n-2}}$$ 

Note: Since the $$\sigma$$ is an estimate. Strictly, it is correct to the calculated $$SE$$ as $$\hat{SE}$$. For simplicity, I will not include the $$\hat{}$$ in the contents below. 



1. With standard error we can compute confidence intervals of the $$\beta$$s. 

* 95% confidence interval is the range where the true parameter is in with a 95% chance.

For linear regression the 95% confidence interval for $$\beta_0$$ and $$\beta_1$$ are:

* $$\hat{\beta_0} \pm 2SE(\hat{\beta_0})$$
* $$\hat{\beta_1} \pm 2SE(\hat{\beta_1})$$

2. Standard error can also be used for hypothesis testing on the coefficients where:

   $$H_0 : \beta_1 = 0$$	i.e) There is no relation between $$X$$ and $$Y$$.

   $$H_1 : \beta_1 \neq 0$$	i.e) There is relation between $$X$$ and $$Y$$.

   In words: to determine if $$\hat{\beta_1}$$ is sufficiently far enough for zero so that we are confident that $$\beta_1$$ is non-zero.

   to test the hypothesis, we compute a t-statistic. $$t = \frac{\hat{\beta_1}-0}{SE(\hat{\beta_1})}$$

   ​	Note: $$SE(\hat{\beta_1})$$ is the average amount that an estimate $$\hat{\beta_1}$$ differ from the $$\beta_1$$ . 

   ​	and the t-statistic measure the number of standard deviation that $$\hat{\beta_1}$$ is away from 0. 

   ​	If there is no relationship between $$X$$ and $$Y$$, then we use expect $t$ to have a t-distribution with 

   ​	$$n-2$$ degree of freedom.  

   We calculate the probability of a number that equal $$|{t}|$$ or larger in absolute value. (p-value)

   *a small p-value*

   $$\rarr$$ the probability of observing this t is small if $$\hat{\beta_1}=0$$

   $$\rarr$$ declare $$\beta_1 \neq 0 $$ (reject $$H_0$$) 

**Example)**

![1552832666531](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1552832666531.png)

​	WE reject $$H_0$$ . Notice that Std.error is much smaller than the coefficients. 



### 3.1.3 Assessing the Accuracy of the Model

​	After assessing the accuracy of the coefficient estimates, we want to quantify the extent which the model fits the data.

​	The quality of a linear regression model's fit is typically assessed using:

​	1. $$RSE$$ - the residual standard error

​	2. $$R^2$$ statistic

---

**Residual Standard Error**

​	RSE is an estimate of the standard deviation of $$\epsilon$$.

​	*in words* : it is the average amount that response will deviate from the true regression line.

​	$$RSE = \sqrt{\frac{1}{n-2}RSS}=\sqrt{\frac{1}{n-2}\sum_{i=1}^{n}(y_i-\hat{yi})^2} $$

---

*e.g* Table 3.2

![1552879686489](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1552879686489.png)

​	Ad - sales data.

​	The table indicates:

1. actual sales deviate from the true regression line by 3260 units on average.
2. even if the model were correct, prediction will be off by 3260 on average

---

​	The RSE is measure of lack of fit,

if $$\hat{y_i} \approx y_i$$ for $$i=1,2,...,n$$, then RSE will be small. (Vice Versa)



**$$R^2\space statistic$$**

​	*RSE is measured in units of Y, it is not always clear what constitutes a good RSE*

​	Therefore, $$R^2 $$ can be a useful measurement. 

​	$$R^2 = \frac{TSS-RSS}{TSS}=1-\frac{RSS}{TSS}$$

​	$$TSS(SST)=\sum(y_i - \bar{y})^2$$

​	$$RSS(SSE)=\sum(y_i - \hat{y_i})^2$$

​	TSS measure the total variance of Y.

​	RSS measure the amount of variability that is left unexplained after performing a regression.

​	$$R^2$$ explains the proportion of variability in $$Y$$ that can be explained using $$X$$.

*Note*: $$R^2$$ is a measure of the linear relationship between $$X$$ and $$Y$$.

​	  Correlation is also a measure of the linear relationship between $$X$$ and $$Y$$.

​	  $$r = Cor(X,Y) = \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}}$$

​	In fact, in simple linear regression setting $$r^2 = R^2$$.

## 3.2 Multiple Linear Regression

​	Multiple Linear Regression is an extension of simple linear regression to accommodate multiple predictors. 

​	The model is given by:

​	$$Y = \beta_0 + \beta_1X_1+ \beta_2X_2 + ... +\beta_pX_p $$

​	where $$X_j$$: $$jth$$ predictor, $$\beta_j$$: $$jth$$ coefficient. 

​	In word: $$B_j$$ is the average effect on $$Y$$ when $$X_j$$ is increased by 1, if other $$X$$s are held constant. 

### 3.2.1 Estimating the Regression Coefficients

​	coefficients $$\beta_0$$,$$\beta_1$$,...,$$\beta_p$$ are unknown. we need find estimates $$\hat{\beta_0},\hat{\beta_1},...,\hat{\beta_p}$$. With these coefficients, we can make prediction based on the model:

​	$$ \hat{Y} = \hat{\beta_0} + \hat{\beta_1}x_1+ \hat{\beta_2}x_2 + ... + \hat{\beta_p}x_p$$

---

​	In the same way as SLR, parameters of multiple linear regression are estimated by minimizing the $$RSS$$.

​	$$RSS = \sum_{i=1}^{n}(y_i-\hat{y_i})^2 = \sum_{i=1}^{n}(y_i - \hat{\beta_0} - \hat{\beta_1}x_{i1} - \hat{\beta_2}x_{i2} - ... - \hat{\beta_p}x_{ip})^2$$

​	Note: p must be smaller or equals to n. 

​	![1552972347753](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1552972347753.png)

![1552972367548](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1552972367548.png)

![1552972446329](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1552972446329.png)

![1552972523393](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1552972523393.png)

---

### Visual Example

![1552972636833](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1552972636833.png)

* When there are 2 $$X$$. The least squares regression line becomes a plane
* the plane is chose to minimize the RSS (the vertical lines)

---

![1552982422933](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1552982422933.png) ![1552982440767](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1552982440767.png)

Note: p-value of a $$X$$ can be large in Multiple linear regression, even if the p-value for the same $$X$$ is small when it is involved in a simple linear regression. (Due to correlation)

### 3.2.2 Some Important Questions

​	When we perform multiple linear regression, we are interested in a few important questions.

1. Is at least one of the predictors $$X_1,X_2,...,X_p$$ useful in terms of predicting $$Y$$?

2. Do all the predictors help to explain $$Y$$?

3. How well does the model fit the data?

4. Given a set of predictor values, what response value should we predict and how accurate is our prediction?

---

**1. Is there a Relationship between the Response and Predictors? (Is at least one of the predictors $$X_1,X_2,...,X_p$$ useful in terms of predicting $$Y$$?)**

​	To answer this question, we need to whether all of the regression coefficients are zero? We test the null hypothesis:

​	$$H_0: \beta_1=\beta_0=...=\beta_p=0$$

​	$$H_1: not \space H_0$$

​	This test is performed by using the $$F-statistic$$

​	$$F=\frac{(TSS-RSS)/p}{RSS/(n-p-1)}$$

​	If the linear model assumptions are correct, one can show that:

​	$$E[RSS/(n-p-1)]=\sigma^2$$

​	Also, if $$H_0$$ is true, then:	

​	$$E[TSS-RSS/p]=\sigma^2$$	

​	Thus, if $$H_0$$ is true, then the F-value will be close to 1. 



​	if $$H_0$$ is not true, then $$E[TSS-RSS/p] > \sigma^2$$ $$\rarr$$ F value would be larger than 1. 

​	Thus, if the F value is larger than 1, we reject $$H_0$$. 



​	How much larger does the F value has to be in order to reject $$H_0$$? 

​	1. if n is large, it can be just a little larger than 1 (relatively)

​	2. if n is small, it has to be much large than 1. 

​	

​	Note: when $$H_0$$ is true and the error $$\epsilon_i$$ follows a normal distribution . the F-statistic follows a F-distribution with a degree of freedom of $$(p,n-p-1)$$. we then calculate the p-value, and test the hypothesis.

---

​	**we can also test a subset of p coefficient** , that is:

​	$$H_0: \beta_{p-q+1} = \beta_{p-q+2} = ...=\beta_{p} =0 $$

​	$$H_1: not \space H_0$$

​	Here $$F=\frac{(RSS_0-RSS)/q}{RSS/(n-p-1)}$$ ~ $$F(q,n-p-1)$$

​	$$RSS_0$$ is the residual sum of squares for a multiple linear regression model that does not use last p coefficients. 

​	*Note*: 

​	If only one variable is omitted from the model above, then the p-value yielded from the F-test is exactly the same as the p_value yielded from individual t-test of the entire multiple linear regression model. 

​	The F-statistic reports the partial effect of adding that variable to the model. (need more 설명)

---

**Why F-test?**

​	It might seem that: if we perform t-test for all the $$X$$ variables and at least one of the p-values are small, we can conclude that at least one of the predictors is related to the response. 

​	However, this logic is flawed. (Problems can arise when the number of predictors $$p$$ is large)

​	*e.g*

​	$$p=100$$ 

​	$$H_0: \beta_1=\beta_0=...=\beta_p=0$$  is True. 

​	In this case, about 5% of p-values associated with t-statistics in MLR will be below 0.05 by chance. 

​	The F-test can avoid this problem. (it adjusts for the number of p variables)



**2. Deciding on important variables (Do all the predictors help to explain $$Y$$?)**

​	It is possible that all of the predictors are associated with the response. However, usually only a subset of predictors are related to the response. 

​	The task of determining which predictors are associated with the response, in order fit a single model involving only those predictors,  is called **variable selection**. 

​	

 1. Exhaustive search:

    ​	Ideally, it would be perfect to perform variable selection by trying out all possible combination of variables. However, since it requires $$2^p$$ tests, this method is not practical

2. Forward selection:

   ​	Begin with null model (a model that contains an intercept but no predictors)

   ​	fit p simple linear regression model

   ​	add the predictor that yields lowest RSS to the model.

   ​	add the predictor that yields lowest RSS in the new two-variable model.

   ​	....

   ​	Stop when a stopping rule is reached. 

3. Backward elimination:

   ​	Start with all variables in the model. 

   ​	remove the variable with the largest p-value. 

   ​	...

   ​	continue until a stopping rule is met. 

4. Mixed selection: 

   ​	combination of forward selection and backward elimination. 

---

**3. Model fit (How well does the model fit the data?)**

​	Two of the most common numerical measures of model fit are the $$RSE$$ and $$R^2$$. ($$R^2$$ is the fraction of variance explained by the regression model)

---

In simple linear regression: $$R^2$$ is $$Cor(Y,X)^2$$ 

In multiple linear regression: $$R^2$$ is $$Cor(Y,\hat{Y})^2$$ 

---

​	A large $$R^2$$ close to 1 indicate the model explains a large proportion of variance in the response variable. vice versa. 

​	*Caution*: $$R^2$$ will always increase when a variable is added to the model, even if the variables are weakly associated with the response. 

​		       But, inclusion of too many unassociated variables will likely lead to poor results on test samples due to overfitting. 

---

​	*Graphical summaries*



​	![1553040752733](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1553040752733.png)

​	It can reveal synergy or interaction effect between the advertising media. Dealing with this problem is mentioned at 3.3.2

---

**4. Predictions (Given a set of predictor values, what response value should we predict and how accurate is our prediction?)**

​	Once the multiple linear regression model is estimated, it is straightforward to predict the response $$Y$$. 

​	However, There are 3 uncertainties with the prediction.

 1. $$ \hat{Y} = \hat{\beta_0} + \hat{\beta_1}x_1+ \hat{\beta_2}x_2 + ... + \hat{\beta_p}x_p$$

    This model is only an estimate of the true population line. 

    The inaccuracy in the coefficient estimates is related to the reducible error. 

    We can compute a **Confidence interval**. in order to determine how close $$\hat{Y}$$ will be to $$f(x)$$.

2. In practice, assuming a linear model for $$f(x)$$ is almost always an approximation of reality. 

   This is called *model bias*.

   However, we will ignore this discrepancy and operate as if the linear model is correct. 

3. Even if we can perfectly predict $$f(x)$$, the $$Y$$ value cannot be perfectly predicted due to $$ \epsilon$$.(irreducible error)

   We use *prediction interval* to answer how much will $$Y$$ vary from $$\hat{Y}$$. 

   *prediction interval* is alway wider than confidence intervals. (They incorporate both the error in the estimate for $$f(x)$$ and the irreducible error.

   

## 3.3 Other Considerations in the Regression Model

### 3.3.1 Qualitative Predictors

​	In practice, there exists qualitative(categorical) predictors. (e.g gender, status, ethnicity)

---

**Predictors with only 2 levels**

​	If a qualitative predictor only has two levels(possible values). We simply create an indicator or dummy variable. takes on two possible numerical values.

​	*e.g* dummy variable of gender.

​	$$ x_i = 1 $$ if ith person is female

​	     $$=$$ 0 if ith person is male. 

​	we use this variable as a predictor in the regression equation. 

​	$$y_i = \beta_0 + \beta_1x_i + \epsilon_i = \beta_0 + \beta_1 + \epsilon_i$$ if ith person is female

​					  $$= \beta_0 + \epsilon_i$$         if ith person is male.

​	

​	It does not matter how we encode the categorical variables. if male = 1 and female = 0. There is no difference in terms of regression fit. 

​	However, there will be interpretation difference for $$\beta$$. 

​	Alternatively, we could also encode gender as: female = 1, male = -1

---

**Predictors with more than 2 levels**

​	When there are more than 2 possible values of a qualitative predictor, a single dummy variable cannot represent all possible values. 

​	we need to create addition dummy variable.

​	*e.g* ethnicity: Asian, Caucasian, African American

​	$$x_i1 = 1$$ (if ith person is Asian)

​	       $$= 0 $$ (if ith person is not Asian)

​	$$x_i2= 1$$ (if ith person is Caucasian)

​	       $$= 0 $$ (if ith person is not Caucasian)

​	then the regression equation becomes:

​	$$ y_i = \beta_0 + \beta_1x_i1 + \beta_2xi2 +\epsilon$$ 

​	There will always be one fewer, dummy variable than the number of levels. 

### 3.3.2 Extensions of the Linear Model

​	The standard linear regression model provides interpretable results and works  quite well on many real world problems. 

​	However, it makes several highly restrictive assumptions that are often violated in practice.

​	**2 important assumptions**

 1. the predictors and response are additive:

    predictors are independent of each other.

	2. the predictor and response are linear:

    the change in $$Y$$ cause by 1 unit of $$X$$ is constant. 

---

**Removing the Additive Assumption**

​	In our previous analysis of the Advertising data. We assumed that the effect effect on sales of increasing one ad-medium is independent of other ad-mediums. (The effect of a variable is constant)

​	However, this simple model may be incorrect. (notice, when levels of either TV or radio are low. the true sales are lower than predicted by the linear model.) a value of X can alter the effect ($\beta$).

​	In marketing, this is called synergy effect.

​	In statistics, this is called interaction effect.

​	We can account for this interaction effect by adding an *interaction term*. 

​	$Y = \beta_0 + \beta_1X_1 + \beta_2X_2 +\beta_3X_1X_2 +\epsilon$

​	    $=\beta_0 + (\beta_1 +\beta_3X_2)X_1 + \beta_2X_2 + \epsilon$ 

​	    $= \beta_0 + \tilde{\beta_1}X_1 + \beta_2X_2 + \epsilon$

​	Since, $\tilde\beta_1 = \beta_1 + \beta_3X_2$, it changes with $X_2$, the effect of $X_1$ on $Y$ is no longer constant. 

​	![1553052404611](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1553052404611.png)

​	Adding a interaction term can increase the performance of a model. $R^2 = 0.897 \rarr R^2 = 0.968$

​	This mean that $(96.8 - 0.897)/(100-0.897)=69%$% of the variability in sales that remains after fitting the additive model has been explained by the interaction term. 

​	the p-value of the interactive term suggest that $\beta_3 \neq 0$, in other words, it is clear that the true relationship is not additive. 

---

​	Note: Sometimes the p-value of individual $X$s can be large. However, their interaction term's p-value can be small. 

​		  Then we must include both the $X$s in the linear model. 

---

​	The concept of interactions can also be applied to qualitative variables. 

​	In fact the interaction between a qualitative and a quantitative variable is particularly nice. 

​	*e.g*

​	![1553053969724](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1553053969724.png)

​		MLR with categorical variable without interaction term. (it yields 2 parallel lines)

​	![1553061962462](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1553061962462.png)

​		MLR with categorical variable and with interaction term.

​		Likewise, there are two different regression lines for the student and non-student.

​		However, the two lines have different intercept as well as different slopes. 

​		This allows for the change in income($Y$) to be different among student and non-students. 

![1553062460332](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1553062460332.png)



---

**Non-linear Relationship**

​	The linear regression model assumes a linear relationship between $Y$ and $X$s. 

​	*e.g*

​	![1553062247040](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1553062247040.png)

​	The example above suggest that the relationship between Miles/gallon and Horsepower is nonlinear. 

​	A simple approach to deal with non-linear relation is to include transformed versions of the predictors in the model. 

​	example)

​	$mpg = \beta_0 + \beta_1 horsepower + \beta_2 horsepower^2+\epsilon$

​	*polynomial regression* methods will be discussed in **Chapter 7**



---

### 3.3.3 Potential Problems

​	When we fit a linear regression model to a particular data set, many problems occur. 

​	These are the 6 most common problems:

​	1. Non-linearity of the response-predictor relationships

​	2. Correlation of error terms

​	3. Non-constant variance of error terms.

​	4. Outliers

​	5. High-leverage points.

​	6. Collinearity

​	Note: overcoming these problems is more art-like than scientific.

---

**1.Non-linearity of the Data** 

​	If the true relationship of the response and the predictors are far from linear. 

​	The insight we gained from the regression model is uncertain.

​	Also, the prediction accuracy of the model is significantly reduced. 

​	

​	Residual plots can be used to identify non-linearity. 

![1553062987186](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1553062987186.png)

​	The residual plot of mpg and horsepower. (left is vanilla simple linear regression, right is regression with $horsepower^2$ term. 

​	The U shape of the residual plot suggests there is non-linear associations. 

​	The transformed residual plot seems to have reduced this effect.

​	*Note: there are more than 1 Fitted values for MLR.*

​	*Therefore in MLR we plot the residual $e$ versus $\hat{y_i}$*

​	When non-linear association is suggested by the residual plot. Then, we can use simple transformations of the predictors (*e.g $logX,\sqrt{X},X^2$*)

​	More will be discussed in Chapter 7. 

---

**2. Correlation of Error Terms**

​	linear regression assumes that the error terms $\epsilon_1,\epsilon_2,...,\epsilon_n$ are uncorrelated.

​	that is the sign of the value of $\epsilon_i$ has no effect sign of the $\epsilon_{i+1}$.

​	

​	If there is correlation among the error terms, the estimated standard error will tend to underestimate the true standard error. 

​	As a result, confidence and prediction intervals will be narrower than they should be. 

​	*e.g* 95% confidence interval may in reality have a much lower probability than 0.95 of containing the true value of the parameter. This may lead us to erroneously conclude that  a parameter is statistically significant.

​	Correlations among error terms frequently occur in the context of time series data. (It occurs in other places as well)

![1553069754128](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1553069754128.png)

​	Plotting residuals with observation:

 	1. check for tracking, that is if adjacent residual have similar values. 



​	Many methods have been developed to properly take account of correlations in the error terms in time series data. 

---

**3. Non-constant variance of error terms.**

​	Linear regression model also assumes the error term to have a constant variance.

​	that is, $Var(\epsilon_i) = \sigma^2$

​	standard error(RSE), confidence intervals, and hypothesis tests used in linear model rely on this assumption. 

​	However, often the variances of the error terms are non-constant. 

​	We can identify non-constant variances in the errors, (*heteroscedasticity*), by looking at the residual plot.

​	![1553070500135](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1553070500135.png)

​	The funnel shape indicates heteroscedasticity (not constant residual).

​	We can solve this problem by transforming $Y$ using a concave function such as $logY$ or $\sqrt{Y}$ .

​	(this results a greater amount of shrinkage of the larger responses.)

​	This can be confirmed on the left plot. 





---

**4. Outliers**

​	An outlier is a point, where the $y_i$ value is far from the value predicted by the model. 

​	this might occur for a variety of reason. (incorrect recording, missing 0, etc...)

![1553071286599](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1553071286599.png)

​	left plot: 

​		The red point in the left panel illustrates a typical outlier. 

​		Red line is is the regression model. and the red point is a outlier. 

​		(Blue line)In this case, removing the outlier does not have a big effect on the regression line. (this is typical)

​		Caution: removing an outlier will increase $R^2$ or decrease $RSE$.

​	Center plot:

​		Residual plots can be used to identify outliers. 

​		But in practice, it is difficult to decide how large does a residual has to be.

​	Right plot:

​		To solve the problem of residual plot.  we plot the studentized residuals.

​		(simply divide each $e_i$ by its estimated standard error.)

​		The obs where the studentized residual exceed 3 in absolute value is classified as outliers. 

​		*The red dot is clearly an outlier by this standard*



​		We can simply delete the outliers, if we believe them to be measurement errors. 

​		However, be careful. an outlier may indicate problems within the model. (missing predictor. etc...)

---

**5. High-leverage points.**

​	High-leverage points are points with unusual $x_i$ values. 

![1553072488450](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1553072488450.png)

Left picture: 

​	Obs 41 is a high leverage point. Removing it has a large impact on the regression line. 

​	Removing a high leverage point $\rarr$ big impact on regression line. 

Center picture: 

​	In SLR, a point with unusual $X$ value is the high leverage point. 

​	But, in MLR it is less obvious. 

​	In the picture, by plotting $X_1,X_2$. we identify the red dots as the high leverage point. unusual in terms of the full set of predictors.  

​	Note: its $X_1,X_2$ range is acceptable.

​	This is a problem, we cannot always plot all $X$s in MLR. 

​	to quantify an obs' leverage, compute the *leverage statistic*:

​	$h_i=\frac{1}{n} + \frac{(x_i-\bar{x})^2}{\sum_{i'=1}^{n}(x_{i'}-\bar{x})^2}$ (leverage statistic for single predictor) $1/n<h_i<1$

​	There is a simple extension for multiple predictors (not provided in book)

​	if $h_i$ greatly exceeds (p+1)/n, we suspect that the corresponding obs has high leverage. 

Right picture:

​	Studentized residual vs $h_i$. 

​	obs 41 has high values of sr and $h_i$. (both an outlier and a high leverage point)

---

**6. Collinearity**

​	Collinearity is the situation where two or more predictor variables are closely related to one another. 

![1553074329679](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1553074329679.png)

Left: no obvious relation. Right: collinear. 

​	The presence of collinearity can be a problem in linear regression. 

​	Since, $X_1$ and $X_2$ increase and decrease together. it is difficult to determine how each one is separately associated with $Y$. 

![1553074505991](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1553074505991.png)

Right: contour plot of collinear $X$s. 

Each ellipse represents the set of $\beta$s that yields the same value of $RSS$. closer to the center the smaller the RSS.

collinearity exists $\rarr$ small change in the data could cause the optimal coefficient values to move anywhere along this long valley. 

(because, the range of $\beta$s that yields the same RSS is now much larger)

Notice, that the $\beta_{Limit}$ 's range has increased significantly. 

![1553075172325](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1553075172325.png)

collinearity increase the standard error for $\hat{\beta_j}$ to increase. 

collinearity also decrease the power of t-tests. (probability of correctly detecting a non-zero coefficient is reduced) (t = $\hat{\beta_j} / SE$ )

We might make wrong decisions (picture above p-value)

---

A simple way to detect collinearity $\rarr$ correlation matrix.  (However, it cannot detect multicollinearity)

A better way to assess multicollinearity: *vairance inflation factor* (VIF)

![1553075565839](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1553075565839.png) $\geq$ 1 ; $R^2$ here: $Y$ is $X_j $ , $Xs$ are other $Xs$ 

VIF is the ratio of the variance of $\hat{\beta_j}$ in full model devided by variance of $\hat{\beta_j}$ when it is fit on its own. 



$VIF =1 $ indicate the complete absence of collinearity. (does not happen in practice)

Rule of thumb:

​	if $VIF \geq 5,\space or \geq 10$ $\rarr$ problematic collinearity. 



When faced with collinearity. 

 	1. drop one of the problematic variables. 
 	2. combine the collinear variables. (take average)

## 3.5 Comparison of Linear Regression with K-NN

​	Linear regression is a *parametric* approach. (assumes a linear form of $f(X)$)

​	adv: easy to fit. (only need to estimate a small number of coefficients)

​		(linear regression)easy to interpret 

​		statistical tests can be easily performed

​	dis-adv: 

​		make strong assumptions about the form of $f(X)$, if not true $\rarr$ low performance. 

​		

​	*non-parametric methods*:

​	no assumption about the form of $f(X)$. 

​	more flexible. 



*e.g* K-nearest neighbors regression. (KNN regression)

![1553077348565](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1553077348565.png)

![1553077383345](C:\Users\naval\AppData\Roaming\Typora\typora-user-images\1553077383345.png)



​	the parametric approach will outperform the nonparametric
approach if the parametric form that has been selected is close
to the true form of f.

​	Also, when there are noise, parametric methods will perform better. 

​	KNN-regression is vulnerable to high dimensional data.

​	Note: Even when dimensions are low, KNN is hard to interpret. Therefore, linear regression can be preferred. 

​	Generally, parametric methods will tend to outperform non-parametric approaches when there is a small number of observation per predictor.

















