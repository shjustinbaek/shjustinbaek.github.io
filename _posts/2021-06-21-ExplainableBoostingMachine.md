---
layout: single
title:  "설명가능한 부스팅 모델 - Explainable Boosting Machine"
date:   2021-06-21
mathjax: true
tags: ML xai ebm gam ga2m ExplainableBoostingMachine
categories: ML
toc: true
---

>설명하기 어려운 XGBoost, LightGBM, RandomForest와 대등한 성능을 가지면서, 설명 가능한 Explainable Boosting Machine의 원리, API 사용 방법, 활용 사례에 대해 알아보자.


## accuracy vs interpretability trade-off
![img](https://miro.medium.com/max/978/1*SI3wAOvfTQrLl5NXQHwuxA.png)

머신러닝에서 모델의 예측 정확도와 설명 가능성이 trade-off 관계를 보인다는 것은 정설처럼 받아들여졌다.

부스팅, 딥러닝 모델처럼 높은 예측 정확도를 보이는 모델들은 각 feature가 의사결정에 어떻게 관여하는지 파악하기 어렵고

선형회귀, 의사결정나무 같이 각 feature의 의사결정 기여를 설명하기 쉬운 모델들은 보편적으로 낮은 예측 정확도를 보인다.



즉, 우수한 예측 성능과 설명 가능성을 모두 확보하는 것은 불가능하다고 여겨져왔다.


<br>
## Explainable Boosting Machine (EBM)
그러나, 마이크로소프트의 interpretML 패키지에 포함된 Explainable Boosting Machine(EBM)은 우수한 예측 성능과 설명 가능성 두마리 토끼를 모두 잡은 것처럼 보인다.

![image-20210621230204530](https://user-images.githubusercontent.com/46898478/125193853-9bf73b00-e289-11eb-9538-3133d7c84f2d.png)



5개의 테스트 데이터셋에 대해 EBM, LightGBM, 로지스틱 회귀, 랜덤 포리스트, XGBoost 모델을 학습시킨 결과이다.

EBM은 높은 설명 가능성을 가지고 있음에도 불구하고, LightGBM, 랜덤 포리스트, XGboost같은 모델들과 동등한 수준의 성능을 보인다.

이는 EBM을 활용한다면 우리는 더이상 높은 예측 정확도를 확보하기 위해 설명 가능성을 포기할 필요가 없다는 것을 의미한다!


<br>
**그렇다면  설명 가능성을 포기하지 않는게 왜 큰 장점인가?**
<br>
- 모델 디버깅

>설명 가능성이 높은 모델을 사용한다면, <br>
>단순히 모델의 잘못된 예측을 관찰하는것을 넘어서, 잘못된 예측의 의사결정 과정을 확인할 수 있다<br>
>이를 기반으로 overfitting의 징후를 찾고, 문제를 일으키는 데이터 전처리 과정을 찾는 등 모델 디버깅을 수월하게 진행할 수 있다.

- 공정성 이슈 확인

>모델에 공정하지 못한 bias가 존재하는지 확인할 수 있다.

- 모델 신뢰성

> 의사결정 과정을 확인할 수 있는 모델은 사람이 더 신뢰할 수 있고, 머신러닝 도메인 지식이 없는 사람들에게 의사결정 과정을 설명하기 용이하다.

- 하이 리스크 의사결정

> 의료, 금융같은 산업 분야에서 머신러닝 모델을 사용한다면, 잘못된 예측에 대한 리스크가 매우 크다.<br>
> 설명 가능성이 높은 모델을 사용한다면 모델의 예측을 무작정 믿지 않고, 모델의 의사결정 과정을 검토하며  sanity check을 진행할 수 있다.



<br>
## EBM 모델 구조
EBM은 Generalized Additive Model (GAM)의 발전된 형태인 GA2M 모델에 속하는 알고리즘이다.
조금 더 자세히 설명하자면, FAST알고리즘으로 pairwise feature interaction term을 선택하는 GA2M 알고리즘이다.

<br>
**Generalized Additive Model (GAM)**
<br>

아래의 형태로 구성된 예측 모델을 Generalized Additive Model (GAM)이라 부른다.

> $g(E(y)) = \beta_0+ \Sigma f_i(X_i)$

$g$를 link function으라 지칭하고, 1개의 feature를 입력으로 받는 $f_i$를 shape function이라 지칭한다.

regression task에서 $g$는 identity function이고:	$E(y)=\beta_0+\Sigma f_i(Xi)$

classification task에서 $g$는 logistic function이다:	$\log \frac{p}{1-p}=\beta_0+\Sigma f_i(Xi)$

전통적으로 backfitting 알고리즘, regression spline이 GAM 학습에 사용되어왔다.

<br>

GAM은 Generalized Linear Model과 형태가 흡사하다.

multiple linear regression:	$E(y)=\beta_0+\Sigma \beta_i Xi$

logistic regression:	$\log \frac{p}{1-p}=\beta_0+\Sigma \beta_i Xi$

<br>

GAM의 형태는 Generalized Linear Model처럼 각 feature에 대한 연산값의 합을 의사결정에 활용하기 때문에 각 feature가 의사결정에 어떤 영향을 미치는지 파악하기 쉬우면서도, Generalized Linear Model에 비해 feature와 target값의 비선형적인 관계를 더 잘 파악할 수 있어, linear model에 비해 예측 성능이 훨씬 좋다.

*Note: Gradient Boosting같은 Full Complexity Model은 $y=f(x_1,...,x_n)$ 형태로 예측을하기 때문에 각 feature가 의사결정에 어떻게 기여하는지 쉽게 파악할 수 없다.*


하지만, GAM은 feature $x_i,x_k$의 pairwise interaction을 모델에 포함할 수 없다는 한계점 있다.

<br>
**Generalized Additive Model plus Interactions (GA2M)**
<br>

GAM의 한계점을 극복하기 2 feature간의 pairwise interaction을 담을 수 있는 pairwise interaction term이 추가된 알고리즘이 GA2M이다.

아래의 형태로 구성된 예측 모델을 Generalized Additive Model plus Interactions (GA2M)이라 부른다.

> $g(E(y)) = \beta_0+ \Sigma f_i(X_i) + \Sigma f_{i,j}(x_i,x_j)$

pairwise interaction term $f_{i,j}(x_i,x_j)$은 $x_i,x_j$평면에 heatmap 형태도 시각화할 수 있고.

$x_i,x_k$의 pairwise 값이 모델의 의사결정에 어떤 영향을 미치는지 비교적 쉽게 파악할 수 있다.

<br>

feature의 수가 많아지면 pairwise interaction term의 수도 폭발적으로 증가한다.

GA2M 모델을 생성할 때 모델 싸이즈가 과도하게 커지는 것을 방지하지 위해  통계적으로 유의한 수준의 pairwise interaction을 가지는 $x_i,x_j$에 해당하는  $f_{i,j}(x_i,x_j)$을 선택할 필요가 있다.

전통적으로  ANOVA , Partial Dependence Function, GUIDE, Grove등이 pairwise interaction 검증에 사용되었다.

<br>

**Explainable Boosting Machine**
<br>

Explainable Boosting Machine (EBM)은 현대적인 머신러닝 기법(gradient boosting 등)을 활용해 feature function $f$를 학습하고, FAST 알고리즘으로 pairwise interaction term을 선택하는 GA2M 모델이다.

<br>
## EBM 학습 과정
Explainable Boosting Machine은 2 단계의 과정을 거치며 학습을 진행한다.

1. gradient boosting으로 pairwise interaction을 고려하지 않은 $f_i$ 학습
2. FAST 알고리즘으로 모델에 포함할 $f_{i,j}$ 선택 후 gradient boosting으로 학습

<br>
**1. gradient boosting으로 pairwise interaction을 고려하지 않은 GAM 모델 학습**
<br>

![image-20210622021726012](https://user-images.githubusercontent.com/46898478/125193862-ab768400-e289-11eb-883a-db1c39903b44.png)

EBM은 위 알고리즘을 사용해 pairwise interaction을 고려하지 않은 GAM 모델을 학습한다.

보편적인 부스팅 기법과 유사하게 데이터에 작은 트리를 피팅하는 것으로 학습이 시작하고, 다음 트리는 이전 트리의 residual을 예측하도록 학습된다.
조금 다른 점은 트리를 학습할 때 1개의 feature (feature1)만 사용할 수 있도록 제한한다는 것이다.

![image-20210622024256361](https://user-images.githubusercontent.com/46898478/125193870-b6c9af80-e289-11eb-970b-fef44d88eee5.png)
<br>
마찬가지로, 두번째 트리도 1개의 feature (feature2)만 사용하여 첫번째 트리의 residual을 예측하도록 트레이닝합니다.

![image-20210622024707598](https://user-images.githubusercontent.com/46898478/125193886-c6e18f00-e289-11eb-925b-429b6c1fbe37.png)
<br>
이런 형식으로 모든 feature를 순차적으로 사용해서 base estimator를 학습 시키면 한 iteration이 완료됩니다.

모든 트리를 학습 시킬 때 아주 작은 learning rate를 적용하여 feature에 트리를 적용 시키는 순서가 무의미하도록 유도합니다.

![image-20210622024900319](https://user-images.githubusercontent.com/46898478/125193893-ccd77000-e289-11eb-9c79-0bd9b4a93e51.png)
<br>
이 과정을 M iteration 반복합니다.

![image-20210622031525624](https://user-images.githubusercontent.com/46898478/125193899-d7920500-e289-11eb-85cb-944ea96b7dba.png)
<br>

M iteration이 완료되면 각 feature 만을 독립변수(x)로 받는 트리가 M개의 생성됩니다. 각 feature에 해당하는 트리 예측값의 합을 lookup table (graph)로 변환하고 트리를 삭제합니다. 

이후 $x_1$ 가 주어지면 해당 그래프로 $f_1$값을 계산합니다.

![image-20210622025947705](https://user-images.githubusercontent.com/46898478/125193909-dfea4000-e289-11eb-8dad-33756a11a462.png)
<br>
<br>
**2. FAST 알고리즘으로 모델에 포함할 $f_{i,j}$ 선택 후 gradient boosting으로 학습**
<br>

앞서 설명했듯이 feature의 수가 많을때, 존재하는 feature pair 조합의 수는 매우 크다. 그러므로, 수많은 feature interaction term 중 어떤 interaction term을 모델에 포함할지 결정하는 과정이 필요하다.

EBM은 FAST 알고리즘을 이용하여 어떤 interaction term을 모델에 포함할지 결정하게된다.

FAST 알고리즘의 철학은 다음과 같다: 
<br>
$(x_i,x_j)$feature pair의 interaction이 강하다면, $f_{i,j}$도 낮은 RSS를 보일 것이다.

하지만, 모든 $f_{i,j}$를 부스팅 방식으로 학습 시키는 것은 컴퓨팅 비용이 크기 때문에 적절하지 않다.

FAST는 $x_i, x_j$ 평면에 2개의 cut $c_i,c_j$을 생성하고 각 quadrant에 속하는 데이터의 평균값을 예측값으로 사용하는 간단한 모델을 만들어 부스팅 모델 학습을 대체한다. 이 모델을 $T_{i,j}$(interaction predictor)라고 지칭한다. 

![image-20210622033255318](https://user-images.githubusercontent.com/46898478/125193914-e7114e00-e289-11eb-8e0a-b905684ac335.png)

$T_{i,j}$ 학습은 모든 $(c_i, c_j) $조합을 모두 탐색하여 $RSS = \Sigma (y_k − T_{ij}(x_k))^2$가 가장 낮은 최적 $(c_i, c_j)$를 선택하는 방식으로 진행된다. 

가장 낮은 $RSS$값을 가지는 $T_{i,j}$에 해당하는 $(x_i,x_j)$를 가장 interaction이 높은 feature pair로 간주한다. 

이후 interaction이 높은 상위 K개 feature pair를 선택하고. 이전 방식과 동일한 부스팅 방식으로 모델 잔차에 2개 feature를 사용할 수 있는 tree를 학습 시킨다.

*현재 EBM 학습 과정 1,2가 1 iteration을 구성하는지. 1에 대한 iteration이 완료된 후 2에 대한 iteration이 진행되는지 명확하게 확인하지 못했습니다.추후에 추가적으로 확인해보고 위에서 설명한 알고리즘과 다르다는 것이 확인된다면 업데이트하겠습니다.*


<br>
## 코드 예제
interpretML은 scikit-learn 스타일의 API를 제공하기 때문에, 익숙한 .fit() .predict() method로 간편하게 모델 학습 및 예측을 진행할 수 있습니다.

아래 코드 예제는 interpretML EBM documentation에서 가져왔습니다.

나이, 학력 등 14개 feature로 연소득이 $50,000 이상인지 아닌지를 분류하는 이진 분류 문제입니다.

```python
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())
```

```python
import pandas as pd
from sklearn.model_selection import train_test_split

from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    header=None)
df.columns = [
    "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
    "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
    "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
]
df = df.sample(frac=0.05)
train_cols = df.columns[0:-1]
label = df.columns[-1]
X = df[train_cols]
y = df[label]

seed = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

ebm = ExplainableBoostingClassifier(random_state=seed)
ebm.fit(X_train, y_train)
```

모델 트레이닝이 완료되면 .explain_global() 메소드로 각 feature값이 EBM의 예측값에 어떻게 기여하는지 확인할 수 있습니다.
```python
ebm_global = ebm.explain_global()
show(ebm_global)
```
다른 트리 기반 앙상블 모델과 마찬가지로, feature importance를 확인할 수 있습니다.
![image-20210622035103270](https://user-images.githubusercontent.com/46898478/125193931-f6909700-e289-11eb-9878-6e085c914b03.png)

<br>
모든 feature에 해당하는 $f_i$ shape function을 시각적으로 확인할 수 있습니다. 
![image-20210622035128181](https://user-images.githubusercontent.com/46898478/125193939-027c5900-e28a-11eb-9a7f-5a61b6efce06.png)
<br>
모델에 포함된 interaction term $f_{i,j}$도 시각적으로 확인할 수 있습니다.
![image-20210622035253207](https://user-images.githubusercontent.com/46898478/125193949-0d36ee00-e28a-11eb-8955-14e705f3e007.png)

<br>


EBM의 .explain_local() 메소드로 instance의 예측 결과에 대한 설명을 볼 수 있습니다.
```python
ebm_local = ebm.explain_local(X_test[:5], y_test[:5])
show(ebm_local)
```
![image-20210622035405893](https://user-images.githubusercontent.com/46898478/125193956-17f18300-e28a-11eb-99ab-e7e40dec7864.png)
<br>
4번째 index 값을 EBM은 0으로 예측했고, 그 예측을 어떻게 했는지 확인할 수 있습니다.

위 그림의 각 막대는 각 shape function에 feature 값을 대입한 연산 결과를 나타냅니다.


EBM regressor에서 모델의 예측값은 위 그림에서 시각화된 막대 값의 합이고.
<br>
EBM Classifier에서 모델의 예측 positive class 확률은 위 그림에서 시각화된 막대 값의 합의 inverse-logit 값입니다.
<br>

다른 부스팅 모델과 달리, EBM을 사용하면 모델이 특정 예측 결과에 어떻게 도달했는지 매우 명확하게 확인할 수 있습니다.

<br>
## EBM을 활용한 모델 디버깅 사례
![image-20210622034023296](https://user-images.githubusercontent.com/46898478/125193968-2d66ad00-e28a-11eb-8e0c-f5424b1a0524.png)
간략하게 EBM을 디버깅한 사례를 소개하고 글을 마무리하겠습니다.

위 그래프는 폐렴 사망 여부 예측 EBM 모델링 과정에서 발견되었습니다.

폐렴 사망 여부 예측에 사용된 feature 중 흡입산소분율(FiO2)동맥혈산소분압(PaO2)비율이 모델의 예측 결과에 어떤 영향을 미치는지 확인하고자 해당 shape function 그래프를 확인해보았는데 이상한 점이 발견되었습니다.

보편적으로 폐렴 환자들은 흡입산소분율(FiO2)동맥혈산소분압(PaO2)비율이 낮을 수록 위급한 상태입니다. 하지만, 이상하게도 FiO2/PaO2비율이 300일 때 EBM은 사망 확률이 낮아진다고 예측하다는 점을 발견했습니다.

데이터 전처리 과정을 검토해보니, 데이터셋 FiO2/PaO2비율 feature의 평균값이 300이었고. 데이터의 결측치를 이 평균값으로 채웠다는 사실을 발견했습니다.

즉, 폐렴의 정도가 심하지 않아 병원에서 FiO2/PaO2비율을 굳이 측정하지 않은 환자들의 FiO2/PaO2비율이 300으로 EBM 모델에 주어져서 FiO2/PaO2비율이 300정도 일때 EBM 모델은 사망 확률이 낮아진다고 예측하는 현상이 발생한다는 것을 확인할 수 있었습니다.

이 사례에서 볼 수 있듯이 EBM은 높은 예측 성능과 설명 가능성을 모두 가지고 있는 훌륭한 모델입니다. 

<br>
## 참고 자료
- 
[InterpretML documentation](https://interpret.ml/docs/ebm.html?fbclid=IwAR0P7TwfiWMBpbY1EU7YhDrtjM4wPbpiV-qh211mMoaTK9O4q3bxTSk8VXI#id7)
- [Yin Lou, Rich Caruana, Johannes Gehrke, and Giles Hooker. Accurate intelligible models with pairwise interactions. In *Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining*, 623–631. 2013.](https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf)
- [Y. Lou, R. Caruana, and J. Gehrke. Intelligible models for classification and regression. In KDD, 2012.](https://www.cs.cornell.edu/~yinlou/papers/lou-kdd12.pdf)
- [InterpretML: A Unified Framework for Machine Learning Interpretability](https://arxiv.org/pdf/1909.09223.pdf)
- [The Science Behind InterpretML: Explainable Boosting Machine](https://www.youtube.com/watch?v=MREiHgHgl0k)
- [An introduction to Explainable AI (XAI) and Explainable Boosting Machines (EBM)](https://www.kdnuggets.com/2021/06/explainable-ai-xai-explainable-boosting-machines-ebm.html)

