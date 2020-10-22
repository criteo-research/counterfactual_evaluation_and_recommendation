---
layout: post
title: "Beyond the capped estimator"
order: 4
show: false
date: 2020-09-07
---

> This post will describe some ideas to build a low variance estimator with an "acceptable" bias, and detail a variant of the estimator we found useful at Criteo.
Its content is closely related to what I published with my co-authors on [Offline  A/B  Testing  for  Recommender  Systems](https://arxiv.org/abs/1801.07030), but I tried here to put more focus on the underlying intuitions.
<!--more-->

In the previous post:
- we defined the capped importance sampling estimator.
- we explained that capping trades some variance for a bias.
- we concluded that when $\pi_{test}$ is too far from $\pi_0$, there is no way to build a low variance unbiased estimator, because we just did not collect enough samples of what happens after the actions favored by $\pi_{test}$.

To quickly remind our notations, we have a counterfactual bandit dataset $(x_i, a_i, r_i)$ where:
* $x_i$ are iid samples of the state X.
* action $a_i$ is a sample from $\pi_0$ on state $x_i$
* $r_i$ is a sample of the reward in the state $x_i$ after action $a_i$
* the importance weight is $w(a,x) := \frac{ \pi_{test}(a,x)}{ \pi_{0}(a,x)} $ 


# Two ideas to go further

We observed that we cannot get a low variance unbiased estimator for what happens when applying $\pi_{test}$, because we do not have enough data on some of the actions it selects. So what we do ? We would like to propose two viewpoints here:
 - We could make a hypothesis on the average reward of those unobserved actions.
 - Or we could replace $\pi_{test}$ by an approximate policy $\pi_{test'}$ which should be as similar to $\pi_{test}$ as possible, while staying close enough from $\pi_0$ to keep low importance weights.
 
Interestingly enough, both ideas will lead us to very similar estimators.

## An example

Before jumping into the equations, let's look at how those ideas could unfold in a simple example.

|              | Action $a_1$ | Action $a_2$ | Action $a_3$|
| $\pi_0$      | 0.8 | 0.2 | 0.00000001 |
|$\pi_{test}$  | 0.1 | 0.8 | 0.1 |
|$w$  | 0.125 | 4.0 | 10000000 |
|Average reward on observed data | 0.05 | 0.055 | Unobserved! |


The action $a_3$ clearly would make the variance too high.
Instead of capping we could:
- Make a hypothesis one the average reward of action $a_3$. Looking at the other actions, maybe a reasonable guess would be around $0.05$ ?
- Or we could define a policy $\pi_{test'}$ which looks like $\pi_{test}$  but avoids the action $a_3$, like this one:

|              | Action $a_1$ | Action $a_2$ | Action $a_3$|
|$\pi_{test'}$  | 0.11 | 0.89 | 0 |
| $w'$  | 0.1375 | 4.45 | 0 |

We can think of $\pi_{test'}$ as an approximation of $\pi_{test}$, and compute an (unbiased ! ) estimation of what would happen if we were using $\pi_{test'}$

We will now formalize a bit more what we did in those examples.


## Hypothesis on the unexplored actions

### Approximating unexplored actions by 0

As already explained, high variance of the $IPS$ is caused by actions with a large associated weight $w(a)$.


But let's assume for a moment that the reward $R(a)$ on those actions (let's say those with $w \geq 100$) is always 0. In this case, the $ips$ would become:

$$IPS = \frac{1}{n} \times  \sum_\limits{ i \in {1...n} } {W_i} \times 1_{ W_i < 100 }  \times  R_i  $$

What we recognize here is exactly the filtered $IPS$ estimator, where we dropped all the samples with $w_i > 100$.

_Using the filtered estimator is therefore equivalent to approximating the reward of the filtered actions by 0_.

### Using some other approximation

Now, let's assume instead that we would like to approximate average reward of the "filtered" actions (still those with $w>100$) not by 0 but by some function $f(x,a)$ instead.

We can decompose expected reward under $\pi_{test}$ as:
- the expected reward from playing non filtered actions. This is the part estimated by filtered $IPS$.
- plus the expected reward from playing filtered actions.

This second part can be written explicitly as :

 $$\sum_\limits{a  } \pi_{test}(a,x) \times 1_{ W_i > 100 } \times  \mathbb{E}(R | A=a, X=x) $$


The trick here is to notice that, since our hypothesis gives us the expected reward of those actions, we do not need importance weighting to estimate this second. According to our hypothesis, it should be equal to:

 $$\sum_\limits{a | w(a,x) > 100   } \pi_{test}(a,x) f(a,x) $$

We can then correct the filtered $IPS$ by adding this additional term:

$$\frac{1}{n} \times  \sum_\limits{ i \in {1...n} } \left( w_i \times 1_{ w_i < 100 }  \times  r_i  + \sum_\limits{a | w(a,x_i) > 100   } \pi_{test}(a,x_i) f(a,x_i) \right) $$

Let's see now how we could choose the function 'f'

### Approximating reward on filtered actions with a model

The most obvious solution here is to build some model $f(a,x)$ predicting the reward on context $x$ after action $a$.
The formula above then becomes (well, almost) what is known as the "doubly robust" estimator.

( To be more precise, the "doubly robust" estimator would be usually defined as follow: 

$$\frac{1}{n} \times  \sum_\limits{ i \in {1...n} } \left( w_i \times 1_{ w_i < 100 }  \times  (r_i - f(a_i,x_i) + \sum_\limits{a } \pi_{test}(a,x_i) f(a,x_i) \right) $$

It may be checked that this formula as the same expectation as the formula above, and that both usually have similar variances. )

While the doubly robust may be a very good choice of estimator, it requires to have a model of the reward, which may make it a bit more complicate to compute. On top of that, we want here to evaluate one policy $\pi_{test}(a,x)$ which may typically rely on another model itself. So we are using a model of the reward to evaluate another model of the reward, which is not really satisfying. 
For those reasons, we would like to propose different hypotheses on the reward after a filtered action.

### Approximating reward on filtered action by the average reward on other actions

In a recommender system, the set of actions proposed in a context $x$ typically comes from several heuristics (collaborative filtering, bestofs,...) which are tuned to provide at least 'reasonable' actions.
As a result, the expected reward $\mathbb{E}(R |X=x,A=a)$ in a context $x$ after action $a$ does not vary that much with $a$. 
On the other hand, it may vary by several orders of magnitude with context $x$. So we might write, for the less explored actions: $\mathbb{E}(R\| X=x,A=a)) \approx \mathbb{E}(R\|X=x)  $.

Since the reward $R$ is not exactly the same for each action, we need to be a bit more specific here: To define $\mathbb{E}(R\| X=x)$, we need to specify which policy is used to collect the actions.
The easiest here is to choose the policy $\pi_0$:
- Approximate $\mathbb{E}(R \| X=x, A= a_{filtered}))$ by the expected reward using $\pi_0$

### Approximating reward on filtered action by the average reward from $\pi_0$

We therefore propose here to use:

$$ f(x,a) := \mathbb{E}_{\pi_0}( R | X = x) $$

as an approximation of $ \mathbb{E}(R\|X=x ,A=a) $ when $a$ is a filtered action.

Following the previous section, our estimator should be:

$$\frac{1}{n} \times  \sum_\limits{ i \in {1...n} } \left( w_i \times 1_{ w_i < 100 }  \times  r_i  + \sum_\limits{a | w(a,x_i) > 100   } \pi_{test}(a,x_i) \times \mathbb{E}_{\pi_0}( R | X = x_i) \right) $$

Here we note that $\mathbb{E}_{\pi_0}( R \| X = x_i) )$ is not known, but we have a sample of it: this sample is $r_i$!

Estimator becomes:
 
$$\frac{1}{n} \times  \sum_\limits{ i \in {1...n} } ( w_i \times 1_{ w_i < 100 } + \mathbb{P}_{ a \sim \pi_{test} } ( w(a,x_i) > 100  ) )   \times  r_i  $$

where we noted :

$$ \mathbb{P}_{ a \sim \pi_{test} } ( w(a,x_i) > 100  ) :=  \sum_\limits{a | w(a,x) > 100 } \pi_{test}(a,x_i)  $$

This term may also be problematic to compute explicitly, because the sum may be on a very large number of actions.
But once again, we can sample to estimate it. To do that, we just need to draw one action $a_i'$ from the test policy, and check if this action is filtered.

The final estimator is thus computed as follow:
- On each line of log, draw $a_i' \sim \pi_{test}(x_i)$
- Compute: 

$$\frac{1}{n} \times  \sum_\limits{ i \in {1...n} } ( w_i \times 1_{ w_i < 100 }  + 1_{ w(a_i',x_i) > 100 } )  \times  r_i   $$


#### An Additive correction
The weights of this estimator are thus:

$$  w(a_i,x_i) \times 1_{w(a_i,x_i) < 100} +  + \mathbb{P}_{ a \sim \pi_{test} } ( w(a,x_i) > 100  )$$

It is straightforward to check that those weight have an expectation of $1$.
One way to think to this estimator is that we added  constant to the filtered weights to get back the property $E(W=1$ that we lost with capping.

#### An estimator of a modified policy

It can be verified that this is an unbiased estimator or the following policy:

 - sample one action $a$ using $\pi_{test}$
 - compute the importance weight $w(a) := \frac{\pi_{test}(a)}{\pi_p(a)} $
 - if $w(a) \leq c $ return $a$
 - else return a sample from $\pi_0$
 
"Draw an action from $\pi_{test}$, if it is filtered reject it and replace it with one sample of $\pi_0$ instead".

### Using the few datapoints we have for less explored actions 

In the previous section, we did not use at all the observed reward $r_i$ on the samples where the action $a_i$ is "filtered", that is $w_i > 100$.
We have seen in last post a usually better strategy is capping large importance weights, instead of filtering them.
But how can we use an hypothesis on the reward on "less explored action" when using the capped estimator ?

The solution here is to realize that capping is equivalent (at least in expectation) to _randomly_ filtering large weights.

More precisely, the capped weight $ min(w_i,100) $ is the expectation of the random variable constructed as follow:
- Sample $U$, an uniform random variable in interval $[0;1]$
- if $ w_i *U > 100 $ , return 0  (_the weight is filted_)
- else return $w_i$

The probability that an action $a$ is filtered with this definition is then:

$$ \mathbb{P} \left( w(a,x) \times U > 100 \right) = 1 - \frac{ min(w(x,a),100) }{ w(x,a) }  $$
 
We can now rewrite our corrected estimator, by using the approximation $f(x,a)$ when the action is filtered:

$$\frac{1}{n} \times  \sum_\limits{ i \in {1...n} } \left( min(w_i,100)  \times  r_i  + \sum_\limits{a  } \pi_{test}(a,x_i) \times (1 - \frac{ min(w(x_i,a),100) }{ w(x_i,a) }) \times f(a,x_i) \right) $$

In the case when we choose, as in previous paragraph, $f(x,a) := \mathbb{E}(R\|X=x)$, we finally get the following program: 
- On each line of log, draw $a_i' \sim \pi_{test}(x_i)$
- Compute: 

$$\frac{1}{n} \times  \sum_\limits{ i \in {1...n} } ( min( w_i , 100)  + 1 - \frac{ min(w(x_i,a_i'),100) }{ w(x_i,a_i') }  )  \times  r_i   $$


## Estimator of the reward under an approximation of test policy

Since we cannot precisely estimate the expected reward under $\pi_{test}$, we propose to estimate instead the expected reward under a policy similar to $\pi_{test}$ but close enough from $\pi_0$ to allow an accurate estimation.
This can be done as follow:

 - Define a set $\mathcal{B}$ of policies which can be accurately estimated from the data.
 - Define $\pi_{test}'$ as the "best approximation" in $\mathcal{B}$ of the test policy $\pi_{test}$.
 - Compute the expected reward under $\pi_{test}'$

To guaranty that we have a good estimator for $\pi_{test}'$, it seems natural to ask that it verifies:

$$ \forall x,a \quad  \frac{  \pi_{test}'(x,a) } { \pi_0(x,a)} \leq c $$

In other words, we would like to choose the policy $\pi_{test}'$ so that it does not require capping. Let $\mathcal{B}( \pi_0 , c )$ the set of policies verifying this condition. We can think of $B( \pi_0 , c )$ as a ball around $\pi_0$ in the space of policies.

Now, we want to find $\pi_{test}'$ "as close as possible" from $\pi_{test}$ among acceptable policies (ie in the set $\mathcal{B}( \pi_p , c )$). Measuring the "distance" between policies with the KL divergence, we can now define:

$$ \pi_{test}' := Argmin_{ \pi \in \mathcal{B}( \pi_0 , c ) } KL(\pi_{test} || \pi )  $$

For a fixed $x$, this write:

$$ \pi_{test}' = Argmin_{\pi}  ( \sum\limits_{a} \pi_{test}(a) log( \pi(a) )   ) \quad under \quad \sum\limits_{a} \pi(a) = 1 \quad and \quad \forall a \quad \pi(a) \leq c \times \pi_p(a) $$

Using the Lagrange multiplier, it is straightforward to check that this solution verifies:

$$ \exists \alpha \quad \forall a \quad either \quad \pi_{test}'(a) = c \times \pi_p(a) \quad or \quad \pi_{test}'(a) = \alpha \times \pi_{test}(a)  $$


Now let's define $c' := \frac{c}{ \alpha} $

 $$ \pi_{test}'(a) =  \alpha \times min( c' \pi_0(a) , \pi_{test}(a) ) $$

and thus:

 $$ \frac{\pi_{test}'(a)}{ \pi_0(a) } =  \alpha \times min( c' , w(a) ) $$

The importance weights for $\pi_{test}'$ can be found by capping at $c'$ and normalizing  by multiplying by $\alpha$ to retrieve an importance weight of expectation $1$: here we applied a multiplicative correction to the importance weight.

#### Describing the policy $\pi_{test}'$

A sample of the approximate policy $\pi_{test}'$ can be obtain as follow:
  - sample one action $a$ using $\pi_{test}$
  - compute the importance weight $w(a) := \frac{\pi_{test}(a)}{\pi_p(a)} $
  - draw a random number $u$ uniform in $[0;1]$ 
  - if $w(a) \times u  \leq c $ return $a$
  - else reject the sample and repeat all those steps

Note that if $w(a) \leq c $ the sample is always accepted. This algorithm could be roughly described as "Sample from $\pi_{test}$ until you find an action not capped under $\pi_0$".

Actually, it may be checked that this is equivalent to approximate (with the methods from previous section) the expected reward on "filtered actions" by the expected reward from non filtered actions from $\pi_{test}$; or more formally:

$$ \text{Approximate } \mathbb{E} _{ A \sim \pi_{test} } (R |  X =x , W \times U > 100 )  \text{  by:  }  \mathbb{E} _{ A \sim \pi_{test} } (R |  X =x , W \times U < 100 ) $$



This last estimator is what we described in [Offline  A/B  Testing  for  Recommender  Systems](https://arxiv.org/abs/1801.07030).
We realized after writing the paper that the "additive correction" presented above gave almost exactly the same results on our dataset, but with slightly less variance (The decreased variance was mostly explained by the imperfect computation of the capping threshold $c'$)
It is also much easier to implement. For those reasons, we switched to using this "additive correction" instead of the "multiplicative correction" presented in the paper.
