---
layout: post
title: "Beyond the capped estimator"
order: 4
show: false
date: 2020-09-07
---

> In this post, we will propose some idea to build a low variance estimator with an "acceptable" bias,
and introduce the "pointwise capped normalized" estimator defined in [Offline  A/B  Testing  for  Recommender  Systems](https://arxiv.org/abs/1801.07030) and some possible variants.

<!--more-->

In the previous post:
- we defined the capped importance sampling estimator
- we explained that capping trades some variance for a bias
- we concluded that when $\pi_{test}$ is too far from $\pi_0$, there is no way to build a low variance unbiased estimator, because we just did not collect enough samples of what happens after the actions favored by $\pi_{test}$.


# Two ideas to go further

We observed that we cannot get a low variance unbiased estimator for what happens when applying $\pi_{test}$, because we do not have enough data on some of the actions it selects. So what we do ? We would like to propose two view points here:
 - we can make an hypothesis on the average reward of those unobserved actions
 - or we can replace $\pi_{test}$ by an approximate policy $\pi_{test'}$ which should be as similar to $\pi_{test}$ as possible while staying close enough from $\pi_0$ to keep low importance weights.
 
Interestingly enough, both ideas may lead to the same estimator.

## An example

Before jumping into the equations, let's first look at a simple example to show how we can apply those ideas.

|              | Action $a_1$ | Action $a_2$ | Action $a_3$|
| $\pi_0$      | 0.8 | 0.2 | 0.00000001 |
|$\pi_{test}$  | 0.1 | 0.8 | 0.1 |
|$w$  | 0.125 | 4.0 | 10000000 |
|Average reward on observed data | 0.05 | 0.055 | Unobserved ! |


The action $a_3$ clearly would make the variance too high.
Instead of capping we could:
- make an hypothesis one the average reward of action $a_3$. Looking at the other actions, maybe a reasonable guess would be around $0.05$ ?
- or we could define a policy $\pi_{test'}$ which looks like $\pi_{test}$  but avoids the action $a_3$

|              | Action $a_1$ | Action $a_2$ | Action $a_3$|
|$\pi_{test'}$  | 0.11 | 0.89 | 0 |

We can think of $\pi_{test'}$ as an approximation of $\pi_{test}$, and compute an (unbiased ! ) estimation of what would happen if we were using $\pi_{test'}$

We will now formalize a bit more what we did in those examples.


## Hypothesis on the unexplored actions

### Approximating unexplored actions by 0

As already explained, high variance of the $IPS$ is caused by actions with a large associated weight $w(a)$

But let's assume for a moment that the reward $R(a)$ on those actions (let's say those with $w \geq 100$) is always 0. In this case, the ips would becomes:

$$IPS = \frac{1}{n} \times  \sum_\limits{ i \in {1...n} } {W_i} \times 1_{ W_i < 100 }  \times  R_i  +  $$

What we recognize here is exactly the capped $IPS$ estimator, where we dropped all the samples with $w_i > 100$.

_Using the capped estimator is therefore equivalent to approximating the reward of the capped actions by 0._

### Using some other approximation

Now, let's assume instead that we would like to approximate average reward of the same actions (still those with $w>100$) not by 0 but by some function $f(x,a)$ instead.

We can decompose expected reward under $\pi_{test}$ as:
- expected reward from playing non capped actions. This is the part estimated by capped $IPS$
- plus expected reward from playing capped actions, $\sum_\limits{a \| w(a,x) > 100   } \pi_{test}(a,x) f(a,x) )$

We can then correct the capped $IPS$ by adding this additional term:

$$\frac{1}{n} \times  \sum_\limits{ i \in {1...n} } ( W_i \times 1_{ W_i < 100 }  \times  R_i  + \sum_\limits{a \| w(a,x) > 100   } \pi_{test}(a,x_i) f(a,x_i) ) $$

Let's see now some possible hypothesis we could use.

### Approximating reward on unexplored action by the average reward on other actions

In a recommender system, the set of actions proposed in a context $x$ typically comes from several heuristics (collaborative filtering, bestofs,...) which are tuned to provide at least 'reasonable' actions.
As a result, the expected reward $\mathbb{E}(R |X=x,A=a)$ in a context $x$ after action $a$ does not vary that much with $a$. 
On the other hand it may vary by several orders of magnitude with context $x$. So we might write, for the less explored actions: $\mathbb{E}(R(x,a)) \approx \mathbb{E}(R(x))  $. \\

Since the reward $R$ is not exactly the same for each action, we need to be a bit more specific here: To define $\mathbb{E}(R(x))$, we need to specify which policy is used to collect the actions. So we propose two variations:

- Approximate $\mathbb{E}(R(x,a_{unexplored}))$ by the expected reward using $\pi_0$
- Approximate $\mathbb{E}(R(x,a_{unexplored}))$ by the expected reward on the explored actions chosen by $\pi_{test}$

We will detail the first option in section 3. Here we focus on the second option, and explain how it lead to the "pointwise capped normalized estimator".\\
But let's first formalize it:

We would need to approximate the expected reward of an action when it is not explored enough by $\pi_0$, but happens more often with $\pi_{test}$. Those are exactly the actions where the importance weight $w$ is higher than the chosen threshold $c$. \\
So we propose to approximate, for any context $x$:
  $\mathbb{E}\_{ \pi_{test}} ( R(A,x) \| W > c ) $ by $\mathbb{E}\_{\pi_{test} }( R(A,x) \| W \leq c ) $


Let's see how we can use this assumption to build our estimator:
\begin{aligned}
\mathbb{E}\_{ \pi_{test} }( R(A,x) ) &= \mathbb{E}\_{ \pi_{test} }( R(A,x) \| W(A,x) > c ) P\_{ \pi_{test} }(  W(A,x) > c ) + \mathbb{E}\_{ \pi_{test} }( R(A,x) \| W(A,x) \leq c )P\_{ \pi_{test} }( W(A,x) \leq c )  \\\\   &\approx \mathbb{E}\_{ \pi_{test} }( R(A,x) \| W(A,x) \leq c ) \\\\  &\approx \mathbb{E}\_{ \pi_{test} }( R(A,x) \times( \mathbf{1}\_{ W(A,x) \leq c} ))  \frac{1}{ P\_{ \pi_{test} }(  W(A,x) \leq c )} 
\end{aligned}



The factor $\frac{1}{ P_{ A \sim \pi_{test} }(  W > c )}$ can be either computed explicitly, or estimated with a Monte Carlo method if the action space is too large (With some unbiased estimator of the inverse, such as  [this](https://en.wikipedia.org/wiki/Ratio_estimator#Midzuno-Sen's_method). ) See also [Offline  A/B  Testing  for  Recommender  Systems](https://arxiv.org/abs/1801.07030) for more details) 

The estimator we get is then:

 $$ \frac {1}{n} \sum_\limits{i} w_i \times r_i \times \mathbf{1}_{w_i \leq c} \times \frac{1}{ P_{ A \sim \pi_{test} }(  W > c | X=x_i )}$$

 
### Using the few datapoints we have for less explored actions 

In the previous section, we considered that actions are either 'explored' or 'unexplored'. Obviously, reality is not binary, and even if we do not have collected enough datapoints to estimate accurately the expected reward of the actions where $w > c$, we may still have a few. How do we use those ? \\
This question is actually closely linked to the choice of how to deal with high weights in the capped estimator, where there are typically two options:
 - Discard the lines with high weights, effectively setting the weight to 0, and not using those data at all.
 - Or cap the importance weight, replacing $w$ by $min(x,c)$. This method allows to use reward observed on datapoints where $w>c$, it just does not rely on it too much.  

Can we do something similar when applying our approximation ? The answer is yes. To do that, we need a finer definition of the event 'the action is not explored enough'. We propose the following definition:

Let $U$ an uniform random variable in interval $[0;1]$, independent from anything else. We will say that the action $A$ is 'unexplored' if $w(A) > c \times U $. \\
This definition is motivated by the following equality:

$$ \mathbb{E}( r_i \times min(w_i,c) ) = \mathbb{E}_{U, A \sim \pi_{test} }( R(A,x) \times( \mathbf{1}_{ W(A,x) > c \times U} ))$$ 
In other words, the capped estimator is (in expectation) what we get when we remove the 'unexplored' actions according to this precise definition.

We can now rewrite our hypothesis: 

We will approximate $$\mathbb{E}_{ \pi_{test} }( R(A,x) | W > c \times U )$$ by $\mathbb{E}\_{ \pi_{test} }( R(A,x) \| W \leq c \times U )$. \\
The computation unfold in the same way as above:
$$\mathbb{E}_{ \pi_{test} }( R(A,x) ) 
\approx
\mathbb{E}_{ \pi_{test} }( R(A,x) \times( \mathbf{1}_{ W > c \times U} ))  \frac{1}{ P_{ A \sim \pi_{test} }(  W > c \times U | X=x )} $$

We can notice here that $$ P_{ A \sim \pi_{test} }(  W > c \times U | X= x ) = \mathbb{E}_{\pi_{test}}( \frac{ min(W,c) }{W} | X=x ) = \mathbb{E}_{\pi_{0}} ( min(W,c) | X=x )  $$
and  this quantity can also be either computed explicitly or estimated by Monte Carlo.
The estimator we get is thus:

 $$ \frac {1}{n} \sum_\limits{i} min( w_i , c ) \times \frac{1}{  \mathbb{E}_{\pi_{0}} (min(W_i,c)| X=x_i ) }$$

This is exactly the "pointwise normalised capped estimator" proposed in section 5.4 of [Offline  A/B  Testing  for  Recommender  Systems](https://arxiv.org/abs/1801.07030).

Note that there is actually still a possible problem in practice: $$\mathbb{E}_{\pi_{0}} (min(W_i,c)| X=x_i )$$ may sometimes become very low, leading to a large variance. This can be solved by lowering the capping threshold $c$.


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

For a fixed $x$, this writes:
$$ \pi_{test}' = Argmin_{\pi}  ( \sum\limits_{a} \pi_{test}(a) log( \pi(a) )   ) \quad under \quad \sum\limits_{a} \pi(a) = 1 \quad and \quad \forall a \quad \pi(a) \leq c \times \pi_p(a) $$

Using the Lagrange multiplier, it is straightforward to check that this solution verifies:
$$ \exists \alpha \quad \forall a \quad either \quad \pi_{test}'(a) = c \times \pi_p(a) \quad or \quad \pi_{test}'(a) = \alpha \times \pi_{test}(a)  $$


Now let's define $c' := \frac{c}{ \alpha} $
 $$ \pi_{test}'(a) =  \alpha \times min( c' \pi_0(a) , \pi_{test}(a) ) $$. So the importance weights for $\pi_{test}'$ can be found by capping at $c'$ and normalizing by multiplying by $\alpha$. It is left as an exercise to check that the estimator we build in the previous section, when we lower the capping value to $c'$, is exactly the unbiased $IPS$ estimator for the reward of $\pi_{test'}$ 
 

#### Describing the policy $\pi_{test}'$
We explained that the "pointwise capped normalised" estimator is actually the unbiased importance weighting estimator for a policy $\pi_{test'}$. Let's describe explicitly this policy.


A sample of the approximate policy $\pi_{test}'$ can be obtain as follow:
  - sample one action $a$ using $\pi_{test}$
  - compute the importance weight $w(a) := \frac{\pi_{test}(a)}{\pi_p(a)} $
  - draw a random number $u$ uniform in $[0;1]$ 
  - if $w(a) \times u  \leq c $ return $a$
  - else reject the sample and repeat all those steps

Note that if $w(a) \leq c $ the sample is always accepted. This algorithm could be roughly described as "Sample from $\pi_{test}$ until you find an action not capped under $\pi_0$"

## Additive correction

The "pointwise capped normalised" estimator corrects the capped importance weight by a constant factor to ensure that the expectation of the importance weight is 1. What if instead of a multiplicative factor, we used an additive correction ?
We can therefore defined the "additive corrected capped normalised" estimator as follow:

$$ \sum\limits_i r_i \times ( min( c, w(a_i,xi) ) + 1 - \mathbb{E}_{a \sim \pi_p}  min( c, w(a,xi) )  ) $$

### Underlying hypothesis on expected reward of unexplored actions 
It can be checked that this estimator is what we get when we approximate the expected reward of 'unexplored' actions (with the definition allowing to use available data on all actions) by the expected reward when following $\pi_0$


### An estimator of a modified policy

It can be verified that this is an unbiased estimator or the following policy:

 - sample one action $a$ using $\pi_{test}$
 - compute the importance weight $w(a) := \frac{\pi_{test}(a)}{\pi_p(a)} $
 - draw a random number $u$ uniform in $[0;1]$ 
 - if $w(a) \times u  \leq c $ return $a$
 - else return a sample from $\pi_0$

'Draw an action from $\pi_{test}$', if it is capped reject it and replace it with one sample of $\pi_0$ instead.

### Implementation

We could compute the additive correction $$1 - \mathbb{E}_{a \sim \pi_0}  min( c, w(a,xi) ))$$ explicitly by iterating on all actions, but this may be way too costly when the number of actions is large.

Instead, we propose to compute it by Monte Carlo. This can be done with a very low variance by taking advantage of the following formula:

$$ \mathbb{E}_{ \pi_0}  min( c, w(A,xi) )) = \sum_\limits{a} \pi_0(a) \times min( c, w(a,xi) )) = \sum_\limits{a} \pi_{test}(a)  \times \frac{ min( c, w(a,xi) ))}{w(a,x_i)}  = \mathbb{E}_{ \pi_{test}}  \frac{ min( c, w(A,xi) ))}{w(A,x_i)}  $$

We therefore compute the estimate as follows:
 - For each line $i$ of the dataset, sample $$a'_i$$ from the policy $\pi_{test}$
 - Compute: $$ \sum\limits_i r_i \times ( min( c, w(a_i,xi) ) + 1 - \frac{min( c, w(a'_i,xi) )}{ w(a'_i,xi) }) $$

Since the term we added to the capped importance weight is in $[0;1]$, the variance of this estimator is very close to the variance of the capped estimator.

### Experimental results

We implemented this estimator and compared its results with the previously proposed "pointwise capped normalised".
We observed that:
- It was giving almost the same results (Correlation was about 0.99)
- But its variance was a bit lower. This was because in "pointwise capped normalised", when the normalisation in too high, it may become difficult to find the correct capping value without drawing a lot of samples.
- It requires one single sample of $\pi_{test}$, whereas "pointwise capped normalised" may need a lot of samples to be accurate.
- Implementation is also significantly easier.

The fact that the results are closely correlated is not really a surprise:
The policy $\pi_{test}$ we test are usually not that far from $\pi_0$, which means that:

- Probability of the "unexplored" actions is typically a few %
- Difference in expected reward between both policies is typically no more than 1 or 2%.

Overall, this mean that the difference in the hypothesis we used just change the result by a few %  on a few % of the dataset: overall, it is a second order effect which is usually far below the noise level.
For all those reason, we ended using mostly this version of the estimator.
