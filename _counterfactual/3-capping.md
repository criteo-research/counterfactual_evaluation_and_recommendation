---
layout: post
title:  "Capping the importance sampling estimator"
order: 3
show: false
date: 2020-09-07
---

> This post explains why 'importance sampling' suffers from high variance, and the consequence of the usual capping method.

<!--more-->

## Summary of previous post

As a reminder from previous post, we have a counterfactual bandit dataset $(x_i, a_i, r_i)$ where:

* $x_i$ are iid samples of the state X.
* $a_i$ is a sample from $\pi_0$ on state $x_i$
* $r_i$ is a sample of the reward when the state is $x_i$ and the action $a_i$

We defined:
- the importance weight $$ W_i := w(A_i,X_i) := \frac{ \pi_{test}(X_i,A_i) }{ \pi_0(X_i,A_i) } $$
- and the IPS estimator $$ IPS := \frac{1}{n} \times  \sum_\limits{ i \in {1...n} } W_i \times  R_i $$
 
and proved it is an unbiased estimator of the number of clicks when applying policy $\pi_{test}$ instead of $\pi_0$

## Variance of the importance sampling estimator

- $IPS$ is an average on $n$ independent samples of $W \times R$ , where we noted $W := w(A,X)$; so $Var(IPS) = \frac{1}{n} \times Var( W \times R ) $ 
- while there is no simple formula for $Var( W \times R)$ , we may approximate it by assuming that $R$ and $w(A,X)$ are independent. (This is wrong, but should give us the correct magnitude)

We can then write:

\begin{aligned}
  Var(IPS)  &= \frac{1}{n} \mathbb{E}(W²R²) - \frac{1}{n} \mathbb{E}(WR)² \\\\  &\approx \frac{1}{n} \mathbb{E}(W² ) \mathbb{E}(R²) - \frac{1}{n} \mathbb{E}(W)² \mathbb{E}(R)² \\\\           &\approx \frac{1}{n} \mathbb{E}(W² ) \mathbb{E}(R)
\end{aligned}

(on the last step, we used  $R=R²$ because $R$ is binary, and dropped the second term which is typically much smaller)

Variance of $IPS$ is more or less proportional to $ \mathbb{E}(W²) $ . How big is this ?

### Variance of the importance weight

Let's note that the expectation of the importance weight is always $1$, so $ \mathbb{E}(W²) = Var(W) +1 $  
( here is the proof: 
$$ E(W) = \sum_\limits{a} \pi_0(a) w(a) = \sum_\limits{a} \pi_0(a) \frac{\pi_{test}(a)}{\pi_0(a)}  = \sum_\limits{a} \pi_{test}(a) = 1 $$ )


It's variance however depends on how different $\pi_0$ and $\pi_{test}$ are.

Let's look at what happen on a few examples:

![w examples]({{site.repo_name}}/assets/images/reco_problem/w_with_different_pi.png){:class="img-responsive"}

We can see on those examples that:
- when $\pi_0$ is close to $\pi_{test}$ , the ratio  $\frac{\pi_{test}(a)}{\pi_0(a)}$  is always close from 1, and variance is low.
- if $\pi_0$ and $\pi_{test}$ are very different, the weight is almost 0 with a large probability, but may take (with a low probability) some huge value. The variance is then driven by those outliers and is large.

The worst case happens when $\pi_{test}$ puts all the mass on the action less likely according to $\pi_0$.
The $w$ is then either $\frac{1}{ min_a(\pi_0(a)}$ , with probability $ min_a(\pi_0(a)$ , or 0, and the variance is   $\frac{1}{ min_a(\pi_0(a)}$ -1 $

In practice, it is difficult to avoid having any actions with very low propensity. Indeed:
 - assigning significant propensity to some actions which are known (or strongly suspected) to perform badly would degrade the whole system performances
 - when he action space is large, it is just not possible to assign a large probability to every action.


### Hidden variance

When the probability of an action is really low, the associated weigh can grow really large, but is almost never observed. 

For example, let's assume that an action $a_0$ has:
- probability $0.1$ on $\pi_{test}$ 
- probability $10^{-10}$ on $\pi_0$
- associated weight is then $w(a) = 10^9$
- the other actions have reasonable weights (lets say less than 10)
- and we collect $10^7$ samples of $A$ following $\pi_0$

What would we observe in such a case ?
- With a probability of about $0.01$, we would observe one sample with the action $a_0$ and the giant $10^9$ weight. In such case, we would conclude that the variance is crazily high (this single sample has more total weight than the 9999999 other ! ) and that the estimator is not usable.
- But with a high probability (around $0.99$), we won't observe the action $a_0$ at all. In such a case, the empirical variance of $w$ might look low, because we observed only some $w < 10$. It is important to realize that this is wrong ! Indeed, the estimated value we get from the $ips$ is in the case underestimating the number of clicks we would get with $\pi_{test}$, because it does not account for the value we would get from playing action $a_0$.

This "hidden" variance is actually behaving like a bias !

Let's also note that in the limit case, when the probability of an action following $\pi_0$ becomes exactly 0, the variance becomes low, but the estimator is now biased.

### Capped importance weight

When the variance is too large, the estimator is no longer useful in practice. Can we lower its variance ?

Since the variance is driven by some outliers, we can lower the variance by removing those outliers. For example removing all samples where $w$ is above a threshold, or replacing $w$ by some maximum value when the true value is higher.

 We can then defined the capped IPS estimator:

$$ capped IPS := \frac{1}{n} \times  \sum_\limits{ i \in {1...n} } \overline{W_i} \times  R_i $$

with:  $ \overline{W_i} $ defined either as:
 -  $ \overline{W_i} := min( W_i , c ) $     (capped weight)
 -  or $ \overline{W_i} :=  W_i \times \mathbb{1}_{ W_i < c  } $    (filtering out large weights)

and $c$ is the capping threshold.

By choosing the capping threshold $c$ low enough (typically somewhere between 10 and 1000, depending on how much variance you are ready to accept), it is possible to get a variance low enough to use this estimator.

... but of course, the capped estimator is no longer unbiased. Choice of the capping threshold $c$ is therefore a [bias-variance tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)
 Instead, since $\overline W \leq W $ (and $R \geq 0$ ), the capped IPS underestimate the outcome of the tested policy.
This underestimation is all the more important when:
 - capping threshold $c$ is smaller
 - $\pi_{test} is further from \pi_0$
 
To summarize:

|            | $\pi_{test}$ not far from $\pi_0$  | $\pi_{test}$ far from $\pi_0$ |
| IPS        |  unbiased & low variance | unbiased, <span style="color:red">High variance</span>
| capped IPS | slighlty biased, very low variance | <span style="color:red">Biased</span>, low variance

Ideally, we would like a low variance unbiased estimator for all policies. But is this possible ?


### No unbiased low variance estimator when $\pi_{test}$ is far from $\pi_0$

Unfortunately, the answer is No, unless we make some additional hypothesis.

Indeed, having  some  large  importance  weights  means  that  the  test  policy  takes  some  actions  which  were  very uncommon under the logging policy.  We just did not collect enough data on those actions to get any low variance estimate of what would happen when they are chosen.

In the next post, we will propose some possible additional hypothesis which seemed quite reasonable on our data at Criteo, and allowed to build some usable estimators for policies that are a bit further from $\pi_0$ (well, still not *too* far, there is just no magic for that)

