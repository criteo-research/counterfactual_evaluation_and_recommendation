---
layout: post
title:  "Capping the importance sampling estimator"
order: 3
show: false
date: 2020-09-07
---

> This post explains why "importance sampling" suffers from high variance, and the consequences of the usual capping method.

<!--more-->

## Summary of previous post

As a reminder from previous post, we have a counterfactual bandit dataset $(x_i, a_i, r_i)$ where:

* $x_i$ are iid samples of the state X.
* action $a_i$ is a sample from $\pi_0$ on state $x_i$
* $r_i$ is a sample of the reward in the state $x_i$ after action $a_i$

We defined:
- the importance weight $ W_i := w(A_i,X_i) := \frac{ \pi_{test}(X_i,A_i) }{ \pi_0(X_i,A_i) } $
- and the $IPS$ estimator $ IPS := \frac{1}{n} \times  \sum_\limits{ i \in {1...n} } W_i \times  R_i $
 
and we proved it is an unbiased estimator of the number of clicks when applying policy $\pi_{test}$ instead of $\pi_0$.

## Variance of the importance sampling estimator

To study the variance of $IPS$, we will look at the expectation and variance of $W$. Let us start with the expectation of $W$:

### A useful lemma

Expectation of $W$ is easy to compute:

 $$ E(W) = 1 $$

The proof is basically the same as the proof that <script type="math/tex">  \mathbb{E}_{\pi_{0}}(R \times W) = \mathbb{E}_{\pi_{test}}(R)  </script>.
Here it is:

$$ 
 \begin{aligned}
  \mathbb{E}(W)  &= \sum_\limits{a} \pi_0(a) w(a) \\\\ &= \sum_\limits{a} \pi_0(a) \frac{\pi_{test}(a)}{\pi_0(a)} \\\\ &= \sum_\limits{a} \pi_{test}(a) \\\\ &= 1 
\end{aligned}
$$

Or we could also redefine $R$ as 1 and apply the previous result, which would then writes in this case:  <script type="math/tex">  \mathbb{E}_{\pi_{0}}(1 \times W) = \mathbb{E}_{\pi_{test}}(1) </script> , and the right term is obviously $1$.

### Variance of $IPS$

- $IPS$ is an average on $n$ independent samples of $W \times R$ , where we noted $W := w(A,X)$; so $Var(IPS) = \frac{1}{n} \times Var( W \times R ) $ 
- While there is no simple formula for $Var( W \times R)$ , we may approximate it by assuming that $R$ and $W$ are independent. (This is wrong, but should give us the correct magnitude.)

We can then write:

$$ 
 \begin{aligned}
  Var(IPS) &= \frac{1}{n} \times  ( \mathbb{E}(W²R²) - \mathbb{E}(WR)² ) &\\\\
           &\approx \frac{1}{n} \times  ( \mathbb{E}(W²)\mathbb{E}(R²) - \mathbb{E}(W)²\mathbb{E}(R)² ) & \text {assuming that } W \text{ and } R \text { are independent} \\\\
           &\approx \frac{1}{n} \times  ( \mathbb{E}(W²)\mathbb{E}(R) - \mathbb{E}(R)² ) \qquad& \text{noting that R is binary, so }R²=R \text{; and } \mathbb{E}(W)=1  \\\\
           &\approx \frac{1}{n} \times \mathbb{E}(W²)\mathbb{E}(R)		  \qquad & \text{ because } \mathbb{E}(W²) >> \mathbb{E}(R)  
\end{aligned}
$$


Variance of $IPS$ is more or less proportional to $\mathbb{E}(W²)$.

### Variance of the importance weight

So we should study $\mathbb{E}(W²) = Var(W)+1$.
Let's look at this term on a few examples:

![w examples]({{site.repo_name}}/assets/images/reco_problem/w_with_different_pi.png){:class="img-responsive"}

We can see on those examples that:
- when $\pi_0$ is close to $\pi_{test}$ , the ratio  $\frac{\pi_{test}(a)}{\pi_0(a)}$  is always close from 1, and variance is low.
- if $\pi_0$ and $\pi_{test}$ are very different, the weight is almost 0 on most samples collected from $\pi_0$, but may take (with a low probability) some huge value. The variance is then driven by those outliers and is large.

The worst case happens when $\pi_{test}$ puts all the mass on the action less likely according to $\pi_0$.
The weight $w$ is then either $\frac{1}{ min_a(\pi_0(a))}$ , with probability $ min_a(\pi_0(a))$ , or 0, and the variance is then $\frac{1}{ min_a(\pi_0(a))} -1 $.

To summarize: <b> $IPS$ variance is high when the test policy $\pi_{test}$ assigns a significant probability to actions very unlikely under $\pi_0$. </b>

The reason is intuitively clear: such actions are not explored much by $\pi_0$, and therefore what happens when they are chosen is not well known.


### Hidden variance

There is one common pitfall when estimating the variance of $IPS$ estimator:
sometimes the *empirical* variance of $IPS$, on some sample, may look small, while the true variance is not.

Let us indeed look at what may happen on an example where an action has a probability very close to 0 under $\pi_0$.

| Policy | Action $a_1$ | Action $a_2$ |Action $a_3$ |
| $\pi_0$ | $10^{-10}$ | $0.5-10^{-10}$ | 0.5 |
| $\pi_{test}$ | 0.1 | 0.4 | 0.5 |
| w | $10^{9}$ | $ \approx 0.8$ | $1.0$ |

In this example,
$E(W²) = 10^{-10} \times (10^{9})² + 0.4999... \times 0.8² + 0.5 \times 1² \approx 10^{8}  $.

The key point here is that the variance is driven by some event (here observing action $a_1$) which almost never happens. 


Let us now assume we draw a fairly large sample, say of size $10^7$, of those data, following $\pi_0$.
Usually (well, with probability $\approx 0.999$), this sample will not contain a single line with action $a_1$, and thus not a single large weight. 
On this "typical" sampled dataset, 
- the *empirical* variance would be low. It is important to note that it happens because the sample size is here too small for the *empirical* variance to estimate correctly the *true* variance. 
- the *empirical average* of $w$ would be around $0.9$, whereas $E(W)=1$:
this "hidden" variance is behaving like a bias!

Let's also note that in the limit case, when the probability of an action following $\pi_0$ becomes exactly 0, the variance becomes low, but the estimator is now biased.

### Capped importance weight

When the variance is too large, the estimator is no longer useful in practice. Can we lower its variance?

Since the variance is driven by some outliers, we can lower the variance by removing those outliers. For example, removing all samples where $w$ is above a threshold, or replacing $w$ by some maximum value when the true value is higher.

We can then define the capped $IPS$ estimator:

$$ capped IPS := \frac{1}{n} \times  \sum_\limits{ i \in {1...n} } \overline{W_i} \times  R_i $$

where the _capped_ weight $ \overline{W_i} $ may be defined either as:
 -  $ \overline{W_i} := min( W_i , c ) $     (capped weight)
 -  or $ \overline{W_i} :=  W_i \times \mathbb{1}_{ W_i < c  } $  (filtering out large weights)

and $c$ is the capping threshold.

By choosing the capping threshold $c$ low enough (typically somewhere between 10 and 1000, depending on how much variance you are ready to accept), it is possible to get a variance low enough to use this estimator.

... but of course, the capped estimator is no longer unbiased. Choice of the capping threshold $c$ is therefore a [bias-variance tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff).
 Instead, since $\overline W \leq W $ (and $R \geq 0$ ), the capped $IPS$ underestimate the outcome of the tested policy.
This underestimation is all the more important when:
 - capping threshold $c$ is smaller
 - $\pi_{test}$ is further from $\pi_0$
 
To summarize:

|            | $\pi_{test}$ not far from $\pi_0$  | $\pi_{test}$ far from $\pi_0$ |
| $IPS$        |  unbiased & low variance | unbiased, <span style="color:red">high variance</span>
| capped $IPS$ | slightly biased, very low variance | <span style="color:red">biased</span>, low variance

#### Capping or filtering ?

In theory, capping is a slightly better bias-variance tradeoff, and should therefore be chosen.
In practice, it made little difference on the dataset I observed. 
"Filtering" is also easier to reason with, and in next post we will use it to get some intuitions on some methods to mitigate the bias. (We will also explain how to generalize those reasonings to "capping")

### No unbiased low variance estimator when $\pi_{test}$ is far from $\pi_0$

Ideally, we would like a low variance unbiased estimator for all policies. But is this possible?
The answer is unfortunately No, unless we make some additional hypotheses.

Indeed, having some large importance weights means that the test policy takes some actions which were very uncommon under the logging policy.  We just did not collect enough data on those actions to get any low variance estimate of what would happen when they are chosen.

The only way to ensure that the reward under any policy $\pi_{test}$ may be estimated with a low variance would be to ensure that all actions have a high enough probability under $\pi_0$. But this is certainly not realistic:
 - assigning significant propensity to some actions which are known (or strongly suspected) to perform badly would degrade the whole system performances.
 - when the action space is large, it is just not possible to assign a large probability to every action.


In the next post, we will propose some additional hypotheses which seemed quite reasonable on our data at Criteo, and allowed to build some usable estimators for policies that are a bit further from $\pi_0$. (Well, not *too* far either, there is just no magic for that! )

