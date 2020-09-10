---
layout: default
title: Contact
---

I am Alexandre Gilotte, from the research team at Criteo, where I spent some time working on different metrics to evaluate and improve our recommender system. 
One family of metrics of interest in this context are the so called "importance weighting" metrics, which enable to estimate what would have happened if we had run a modified version of our recommendation system.

While the basic ideas of those metrics are quite simple, I could not find an easy-to-read introduction to this topic, presenting both the mathematics and the intuition. This is why I started writing this blog. 
I will also use this as an opportunity to describe how we adapted at Criteo the existing methods to deal with the bias/variance tradeoff which quickly appears when using "importance weighting" on real data. What we use indeed is similar, but simpler, to the estimator proposed in [an article we published a few years ago at WSDM](https://arxiv.org/abs/1801.07030), and this variant is I believe not published anywhere.

You can contact me here: a.gilotte _at_ criteo.com

Thanks also to David Rohde who helped a lot by discussing and reviewing the present blog.
