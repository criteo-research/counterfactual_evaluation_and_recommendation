---
layout: default
title: Importance weighting estimators
---

This blog will contain a series of posts describing the use of the _importance sampling estimator_ in the context of a recommender system.
The first posts introduce the topic, and should be readable with only basic knowledge of probabilities theory. After that, I would like to explain some more advanced details on bias/variance tradeoff which typically arises when using this kind of estimator, and describe a variant we found useful in practice at Criteo.

<div class="home other-pages">
  <ul class="post-list">

{% assign orderedpages = site.counterfactual | sort:"order" %}
{% for post in orderedpages %}
  {% if post.show == true %}
    <li>
	    {% assign date_format = site.minima.date_format | default: "%b %-d, %Y" %}
        <span class="post-meta">
          {{ post.date | date: date_format }}
        </span>
	
	<h2>
      <a href="{{site.repo_name}}{{ post.url }}">
          {{ post.title }}
      </a>
      {% if post.excerpt %}
	  	<span>
          {{ post.excerpt }}
		</span>
      {% endif %}

    </h2> </li>

  {% endif %}
{% endfor %}

  </ul>
 </div>
 
 
