---
title:
layout: default
permalink: /research/loss_function
published: true
---


$$D_{KL}(P^*|P) = \sum_y{P^*(y|x_i)log\frac{P^*(y|x_i)}{P(y|x_i; \theta)}} = \sum_y{P^*(y|x_i)[logP^*(y|x_i) - logP(y|x_i; \theta)]} = \underbrace{\sum_y{P^*(y|x_i)logP^*(y|x_i)}}_\text{Entropy and does not depend on $\theta$} \underbrace{- \sum_y{P^*(y|x_i) logP(y|x_i; \theta)}}_\text{Cross entropy} $$,

where $(x, y)$ is the feature and corresponding label, $\theta$ is the model parameter, the $P$ is the predicted class distribution and $P^*$ is the true class distribution.

> Therefore, minimizing $D_{KL}(P^*|P)$ is equivalent to minimizing **Cross entropy**.