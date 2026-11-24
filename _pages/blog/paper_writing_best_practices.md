---
title: Best Practices in Paper Writing
layout: default
permalink: /blog/paper_writing_best_practices
published: true
---

# Best Practices in Paper Writing

- First thought, with Ted Adelson, for preparation of a paper:

  + State the problem we want to solve.

  + Review the current solutions to the problem and ask whether they are satisfactory. If they are, we do not need to work on it.

  + If they are not, present our solution and compare it with the existing ones.

  + Finally, in the related work section, state the proposed techniques that have been used for different problems.

- What can we do for each section in a paper: 

  + Introduction: a) What problem we want to solve, b) why is it important and interesting, and c) what is/isn't new in our paper? (***claims***, ***problems***, ***novelty***, ***significance***)

  + Related work: surveys and discusses how these work are related, what have been done/have not done from the big to narrow pictures of the targeting topic (***context***).

  + Main idea & Algorithm: can be foundational conceptual framework (e.g., underlying model/theory which can be theoretical or systemic) and methodology (e.g., structured, strategic and procedure,...). This part can be better with a toy example or a figure to illustrate components, how they work and interact with others. Besides, the paper should explicitly state the technical assumptions/limitations (***honesty***).

    + *Workflow vs. Algorithm*: while workflow provides a marco-level conceptual understanding of the system model/theory but fail to show edge cases, stop conditions,..., algorithm can introduce micro-level implementation blueprint (e.g, how to implement the proposed solution: control flow, loop bounds,... with complexity).

  + Experiments: are the place we need to show the feasibility and efficiency of the proposed solution, comparing with existing SOTAs, under some (controlled) settings. On the other hands, we need to show the ***soundness***: results need to substantiate the claims/contributions.

    + *If the paper improves results compared to SOTAs, the paper clearly explains the reason of what made this improvement possible, or where it is coming from: What is improved in the algorithm/analysis?*

  + Discussion: is to end with a conclusion/summary, or to show what our contributions open up and how the proposed solution changes the way we approach the current problem.

- Some tips for better scientific writing:

  + **Principle**: What does the reader know so far, and what does the reader expect next and why?

  + Readers look at, in decreasing numbers, 1) the title, 2) the abstract, 3) a few result figures, and 4) every word. As a result, figure captions need to be self-contained, and the lead-in to the formulation should be precise and smooth.

- How do we decide whether a paper should be accepted or rejected?

  + Are any important references missing, so that the state of the art is not properly covered?

  + Do the authors fail to deliver what the stated contributions promise?

  + Are the results incremental or implausible?

  + Is the writing poor?

**References**

[1]. [Paper Writing Best Practices](https://icml.cc/Conferences/2022/BestPractices)

[2]. [Bill Freeman's slides](https://drive.google.com/file/d/1sE4ZCHkU65J6ZFjOK4dWGIiuPygkM6ZW/view)
