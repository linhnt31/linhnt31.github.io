---
title: Lessons learned from my rebuttals
layout: default
permalink: /blog/rebuttal_lessons_learned
published: true
---

<span style="color: red;"><u>**Publication Date</u>**</span>: Nov. 24, 2026

So the reviews are out. Maybe you got a "weak accept" that makes you smile, a "reject" that makes you stare at the wall for a while, and that one reviewer who clearly read only the abstract. Welcome to the club! ☕

This post collects what I have learned (from both successful and failed attempts) about writing **response letters** for journals (in my case, Networking and Communications, e.g., IEEE Transactions) and **rebuttals** for AI conferences (e.g., ICLR, NeurIPS). They look like the same job, but the rules of the game are quite different. Grab a coffee, and let's go.

> **TL;DR** for busy PhD students: (1) read the rules first, (2) sort the comments by how much they can move your score, (3) answer with evidence instead of arguments, (4) stay polite even when it hurts, (5) post early, then let it go.

---

### Step 0: Read the rules of the game first

Before writing a single word, check the policy of your venue. It decides your whole strategy.

- **How much time do you have?** For IEEE Transactions, a major revision usually gives you around two to three months. For AI conferences such as [ICLR](https://iclr.cc/), the whole rebuttal/discussion period is only about **two weeks**. Same task, very different speed.

- **How much can you write?** Journals usually let you write a response letter in your own format with no strict length limit. Conferences may have a character limit per reviewer, and some do not allow you to upload a revised PDF or share external links. Check this *before* you plan a beautiful new figure that you cannot actually show.

- **Respect the double-blind rule.** No links to your GitHub, lab website, or anything that reveals who you are. A desk rejection during the rebuttal is a plot twist nobody wants.

- **The two-week window is shorter than it looks.** At AI conferences, those two weeks include the *discussion* with reviewers, *not only your writing time*. So try to **post your responses within the first 5 to 8 days**. If you post on the last day, reviewers have no time to ask follow-up questions or update their scores. And remember that many reviewers are also authors fighting with their own rebuttals at the same time. Make it easy for them to engage early.

### Step 1: Take a breath, then triage

Do not reply on the day the reviews come out. Read everything, close the laptop, and sleep on it. Comments that feel like an insult at 11 PM often look like a fair (or at least fixable) point the next morning.

Then read all comments from editors, area chairs, and reviewers again, and sort them into three buckets:

- **#Q1: Score-moving concerns** (the ones that decide accept or reject)

    - Doubts about novelty and contributions

    - Flaws in theory or evaluation

    - Missing baselines

- **#Q2: Fixable misunderstandings**

    - The reviewer missed (important) parts of the paper, or understood them differently from what you meant. This is also a quiet hint that those parts could be written more clearly.

- **#Q3: Minor points**

    - Typos, formatting, notation, missing references

One more trick: **if two or more reviewers raise the same concern, it automatically becomes a #Q1.** It is a strong signal to the area chair, so treat it seriously.

### Step 2: Answer with evidence, not arguments

Reviewers are hard to convince with "we respectfully disagree" alone. They are much easier to convince with a number, a table, or a pointer to a specific line. For each concern, I use this simple format:

- **Who** raised it (e.g., *R2-Q3*), and a short quote or summary of the concern,

- **What** we did to address it,

- **The evidence** (new results, a proof sketch, a reference, or "see Section 4.2, lines 210 to 215"),

- **What changed** in the paper (or *will* change in the camera-ready version).

For journal response letters, it also helps a lot to highlight the revised text in the manuscript (e.g., in blue) and point to the exact page and line numbers. Editors love it when they do not need to play "find the change".

### Step 3: Plan your energy (and your GPUs)

- **Clear #Q2 and #Q3 early.** They are cheap to fix and give you a quick feeling of progress. More importantly, they save your energy for **#Q1**.

- **Decide on new experiments fast.** If a #Q1 needs additional experiments, decide on day one or two whether you will run them. If yes, start immediately. During the rebuttal period, the shared GPU cluster of your school or group becomes very crowded, because everyone else is running "just one more experiment" too. 🐢

- **Do not promise what you cannot deliver.** "We will add this in the final version" is fine only if you really will. Reviewers and area chairs do remember.

- **Order matters.**
    - For AI conferences, put the **#Q1** concerns (and the shared concerns) at the **top** of your response. Reviewers and area chairs may only read the first part carefully.

    - For IEEE Transactions, I recommend answering the comments **in their original order**. Editors expect a complete, point-by-point letter.

### Step 4: Mind your tone

- **Thank the reviewer in one sentence, then get to the point.** Whatever the reviewer says, never attack them. The rebuttal is not the place to show your feelings. Write the angry version in a private note if you need to, delete it, and then write the real one. 😄

- **Turn "you are wrong" into "we should have been clearer".** For example: *"We apologize for the confusion. We have clarified this in Section 3.1."* It solves the problem and keeps everyone happy.

- **Be specific, concise, and self-contained.** Reviewers (and we too) are busy and tired, and they have probably forgotten your notation and acronyms. Re-define key terms briefly instead of writing "as shown in Eq. (7)" with nothing else.

- **Spend most of your space and time on #Q1.** Do not write three paragraphs about a typo.

- **Do not forget the summary for the decision makers.** At the end (or at the beginning, in a general response), summarize the main changes and how the concerns were addressed for the editors or area chairs. They make the final call, and they are reading many papers at the same time.

- **Ask a co-author or your advisor to read it before you post.** A fresh pair of eyes catches both the typos and the accidentally passive-aggressive sentences.

### Step 5: Lower your expectations and get back to life

- Writing rebuttals is tiring, especially for AI conferences, and you cannot control whether reviewers and area chairs will actually engage. If you have heard nothing after posting, a short and polite reminder about **two days before the deadline** is reasonable. (From the reviewer side: some of them may have had their own papers rejected or withdrawn, and simply lost motivation. That is not good reviewing practice, but it helps to understand that silence is not always about your paper.)

- Do your best, submit, and then **move on to your next work** while waiting for the decision. Refreshing OpenReview or the IEEE/ACM submission portal every hour will not change the score. Trust me, I have tested this hypothesis many times, and the result is always negative. 📉

- Whatever the outcome, ***the reviews usually make the next version of the paper better***. A rejection with good reviews is still useful data.

Good luck to all of us PhD-ing out there! 🍀

---

### References

[1]. [How to write a NeurIPS/ICML rebuttal (ResearchCraft)](https://researchcraft.dev/guides/how-to-write-a-neurips-icml-rebuttal)

[2]. [How we write rebuttals, by Devi Parikh](https://deviparikh.medium.com/how-we-write-rebuttals-dc84742fece1)
