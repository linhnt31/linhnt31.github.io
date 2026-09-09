---
title: Misc
layout: default
permalink: /misc/
published: true
---

<nav class="toc" markdown="1">
**On this page**

- [Cambridge FL Workshop &amp; Flower Hackathon — Aug. 2026](#fl-cambridge-2026)
- [ADVANCE CRT Research Colloquium — Feb. 2026](#advance-colloquium-2026)
- [ADVANCE CRT 3rd Cohort Celebration — Oct. 2025](#advance-cohort3-2025)
- [ADVANCE CRT Research Summer School — Jun. 2024](#advance-summer-school-2024)
- [ADVANCE CRT Research Colloquium — Apr. 2024](#advance-colloquium-2024)
- [ADVANCE CRT Induction — Jan. 2024](#advance-induction-2024)
- [ADVANCE CRT Workshop — Nov. 2023](#advance-workshop-2023)
- [9th Vietnam Summer School of Science — Aug. 2022](#vsss-2022)
</nav>

## [Federated Learning in Healthcare System Workshop](https://luma.com/mm2ruxjk) & [Flower Collaborative Agent Hackathon](https://flower.ai/events/collaborative-agent-hackathon), Cambridge, 23–26 August 2026 {#fl-cambridge-2026}

From 23 to 26 August 2026, I had the chance to visit Cambridge, UK, to attend an FL hackathon and workshop. Cambridge is walkable, quiet, and full of academic atmosphere, with historic colleges, the beautiful [River Cam](https://en.wikipedia.org/wiki/River_Cam), and lovely bridges. It was a perfect place to pause, exchange ideas, and reflect on the current state and future of federated learning (FL).

The workshop focused on applying FL in healthcare systems. Many of the projects presented appeared to operate through established consortia, where hospitals, universities, and other institutions collaborate to train a shared model while keeping their data local. Agreements govern participation, access to the model, and, in some cases, how commercial benefits are shared. My impression was that making FL work in practice depends as much on these arrangements as on the learning algorithms themselves.

I was also interested in the campus applications presented, including [FedCampus 1.0 and 2.0](https://github.com/FedCampus), [FedKit](https://github.com/FedCampus/FedKit), and [ChatDKU](https://chatdku.com/). These examples prompted me to think about how FL and agentic AI could support students’ academic, administrative, and health needs while keeping sensitive data local. They also raised a question worth asking of such applications: where does collaborative learning happen, and where/how are agents simply coordinating queries or exchanging insights?

Beyond individual projects, I saw a growing FL ecosystem, including [FLIP](https://github.com/londonaicentre/FLIP) (see the photo below). FLIP supports federated training and evaluation across healthcare institutions using FL engines such as Flower and NVIDIA FLARE. As described at the workshop, its architecture uses secure environments at client sites and limits the information retained centrally.

<figure class="photo">
  <a href="{{ site.baseurl }}/assets/images/FLworkshop26/flip.jpeg" target="_blank" rel="noopener">
    <img src="{{ site.baseurl }}/assets/images/FLworkshop26/flip.jpeg"
         width="2046" height="1130" loading="lazy" decoding="async"
         alt="Workshop slide grouping federated learning tools into deployed platforms, federated analytics, FL engines, and peer-to-peer approaches.">
  </a>
  <figcaption>The FL platform landscape as presented at the workshop: deployed platforms, federated analytics, FL engines, and peer-to-peer approaches.</figcaption>
</figure>

These developments were encouraging, but they left me wondering how FL could become more accessible beyond established collaborations. Although FL already extends beyond healthcare and institutional consortia, many of the examples I encountered relied on agreements among a relatively small group of organizations. How could individuals and smaller organizations participate without first building such a consortium?

> Can we create an FL-based marketplace where individuals and organizations are rewarded for useful contributions from their local data and computing resources, while retaining control over how those contributions are used?

I imagine a platform where a buyer proposes a learning task, defines how model improvement will be evaluated, and invites eligible participants to contribute without transferring their raw data. In return, participants might receive payment, access to the trained model, or a share of the value it creates.

However, computation and data contributions should not be valued in the same way. A participant may perform substantial computation while contributing little new information, whereas a small dataset may contain rare and valuable examples. Broader participation could improve coverage and generalization, but this is not automatic: contributions must be relevant, and differences in data quality and distribution must be handled carefully. The challenge is to make participation accessible while ensuring that contributions are useful and rewards are fair.

These questions stayed with me during the hackathon organized by [Flower](https://flower.ai/), where we developed agent applications and deployed them on the [Flower SuperGrid](https://flower.ai/docs/framework/how-to-run-flower-apps-on-supergrid.html). It was exciting to team up with people from different backgrounds and build a prototype exploring how federated insights across facilities could support better post-operative outcomes for cataract surgery without sharing patient records across regions (see [Flower-Seed](https://github.com/linhnt31/hackathon_floweai_team) and the [Flower-Seed App](https://flower.ai/apps/linhnt/Flower-Seed)).

The emphasis on agentic AI initially made me wonder whether FL was struggling to find its own position amid the attention given to AI agents and LLMs. It sometimes feels as though every project needs these keywords nowadays. On reflection, however, becoming the backbone of other applications could be a strong position for FL. Like microservices, it could become widely used without being visible to end users. What matters is whether it enables useful learning and collaboration that would otherwise be difficult to achieve.

My vision is therefore an FL ecosystem that makes collaboration easier to enter, sustain, and leave. Participants should understand what they contribute, how they benefit, and what happens if they later withdraw. This connects directly to my research interests in incentives and federated unlearning: *if someone is rewarded for contributing and later requests the removal of their influence, who performs the unlearning, who pays for it, and how are the remaining participants affected?*

For me, these are central questions for the future of FL. Better algorithms matter, but broader adoption also requires credible rewards, clear participation rules, and meaningful control over contributions throughout their lifecycle.

## Research Ireland ADVANCE CRT Research Colloquium, 10th & 11th February 2026 {#advance-colloquium-2026}

I just had a wonderful time with my ADVANCE CRT friends and speakers during the Research Colloquium in the quiet city of [Portlaoise](https://maps.app.goo.gl/7pFsF2wcVbasgF4P7), where I had a chance to listen, learn, understand, and discuss Horizon Europe funding which is vital for the future careers of fresh PhDs in both academia and industry.

Here are a few lessons I personally observed and conceived from this event, naturally from my current perspective as a 3rd year PhD student.

+ I was provided with necessary skills and information to see the differences between Horizon Europe **PILLARS** and **WORK PROGRAMS**, how to search for information related to Horizon Europe funding calls, and how to read them efficiently to save time while grasping writers' key visions.

+ To have a competitive research proposal, we need to know ***who wrote the topics*** and what they are looking for (see the photo below). Specifically, they are politicians with broad and long-term visions, coming up with strategic plans (e.g., EU policies).

+ It is essential to choose the right partners with a proper matrix of skills, who can complement work packages and ***impacts*** from different perspectives that align with EU policies. 

<figure class="photo">
  <a href="{{ site.baseurl }}/assets/images/Portlaoise2026/2.jpeg" target="_blank" rel="noopener">
    <img src="{{ site.baseurl }}/assets/images/Portlaoise2026/2.jpeg"
         width="2048" height="1518" loading="lazy" decoding="async"
         alt="Slide from the Horizon Europe funding talk at the ADVANCE CRT Research Colloquium in Portlaoise.">
  </a>
  <figcaption>Photo taken from the talk of <a href="http://www.hyperion.ie/seanmccarthy.htm">Dr. Seán McCarthy</a> in our event.</figcaption>
</figure>

## Research Ireland ADVANCE CRT 3rd Cohort Celebration, Oct. 2025 {#advance-cohort3-2025}

It has been a long time since I spent time writing about activities outside research. The year 2025 has been so busy so far with paper submissions, confirmation report after 18 months of PhD, and a placement at TU Delft (I actually wrote some diaries and reflections for this period, but I kept them private in Notion; I will release them in the near future).

At this moment, sitting in the Waterford Plunkett train station, I want to write down some of my thoughts about the event on the 28th and 29th of October in Waterford, at which we celebrated the graduation of the 3rd Cohort from my funding body, Research Ireland ADVANCE CRT. Waterford is the oldest city in Ireland, situated on the River Suir.

<figure class="photo">
  <a href="{{ site.baseurl }}/assets/images/Waterford2025/waterford1.jpg" target="_blank" rel="noopener">
    <img src="{{ site.baseurl }}/assets/images/Waterford2025/waterford1.jpg"
         width="2040" height="1148" loading="lazy" decoding="async"
         alt="Speaker presenting to the ADVANCE CRT cohorts at the 3rd Cohort Celebration in Waterford.">
  </a>
  <figcaption>Photo taken from the talk of <a href="https://www.badboyofscience.com/about">Dr. Sam Gregson</a> in our event.</figcaption>
</figure>

It’s always a joy to meet people who share the same mindset and the same struggles of a PhD journey. With them, you don’t need to explain much about your work, challenges, or little moments of happiness; they simply understand. Over 2 days, here are a few things I heard from my peers, as well as talks from keynote speakers and the 3rd and 2nd cohorts.

+ I am happy to see people doing well, maintaining passion for their research, and maintaining good relationships with their supervisors.

+ Many also shared their struggles with loneliness and how they’ve learned to cope with it, e.g., working in the park, resting, traveling around, calling family, etc. While most people like working from home due to constraints like living far from the office or transportation issues, I also met people whose offices are similar to ours, where people enjoy coming to the office and talking to others ^^.

+ Your success often depends on having a kind supervisor, someone who treats you with mutual respect, listens, is patient, and truly cares about your growth.

> Last but not least, everyone has their own challenges and struggles, so **none of us are alone in this interesting yet demanding journey**. Be brave, stay resilient, and don’t hesitate to open up to people you trust, to seek advice, support, or simply a few words of encouragement.

## SFI ADVANCE CRT Research Summer School, Jun. 2024 {#advance-summer-school-2024}

From June 10 to 14, I had the privilege of attending a remarkable 5-day summer school organized by SFI ADVANCE CRT in the picturesque town of Killarney, County Kerry, Ireland. This intensive program offered two training tracks for PhD students: Social and STEM. What made this experience truly enriching was the interdisciplinary collaboration, where we formed groups with members from both tracks to propose case studies.

<figure class="photo">
  <a href="{{ site.baseurl }}/assets/images/Killarney2024/allteam.jpeg" target="_blank" rel="noopener">
    <img src="{{ site.baseurl }}/assets/images/Killarney2024/allteam.jpeg"
         width="800" height="450" loading="lazy" decoding="async"
         alt="Group photo of all participants at the SFI ADVANCE CRT Research Summer School in Killarney.">
  </a>
</figure>

**Day-by-Day Learning Journey:**

+ Day #0: I traveled with my friend from Trinity College Dublin to Killarney. After lunch at [Muckross Park Hotel & Spa](https://www.muckrosspark.com/?gad_source=1&gclid=CjwKCAjw1K-zBhBIEiwAWeCOF6OjC_EYqY2k8kVrOaSTvzFu0Phe3tJeZCifI0FOoGw5NRNGvwPF6hoCxCUQAvD_BwE), we had two keynote speakers talking about the impacts of AI on Networking and society.

+ Day #1: We delved into Logistic Regression, Neural Networks, and Large Language Models (LLMs).

+ Day #2: The focus was on Reinforcement Learning.

+ Day #3: We explored Data Visualization techniques.

+ Day #4: We presented our group's case study. 

One of the highlights was working on a collaborative project where we proposed an LLM-based mobile application for mental health support. This application is designed as a supplemental tool, not for medical treatment. Collaborating with peers from various fields such as computer vision, e-Health, and neuro-psychology broadened my perspective and enhanced my learning experience.

**Exploring the Natural Beauty of Killarney**
Despite the intensive schedule, I made sure to enjoy the stunning nature around Killarney. Here are some of the memorable spots I visited:

+ [Killarney National Park ](https://www.nationalparks.ie/killarney/): A beautiful expanse of greenery and serene lakes.

+ [Muckross Abbey](https://en.wikipedia.org/wiki/Muckross_Abbey): A short visit to this historic site was quite enriching.

+ [Lough Leane](https://killarneylaketours.ie/): We took a delightful lake cruise, soaking in the scenic beauty of the islands.

Unfortunately, I couldn’t visit the [Torc Waterfall](https://www.kerrygems.com/kerry-gems-app/the-best-walks-in-kerry/torc-waterfall-walk/) this time, but I hope to return soon to explore more of this beautiful region.

Overall, this summer school was not only a great learning opportunity but also a chance to connect with amazing people and enjoy the breathtaking landscapes of Killarney.

## SFI ADVANCE CRT Research Colloquium, Apr. 2024 {#advance-colloquium-2024}

We had a great time at the Research Colloquium event organized by SFI ADVANCE CRT in Tralee, County Kerry. On the first day, we showcased our posters in a *presenters* and *hunters* format, where we could talk about research while receiving lots of advice and feedback from fellows. Besides, we participated in outdoor activities together, including pedalo boats, a climbing wall, and the Tower of Hanoi quizzes at [Tralee Bay Wetlands](https://traleebaywetlands.org/). On the second day, we had two sessions on entrepreneurship. I was quite impressed by the sharing on *how to think like an entrepreneur*, the mindset of *thinking ahead and focusing on scalability rather than just starting up*, and *insisting on doing hard/uncomfortable things* on the long-term road during the first session.

<figure class="photo">
  <a href="{{ site.baseurl }}/assets/images/Tralee2024/ADVANCE_CRT_Cohorts.jpg" target="_blank" rel="noopener">
    <img src="{{ site.baseurl }}/assets/images/Tralee2024/ADVANCE_CRT_Cohorts.jpg"
         width="1600" height="1200" loading="lazy" decoding="async"
         alt="Group photo of the SFI ADVANCE CRT cohorts at the Research Colloquium in Tralee.">
  </a>
</figure>

## SFI ADVANCE CRT Induction, Jan. 2024 {#advance-induction-2024}

> In the early of 2024, my fellowship body, i.e., [SFI ADVANCE CRT](https://www.advance-crt.ie/), organized induction event for the 5th cohort at [University College Cork](https://www.ucc.ie/en/) (UCC), Cork, Ireland. This was a great opportunity for the new cohort to connect, study, and plan for next steps in their PhD journey in Ireland. 

### I. Traveling from Dublin to Cork - Day 1 (09/01/2024)

I started by going to Dublin Pearse station by Dart at 07:20, then took the No.26 bus to Dublin Heuston, and went to Cork Kent station by train at 09:00. After arriving at Cork Kent station around 12:00, I took the No.215 bus to go to [Blarney Castle & Gardens](https://blarneycastle.ie/), which is a beautiful and attractive place for tourists. No wonder, the nature is always the most attractive part of Ireland for me.

After 3 hours of walking around, I went back to the [Kingsley Hotel](https://www.thekingsley.ie/?gad_source=1&gclid=CjwKCAiA-vOsBhAAEiwAIWR0TZH4N4x6T3s6syCxmXEq7f4hEkChI2hXGBLpZRgqm3USX_m5LYJSOBoCcOIQAvD_BwE), where the organisers had booked rooms for us from the 9th to the 13th (The scholarship agent is so generous. They also booked us a 4 star hotel in Clayton for the Limerick trip ^^).

### II. Induction - Day 2 (10/01/2024)

This day was all about information about our PhD training process, e.g. responsibilities, benefits,... and then each PhD student gave a short presentation about themselves, their research topics and hobbies. Finally, we had the opportunity to listen to the experiences of the veterans of the 1st cohort.

### III. Personal Development Training - Day 3 (11/01/2024)

The third day was the day I feel I gained the most from this event, where we were introduced to the [Vitae Researcher Development Framework](https://www.vitae.ac.uk/vitae-publications/vitae-library-of-resources/about-vitae-researcher-development-programmes/effective-researcher) (RDF). This was the time where I had to revisit the reasons why I chose to do a PhD, know the skills needed to become an independent researcher, and understand the role of the supervisor in my career and professional development by listening to the opinions of professors and peers. In addition, [**STAR**](https://nationalcareers.service.gov.uk/careers-advice/interview-advice/the-star-method) and [**SMARTER**](https://solocom.ca/en/smarter-method/#:~:text=The%20SMART%20concept%20to%20achieving,evaluate%20and%20revise%20your%20goals.) techniques are also introduced to us for *self-assessment* and *research topic/career goal finding*.

Here are few key takeaways condensed from these two techniques:

\- `STAR`: for self-assessment

+ <u>S</u>ituation: the situation you had to deal with, i.e., when, where, with whom. E.g., You are a PhD student.

+ <u>T</u>ask: the task you were given to do. E.g., you need to publish papers and articles to graduate in 4 years.

+ <u>A</u>ction: the action you took to achieve your goals. E.g., Planning, learning fundamentals, applying learned things to your problems, running experiments, and publish work.

+ <u>R</u>esult: what happened as a result of your action and what you learned from the experience. E.g., Your published work are enough for graduation?

\- `SMARTER`: for planning, evaluating, and revising your goals

+ <u>S</u>pecific:

    + What do I want to achieve? E.g., Receive a PhD degree & Land a good industrial job after graduating

    + How will I achieve this? E.g., Publish papers and articles at top-tier conferences and journals & Have solid technical skills such as designing architecture, building projects using programming languages and open-sources, and grasping math foundation.

    + Who is involved in this project? E.g., You, your supervisor, and collaborators 

+ <u>M</u>easurable:

    + What are your metrics to reach your goals? E.g., 2 top-tier conference papers and 2 transactions articles

    + How do I know if I’ve reached my objective? E.g., Your work are accepted by publication venues

    + When do I want to reach this objective? E.g., December, 2026

+ <u>A</u>chievable:

    + Do I have the required human resources? E.g., Yes, your supervisor and collaborators

    + Do I have the financial means to reach my goal? E.g., Yes, funding from agents and other work like demonstrators or teaching assistant

    + Do I have the necessary technology or equipment? E.g., Yes, laptops, servers (?) and other devices (?)

+ <u>R</u>elevant:

    + Are my goals relevant to my business context? E.g., Yes, it is good for both academic or professional career

    + Do my goals align with those set by my supervisor, department, or organization? E.g., Yes, publications are all you need

    + Is the market ready or saturated for my offer? E.g., No, currently, i.e., Jan. 2024, the job market for blockchain and federated learning is still hectic. There are shortage of skilled workers.

+ <u>T</u>imely:

    + What can I achieve in 3, 6, and 12 months? E.g., In the next 3 months, you will try to submit a conference paper, then extend it to a journal version in the next 3 months.

    + What’s my deadline? E.g., conference/journal deadlines

    + Do I need to establish a timeline? If so, what are my smaller objectives? E.g., Yes, firstly, you need to finish problem formulation, then solve it using mathematical knowledge (you can divide this part into smaller objectives like time for learning and applying knowledge to your problems). After that, you can build your experimental framework to run your scenarios using your proposed solutions.

+ <u>E</u>valuate:

    + Weekly evaluation

    + Monthly evaluation

+ <u>R</u>eadjust:

    + Is based on the *Evaluation* step.

In the evening, we drank beer and went bowling with another guy at the [Mardyke Entertainment Complex](https://www.google.ie/maps/place/Mardyke+Bowl/@51.8981628,-8.4837204,17z/data=!4m14!1m7!3m6!1s0x48449017a7759863:0xdaa03b25dc975d26!2sMardyke+Bowl!8m2!3d51.8981628!4d-8.4811455!16s%2Fg%2F11gfp52wqb!3m5!1s0x48449017a7759863:0xdaa03b25dc975d26!8m2!3d51.8981628!4d-8.4811455!16s%2Fg%2F11gfp52wqb?entry=ttu). I also caught up with trying *chicken ball, spring roll, braised beef noodle and Chicken Yakitori rice* at [Noodle-UCC restaurant](https://maps.app.goo.gl/1KuxdNfubXK9vkij8) (You know I am obssed with noodles, so I always want to try every kind of noodle wherever I visit ^^). 

### IV. Systems Thinking - Day 4 (12/01/2024)

Systems Thinking

### V. Coming back to Dublin - Day 5 (13/01/2024)

After breakfast at 09:00, I checked out and walked to some places, i.e. the [Glucksman Gallery](https://www.glucksman.org/) in UCC and the [English Market](https://www.corkcity.ie/en/english-market/) in the city centre, before going to Cork Kent Station. Finally, I went home at 16:00 and official ending my journey to theADVANCE CRT Induction event. 

> Such a great trip!

## SFI ADVANCE CRT Workshop, Nov. 2023 {#advance-workshop-2023}

<figure class="photo-grid">
  <a href="{{ site.baseurl }}/assets/images/Limerick2023/1.jpg" target="_blank" rel="noopener">
    <img src="{{ site.baseurl }}/assets/images/Limerick2023/1.jpg"
         width="1000" height="750" loading="lazy" decoding="async"
         alt="Photo from the SFI ADVANCE CRT workshop in Limerick, November 2023 (1 of 4).">
  </a>
  <a href="{{ site.baseurl }}/assets/images/Limerick2023/2.jpg" target="_blank" rel="noopener">
    <img src="{{ site.baseurl }}/assets/images/Limerick2023/2.jpg"
         width="750" height="1000" loading="lazy" decoding="async"
         alt="Photo from the SFI ADVANCE CRT workshop in Limerick, November 2023 (2 of 4).">
  </a>
  <a href="{{ site.baseurl }}/assets/images/Limerick2023/3.jpg" target="_blank" rel="noopener">
    <img src="{{ site.baseurl }}/assets/images/Limerick2023/3.jpg"
         width="750" height="1000" loading="lazy" decoding="async"
         alt="Photo from the SFI ADVANCE CRT workshop in Limerick, November 2023 (3 of 4).">
  </a>
  <a href="{{ site.baseurl }}/assets/images/Limerick2023/4.jpg" target="_blank" rel="noopener">
    <img src="{{ site.baseurl }}/assets/images/Limerick2023/4.jpg"
         width="750" height="1000" loading="lazy" decoding="async"
         alt="Photo from the SFI ADVANCE CRT workshop in Limerick, November 2023 (4 of 4).">
  </a>
</figure>

## 9th Vietnam Summer School of Science, Aug. 2022 {#vsss-2022}

<figure class="photo-grid">
  <a href="{{ site.baseurl }}/assets/images/VSSS09/1.jpg" target="_blank" rel="noopener">
    <img src="{{ site.baseurl }}/assets/images/VSSS09/1.jpg"
         width="1000" height="682" loading="lazy" decoding="async"
         alt="Photo from the 9th Vietnam Summer School of Science, August 2022 (1 of 4).">
  </a>
  <a href="{{ site.baseurl }}/assets/images/VSSS09/2.jpg" target="_blank" rel="noopener">
    <img src="{{ site.baseurl }}/assets/images/VSSS09/2.jpg"
         width="1000" height="750" loading="lazy" decoding="async"
         alt="Photo from the 9th Vietnam Summer School of Science, August 2022 (2 of 4).">
  </a>
  <a href="{{ site.baseurl }}/assets/images/VSSS09/3.jpg" target="_blank" rel="noopener">
    <img src="{{ site.baseurl }}/assets/images/VSSS09/3.jpg"
         width="750" height="1000" loading="lazy" decoding="async"
         alt="Photo from the 9th Vietnam Summer School of Science, August 2022 (3 of 4).">
  </a>
  <a href="{{ site.baseurl }}/assets/images/VSSS09/4.jpg" target="_blank" rel="noopener">
    <img src="{{ site.baseurl }}/assets/images/VSSS09/4.jpg"
         width="1000" height="750" loading="lazy" decoding="async"
         alt="Photo from the 9th Vietnam Summer School of Science, August 2022 (4 of 4).">
  </a>
</figure>
