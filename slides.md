---
marp: true
theme: default
paginate: true
header: 8BE130 | AI for Medical Imaging
footer: TU/e Eindhoven University of Technology
style: |
  section {
    color: #000;
  }
  h1, h2 {
    color: #C3002D;
  }
  footer, header {
    color: #666;
  }

---

# AI for Medical Imaging: 8BE130
### Course Introduction and Logistics

---

# Part 1: Introduction
## The Importance of AI in Medical Imaging

---

# Why Medical AI Matters
* State-of-the-art AI models are **very accurate** for many medical image analysis tasks (e.g. cancer diagnosis or prognosis from tissue biopsies, radiotherapy planning, etc).
    * Some models have performance comparable to inter-observer agreement of expert clinicians.
* AI models can be used as:
    * Copilot assistants, to increase the throughput and/or accuracy of currently established workflows.
    * Triaging tools, to help clinicians prioritize cases.
    * Inexpensive second opinion.
    * And more...

---

![bg contain 50%](assets/nuclei_detection.png)

---

![bg contain 80%](assets/triaging.png)

---

![bg contain 80%](assets/report_generation.png)

---

![bg contain 50%](assets/multimodal.png)

___
# Figure Sources

* Schuiveling et al, Artificial Intelligence–Detected Tumor-Infiltrating Lymphocytes and Outcomes in Anti–PD-1–Based Treated Melanoma, JAMA Oncology, 2025
* Lucassen et al, Artificial intelligence-based triaging of cutaneous melanocytic lesions, npj Biomedical Innovations, 2025
* Lucassen et al, Pathology Report Generation and Multimodal Representation Learning for Cutaneous Melanocytic Lesions, MICCAI, 2025
* Liu et al, Adaptive Prototype Learning for Multimodal Cancer Survival Analysis, arXiv, 2025

---
# From Vision to Language (and Beyond)
* Increasingly medical image analysis is not just about "seeing." 
* It is about **interpreting** images, often in combination with other modalities (e.g. text, lab tests, etc.)
    * We call this **multi-modal** analysis.
* **Vision-Language Models (VLMs):** Multimodal models that can process both images and text.
* In principle, models for any combination of biomedical modalities are possible and can be useful in practice.

---

# The Motivation for This Course
* Engineers are increasingly seen as "users of AI".
* However, AI is just another technology, like a computer or a microscope.
* Engineers should understand the inner workings of AI, this will enable them to:
    * Better use AI in their work.
    * Design and improve AI systems as part of their work.
    * Critically evaluate the use of AI in their work.

---

# Your Assumed Background

* Familiarity with Python programming.
* Fundamental Machine Learning and Deep Learning (e.g. 8BB020 Machine Learning).
* Some familiarity with Medical Image Analysis (e.g. 8DC000 Medical Image Analysis).

---

# Part 2: Course Structure
## Based on Course Syllabus (README)

---

# Course Overview
The course is divided into two phases:
1.  **Skill Building (Assignments 1 & 2):** Foundational work on Transformers, GPT, and Vision-Language models.
2.  **Application (Open Project):** Research-based project using accumulated knowledge to answer a specific clinical or technical question.

---

# Essential Course Files
| File | Description |
| :--- | :--- |
| **README.md** | Logistics, schedule, and guidelines. |
| **Assignment 1** | Jupyter notebook focusing on GPT language models. |
| **Assignment 2** | Jupyter notebook focusing on vision-language models. |
| **Open Assignment** | Instructions for the research project. |

---

# Teaching Method: Flipped Classroom

* This is a **project-based course** with a 140-hour workload (5 ECTS).
* **Preparation:** Watch videos and read papers *before* the lecture.
* **In-Class:** Active discussion and peer consultancy. Not a traditional lecture.

---

# Peer Consultancy Model
Project groups rotate roles to facilitate learning:

* **Clients:** Identify blockers and prepare theory or coding questions.
* **Consultants:** Review material, so you are able to brainstorm solutions and provide resources.
* **Plenary Synthesis:** "Unsolvable" blockers are addressed in a plenary discussion.

⚠️ In principle, the preparation is the same for everyone. Different roles are assigned purely to facilitate participation and streamline our discussion.

---

# Course Schedule
* **Week 1:** Kick-off Meeting.
* **Weeks 2 to 3:** Assignment 1 (GPT) Flipped Classrooms and Submission.
* **Weeks 4 to 5:** Assignment 2 (VLM) Flipped Classrooms and Submission.
* **Week 6:** **The Pitches** for the Open Assignment.
* **Weeks 7 to 8:** Office Hours (Independent project work).
* **Exam Week 2:** Final Open Assignment Deadline.

---

# The Pitch Session (Week 6)
To ensure project feasibility, every group presents a **2-minute pitch**:
* **The Slide:** Single slide outlining the research question and methodology. You must use TU/e house style. 
* **Feedback:** 3 to 5 minutes of "critical friend" review from peers and instructors.
* **Outcome:** "Green Light" to proceed or an advice for revision.

---

# Deliverables
* **Assignments 1 & 2:**
    * Completed Jupyter notebook exercises.
    * Flipped Classroom Log (documented preparation and participation). Keep it **short**, 600 words max. Goal is to show engagement with the material. 
* **Open Project:**
    * A scientific poster summarizing your research questions, experiments, and results.
* **Submission:** One set of files per group via Canvas.

---

# Grading Breakdown
* **Assignment 1 (25%):** 20% Exercises, 5% Logs.
* **Assignment 2 (25%):** 20% Exercises, 5% Logs.
* **Open Project (50%):** Based on research design, experiments, and poster quality.

---

# Assessment Criteria: Assignments 1 & 2
* **Understanding:** Move beyond "copy and paste". Demonstrate understanding and insightful connections.
* **Code Quality:** Should be modular (if applicable), well-structured, and easy to reproduce.
* **Analysis:** Professional visualizations and insightful interpretation of model behavior.
* **Flipped Classroom Logs:** Documented preparation and participation. 

---

# Assessment Criteria: Open Project
* **Research Design:** Looking for novel or challenging questions and state-of-the-art methodology. But do keep in mind time and computational resources.
* **Experimental Design:** Comprehensive validation of your hypothesis.
* **Poster Design:** Clear visual narrative, professional language. You must use TU/e house style.

---

# Next Steps
* Form your group of 4 on Canvas.
* Download **Assignment 1** and begin the prep material.
* Review the **Odd/Even** group schedule to determine your role for Week 2.

---

# Assignment 1: GPT from Scratch
## Skill Building: Generative Language Modeling

* **Core Objective:** Study the Transformer architecture by building a decoder-only model from scratch.
* **Architecture:** Implementation of self-attention mechanisms, including Query, Key, and Value matrices.
* **Autoregressive Learning:** Training the model to predict the next token in a sequence using character-level tokenization.
* **Medical Context:** Use of textual figure captions from the Open-MELON dataset to learn medical terminology.

---

# Assignment 1: Rationale
## Why Build a "NanoGPT"?

* **Technical Depth:** Moving beyond use of AI models via APIs or chatbots to understand the exact data flow within a Transformer block.
* **Practical Training:** Learning how hyperparameters like learning rate, batch size, and temperature affect training and output quality.
* **Foundational Knowledge:** Learning "the basics" before moving to multi-modal vision-language tasks.
* **Model Evaluation:** Understanding how models prioritize "plausible sounding" text over factual truth in medical reports.

---

# The Open-MELON Dataset

![bg right 90%](assets/open-melon.png)
