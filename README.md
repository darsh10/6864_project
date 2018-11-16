# Question answering in online forums
Irene Chen, Rohan Chitnis, Darsh Shah

## Intro
For this project, we will focus on how to answer a question from an online forum (e.g. Cisco, Apple) with help from prior questions and online support documents and manuals. 

For more frequent updates, check out [the Google Doc.](https://docs.google.com/document/d/1IG29vpl3IevXwyaqfyhOvg2fpJlST9OT6ap5wcd5dSk/edit)

## Dataset

We are currently investigating the Apple dataset (93k discussion threads scraped from online) in order to suggest an answer that leverages support documents as well as prior answers. 

The scraped dataset can be re-collected using our scraping script `scraping/apple_discussion_threads.py`, which includes a randomized delay in order to avoid being blocked.

## Baselines

Our initial baseline investigates suggesting a naive model that suggests a support document based on our labeled training data. If we ignore the content of the support document, we can first approach the problem as a multi-class classification problem.

Below we see that logistic regression performs the best, but we have plenty of room for improvement. More data improves the performance of Logistic Regression and Decision Tree but decreases the performance of Cosine Similarity models.

![Logistic regression performs best, but plenty of room for improvement](baseline/scores_partial_full.png? "Classifier Scores")