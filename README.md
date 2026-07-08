# Amazon Customer Review Text Classification using Naive Bayes

A Machine Learning project that classifies Amazon customer reviews into **Positive** and **Negative** sentiment using **Natural Language Processing (NLP)** techniques and the **Multinomial Naive Bayes** algorithm.

---

## Project Overview

Amazon receives millions of customer reviews every day. Manually analyzing these reviews is difficult due to the large volume of data. This project demonstrates how **text classification** can automatically classify customer reviews based on their sentiment.

The project uses **TF-IDF Vectorization** to convert review text into numerical features and **Multinomial Naive Bayes** for sentiment prediction.

---

## Business Problem

Customer reviews directly influence:

- Product ratings
- Customer purchase decisions
- Seller reputation
- Product quality improvement
- Business intelligence

Manual review analysis is time-consuming and impractical. Therefore, an automated text classification model is developed to classify customer opinions efficiently.

---

## Objectives

- Build a text classification model for Amazon reviews.
- Convert text into numerical features using TF-IDF.
- Train a Multinomial Naive Bayes classifier.
- Predict sentiment for unseen customer reviews.
- Generate vocabulary and document-term matrix for analysis.

---

## Dataset

The dataset consists of Amazon customer reviews.

### Training Data

- 8 labeled reviews
- Positive Reviews (Class = 1)
- Negative Reviews (Class = 0)

### Testing Data

- 2 unlabeled customer reviews
- The trained model predicts their sentiment.

---

## Project Workflow

- Amazon Reviews
- Text Preprocessing
- TF-IDF Vectorization
- TF-IDF (Document-Term Matrix)
- Multinomial Naive Bayes
- Prediction
- Classification Results

---

## Technologies Used

- Python 3.11
- Pandas
- Scikit-learn
- Regular Expressions (re)

---

## Machine Learning Algorithm

### Multinomial Naive Bayes

The classifier estimates the probability that a review belongs to either:

- Positive
- Negative

based on the occurrence of words in the review.

---

## Feature Extraction

The project uses

- TF-IDF (Term Frequency–Inverse Document Frequency) (Document-Term Matrix)

TF-IDF assigns higher importance to informative words while reducing the impact of frequently occurring words.

---

## Text Preprocessing

The following preprocessing steps are applied:

- Convert text to lowercase
- Remove punctuation
- Remove numbers
- Remove special characters using Regular Expressions (re)
- Remove English stop words

---

## Project Output

The model generates

- Vocabulary Dictionary
- TF-IDF Document-Term Matrix
- Predicted Sentiment
- Prediction Probability
- Classification Results CSV

---

## Sample Prediction

 Fire truck toy review - Positive
 Journal layout review - Negative 

---

## Results

The trained model successfully classifies unseen Amazon reviews into:

- Positive
- Negative

using Natural Language Processing and the Multinomial Naive Bayes algorithm.

Generated outputs include:

- Vocabulary Dictionary
- TF-IDF Matrix
- Classification Results
- Prediction Probabilities

---

## Future Enhancements

- Larger Amazon review dataset
- Deep Learning (LSTM, GRU)
- BERT-based sentiment classification
- Multi-class sentiment analysis
- Spam review detection
- Real-time review prediction

---

## Author

**R.P.Keerthana**

PGDSBA - Data Science and Business Analytics 

Thiagarajar School of Management,Madurai

---

## License

This repository is intended for **academic and educational purposes only**.
