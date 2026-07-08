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
## Manual Working

This repository also includes a manual implementation of the Naive Bayes text classification process prepared as part of the academic case study. The manual working demonstrates each step of the algorithm and helps in understanding how the classifier works before implementing it in Python.

The workbook contains the following worksheets:

**Question** -	Contains the original case study problem statement along with the Amazon customer review dataset used for the classification task.
**Preprocessing** - 	Shows the text preprocessing steps performed on the reviews, including cleaning, tokenization, stop-word removal, and preparation of text for feature extraction.
**Naive Bayes Classification** -	Demonstrates the manual calculation of the Naive Bayes classifier, including vocabulary construction, feature representation, probability calculations, and prediction of the unknown reviews.
**Desired Class** -	Contains the expected (reference) class labels for the unknown reviews, which are used to verify the correctness of the manual calculations and Python implementation.

The manual calculations complement the Python implementation and provide a step-by-step explanation of the text classification workflow using the Multinomial Naive Bayes algorithm.

---

## Result Comparison

The manual calculation and the Python implementation produced different prediction results for the two unknown reviews.

This difference is expected because the two approaches do not use exactly the same feature representation and probability estimation methods.

- The manual calculation follows the step-by-step procedure provided in the academic workbook.
- The Python implementation uses TF-IDF Vectorization with Multinomial Naive Bayes from the Scikit-learn library.
- TF-IDF assigns different weights to words based on their importance, whereas the manual method relies on manually computed word frequencies and probabilities.
- Consequently, slight differences in probability values and final class predictions may occur.

The Python implementation follows a standard machine learning workflow and demonstrates how text classification is performed using modern Natural Language Processing (NLP) techniques.

Note: The difference in predictions does not indicate an error in either implementation. It results from differences in feature extraction, probability estimation, and implementation methodology between the manual academic approach and the Scikit-learn implementation.

## Author

**R.P.Keerthana**

PGDSBA - Data Science and Business Analytics 

Thiagarajar School of Management,Madurai

---

## License

This repository is intended for **academic and educational purposes only**.
