"""
Amazon Customer Review Text Classification
------------------------------------------
Algorithm : Multinomial Naive Bayes
Vectorizer: TF-IDF
Objective:
1. Build vocabulary (dictionary)
2. Convert text into TF-IDF features
3. Train classifier
4. Predict unknown reviews
5. Save results
"""
import os
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------------------------------------------
# Download NLTK resources (Run only first time)
# --------------------------------------------------------

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# --------------------------------------------------------
# Training Data
# --------------------------------------------------------

train_reviews = [

"""This book is perfect! I'm a first time new mom, and this book made it
so easy to keep track of feedings, diaper changes, sleep.
Definitely would recommend this for new moms.
Plus it's small enough that I throw in the diaper bag for doctor visits.""",

"""I use this so that our babysitter (grandma) can keep track of what
goes on during the day. We've recorded so many milestones since we
received this. I'll definitely get another one in 6 months when we
finish this book.""",

"""I have used this book since my son was born.
It's a great and handy way to keep track of when and how much eats
and when I change his diaper.
Makes it easy to discuss his progress with his doctor.
Highly recommend this journal for parents.""",

"""My friend recommended this teether because her daughter loves it.
However, my daughter loves throwing it on the floor.
Hopefully it will work better for you.""",

"""Don't waste your money.
My son would rather suck on his fingers than use this teether.
The massager really has to be squeezed in order to work.""",

"""This is the best way to keep track of when your baby eats,
sleeps, poops, gets a bath.
Great gift!""",

"""Inside is not nice and clean.
All in all, it does not do what is advertised.""",

"""We bought these trucks for our daughter.
We love them!
Perfect first trucks!"""

]

train_labels = [1,1,1,0,0,1,0,1]

# --------------------------------------------------------
# Reviews to Classify
# --------------------------------------------------------

test_reviews = [

"""My son's therapist brings this toy over and he just loves it.
He loves to move the ladder on the fire truck up and down.
This is one of his favorite toys and he doesn't play with a lot
of baby and toddler toys.""",

"""The layout of the hours are a little tricky.
The print on each page is light grey.
Trying to journal diapers, bottles and meds during the night
is difficult.
I could not use the space to make more notes."""

]

# --------------------------------------------------------
# Text Preprocessing
# --------------------------------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z ]',' ',text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

train_clean = [clean_text(i) for i in train_reviews]
test_clean = [clean_text(i) for i in test_reviews]

# --------------------------------------------------------
# TF-IDF Vectorizer
# --------------------------------------------------------

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(train_clean)

X_test = vectorizer.transform(test_clean)

# --------------------------------------------------------
# Vocabulary (Dictionary)
# --------------------------------------------------------

vocabulary = vectorizer.get_feature_names_out()

print("="*60)
print("VOCABULARY")
print("="*60)

for i, word in enumerate(vocabulary,1):
    print(f"{i:3d}. {word}")

print("\nVocabulary Size :",len(vocabulary))

# Save dictionary

with open("vocabulary.txt","w") as f:

    for word in vocabulary:
        f.write(word+"\n")

print("\nVocabulary saved as vocabulary.txt")

# --------------------------------------------------------
# Document-Term Matrix
# --------------------------------------------------------

dtm = pd.DataFrame(
    X_train.toarray(),
    columns=vocabulary
)

print("\nDocument Term Matrix\n")
print(dtm)

dtm.to_csv("document_term_matrix.csv",index=False)

print("\nDocument-Term Matrix saved.")

# --------------------------------------------------------
# Train Naive Bayes
# --------------------------------------------------------

model = MultinomialNB()

model.fit(X_train,train_labels)

# --------------------------------------------------------
# Training Accuracy
# --------------------------------------------------------

train_pred = model.predict(X_train)

accuracy = accuracy_score(train_labels,train_pred)

print("\nTraining Accuracy :",round(accuracy*100,2),"%")

# --------------------------------------------------------
# Prediction
# --------------------------------------------------------

predictions = model.predict(X_test)

probability = model.predict_proba(X_test)

label = {

1:"Positive",

0:"Negative"

}

print("\n"+"="*60)
print("PREDICTION RESULTS")
print("="*60)

for i,review in enumerate(test_reviews):

    print("\nReview",i+1)

    print("-"*60)

    print(review)

    print("\nPredicted Class :",predictions[i])

    print("Predicted Label :",label[predictions[i]])

    print("Positive Probability :",
          round(probability[i][1],3))

    print("Negative Probability :",
          round(probability[i][0],3))

# --------------------------------------------------------
# Save Results
# --------------------------------------------------------

import os

os.makedirs("Results", exist_ok=True)

# --------------------------------------------------------
# Save Vocabulary
# --------------------------------------------------------

with open("Results/vocabulary.txt", "w", encoding="utf-8") as f:
    for word in vocabulary:
        f.write(word + "\n")

print("Vocabulary saved successfully.")

# --------------------------------------------------------
# Save Document-Term Matrix
# --------------------------------------------------------

dtm.to_csv("Results/document_term_matrix.csv", index=False)

print("Document-Term Matrix saved successfully.")

# --------------------------------------------------------
# Save Classification Results
# --------------------------------------------------------

results_df = pd.DataFrame({

    "Review": test_reviews,

    "Predicted_Class": predictions,

    "Predicted_Label": [
        "Positive" if p == 1 else "Negative"
        for p in predictions
    ],

    "Positive_Probability": probability[:,1],

    "Negative_Probability": probability[:,0]

})

results_df.to_csv