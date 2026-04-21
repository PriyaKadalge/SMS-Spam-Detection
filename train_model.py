import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = [
    "free entry win cash now",
    "urgent claim your prize",
    "meeting at 5 pm",
    "let us go tomorrow"
]

labels = [1, 1, 0, 0]

# create vectorizer
tfidf = TfidfVectorizer()

# fit vectorizer
X = tfidf.fit_transform(texts)

# train model
model = MultinomialNB()
model.fit(X, labels)

# save files
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Files created successfully")
