import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# sample training data
data = [
    "free entry win money now",
    "call me tonight",
    "urgent claim prize",
    "let us meet tomorrow"
]

# fit vectorizer
tfidf = TfidfVectorizer()
tfidf.fit(data)

# save again
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))

print("vectorizer fixed successfully")
