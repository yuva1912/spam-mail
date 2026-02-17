import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import os
print('CWD:', os.getcwd())
print('exists spam.csv:', os.path.exists('spam.csv'))
df = pd.read_csv('spam.csv', encoding='latin-1')
print('columns:', df.columns.tolist())
try:
    data = df[['v1','v2']]
except Exception as e:
    print('select error', e)
    raise
print('data shape', data.shape)

data.columns = ['label','message']
print('mapped labels unique before mapping:', data['label'].unique())
data['label'] = data['label'].map({'ham':0,'spam':1})
print('labels after mapping unique:', data['label'].unique())

X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2)
print('train/test sizes', len(X_train), len(X_test))

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
print('X_train_vec shape', X_train_vec.shape)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

out_model = os.path.abspath('spam_model.pkl')
out_vect = os.path.abspath('vectorizer.pkl')
print('will write', out_model, out_vect)
pickle.dump(model, open(out_model, 'wb'))
pickle.dump(vectorizer, open(out_vect, 'wb'))
print('written files exist:', os.path.exists(out_model), os.path.exists(out_vect))
