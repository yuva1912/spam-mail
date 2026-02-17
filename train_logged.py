import os, traceback
log_path = os.path.join(os.path.dirname(__file__), 'train_output.txt')
with open(log_path, 'w', encoding='utf-8') as log:
    def L(msg=''):
        print(msg)
        log.write(str(msg)+'\n')
    try:
        L('START')
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import MultinomialNB
        import pickle
        p = os.path.join(os.path.dirname(__file__), 'spam.csv')
        L('reading csv at: '+p)
        df = pd.read_csv(p, encoding='latin-1')
        L('read rows:'+str(len(df)))
        L('columns:'+str(df.columns.tolist()))
        data = df[['v1','v2']]
        data.columns = ['label','message']
        L('labels unique before mapping:'+str(data['label'].unique()))
        data['label'] = data['label'].map({'ham':0,'spam':1})
        L('labels after mapping:'+str(data['label'].unique()))
        X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2)
        L('split sizes: train='+str(len(X_train))+' test='+str(len(X_test)))
        vectorizer = TfidfVectorizer(stop_words='english')
        X_train_vec = vectorizer.fit_transform(X_train)
        L('X_train_vec shape:'+str(X_train_vec.shape))
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)
        out_model = os.path.join(os.path.dirname(__file__), 'spam_model.pkl')
        out_vect = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
        L('writing to:'+out_model+' and '+out_vect)
        pickle.dump(model, open(out_model, 'wb'))
        pickle.dump(vectorizer, open(out_vect, 'wb'))
        L('written files exist:'+str(os.path.exists(out_model))+' '+str(os.path.exists(out_vect)))
        L('DONE')
    except Exception as e:
        L('EXCEPTION')
        traceback.print_exc(file=log)
        L(str(e))
