from flask import Flask,render_template,url_for,request
import pickle
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')

app = Flask(__name__)

@app.route('/')
def index():
    # return '<h1>Hello</h1>'
    return render_template('home.html')

@app.route('/predict',methods=["POST"])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        model = pickle.load(open('prediction.pkl','rb'))
        
        # cleaning the text
        message = clean_text(message)
        
        cv = pickle.load(open('transform.pkl','rb'))
        vect = cv.transform([message]).toarray()
        prediction = model.predict(vect)
        return render_template('result.html',prediction = prediction)

def clean_text(text):
    nltk.download('wordnet')
    wl = WordNetLemmatizer()
    text = re.sub('[^a-zA-Z]',' ',text)
    text = text.lower()
    text = text.split()
    text = [wl.lemmatize(t) for t in text if t not in stopwords.words('english')]
    return(' '.join(text))

if __name__ == '__main__':
	app.run(debug=True)