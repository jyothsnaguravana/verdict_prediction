from flask import Flask, render_template, request, redirect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
import tensorflow as tf
import lime
from lime.lime_text import LimeTextExplainer
from tensorflow import keras
from keras import layers
import random
import nltk
from nltk.stem import WordNetLemmatizer
from flask_mysqldb import MySQL
import mysql.connector
nltk.download('punkt')
nltk.download('wordnet')


intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('court.h5')
lemmatizer = WordNetLemmatizer()

# punishment or article violation##################################################################### 

def db_connect(app):
    connection = mysql.connector.connect(host = 'localhost', port = 3306, user = 'root', password = '', database = 'docket')
    return connection

def insert_data(name, lawyerName, address, caseType, contact):
    cursor = conn.cursor()
    cursor.execute('INSERT INTO `docket`.`data` (`name`, `lawyerName`, `address`, `caseType`, `contact`) VALUES (%s, %s, %s, %s, %s);',(name, lawyerName, address, caseType, contact))
    conn.commit()
    cursor.close()

def get_data():
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM `docket`.`data`;')
    data = cursor.fetchall()
    cursor.close()
    return data

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word)  for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words= clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda  x:x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list,intents_json):
    tag= intents_list[0]['intent']
    list_of_intents =intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

#################################################################################################### 

def initiate(temp):
    text_vectorization_bi_tfidf = keras.layers.TextVectorization(
        ngrams=2,
        max_tokens=20000,
        output_mode = "tf_idf"
    ) 
    X_train = pd.read_csv('X_train.csv')
    text_vectorization_bi_tfidf.adapt(X_train)   
    model_bi_tfidf = tf.keras.models.load_model("model_bi_tfidf.h5")
    class_names=['petitioner_winning','respondent_winning']
    explainer=LimeTextExplainer(class_names=class_names)

    def new_predict(text):
        vectorized = text_vectorization_bi_tfidf(text) 
        padded = keras.preprocessing.sequence.pad_sequences(vectorized, maxlen=20000,padding='post')
        pred = model_bi_tfidf.predict(padded) 
        pos_neg_preds = [] 
        for i in pred:
            temp=i[0] 
            pos_neg_preds.append(np.array([1-temp,temp])) 
        return np.array(pos_neg_preds)    
    
    explanation = explainer.explain_instance(temp[0], new_predict) 
    explanation_image = explanation.as_pyplot_figure()
    explanation_image.savefig('static/assets/explanation_image.png', bbox_inches='tight', pad_inches=0)
    return 'static/assets/explanation_image.png'
    


###################################################################################################

app = Flask(__name__)

#Connecting to the Database
conn = db_connect(app)

@app.route("/")
def index():
    return render_template('index.html', data = get_data())
    # return render_template('index.html', data = {})

@app.route("/predict", methods=["GET", "POST"])
def home():
    return render_template('predictVerdict.html')
    
@app.route("/predict1", methods=["GET", "POST"])
def predict1():
    return render_template('verdictPredict.html')

@app.route("/verdict", methods=["GET", "POST"])
def submit():
    res = ""
    if request.method=="POST":
        request_data = request.get_json()
        petitioner_name = request_data["petitioner_name"]
        respondent_name = request_data["respondent_name"]
        facts = request_data["facts"]
        facts = petitioner_name + " " + respondent_name + " " + facts  
        try:
            img = initiate([facts])
            ints = predict_class(facts)
            res = get_response(ints, intents)
            response = {"verdict": res, "image": img}
        except:
            res="Unable to make decision !!!"
            response = {"verdict": res, "image": "static/assets/invalid.jpg"}
    return response

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method=="POST":
        firstName = request.form.get("firstName")
        lastName = request.form.get("lastName")
        lawyerName = request.form.get("lawyerName")
        street = request.form.get("street")
        city = request.form.get("city")
        state = request.form.get("state")
        zip = request.form.get("zip")
        caseType = request.form.get("caseType")
        contact = request.form.get("contact")
        name = firstName + " " + lastName
        address = street + ", " + city + ", " + state + ", " + zip
        insert_data(name, lawyerName, address, caseType, contact)
        return render_template('index.html', status="success")
    return render_template('index.html', status="failed")

if __name__=="__main__":
    # app.run(debug = True, host='192.168.137.129', port=5000)
    # app.run(debug = True)
    app.run()