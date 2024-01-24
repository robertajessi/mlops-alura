
from flask import Flask, request,jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

colunas=['tamanho','ano','garagem']
model=pickle.load(open('../../models/modelo.sav','rb'))
# criando nossa app
app = Flask(__name__)
#autentificacao
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD']=os.environ.get('BASIC_AUTH_PASSWORD')

basic_aut = BasicAuth(app)

# rota
@app.route('/')

def home():
    return "Minha primeira API"

@app.route('/sentimento/<frase>')
#colocar autotificacao de cima
@basic_aut.required
def sentimento(frase):
    tb=TextBlob(frase)
    tb_en = tb.translate(from_lang='pt',to='en')
    polaridade = tb_en.sentiment.polarity
    return "polaridade: {}".format(polaridade)

@app.route('/predicao/', methods=['POST'])
@basic_aut.required
def predicao():
    df = request.get_json()
    dados_input = [df[col] for col in colunas]
    preco = model.predict([dados_input])
    return jsonify(preco=preco[0])
app.run(debug=True,host='0.0.0.0')

