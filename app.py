import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

##Load the model
clsmodel=pickle.load(open('clsmodel.pkl','rb'))
Diseases=pickle.load(open('Plant Diseases.pkl','rb'))
sc=pickle.load(open('scaling.pkl','rb'))

@app.route("/images_mat")
def diseases():
    return render_template('materological_conditions/images_mat.html')

@app.route("/home")
def home():
    return render_template('materological_conditions/home.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    
    data=request.json['data'] 
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=sc.transform(np.array(list(data.values())).reshape(1,-1))
    output=clsmodel.predict(new_data)
    print(output[0])
    return jsonify(str(output[0]))

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=sc.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=clsmodel.predict(final_input)[0] 

    
    if(int(output)==0):
        prediction = "apple"
    elif(int(output)==1):
        prediction = "banana"
    elif(int(output)==2):
        prediction = "blackgram"
    elif(int(output)==3):
        prediction = "chickpea"
    elif(int(output)==4):
        prediction = "coconut"
    elif(int(output)==5):
        prediction = "coffee"
    elif(int(output)==6):
        prediction = "cotton"
    elif(int(output)==7):
        prediction = "grapes"
    elif(int(output)==8):
        prediction = "jute"
    elif(int(output)==9):
        prediction = "kidneybeans"
    elif(int(output)==10):
        prediction = "lentil"
    elif(int(output)==11):
        prediction = "maize"
    elif(int(output)==12):
        prediction = "mango"
    elif(int(output)==13):
        prediction = "mothbeans"
    elif(int(output)==14):
        prediction = "mulberry"
    elif(int(output)==15):
        prediction = "mungbean"
    elif(int(output)==16):
        prediction = "muskmelon"
    elif(int(output)==17):
        prediction = "orange"
    elif(int(output)==18):
        prediction = "papaya"
    elif(int(output)==19):
        prediction = "pigeonpeas"
    elif(int(output)==20):
        prediction = "pomegranate"
    elif(int(output)==21):
        prediction = "potato"
    elif(int(output)==22):
        prediction = "ragi"
    elif(int(output)==23):
        prediction = "rice"
    elif(int(output)==24):
        prediction = "watermelon"                                                                

    # return render_template('materological_conditions/home.html',prediction_text="The prediction is {}".format(prediction))
    return jsonify(prediction_text="The prediction is {}".format(prediction));

@app.route('/predict_diseases',methods=['POST'])
def predict_diseases():
    '''
    For Rendering results on html gui'''

    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    output=Diseases.predict(final_features) 

    if(int(output)==0):
        prediction = "Blast"
    elif(int(output)==1):
        prediction = "Hispa"
    elif(int(output)==2):
        prediction = "False Smut"
    elif(int(output)==3):
        prediction = "Stem Rot"    


    return render_template('materological_conditions/images_mat.html',prediction_text=prediction)

if __name__=="__main__":
    app.run(debug=True)

