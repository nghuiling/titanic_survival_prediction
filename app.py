from flask import Flask, render_template,request
import pickle
import numpy as np


app = Flask(__name__)


model = pickle.load(open('finalized_model.p','rb'))



def model_predict(data):
    probas = model.predict_proba(data)[0]
    max = np.argmax(probas)
    return int(max)


@app.route('/', methods=['GET','POST'])
def home():
	return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_feature = [np.array(int_features)]

    Age = final_feature[0][0]
    Sex = final_feature[0][1]
    Class = final_feature[0][2]
    Embarked = final_feature[0][3]
    Age_Class = Age*Class


    final_data = [Class, Sex, Age, Embarked, Age_Class]
    data  = np.array(final_data).reshape(-1, 5)


    prediction = model_predict(data)


    result = ['Did not Survive','Survived']

    age1 = ['<= 16', '17-32', '33-48' , '49-64','>= 65']
    sex1 = ['Male','Female']
    class1 = ['', 'First Class','Second Class','Third Class']
    embarked1 = ['Southampton','Cherbourg','Queenstown']

    output1 =  age1[Age], sex1[Sex], class1[Class], embarked1[Embarked]
    output2 = result[prediction]


    return render_template('index.html',prediction_text1='Input :  {}'.format(output1),prediction_text2='Result :  {}'.format(output2))


if __name__ == '__main__':
   app.run(debug=True)