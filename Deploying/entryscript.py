from sklearn.externals import joblib
import pandas as pd
import numpy as np

# Called when the deployed service starts
def init():
    global loaded_model

    # Get the path where the model(s) registered as the name 'sentiment' can be found.
    model_path = Model.get_model_path('LogisticRegression_Initial')
    # load models
    loaded_model=joblib.load(model_path+"/finalized_model.pkl")
    # model = load_model(model_path + '/model.h5')
    # w2v_model = Word2Vec.load(model_path + '/model.w2v')

    # with open(model_path + '/tokenizer.pkl','rb') as handle:
    #     tokenizer = pickle.load(handle)

    # with open(model_path + '/encoder.pkl','rb') as handle:
    #     encoder = pickle.load(handle)

# Handle requests to the service
def run(data):
    try:
        # data_test=pd.read_csv('testms.csv')
        # load the model from disk
        # filename = 'finalized_model.pkl'
        # loaded_model = joblib.load("C:/Users/Mohith/Desktop/Datasets/models/"+filename)

        # Pick out the text property of the JSON request.
        # This expects a request in the form of {"text": "some text to score for sentiment"}
        data = json.loads(data)

        one_array=np.array(data)
        one_array=np.reshape(one_array,(-1,1),order='C').T
        # one_value=logreg.predict(np.array([one_array[0][0]]).reshape((-1,1),order='C'))

        # data['mental_health_interview']=data['mental_health_interview'].map({'Yes':1,'No':0,'Maybe':0.5})
        # prediction = predict(data)
        result = loaded_model.predict(np.array([one_array[0][0]]).reshape((-1,1),order='C'))
        #Return prediction
        return result
    except Exception as e:
        error = str(e)
        return error
