from flask import Flask
import flask
import tensorflow as tf
from keras.models import load_model
import numpy as np

app = Flask(__name__)

def init():
  global model
  model = load_model('assets/pima-indians-diabetes-model.h5')
  
@app.route('/predict')
def predict():
  X = np.array([1,85,66,29,0,26.6,0.351,31]).reshape(1, 8)
  y_prob = model.predict(X).tolist()
  y_pred_class = model.predict_classes(X).tolist()

  return 'Predicted class: ' + str(y_pred_class[0][0]) + ' with probability: ' + str(y_prob[0][0])

if __name__ == "__main__":
  init()
  app.run(debug=True)
