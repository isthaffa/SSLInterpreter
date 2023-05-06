from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input

from tensorflow.keras import models
from PIL import Image
import numpy as np
import flask
from flask_cors import CORS

import io

# Boot up production server with:
# gunicorn -b 127.0.0.1:5000 main:app

# Boot up development server with:
# py main.py

# Make request with curl (or programmatically with testRequest.py:
# curl -X POST -F image=@H2.jpg 'http://localhost:5000/predict'
# curl -X POST -F image=@H2.jpg 'http://35.198.151.110/predict'
# Gives Response:
# {"letterSent":"H","predictions":[["H",100.0],["C",0.0],["F",0.0]]}

app = flask.Flask(__name__)
CORS(app)
model = None

def load_model():
    global model
    model = models.load_model("./assets/models/Inceptionresnet.h5")


def prepare_image(image, target):

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    return image


@app.route("/", methods=["GET"])

def index():
    return "<p>Welcome to Sign Interpreter SSL API.<p>"

@app.route("/predict", methods=["POST"])

def predict():
    response = {'success': False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            image = prepare_image(image, target=(224, 224))

            predictions = model.predict(image)

            predictions = getTopPredictions(predictions[0])
            predictions = serialisePreds(predictions)
            response = {
                'success': True,
                'predictions': [{
                    'letter': predictions[0][0],
                    'confidence': predictions[0][1]},{
                    'letter': predictions[1][0],
                    'confidence': predictions[1][1]},{
                    'letter': predictions[2][0],
                    'confidence': predictions[2][1]
                    }]
            }

    return flask.jsonify(response)

def getTopPredictions(preds):
    class_labels ={'A': 0,
 'Aah': 1,
 'Ae': 2,
 'Aeh': 3,
 'Ah': 4,
 'Ee': 5,
 'Eeh': 6,
 'Ig': 7,
 'K': 8,
 'O': 9,
 'Ohh': 10,
 'T': 11,
 'Uh': 12,
 'Uhh': 13}
    # map preds index with probability to correct letter from dictonary
    sorted_indices = np.argsort(preds)
    top_three_indices = sorted_indices[-3:]
    label1 = list(class_labels.keys())[list(class_labels.values()).index(top_three_indices[0])]
    label2 = list(class_labels.keys())[list(class_labels.values()).index(top_three_indices[1])]
    label3 =list(class_labels.keys())[list(class_labels.values()).index(top_three_indices[2])]
    top_preds = [[label1,preds[top_three_indices[0]]],[label2,preds[top_three_indices[1]]], [label3,preds[top_three_indices[2]]]]
    return top_preds, preds

def serialisePreds(predictions):
    topPreds = []
    for i, pred in enumerate(predictions[0], start=0):
        letter, confidence = predictions[0][i]
        topPreds.append((letter, round(confidence*100, 8)))
    return topPreds

@app.errorhandler(500)
def internal_server_error(error):
    response = {'success': False}
    print(error)
    return flask.jsonify(response)

print("Loading Model...")
load_model() #Load model before apps run to prevent long loading time
print("Model Loaded. Server Running.")

if __name__ == "__main__":  #if running on development
    app.run(debug=True, threaded=False, host='0.0.0.0')

