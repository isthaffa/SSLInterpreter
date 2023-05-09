import os
import io
import mediapipe as mp
import cv2
import keyboard
from flask import Response

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
import time

from tensorflow.keras import models
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import flask
from flask_cors import CORS

# from my_functions import *


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

# camera = cv2.VideoCapture(0)
time.sleep(2.0)


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def draw_landmarks(image, results):
    mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
    mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)


def image_process(image, model):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results

def keypoint_extraction(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([lh, rh])

def action_detections(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, results

def drawing_image_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color=(242,172,188), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(149,249,100), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def extracting_image_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


'''
for ip camera use - rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' 
for local webcam use cv2.VideoCapture(0)
'''

actions = np.array([
    'ah',
      'aah',
      'aeh','ee','uh','a','ae','k','t','oh','o','eeh','ig'
      
    ])
letter={
"ah": "අ", "aah": "ආ", "aeh": "ඇ",  "ee": "ඉ", "eeh": "ඊ", "uh": "උ", "uhh": "ඌ", "a": "එ",
	    "ae": "ඒ", "o": "ඔ", "oh": "ඕ","k":"ක්","ig":"ග්","t":"ටී"
}

cnn_model = load_model('cnn')
sentence, keypoints = [' '], []

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

# def gen_frames():  
#     while True:
#         success, frame = camera.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
            
            
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


# def gen_frames():  
#     while True:
#         success, frame = camera.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
            
            
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/video_feed')
def video_feed():
    return Response(detect(), mimetype='multipart/x-mixed-replace; boundary=frame')

cap = cv2.VideoCapture(0)
def detect():
    sequence = []
    sentence = []
    threshold = 0.4
    
    mp_holistics = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    with mp_holistics.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # while cap.isOpened():
        while True:
            ret, frame = cap.read()

            image, results = action_detections(frame, holistic)
            print(results)
            drawing_image_landmarks(image, results)
            keypoints = extracting_image_keypoints(results)
            # sequence.insert(0, keypoints)
            # sequence = sequence[:10]
            
            if len(keypoints) == 10:
                keypoints = np.array(keypoints)
                prediction = cnn_model.predict(keypoints[np.newaxis, :, :])
                keypoints = []
                print(actions[np.argmax(prediction)])
                print('prediction')
                print(prediction)
                if np.amax(prediction) > 0.9:
                    if sentence[-1] != actions[np.argmax(prediction)]:
                        sentence.append( actions[np.argmax(prediction)])

            if len(sentence) > 7:
                sentence = sentence[-7:]
            
            if keyboard.is_pressed(' '):
                sentence = [' ']

            textsize = cv2.getTextSize(' '.join(sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_X_coord = (image.shape[1] - textsize[0]) // 2
            
            print(text_X_coord)

            # if len(sequence) == 30:
            #     res = cnn_model.predict(np.expand_dims(sequence, axis=0))[0]
            #     #             print(actions[np.argmax(res)])
            #     print(actions[np.argmax(res)])
            #     print(sentence)

            #     if res[np.argmax(res)] > threshold:
            #         if len(sentence) > 0:
            #             if actions[np.argmax(res)] != sentence[-1]:
            #                 sentence.append(actions[np.argmax(res)])
            #         else:
            #             sentence.append(actions[np.argmax(res)])

            # if len(sentence) > 5:
            #     sentence = sentence[-5:]

            # response.append(sentence)
            ret, buffer = cv2.imencode('.jpg', frame)
            image = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

# @app.route("/stream", methods=["GET"])
def stream():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera.")
        exit()

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
        while cap.isOpened():
            _, image = cap.read()
            results = image_process(image, holistic)
            draw_landmarks(image,results)
            keypoints.append(keypoint_extraction(results))

            if len(keypoints) == 10:
                keypoints = np.array(keypoints)
                prediction = cnn_model.predict(keypoints[np.newaxis, :, :])
                keypoints = []
                print(actions[np.argmax(prediction)])
                if np.amax(prediction) > 0.9:
                    if sentence[-1] != actions[np.argmax(prediction)]:
                        sentence.append( actions[np.argmax(prediction)])

            if len(sentence) > 7:
                sentence = sentence[-7:]
            
            if keyboard.is_pressed(' '):
                sentence = [' ']

            textsize = cv2.getTextSize(' '.join(sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_X_coord = (image.shape[1] - textsize[0]) // 2
                
            cv2.putText(image, ' '.join(sentence), (text_X_coord, 470), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
            cv2.imshow('Camera', image)
            
            cv2.waitKey(1)
            if cv2.getWindowProperty('Camera',cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()


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
    app.run(debug=True, threaded=False, host='0.0.0.0', port=5001)

