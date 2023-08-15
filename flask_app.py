from flask import Flask, render_template, request, Response
import cv2
from src.utils.all_utils import allowed_file
from flask_app_functions import save_uploaded_image, recommend, read_yaml, extract_features
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import numpy as np
import base64
import face_recognition

config = read_yaml('config/config.yaml')
params = read_yaml('params.yaml')

artifacts = config['artifacts']
artifacts_dir = artifacts['artifacts_dir']

app = Flask(__name__)

static_dir = artifacts['static_dir']
#upload
upload_image_dir = artifacts['upload_image_dir']  # the folder where uploaded images will be stored
uploadn_path = os.path.join(static_dir, upload_image_dir)

# Pickle format data directory
pickle_format_data_dir = artifacts['pickle_format_data_dir']
img_pickle_file_name = artifacts['img_pickle_file_name']

raw_local_dir_path = os.path.join(artifacts_dir, pickle_format_data_dir)
pickle_file = os.path.join(raw_local_dir_path, img_pickle_file_name)

feature_extraction_dir = artifacts['feature_extraction_dir']
extracted_features_name = artifacts['extracted_features_name']

feature_extraction_path = os.path.join(artifacts_dir, feature_extraction_dir)
features_name = os.path.join(feature_extraction_path, extracted_features_name)

feature_list = pickle.load(open(features_name, 'rb'))
filenames = pickle.load(open(pickle_file, 'rb'))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/home")
def homee():
    return render_template("index.html")

@app.route("/choose_computer", methods=["GET", "POST"])
def choose_computer():
    if request.method == "POST":
        uploaded_image = request.files["uploaded_image"]
        if uploaded_image and allowed_file(uploaded_image.filename):
            image_path = save_uploaded_image(uploaded_image)
            if image_path:
                features = extract_features(image_path)
                if features is not None:
                    index_pos = recommend(feature_list, features)
                    predicted_actor = " ".join(filenames[index_pos].split("/")[2].split(".")[0])
                    return render_template(
                        'choose_computer.html',
                        uploaded=True,
                        uploaded_image=uploaded_image.filename,
                        predicted_actor=predicted_actor,
                        actor_image=filenames[index_pos]
                    )

    return render_template("choose_computer.html", uploaded=False)

def generate_frames(camera):
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/choose_webcam')
def index():
    return render_template('choose_webcam.html')

@app.route('/video')
def video():
    camera = cv2.VideoCapture(0)  # Initialize the camera here
    return Response(generate_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    frame_name = request.form['frame_name']
    camera = cv2.VideoCapture(0)
    _, frame = camera.read()
    folder_path = uploadn_path
    os.makedirs(folder_path, exist_ok=True)

    image_path = os.path.join(folder_path, f'{frame_name}.jpg')
    cv2.imwrite(image_path, frame)

    features = extract_features(image_path)
    if features is not None:
        index_pos = recommend(feature_list, features)
        predicted_actor = " ".join(filenames[index_pos].split("/")[2].split(".")[0])
        predicted_actor_image = filenames[index_pos]
        _, buffer = cv2.imencode('.jpg', frame)
        captured_frame_base64 = base64.b64encode(buffer).decode('utf-8')
    else:
        predicted_actor = "Unknown"
        predicted_actor_image = ""
        camera.release()

    return render_template(
        'results.html',
        captured_frame=captured_frame_base64,
        predicted_actor=predicted_actor,
        predicted_actor_image=predicted_actor_image
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)







