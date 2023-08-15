from flask import Flask, render_template, request
import cv2
from src.utils.all_utils import allowed_file
import os
from src.utils.all_utils import read_yaml, create_directory
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import numpy as np
import face_recognition

config = read_yaml('config/config.yaml')
params = read_yaml('params.yaml')

artifacts = config['artifacts']
artifacts_dir = artifacts['artifacts_dir']
static_dir = artifacts['static_dir']

# Upload
upload_image_dir = artifacts['upload_image_dir']
uploadn_path = os.path.join(static_dir, upload_image_dir)

# Pickle format data directory
pickle_format_data_dir = artifacts['pickle_format_data_dir']
img_pickle_file_name = artifacts['img_pickle_file_name']

raw_local_dir_path = os.path.join(artifacts_dir, pickle_format_data_dir)
pickle_file = os.path.join(raw_local_dir_path, img_pickle_file_name)

# Feature path
feature_extraction_dir = artifacts['feature_extraction_dir']
extracted_features_name = artifacts['extracted_features_name']

feature_extraction_path = os.path.join(artifacts_dir, feature_extraction_dir)
features_name = os.path.join(feature_extraction_path, extracted_features_name)

feature_list = pickle.load(open(features_name, 'rb'))
filenames = pickle.load(open(pickle_file, 'rb'))

def recommend(feature_list, features):
    similarity = []
    for i in range(len(feature_list)):
        if features is not None and feature_list[i] is not None:
            similarity.append(cosine_similarity([features], [feature_list[i]])[0][0])
        else:
            similarity.append(0.0)  # Set similarity to 0 for cases with missing features

    index_pos = sorted(enumerate(similarity), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos


def extract_features(image_path):
    # Load the image using face_recognition
    img = face_recognition.load_image_file(image_path)
    
    # Find all face locations in the image
    face_encodings = face_recognition.face_encodings(img)
    
    if len(face_encodings) > 0:
        features = face_encodings[0]  # Store the first face encoding
        return features
    else:
        return None

def save_uploaded_image(uploaded_image):
    try:
        create_directory(dirs=[uploadn_path])

        image_path = os.path.join(uploadn_path, uploaded_image.filename)
        uploaded_image.save(image_path)

        return image_path
    except Exception as e:
        print("Error saving uploaded image:", e)
        return ""











