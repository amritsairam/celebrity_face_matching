from src.utils.all_utils import read_yaml, create_directory
import argparse
import os
import logging
import face_recognition
import numpy as np
import pickle
from tqdm import tqdm

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'running_logs.log'), level=logging.INFO, format=logging_str, filemode="a")

def feature_extractor(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config['artifacts']
    artifacts_dir = artifacts['artifacts_dir']
    pickle_format_data_dir = artifacts['pickle_format_data_dir']
    img_pickle_file_name = artifacts['img_pickle_file_name']

    img_pickle_file = os.path.join(artifacts_dir, pickle_format_data_dir, img_pickle_file_name)
    filenames = pickle.load(open(img_pickle_file, 'rb'))

    feature_extraction_dir = artifacts['feature_extraction_dir']
    extracted_features_name = artifacts['extracted_features_name']
    feature_extraction_path = os.path.join(artifacts_dir, feature_extraction_dir)
    create_directory(dirs=[feature_extraction_path])

    features_name = os.path.join(feature_extraction_path, extracted_features_name)
    features = []

    for file in tqdm(filenames):
        img = face_recognition.load_image_file(file)
        face_encodings = face_recognition.face_encodings(img)
        
        if len(face_encodings) > 0:
            features.append(face_encodings[0])  # Store the first face encoding
        else:
            logging.warning(f"No face found in {file}")
            features.append(None)  # Add None for cases where no face is found

    pickle.dump(features, open(features_name, 'wb'))

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()
    
    try:
        logging.info(">>>>> stage_02 started")
        feature_extractor(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info("stage_02 completed!>>>>>")
    except Exception as e:
        logging.exception(e)
        raise e
