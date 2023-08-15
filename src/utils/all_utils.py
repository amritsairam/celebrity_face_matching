import yaml
import os
import logging
from bing_image_downloader import downloader


def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)

    return content


def create_directory(dirs: list):
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Directory is created at {dir_path}")


def allowed_file(filename):
    allowed_extensions = {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
