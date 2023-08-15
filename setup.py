
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",
    version="0.0.1",
    author="amrit sairam",
    description="A small package for Which Bollywood Celebrity You look like? Deep Learning Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amritsairam/Which-Bollywood-Celebrity-You-look-like",
    author_email="amritsairam@gmail.com",
    packages=["src"],
    python_requires=">=3.7",
    install_requires=[
        'face_recognition',
        'tensorflow',
        'keras',
        'keras-vggface',
        'keras_applications',
        'PyYAML',
        'tqdm',
        'scikit-learn',
        'streamlit',
        'bing-image-downloader'
    ]
)