# image_classification_trash

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A project repository containing all codebase works of image classification

## Project Organization

This structure is developed from `cookiecutter`. 

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for image_classification_trash
│                         and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── image_classification_trash                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes image_classification_trash a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```

--------

## Image Classification on Trash

This project is development is conducted with the purpose of demonstrating the entire lifecycle of deep learning with adherance to software development practices. Few steps such as automation with GitHub Action, model tracking and versioning with Weights and Biases are implemented to further promote smooth transitions and flexibility. In this case, deep learning to answer multiclass classification is chosen to properly recognize types of trashes with machine learning engineering practices.

## 1) Introduction
### Problem Context

I came across the idea of making better use of pretrained Keras models that would see itself being able to learn and recognize variational patterns of animals under the same class, particularly class of bird. Acknowledging my liking of bird, I took a decision to put my machine learning engineering ability and knowledge into building project that would share meaningful purpose that machine learning could hope to achieve.

Our focus is set on utilizing a pretrained model ResNet18 to capture and recognize high-level features of six categories: cardboard, glass, metal, paper, plastic, and trash.

ResNet is chosen as a convolutional base and a customised fully-connected layer is placed on the top of this base. This layer is tasked to give output based on probability generated from a `softmax` output layer. Confusion matrix is used for measuring quality.

### Dataset Image 

A collection of trash iamges are available for download from https://huggingface.co/datasets/garythung/trashnet. This set comes in a large dictionary packed with Pillow images at the large size, prompting us to  of modification and file organization so that PyTorch vision models could be able to update weights from images fed from input pipelines.

### Project Description

Sequences of executions begins from image preparation and initial image visualization, deep learning building, training, and assessing performance; these are covered in `Modelling.ipynb`. Trained models, resides in directory 'models/ResNet18', goes through conversion process in which its structures change from `assets`, `variables`, and `savedmodel.pb` to TFlite format and SavedModel suitable for sending prediction in respond to incoming requests across REST API or running docker images.

### Files

- `READNE.md`: A full description of the project for reader to gain a greater picture of this project.
- `data`: The collection of bird images in jpeg format. Breakdowns of each of their image directories are `train`, `val`, and `test`.
- `bird_classification.ipynb` : A jupyter notebook that covers steps from image preparation and analysis to prediction against `test` images. Also included in this file: converting ML to `.tflite` with TFLite along with inference testing.
- `lambda_function.py`: A python app that serves predictive function to respond to requests across AWS Lambda.
- `test_serverless.py`: A python file for testing response of `lambda_function.py`.
- `serverless-bird.dockerfile`: A dockerfile for building docker image containing functions defined from `lambda_function.py`.
- `tf-serving-connect-sequential-model.ipynb`: A jupyter notebook that covers testing on models specifically tailored for TensorFlow serving.
- `gateway_efficient_net.py`: A flask app that makes use of trained EfficientNet in container to make inference of incoming image URL input.
- `test_efficient-net-serving.py`: A python file for testing response of `gateway_efficient_net.py`.
- `image-model.dockerfile`: A dockerfile for building docker image containing trained EfficientNet stored in `efficient-net-dir` and run it with TensorFlow Serving.
- `image-gateway.dockerfile`: A dockerfile for building docker image containing application built from `gateway_efficient_net.py`.
- `docker-compose.yaml`: A docker compose that coordinate and run two images `gateway` and `model` simultaneously. 
- `Pipfile`: A Pipfile for collection of libraries and modules dependencies.
- `list_urls_bird.txt`: A text file serves as a storage of collections of URLs directed to birds.
- `model_seq.h5`: A h5 file that host weights of trained sequential model.
- `model_efficient_net.h5`: A h5 file that host weights of trained EfficientNet model.

### Environment

`Pipenv` is employed for this project. 

Steps to activate pipenv and install required libraries:
```
pipenv shell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
``` 

