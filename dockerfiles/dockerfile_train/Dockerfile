# FROM tensorflow/tensorflow
FROM tensorflow/tensorflow:latest-gpu
# FROM python:3.9-slim-buster

RUN pip3 install --upgrade pip
# RUN pip3 install awscli --upgrade

RUN mkdir -p /home/docker-test
RUN mkdir -p /home/docker-test/output/saved_models
RUN mkdir -p /home/docker-test/config
RUN mkdir -p /home/docker-test/data
RUN mkdir -p /home/docker-test/src/modules

WORKDIR /home/docker-test

COPY requirements-train.txt requirements-train.txt
COPY config/config-train.json config/config-train.json
COPY src/__init__.py src/__init__.py
COPY src/train_worker.py src/train_worker.py
COPY src/modules/__init__.py src/modules/__init__.py
COPY src/modules/botoHelpers.py src/modules/botoHelpers.py
COPY src/modules/trainHelpers.py src/modules/trainHelpers.py

RUN pip3 install -r requirements-train.txt

WORKDIR /home/docker-test/src

CMD ["python3"]
