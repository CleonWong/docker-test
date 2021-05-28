FROM python:3.9-slim-buster

RUN pip3 install --upgrade pip
# RUN pip3 install awscli --upgrade

RUN mkdir -p /home/docker-test
RUN mkdir -p /home/docker-test/config
RUN mkdir -p /home/docker-test/data
RUN mkdir -p /home/docker-test/src/modules

WORKDIR /home/docker-test

# RUN aws s3 cp s3://cleon-docker-test/requirements.txt requirements.txt
# RUN aws s3 cp s3://cleon-docker-test/config/config.json config/config.json
# RUN aws s3 cp s3://cleon-docker-test/src/__init__.py src/__init__.py
# RUN aws s3 cp s3://cleon-docker-test/src/neuralNetwork.py src/neuralNetwork.py
# RUN aws s3 cp s3://cleon-docker-test/src/modules/__init__.py src/modules/__init__.py
# RUN aws s3 cp s3://cleon-docker-test/src/modules/generateData.py src/modules/generateData.py

COPY requirements-data.txt requirements-data.txt
COPY config/config-data.json config/config-data.json
COPY src/__init__.py src/__init__.py
COPY src/data_worker.py src/data_worker.py
COPY src/modules/__init__.py src/modules/__init__.py
COPY src/modules/generateData.py src/modules/generateData.py
COPY src/modules/botoHelpers.py src/modules/botoHelpers.py

RUN pip3 install -r requirements-data.txt

WORKDIR /home/docker-test/src

CMD ["python3"]
