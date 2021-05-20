FROM tensorflow/tensorflow

RUN mkdir -p /home/docker-test
RUN mkdir -p /home/docker-test/output/predictions
RUN mkdir -p /home/docker-test/output/saved_models
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

COPY requirements-pred.txt requirements-pred.txt
COPY config/config-pred.json config/config-pred.json
COPY src/__init__.py src/__init__.py
COPY src/pred_worker.py src/pred_worker.py
COPY src/modules/__init__.py src/modules/__init__.py
COPY src/modules/botoHelpers.py src/modules/botoHelpers.py

RUN pip3 install -r requirements-pred.txt

WORKDIR /home/docker-test/src

CMD ["python3"]
