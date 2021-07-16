# Collection of Chapter 2 Terminal Commands

### Setting Up

```
pwd
cd /opt
ls
sudo mkdir -p ml
ls
sudo chown ubuntu:ubuntu ml
cd ml
mkdir -p input/config
mkdir -p input/data/train
mkdir -p output/failure
mkdir -p model

sudo apt install tree
tree


mkdir -p /home/ubuntu/environment/opt
cd /home/ubuntu/environment/opt
mkdir -p ml-python ml-r

sudo ln -s /opt/ml  /home/ubuntu/environment/opt/ml

<click opt directory in the Cloud9 sidebar to expand and see the tree structure>
<create hyperparameters.json>
<create inputdataconfig.json>

cd /opt/ml/input/data/train
S3_BUCKET="sagemaker-cookbook-bucket"
PREFIX="chapter01"

aws s3 cp s3://$S3_BUCKET/$PREFIX/input/training_data.csv training_data.csv


sudo apt-get -y update
sudo apt-get install -y --no-install-recommends wget
sudo apt-get install -y --no-install-recommends r-base
sudo apt-get install -y --no-install-recommends r-base-dev
sudo apt-get install -y --no-install-recommends ca-certificates

sudo apt install -y jq
```

### Running Python train script

```
python3 --version
    
pip3 install pandas
pip3 install sklearn

cd /home/ubuntu/environment/opt/ml-python
chmod +x train
./train

```

### Running the Python serve script

```
pip3 install flask

cd /home/ubuntu/environment/opt/ml-python
chmod +x serve
./serve

SERVE_IP=localhost
curl http://$SERVE_IP:8080/ping
curl -d "1" -X POST http://$SERVE_IP:8080/invocations
```

### Building and Testing the Custom Python Algorithm Container Image

```
cd /home/ubuntu/environment/opt/ml-python
IMAGE_NAME=chapter02python
TAG=1
docker build --no-cache -t $IMAGE_NAME:$TAG .
docker run --name pytrain --rm -v /opt/ml:/opt/ml $IMAGE_NAME:$TAG train

docker run --name pyserve --rm -v /opt/ml:/opt/ml $IMAGE_NAME:$TAG serve

<open new terminal tab>
SERVE_IP=$(docker network inspect bridge | jq -r ".[0].Containers[].IPv4Address" | awk -F/ '{print $1}')
curl http://$SERVE_IP:8080/ping
curl -d "1" -X POST http://$SERVE_IP:8080/invocations
```

### Running the R train script

```
R --version
sudo apt-get install -y --no-install-recommends libcurl4-openssl-dev
sudo apt-get install -y --no-install-recommends libsodium-dev
sudo R -e "install.packages('rjson',repos='https://cloud.r-project.org')"
sudo R -e "install.packages('here',repos='https://cloud.r-project.org')"
sudo R -e "install.packages('plumber',repos='https://cloud.r-project.org')"


cd /home/ubuntu/environment/opt/ml-r
chmod +x train
./train
```

### Running the R serve script

```
cd /home/ubuntu/environment/opt/ml-r
chmod +x serve
./serve

<open new browser tab>

SERVE_IP=localhost
curl http://$SERVE_IP:8080/ping
curl -d "1" -X POST http://$SERVE_IP:8080/invocations
```

### Building and Testing the Custom R Algorithm Container Image

```
cd /home/ubuntu/environment/opt/ml-r
IMAGE_NAME=chap02_r
TAG=1

docker build --no-cache -t $IMAGE_NAME:$TAG .

docker run --name rtrain --rm -v /opt/ml:/opt/ml $IMAGE_NAME:$TAG train

docker run --name rserve --rm -v /opt/ml:/opt/ml $IMAGE_NAME:$TAG serve

<open new terminal tab>

SERVE_IP=$(docker network inspect bridge | jq -r ".[0].Containers[].IPv4Address" | awk -F/ '{print $1}')
echo $SERVE_IP
curl http://$SERVE_IP:8080/ping
```

<img align="left" width="130" src="https://raw.githubusercontent.com/PacktPublishing/Amazon-SageMaker-Cookbook/master/Extra/cover-small-padded.png"/>

This file contains the commands to help readers work through one of the recipes of the book [Machine Learning with Amazon SageMaker Cookbook: 80 proven recipes for data scientists and developers to perform ML experiments and deployments](https://www.amazon.com/Machine-Learning-Amazon-SageMaker-Cookbook/dp/1800567030)