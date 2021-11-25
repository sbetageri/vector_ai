FROM nvidia/cuda:10.2-base-ubuntu18.04

RUN apt-get update && \
	apt-get install -y git && \
	apt-get install vim -y && \
	apt-get install python3-pip -y

RUN pip3 install pandas 

RUN pip3 install numpy 

RUN pip3 install jupyterlab 

RUN pip3 install --upgrade pip

RUN pip3 install --upgrade Pillow

RUN pip3 install torch torchvision 

RUN pip3 install tqdm

WORKDIR /root/code
