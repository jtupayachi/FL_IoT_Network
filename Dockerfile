# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:2.12.0rc0
# tensorflow/tensorflow:latest

#-gpu
RUN apt-get update && apt-get install -y git
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install flwr
RUN pip3 install pandas
RUN pip3 install numpy matplotlib scikit-learn
RUN yes |  apt update



RUN curl https://install.zerotier.com/ | bash 
# RUN /usr/sbin/zerotier-one -d 
#RUN /usr/sbin/zerotier-cli join c7c8172af153068f

# RUN pip3 install flwr[simulation]
# RUN pip3 install ray==1.11.1
# RUN apt-get install  python3-psutil


CMD ["cat", "/etc/os-release"]

RUN yes |  apt install iproute2
