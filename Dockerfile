FROM ubuntu:20.04
MAINTAINER David Löfstrand <davidlofstrand@gmail.com>
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install git -y
RUN apt-get install gcc-10 -y
RUN apt-get install g++-10 -y
RUN apt-get install cmake -y
# below are convenience for opening a shell terminal in the docker image
RUN apt-get install vim -y
RUN apt-get install curl -y
RUN curl -L https://raw.githubusercontent.com/docker/compose/1.29.2/contrib/completion/bash/docker-compose -o /etc/bash_completion.d/docker-compose

#RUN alias gcc=/usr/bin/gcc-10
#RUN alias g++=/usr/bin/g++-10

#COPY somefolder /destfolder
#COPY some.file  /folder/some.file

