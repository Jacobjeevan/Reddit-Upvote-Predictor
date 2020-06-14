#!/bin/bash

#Script to setup virtual environment and install all the dependencies.

virtualenv nlp_env
source nlp_env/bin/activate

ipython kernel install --user --name=nlp_env

pip install -r requirements.txt --user

