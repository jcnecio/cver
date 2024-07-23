#!/bin/bash

cwd=`pwd`
cd dockerenv
docker build -t run_env .
docker run -v $cwd:/app -it run_env bash 