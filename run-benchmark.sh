#!/bin/bash

# Run the scikit-learn asv benchmark suite against master and commit the new
# result to https://github.com/scikit-learn-inria-fondation/sklearn-benchmark


# Add ssh key to be able to push to github
eval "$(ssh-agent -s)"
ssh-add ${HOME}/.ssh/sklbench
ssh-keyscan github.com >> ${HOME}/.ssh/known_hosts

# Config git
git config --global user.email "sklearn.benchmark.bot@gmail.com"
git config --global user.name "sklearn-benchmark-bot"

# Clone the sklearn-benchmark repo which stores the benchmark results and hosts
# the benchmarks website https://scikit-learn-inria-fondation.github.io/sklearn-benchmark/
git clone git@github.com:scikit-learn-inria-fondation/sklearn-benchmark.git

# Clone scikit-learn. The benchmark suite is the asv_benchmarks/ directory
git clone https://github.com/scikit-learn/scikit-learn.git ~/scikit-learn

pushd scikit-learn/asv_benchmarks

# Get the short hash of the last commit
COMMIT_TO_BENCH=$(git rev-parse HEAD)
COMMIT_TO_BENCH=${COMMIT_TO_BENCH:0:8}

# Get all previous results to regenerate the html
if [[ -d ${HOME}/sklearn-benchmark/results ]]; then
   cp -r ${HOME}/sklearn-benchmark/results .
fi

# Install gcc
sudo apt-get install --assume-yes gcc
sudo apt-get install --assume-yes g++
sudo apt-get install --assume-yes make

# install Conda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${HOME}/miniconda.sh
bash ${HOME}/miniconda.sh -b -p ${HOME}/miniconda
PATH=${HOME}/miniconda/bin:${PATH}

# Create a conda env and install asv
conda create -y -n skl_benchmark python=3.8
source ${HOME}/miniconda/etc/profile.d/conda.sh
conda activate skl_benchmark
pip install git+https://github.com/airspeed-velocity/asv

# Setup asv
asv machine --yes

# Run the benchmarks
asv run $COMMIT_TO_BENCH^!

# Generate html
asv publish

# Move to sklearn-benchmark/ to commit the new result
popd
pushd sklearn-benchmark/
cp -r ${HOME}/scikit-learn/asv_benchmarks/results/ .
git add .
git commit -m 'new result'
git push origin master

git checkout gh-pages
cp -r ${HOME}/scikit-learn/asv_benchmarks/html/* .
git add .
git commit -m 'new result'
git push origin gh-pages
