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
git clone https://github.com/jeremiedbb/scikit-learn.git ~/scikit-learn

# tmp: benchmark suite not merged in master yet.
pushd scikit-learn
git checkout benchmark-bot
pushd asv_benchmarks
# pushd scikit-learn/asv_benchmarks

# Get the short hash of the last commit
COMMIT_TO_BENCH=$(git rev-parse HEAD)
COMMIT_TO_BENCH=${COMMIT_TO_BENCH:0:8}

# We'll save the fitted estimators for the next run
export SKLBENCH_SAVE_ESTIMATORS=true

# Retrieve the fitter estimators from last run
if [[ -d ${HOME}/sklearn-benchmark/last ]]; then
    export SKLBENCH_BASE_COMMIT=$(ls ${HOME}/sklearn-benchmark/last/)
    # Fast exit if no new commit since last run
    if [[ $COMMIT_TO_BENCH == $SKLBENCH_BASE_COMMIT ]]; then
        exit
    fi
    mkdir benchmarks/cache
    mkdir benchmarks/cache/estimators
    mv ${HOME}/sklearn-benchmark/last/$SKLBENCH_BASE_COMMIT benchmarks/cache/estimators/.
else
    # First run.
    mkdir ${HOME}/sklearn-benchmark/last
fi

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

# Create the .asv-machine.json file to avoid manual confirmation
cat <<EOT >> ${HOME}/.asv-machine.json
{
    "sklearn-benchmark": {
        "arch": "x86_64",
        "cpu": "Intel Core Processor (Haswell, no TSX)",
        "machine": "sklearn-benchmark",
        "num_cpu": "8",
        "os": "Linux 4.15.0-20-generic",
        "ram": "16424684"
    },
    "version": 1
}
EOT

# Run the benchmarks
asv run -b KMeansBenchmark $COMMIT_TO_BENCH^!

# Generate html
asv publish

# Move to sklearn-benchmark/ to commit the new result
popd
popd
pushd sklearn-benchmark/
cp -r ${HOME}/scikit-learn/asv_benchmarks/results/ .
cp -r ${HOME}/scikit-learn/asv_benchmarks/benchmarks/cache/estimators/$COMMIT_TO_BENCH last/.
git add .
git commit -m 'new result'
git push origin master

git checkout gh-pages
cp -r ${HOME}/scikit-learn/asv_benchmarks/html/* .
git add .
git commit -m 'new result'
git push origin gh-pages
