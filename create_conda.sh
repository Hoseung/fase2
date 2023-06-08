#!/bin/bash
conda create -n bbs
conda activate bbs
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install pybind11 scipy scikit-learn tqdm pandas matplotlib
