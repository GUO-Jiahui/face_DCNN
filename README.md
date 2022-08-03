# Modeling naturalistic face processing in humans with deep convolutional neural networks

## Overview

This repository contains a python implementation of the key analyses in the manuscript by Guo Jiahui<sup>†\*</sup>, Ma Feilong<sup>†\*</sup>, Matteo Visconti di Oleggio Castello, Samuel A. Nastase, James V. Haxby, & M. Ida Gobbini <sup>\*</sup> (2021). https://doi.org/10.1101/2021.11.17.469009

## System Requirements

### Hardware requirements
A standard computer with enough RAM to support the in-memory operations. Demos should finish in seconds. 

### Software requirements

#### OS Requirements

Our code has been tested on macOS: Big Sur (11.6.8)

#### Python Dependencies
The code was writen in Python 3, Python packages required for running those analyses include `numpy`, `scipy`, and `rpy2`.

#### Other Package Dependencies
R package `vegan` is required for running the variance partitioning analysis.

## Installation, Setup, and Documentation

Running our code only requires a standard Python3 and R environment with dependencies as above. No extra installation is required.

The data folder contains RDMs and masks needed for the analyses, and the code folder contains functions and example scripts for the key analyses. Each subfolder in the data folder also contains a readme.txt file to explain the content of the data files. 

The sample data can be downloaded from the Release of the repository: https://github.com/GUO-Jiahui/not_so_fast/releases/download/1.0.0/data.zip

## License
This repository is covered under the MIT License.
