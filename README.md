# Grammar Induction for Language Modeling

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Model Architecture](#model-architecture)
- [Released Models](#released-models)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This repository contains code and tools for training and evaluating the grammar induction for language modeling models. The project aims to provide a flexible and extensible framework for experimenting with the grammar induction for language modeling task.

## Features

- Support for training and evaluating grammar induction models.
- Use pre-trained grammar induction modules to train language models
- Preprocessing tools for data preparation.
- Simple command-line interface for training and evaluation.
- Example scripts for running experiments on standard benchmark datasets.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager
- GPU (for faster training, optional but recommended)

### Installation

Clone the repository:

```bash
git clone https://github.com/evaportelance/babylm-joint-learning.git
cd babylm-joint-learning
pip insatll requirements.txt
```

## Usage

### Datapreparation

We use the 10 Million word BabyLM task dataset (the strict track small dataset in the BabyLM challenge) to train our language models. Prior to training, we preprocessed the dataset to remove any blank lines or unecessary formatting punctuation (e.g. `== Title ==' became 'Title'). Additionally, we split paragraphs such that each new line represented a single sentence and removed any sentence that was longer than 40 words. The processed dataset can be fetched in [the OSF repository](https://osf.io/qj4uy/?view_only=0436b8c2c1974d879e3a353bae662b4c).

### Training

To train grammar induction module:

```bash
cd vc-pcfg
python train.py
```

To train language model:

```bash
cd vc-pcfg
python train_lm.py --prefix "x" --model_init "path to the grammar induction model" --train_data "train dataset" --val_data "validation dataset" --save_model_path "path to save the model" --logger_name "path to store the logger" 
```

### Evaluation

See more information in the babylm testing [benchmarks](https://github.com/babylm/evaluation-pipeline).

## Model Architecture

The grammar induction module is based on [Kim et al., 2019](https://github.com/harvardnlp/compound-pcfg) and [Zhao and Titov, 2020](https://github.com/zhaoyanpeng/vpcfg); the language model is based on [OPT](https://huggingface.co/docs/transformers/main/model_doc/opt) provided by Huggingface.

## Released Models

Pretrained models can be found in the [OSF site](https://osf.io/qj4uy/?view_only=0436b8c2c1974d879e3a353bae662b4c).

## Contributing
We welcome contributions to improve this project. If you find any issues or want to add new features, please open an issue or submit a pull request. Let's make this project better together!

## License
This project is licensed under the MIT License.
