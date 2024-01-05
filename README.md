# Keyword-Based Emotion Recognition in Conversations from Audio Data

This repository contains the code for the master thesis "Keyword-Based Emotion Recognition in Conversations from Audio Data" by Miriam Repp, 
which was submitted to the Data and Web Science Group, Prof. Dr. Heiner Stuckenschmidt at the University of Mannheim on March 4, 2024. 




## Setup & Installation

To get started with this project, create a new conda environment and install the requirements from the `requirements.txt` file with the following command:

```shell
$ conda create --name <env> --file requirements.txt
```


## Repository Structure & Overview

This repository contains various python scripts used during the creation of thesis. The following lists the most important scripts and files.
To learn about the available parameters of the scripts, please refer to their help commands.


### Dataset Converters

To convert various datasets to the format required by NeMo data loaders, we provide the scripts in `dataconverter`. 
In particular

* `iemocap_to_nemo_manifest.py` can be used to convert the [IEMOCAP dataset](https://sail.usc.edu/iemocap/). 
Please note that access to the IEMOCAP dataset needs to be requested from the creators via their website. 
* `meld_to_nemo_manifest.py` can be used to convert the [MELD dataset](https://affective-meld.github.io/).
As listed on their website, the dataset can be freely downloaded by running the command 
```shell
wget https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz`
```



### Emotion Keyword Extraction

The script `emotion_keyword_extraction/emotion_keyword_extractor.py` is used to extract emotionally relevant keywords from the 
datasets converted with the scripts mentioned above.



### Training Script

With the script `nemo_models/ser_nemo_models/scripts/finetune_conformer_encoder_classification.py`, the training of our models can be performed.
The NeMo model configurations used for our models and required for the training can be found in `nemo_models/ser_nemo_models/config`.
Please refer to the configs matching the dataset you want to train on.



### Evaluation Script

With `nemo_models/ser_nemo_models/scripts/evaluate_model.py`, the trained models can be evaluated.
The evaluation script will generate a directory with a log of the evaluation run, confusion matrices for the train, val and test sets as well as an Excel file containing the results of all metrics.



### Dataset Distribution Visualizations

The script `visualizations/dataset_distribution_visualizer.py` allows to generate the visualizations about a converted dataset's label distribution.



### Model Definitions, Decoder Layers and Data Loaders

All models proposed in this thesis use the [NeMo framework](https://github.com/NVIDIA/NeMo).

* The classes defining our models can be found in the directory `nemo_models/ser_nemo_models/`.
* The decoders used by these models are located in `nemo_models/ser_nemo_models/decoders`.
* The data loaders used to load keyword enhanced input data are provided in directory `nemo_models/ser_nemo_models/data_loaders`.



## Authors
Miriam Repp, 2024