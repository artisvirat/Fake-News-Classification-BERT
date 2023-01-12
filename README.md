# Fake-News-Classification-BERT
Fine Tuning BERT for sequence classification.

## Data source
The data is downloaded from kaggle: [Click me](https://www.kaggle.com/c/fake-news-pair-classification-challenge)
- Download the data and put the csv files in `./data` folder.


## Install
### Major Dependencies:
- transformers
- pyyaml
- torch
- numpy
- pandas

Run requirements.txt in a new conda env.
```
pip install -r requirements.txt
```
## Training
To train the model ensure the paths in cfg.yaml and train.py are correct.
Run
```
python train.py
```
## Prediction
Run
```
python predict.py
```
