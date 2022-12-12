#!/bin/zsh

python regressor.py --file_path model/model.pkl
python load_regressor.py --file_path model/model.pkl