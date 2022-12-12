import pickle
from regressor import args

with open(args.file_path, "rb") as model:
    loaded_model = pickle.load(model)

print(type(loaded_model))
