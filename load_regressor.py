import pickle


with open("model/model.sav", "rb") as model:
    loaded_model = pickle.load(model)
