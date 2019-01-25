from model import build_model, predict
from loadData import load_model

# models = load_model()

models = build_model()

predict(models)