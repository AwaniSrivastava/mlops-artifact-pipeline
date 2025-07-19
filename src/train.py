import json
import pickle
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

with open('config/config.json') as f:
    config = json.load(f)

digits = load_digits()
X, y = digits.data, digits.target

model = LogisticRegression(
    C=config['C'],
    solver=config['solver'],
    max_iter=config['max_iter'],
    random_state=42
)
model.fit(X, y)

with open('model_train.pkl', 'wb') as f:
    pickle.dump(model, f)
