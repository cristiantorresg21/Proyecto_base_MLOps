import json 

from sklearn.linear_model import Ridge

def eval_model(model, X, y):
    score = model.score(X,y)
    print("El R2 es: ", score)
    return score

def save_report(alpha, random_seed, R2, split_test, path):
    metrics = {"alpha":alpha,
               "random_seed":random_seed,
               "split_test":split_test,
               "R2": R2 }
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)