from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler

def get_model(name):
    if name == "logistic":
        return LogisticRegression(max_iter=5000, solver="saga", multi_class="multinomial")

    elif name == "svm":
        return make_pipeline(MaxAbsScaler(), LinearSVC(max_iter=10000))

    elif name == "rf":
        return RandomForestClassifier(n_estimators=100)

    else:
        raise ValueError(f"Model {name} not supported")
