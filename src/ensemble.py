from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler

def build_ensemble():
    svm = make_pipeline(MaxAbsScaler(), LinearSVC(max_iter=10000))
    rf = RandomForestClassifier(n_estimators=100)
    lr = LogisticRegression(max_iter=10000, solver="saga")

    model = StackingClassifier(
        estimators=[
            ("svm", svm),
            ("rf", rf)
        ],
        final_estimator=lr
    )

    return model
