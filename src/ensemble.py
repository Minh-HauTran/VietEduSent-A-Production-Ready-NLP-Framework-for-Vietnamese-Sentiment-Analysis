from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
# 🔬 Experiment: Ensemble Strategy
# Motivation:
# Individual models (SVM, RF) have different inductive biases:
# - SVM: strong in high-dimensional sparse spaces (TF-IDF)
# - Random Forest: captures non-linear feature interactions
#
# Hypothesis:
# Combining heterogeneous models via stacking can improve generalization
# by leveraging complementary strengths and reducing variance
#
# Future Work:
# - Explore weighted averaging vs stacking
# - Investigate Transformer + ML hybrid ensemble
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
        passthrough=True  
    )

    return model
