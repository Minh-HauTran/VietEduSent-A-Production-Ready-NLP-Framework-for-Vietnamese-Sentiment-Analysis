import mlflow

def start_experiment(name="vietedusent"):
    mlflow.set_experiment(name)
    return mlflow.start_run()

def log_metrics(metrics):
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

def log_params(params):
    for k, v in params.items():
        mlflow.log_param(k, v)
