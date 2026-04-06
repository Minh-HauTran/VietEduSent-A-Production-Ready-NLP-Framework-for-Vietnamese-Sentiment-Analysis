import matplotlib.pyplot as plt

def plot_metrics(metrics):
    names = list(metrics.keys())
    values = list(metrics.values())

    plt.bar(names, values)
    plt.title("Model Performance")
    plt.show()
