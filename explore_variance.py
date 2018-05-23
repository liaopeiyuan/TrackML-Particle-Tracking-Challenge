import numpy as np
import pandas as pd

cluster_var = pd.read_csv("cluster_variance.csv", header=None)

track_var = pd.read_csv("track_variance.csv", header=None)

print(cluster_var)
print(cluster_var.describe())

print(track_var)
print(track_var.describe())