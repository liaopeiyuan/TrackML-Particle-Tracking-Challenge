import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utils.session import Session


def subroutine_1(sub_df):
    if sub_df.shape[0] <= 3 or (sub_df["particle_id"] == 0).any() or np.random.rand() > 0.01:
        return
    else:
        # sub_df.sort_values("rt", inplace=True)
        plt.plot(sub_df["rt"].values, sub_df["arctan_z_rt"].values)
        
if __name__ == '__main__':
    s1 = Session("../portable-dataset/")
    for hits, truth in s1.get_train_events(n=1, randomness=True)[1]:
        df = pd.merge(hits, truth, on="hit_id")
        df["rt"] = np.sqrt(df["x"]**2 + df["y"]**2)
        # df["z_div_rt"] = df["z"] / df["rt"]
        df["arctan_z_rt"] = np.arctan2(df["z"], df["rt"])
        df.groupby("particle_id").apply(subroutine_1)
        plt.xlabel("rt")
        plt.ylabel("arctan_z_rt")
        plt.show()