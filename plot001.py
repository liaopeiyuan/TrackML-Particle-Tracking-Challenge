"""
plot001.py

plot the heatmap of functions f(x, y) around the origin
You may use it to explore how feature engineering works

by Tianyi Miao
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_2d_function(func, x_min, x_max, x_n, y_min, y_max, y_n):
    x_list = np.linspace(x_min, x_max, x_n)
    y_list = np.linspace(y_min, y_max, y_n)
    func_map = np.empty((x_list.size, y_list.size))
    for i, x in enumerate(x_list):
        for j, y in enumerate(y_list):
            func_map[i, j] = func(x, y)
    fig = plt.figure()
    s = fig.add_subplot(1, 1, 1, xlabel='x', ylabel='y')
    im = s.imshow(
        np.rot90(func_map),
        extent=(x_min, x_max, y_min, y_max),
        origin=None)
    fig.colorbar(im)
    return fig

# this converts cartesian coordiantes to angle
# plot_2d_function(lambda x, y: (0 if y >= 0 else np.pi) + np.arctan(x/y), -30, 30, 1000, -30, 30, 1000)
plot_2d_function(lambda x, y: np.log(x**4+y**4), -30, 30, 1000, -30, 30, 1000)

plt.show()


