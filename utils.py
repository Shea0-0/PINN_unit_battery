import torch
import os
import numpy as np
import matplotlib.pyplot as plt


# z=1,2,3,4，表示z方向的4层节点
def contour(temp_c, z):
    x = np.linspace(-0.0725, 0.0725, 17)
    y = np.linspace(-0.096, 0.141, 24)
    x_mesh, y_mesh = np.meshgrid(x, y)
    temperature = np.zeros_like(x_mesh, dtype=float)
    temp_c = temp_c[(z - 1) * 388 : z * 388]
    index_c = 0
    for i_c in range(24):
        if i_c < 20:
            for j_c in range(17):
                temperature[i_c, j_c] = temp_c[index_c]
                index_c += 1
        if i_c > 19:
            for j_c in range(17):
                if j_c in {0, 7, 8, 9, 16}:
                    temperature[i_c, j_c] = np.nan
                else:
                    temperature[i_c, j_c] = temp_c[index_c]
                    index_c += 1
    plt.figure(figsize=(14.5, 23.7))
    plt.contourf(x, y, temperature, levels=1000, cmap="jet")
    plt.colorbar(label="T")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("T")
    plt.show()


def file_path(path):
    file_paths = []
    file_num = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            file_name, file_extension = os.path.splitext(file)
            if not file_extension:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
                file_num += 1
    return file_paths, file_num


def min_max(data):
    min_val = torch.min(data)
    max_val = torch.max(data)
    return min_val, max_val, max_val - min_val
