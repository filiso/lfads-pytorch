import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


sns.set_style('white', rc={'axes.spines.right': False, 'axes.spines.top': False})
sns.set_context("notebook")


def get_pca_time_series(x, ncmp):
    pca = PCA(ncmp)
    pca.fit(np.transpose(x, (1, 0, 2)).reshape(-1, x.shape[-1]))
    tpc = pca.transform(np.transpose(x, (1, 0, 2)).reshape(-1, x.shape[-1]))
    tpc = tpc.reshape(x.shape[1], x.shape[0], ncmp)
    return tpc


def plot_3d_pca(cycles, iplot, figsize=None):
    if figsize is None:
        figsize = (30, 10)
    for a1, a2 in [(30, -60)]:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(131, projection='3d')
        for cyc in cycles[iplot]:
            ax.plot(cyc[:, 0], cyc[:, 1], cyc[:, 2], 'k', linewidth=0.1)
            ax.scatter(cyc[:, 0], cyc[:, 1], cyc[:, 2], cmap='viridis', c=np.arange(cyc.shape[0]), s=0.2)
        ax.plot(cycles.mean(0)[:, 0], cycles.mean(0)[:, 1], cycles.mean(0)[:, 2])
        ax.scatter(cycles.mean(0)[:, 0], cycles.mean(0)[:, 1], cycles.mean(0)[:, 2],
                   cmap='viridis', c=np.arange(0, cycles.mean(0).shape[0]), s=150)
        ax.view_init(a1, a2)
        #
        ax = fig.add_subplot(132, projection='3d')
        for cyc in cycles[iplot]:
            ax.plot(cyc[:, 2], cyc[:, 1], cyc[:, 0], 'k', linewidth=0.1)
            ax.scatter(cyc[:, 2], cyc[:, 1], cyc[:, 0], cmap='viridis', c=np.arange(cyc.shape[0]), s=0.2)
        ax.plot(cycles.mean(0)[:, 2], cycles.mean(0)[:, 1], cycles.mean(0)[:, 0])
        ax.scatter(cycles.mean(0)[:, 2], cycles.mean(0)[:, 1], cycles.mean(0)[:, 0],
                   cmap='viridis', c=np.arange(0, cycles.mean(0).shape[0]), s=150)
        ax.view_init(a1, a2)
        #
        ax = fig.add_subplot(133, projection='3d')
        for cyc in cycles[iplot]:
            ax.plot(cyc[:, 0], cyc[:, 2], cyc[:, 1], 'k', linewidth=0.1)
            ax.scatter(cyc[:, 0], cyc[:, 2], cyc[:, 1], cmap='viridis', c=np.arange(cyc.shape[0]), s=0.2)
        ax.plot(cycles.mean(0)[:, 0], cycles.mean(0)[:, 2], cycles.mean(0)[:, 1])
        ax.scatter(cycles.mean(0)[:, 0], cycles.mean(0)[:, 2], cycles.mean(0)[:, 1],
                   cmap='viridis', c=np.arange(0, cycles.mean(0).shape[0]), s=150)
        ax.view_init(a1, a2)
    plt.show()