from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from matplotlib import pyplot as plt

def Plot3dSurf(mapZ):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    shp=mapZ.shape
    X=np.arange(shp[1])
    Y=np.arange(shp[0])
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, mapZ, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
