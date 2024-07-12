import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from collections import Counter

color_enum = ["r", "g", "b", "c", "m", "y", "k", "w"]

dir_name = "/home/xt/Documents/data/surfelwarp/build/apps/surfelwarp_kinectdk/temp/"
#dir_name = "/home/xt/Documents/data/surfelwarp/build/apps/surfelwarp_app/temp/"


# distance analyse
def distance(X, Y):
    # not used
    # d = ((theta * R) cross_dot P) ** 2 + T ** 2
    coord_x, axis_angle_x, trans_x = X[8:11], X[12:15], X[15:18]
    coord_y, axis_angle_y, trans_y = Y[8:11], Y[12:15], Y[15:18]
    return 

for i in range(90, 300, 1):
#for i in range(250, 285, 7):
    data = np.loadtxt(dir_name + "se3_frame_%06d.log" % i)
    se3, coord, angle_axis, trans, displacement = data[:, :8], data[:, 8:12], data[:, 12:15], data[:, 15:18], data[:, 18:21]


    #db = DBSCAN(eps=0.001, min_samples=10, metric="cosine")
    db = DBSCAN(eps=0.005, min_samples=10)
    db = db.fit(displacement)
    cnt = Counter(db.labels_)
    print("%d: %s" % (i, str(cnt)))

    if(0):
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.view_init(elev=20, azim=-90)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        colors = [color_enum[x % len(color_enum)] for x in db.labels_]
        for j in range(len(se3)):
            #ax.quiver(coord[j, 0], coord[j, 1], coord[j, 2], 1, 1, 1, length=0.01, color=colors[j])
            # for view convenient
            ax.quiver(coord[j, 0], coord[j, 2], -coord[j, 1], 1, 1, 1, length=0.01, color=colors[j])
        plt.show()
