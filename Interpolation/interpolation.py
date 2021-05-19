from scipy import interpolate
import numpy as np

class Interpolation:

    def __init__(self):
        pass

    def interpolate_polyline(self, x, y, num_points):
        if(len(x)>2):
            tck, u = interpolate.splprep([x , y], s=0, k=2)
        else:
            tck, u = interpolate.splprep([x , y], s=0, k=1)
        u = np.linspace(0.0, 1.0, num_points)
        return np.column_stack(interpolate.splev(u, tck))