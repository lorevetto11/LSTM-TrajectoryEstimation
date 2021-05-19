import matplotlib.pyplot as plt

class Plotting:

    def __init__(self):
        pass

    def plot(self, x1, y1, x2, y2, x3, y3):
        self.track_margin_left(x1, y1)
        self.track_margin_right(x2, y2)
        self.trajectory_coordinates(x3, y3)
        plt.show()

    def track_margin_left(self, x, y):
        plt.plot(x, y, "bs", linewidth=0.01)

    def track_margin_right(self, x, y):
        plt.plot(x, y, "gs", linewidth=0.01)

    def trajectory_coordinates(self, x, y):
        plt.plot(x, y, "rs", linewidth=0.01)

