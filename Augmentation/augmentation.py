import numpy as np

class Augmentation:

    def __init__(self):
        pass

    def gaussian_noise(self, x):
        
        noise = np.random.normal(0,1,1)
        x += noise[0]

        return x