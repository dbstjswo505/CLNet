import numpy as np
import torch as th

import matplotlib.pylab as plt

# Utility function to convert radar data

def microdoppler_transform(sample, values=None, standard_scaling=False, minmax_scaling=False, local_scaling=False, preprocessing=False):

    if minmax_scaling:
        sample = (sample - values["min"]) / (values["max"] - values["min"])

    if standard_scaling:
        sample = (sample - values["mean"]) / values["std"]

    if local_scaling:
        sample = (sample - sample.min()) / (sample.max() - sample.min())

    if preprocessing:
        sample = np.concatenate((sample[:, 24:127], sample[:, 130:232]), axis=1)

    # print(sample.shape)
    # plt.imshow(sample.T)
    # plt.show()

    return th.from_numpy(sample.astype(np.float32).reshape((1,) + sample.shape))

