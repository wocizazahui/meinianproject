import numpy as np
import matplotlib.pyplot as plt


def distribution_plot(l):
    x = np.array(l)
    mu, sigma = 200, 25
    n, bins, patches = plt.hist(l, bins=len(l))
    plt.show()


if __name__ == "__main__":
    distribution_plot([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5])