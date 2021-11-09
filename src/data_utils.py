import numpy as np
import matplotlib.pyplot as plt

# Set a random seed
np.random.seed(42)

# Make a dataset of points inside and outside of a circle
def circle(samples, center=[0.0, 0.0], radius=np.sqrt(2 / np.pi)):
    """
    Generates a dataset of points with 1/0 labels inside a given radius.

    Args:
        samples (int): number of samples to generate
        center (tuple): center of the circle
        radius (float: radius of the circle

    Returns:
        Xvals (array[tuple]): coordinates of points
        yvals (array[int]): classification labels
    """
    Xvals, yvals = [], []

    for i in range(samples):
        x = 2 * (np.random.rand(2)) - 1
        y = 0
        if np.linalg.norm(x - center) < radius:
            y = 1
        Xvals.append(x)
        yvals.append(y)
    return np.array(Xvals), np.array(yvals)

# Plot dataset
def plot_data(x, y, fig=None, ax=None):
    """
    Plot data with red/blue values for a binary classification.

    Args:
        x (array[tuple]): array of data points as tuples
        y (array[int]): array of data points as tuples
    """
    if fig == None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    reds = y == 0
    blues = y == 1
    ax.scatter(x[reds, 0], x[reds, 1], c="green", s=20)
    ax.scatter(x[blues, 0], x[blues, 1], c="blue", s=20)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

# Create dataset for this specific QNN formulation
def generate_ds(num_samples):
    X, y = circle(num_samples)
    X = np.hstack((X, np.zeros((X.shape[0], 1))))  # padding to reach 3 dimensions
    label_angles = np.asarray([y if y == 1 else -1 for y in y]).reshape(-1, 1)
    X = np.hstack((X, label_angles))
    new_y = np.ones(y.shape)
    return X, y, new_y