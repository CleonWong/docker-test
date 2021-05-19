import numpy as np
import pandas as pd

# ----------


def generate_data(n_samples, csv_path):

    """
    This function generates `n_samples` random data points (i.e. X and Y) from
    the equation y=x^2. Then, it saves these X and y data in a .csv file.

    Parameters
    ----------
    n_samples : int
        Number of data points to generate.
    csv_path : str or Path
        Path to save csv.

    Returns
    -------
    X : numpy.ndarray
    y : numpy.ndarray
    """

    X = np.random.uniform(low=-8, high=8, size=(n_samples, 1))
    y = X * X

    stacked = np.hstack((X, y))
    df = pd.DataFrame(stacked, columns=["x", "y"])

    df.to_csv(csv_path, index=False)

    return X, y
