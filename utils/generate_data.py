import numpy as np


def generate_CS_matrix(M:int, N:int=784):
    # Generate N vectors of dimension M from a standard Gaussian distribution
    A = np.array([np.random.randn(M) for _ in range(N)]).T
    # normalizing each column of A to a unit norm
    A = A / np.expand_dims(np.linalg.norm(A, ord=2, axis=1), axis=1)

    return A


def generate_measurements(image_data: np.ndarray, A: np.ndarray, scale: float = 1.) -> tuple[np.ndarray, np.ndarray]:
    """
        Generate modulo measurements y = [[Ax]], where [[]] denotes the fractional parts of the elements of the vector Ax
        
        Parameters:
            - image_data: np.ndarray flattened vector of the image pixel values
            - A: np.ndarray compressed-sensing matrix
            - scale: float parameter from 0 to 1 which scales observations (used for Multi-Scale Justice Pursuit)

        Returns:
            - np.ndarray: acquired samples y
            - np.ndarray: 2D compressed-sensing matrix    
    """
    # acquiring dense modulo samples from the sparse image vector
    y_dense = scale * (A @ image_data)
    y = y_dense - np.floor(y_dense)

    return y