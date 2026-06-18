import numpy as np

def sample_mnist(N:int, split='test') -> np.ndarray:
    image_set = np.loadtxt(f'data\mnist_{split}.csv', delimiter=',')
    # sampling N images
    np.random.seed(42)
    indices = np.arange(image_set.shape[0])
    np.random.shuffle(indices)
    image_set = image_set[indices[:N], 1:]
    return image_set