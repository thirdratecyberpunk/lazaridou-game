import numpy as np

def prob_range_to_average_distribution(prob_range):
    """
    turns a collection of probability distributions for a word
    i.e. [[0.9,0.1], [0.7,0.3]]
    into a single average log probability distribution
    i.e. [0.8,0.2]
    """
    return np.mean(np.transpose(np.log(prob_range)), axis=1)
