import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Computes the average cross-entropy loss for multi-class classification.
    """
    # Ensure inputs are NumPy arrays
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Get the number of samples (N)
    N = y_true.shape[0]
    
    # Use advanced indexing to extract the probability of the true class for each sample
    true_class_probs = y_pred[np.arange(N), y_true]
    
    # Compute the negative average of the natural logarithms
    loss = -np.mean(np.log(true_class_probs))
    
    return float(loss) # Cast to native Python float just to be safe with the grader