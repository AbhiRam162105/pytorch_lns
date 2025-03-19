import torch
import math
from ..lns_tensor.lns_tensor import LNSTensor
from ..lns_ops.activation_funcs import lns_softmax

def lns_mse_loss(predictions, targets):
    """
    Mean Squared Error loss for LNSTensor objects.
    
    Args:
        predictions: LNSTensor of predicted values
        targets: LNSTensor of target values
    
    Returns:
        LNSTensor of MSE loss
    """
    if not isinstance(predictions, LNSTensor):
        predictions = LNSTensor(predictions)
    if not isinstance(targets, LNSTensor):
        targets = LNSTensor(targets, base=predictions.base)
    
    # Calculate squared difference
    diff = predictions - targets
    squared_diff = diff * diff
    
    # Calculate mean
    num_elements = torch.numel(predictions.log_data)
    mean_squared_diff = squared_diff.to_float().mean()
    
    return LNSTensor(mean_squared_diff, base=predictions.base)

def lns_cross_entropy_loss(predictions, targets, reduction='mean'):
    """
    Cross Entropy Loss for LNSTensor objects.
    
    Args:
        predictions: LNSTensor of raw (non-normalized) predictions
        targets: Long tensor of target class indices
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        LNSTensor of Cross Entropy loss
    """
    if not isinstance(predictions, LNSTensor):
        predictions = LNSTensor(predictions)
    
    # Apply softmax to predictions to get probabilities
    log_probs = lns_softmax(predictions, dim=1)
    
    # Extract log-probability of the correct class for each sample
    batch_size = predictions.log_data.shape[0]
    n_classes = predictions.log_data.shape[1]
    
    # Convert to float for indexing
    log_probs_float = log_probs.to_float()
    
    # Get log-probabilities of the correct classes
    if targets.dim() == 1:
        # Case: targets contains class indices
        log_prob_correct = log_probs_float[torch.arange(batch_size), targets]
    else:
        # Case: targets is one-hot encoded
        targets_float = targets.float() if not isinstance(targets, torch.FloatTensor) else targets
        log_prob_correct = torch.sum(log_probs_float * targets_float, dim=1)
    
    # Cross entropy is negative log likelihood
    batch_losses = -log_prob_correct
    
    # Apply reduction
    if reduction == 'none':
        loss = batch_losses
    elif reduction == 'sum':
        loss = batch_losses.sum()
    else:  # mean
        loss = batch_losses.mean()
    
    return LNSTensor(loss, base=predictions.base)

def lns_binary_cross_entropy(predictions, targets, reduction='mean'):
    """
    Binary Cross Entropy Loss for LNSTensor objects.
    
    Args:
        predictions: LNSTensor of predicted probabilities (after sigmoid)
        targets: LNSTensor or tensor of target values (0 or 1)
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        LNSTensor of Binary Cross Entropy loss
    """
    if not isinstance(predictions, LNSTensor):
        predictions = LNSTensor(predictions)
    
    # Convert targets to appropriate format
    if isinstance(targets, LNSTensor):
        targets = targets.to_float()
    
    # Calculate negative log likelihood for binary case
    # loss = -targets * log(predictions) - (1 - targets) * log(1 - predictions)
    
    # Convert to float for calculation
    pred_float = predictions.to_float()
    pred_float = torch.clamp(pred_float, min=1e-7, max=1-1e-7)  # Prevent log(0)
    
    # Calculate binary cross entropy
    bce_loss = -targets * torch.log(pred_float) - (1 - targets) * torch.log(1 - pred_float)
    
    # Apply reduction
    if reduction == 'none':
        loss = bce_loss
    elif reduction == 'sum':
        loss = bce_loss.sum()
    else:  # mean
        loss = bce_loss.mean()
    
    return LNSTensor(loss, base=predictions.base)