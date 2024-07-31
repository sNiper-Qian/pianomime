import torch
import torch.nn as nn
import robopianist.models.piano as piano
from utils import get_keys_sdf, get_point_sdf

def get_sdf_loss(pred, nobs, q):
    # Get sdf loss of a batch
    # pred: (B, T, 1)
    # nobs: (B, 88) 
    # q: (B, 1)
    # The loss of (T, 1) is calculated by get_keys_sdf()
    target = torch.zeros_like(pred)
    for i in range(nobs.shape[0]):
        sdf = get_keys_sdf(q[i][0], nobs[i])
        target[i] = sdf
    # Calculate mse loss
    loss = nn.functional.mse_loss(pred, target)
    return loss

def get_point_sdf_loss(pred, nobs, q):
    # Get sdf loss of a batch
    # pred: (B, T, 1)
    # nobs: (B, 88) 
    # q: (B, 2)
    # The loss of (T, 1) is calculated by get_keys_sdf()
    target = torch.zeros_like(pred)
    for i in range(nobs.shape[0]):
        sdf = get_point_sdf(q[i], nobs[i])
        target[i] = sdf
    # Calculate mse loss
    loss = nn.functional.mse_loss(pred, target)
    return loss

def get_recall(pred, target):
    # Get recall of a batch
    # pred: (B, T, C)
    # target: (B, T, C)
    # Round pred to 0 or 1
    pred = torch.round(pred)
    # Compare pred and target
    correct = ((pred == target) * target).sum()
    total = target.sum()
    # Get recall
    recall = correct / total
    return recall

def get_accuracy(pred, target):
    # Get accuracy of a batch
    # pred: (B, T, C)
    # target: (B, T, C)
    # Round pred to 0 or 1
    pred = torch.round(pred)
    # Compare pred and target
    correct = (pred == target).sum()
    # Get accuracy
    accuracy = correct / (pred.shape[0] * pred.shape[1] * pred.shape[2])
    return accuracy

class PrecisionRecallLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(PrecisionRecallLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.precision = Precision()
        self.recall = Recall()

    def forward(self, y_true, y_pred):
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)

        custom_loss_value = self.alpha * (1 - precision) + self.beta * (1 - recall)

        return custom_loss_value

class Precision(nn.Module):
    def forward(self, y_true, y_pred):
        true_positive = torch.sum((y_true == 0) & (y_pred == 0)).float()
        predicted_positive = torch.sum(y_pred == 0).float()

        precision = true_positive / (predicted_positive + 1e-8)  # Adding a small epsilon to avoid division by zero

        return precision

class Recall(nn.Module):
    def forward(self, y_true, y_pred):
        true_positive = torch.sum((y_true == 1) & (y_pred == 1)).float()
        actual_positive = torch.sum(y_true == 1).float()

        recall = true_positive / (actual_positive + 1e-8)  # Adding a small epsilon to avoid division by zero

        return recall

if __name__ == "__main__":
    print(piano.keys)


