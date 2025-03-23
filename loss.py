import torch
import torch.nn as nn
import torch.nn.functional as F

class lossAV(nn.Module):
    def __init__(self):
        super(lossAV, self).__init__()
        self.criterion = nn.CrossEntropyLoss()  # Multi-class classification
    
    def forward(self, x, labels=None, r=1):
        # x: (B, N) logits (before softmax)
        # labels: (B,) indices of correct face (0 to N-1)
        if labels is None:
            return torch.softmax(x, dim=1).detach().cpu().numpy()  # For inference
        labels_indices = torch.argmax(labels, dim=1)
        loss = self.criterion(x / r, labels_indices)
        probs = torch.softmax(x, dim=1)
        pred_labels = torch.argmax(probs, dim=1)
        correct = (pred_labels == labels).sum().float()
        return loss, probs, pred_labels, correct