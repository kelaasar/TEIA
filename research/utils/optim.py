'''Model optimization utilities.'''
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AdamW
from torchmetrics.functional import pairwise_cosine_similarity


class PairwisePivotLoss(nn.Module):
    '''Calculate pairwise pivot loss'''

    def forward(self, emb1, emb2):
        '''Calculate pairwise pivot loss'''
        # Input: (batch, emb_dim)
        cos_emb1 = pairwise_cosine_similarity(
            emb1, emb1)  # output: (batch, batch)
        cos_emb2 = pairwise_cosine_similarity(
            emb2, emb2)  # output: (batch, batch)
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(cos_emb1, cos_emb2)
        # kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        # loss = kl_loss(cos_emb1, cos_emb2)
        return loss


class SequenceCrossEntropyLoss(nn.Module):
    '''Sequence Cross Entropy Loss'''

    def forward(self, logits, targets, mask, label_smoothing=-1, reduce=None):
        """
        reduce: None, "batch", "sentence"
        """
        return sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce)


def sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce):
    """
    label_smoothing : ``float``, optional (default = 0.0)
        It should be smaller than 1.
    """
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    if label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / float(num_classes)
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(
            log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(
            -1, keepdim=True)
    else:
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - \
            torch.gather(log_probs_flat, dim=1, index=targets_flat)

    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(
        -1, logits.shape[1])

    # shape : (batch, sequence_length)
    loss = negative_log_likelihood * mask

    if reduce:
        # shape : (batch,)
        loss = loss.sum(1) / (mask.sum(1) + 1e-13)

        if reduce == "batch":
            # shape : scalar
            loss = loss.mean()

    return loss


def prepare_optimizer(model):
    '''Prepare partial weight_decay optimizer for the model'''
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'ln', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    return AdamW(optimizer_grouped_parameters,
                 lr=3e-5,
                 eps=1e-06)
