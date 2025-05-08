import torch
from torch.nn import functional as F
from torch.nn.modules import loss


class RefineryLoss(loss._Loss):
    """The KL-Divergence loss for the model and refined labels output.

    output must be a pair of (model_output, refined_labels), both NxC tensors.
    The rows of refined_labels must all add up to one (probability scores);
    however, model_output must be the pre-softmax output of the network."""

    def forward(self, output):

        assert type(output) == tuple and len(output) == 2 and output[0].size() == \
            output[1].size(), "output must a pair of tensors of same size."

        model_output, refined_labels = output
        if refined_labels.requires_grad:
            raise ValueError("Refined labels should not require gradients.")

        model_output_log_prob = F.log_softmax(model_output, dim=1)
        del model_output

        refined_labels = refined_labels.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        cross_entropy_loss = -torch.bmm(refined_labels, model_output_log_prob)
        
        cross_entropy_loss = cross_entropy_loss.mean()
        
        return cross_entropy_loss
