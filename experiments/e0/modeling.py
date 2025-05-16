import torch
from torch import nn
from transformers import AutoModelForSequenceClassification

from experiments.core.config import CrossEncoderConfig


class CrossEncoder(nn.Module):

    def __init__(self, config: CrossEncoderConfig):
        super().__init__()

        self.sample_num = config.sample_num

        self.lm = AutoModelForSequenceClassification.from_pretrained(
            config.model_name_or_path, num_labels=1, output_hidden_states=True)
        if config.gradient_checkpoint:
            self.lm.gradient_checkpointing_enable()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, batch):
        ret = self.lm(**batch, return_dict=True)
        logits = ret.logits
        if self.training:
            scores = logits.view(-1,
                                 self.sample_num)  # q_batch_size, sample_num
            target_label = torch.zeros(scores.size(0),
                                       dtype=torch.long,
                                       device=scores.device)
            loss = self.cross_entropy(scores, target_label)
            return loss
        return logits
