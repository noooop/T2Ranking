"""Cross-encoder reranker fine-tuning"""
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import torch
from torch import nn
from torch.optim import AdamW
from transformers import (AutoModelForSequenceClassification, Trainer,
                          TrainingArguments, get_cosine_schedule_with_warmup)

root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from experiments.core import dataset
from experiments.core.config import CrossEncoderConfig


class RerankTrainer(Trainer):

    def __init__(self, config, loss_fn: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs)
        self.sample_num = config.sample_num
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.loss_fn = loss_fn

    def compute_loss(self,
                     model: nn.Module,
                     inputs,
                     return_outputs=False,
                     **kwargs):
        outputs = model(**inputs)

        logits = outputs.logits
        scores = logits.view(-1, self.sample_num)
        target_label = torch.zeros(scores.size(0),
                                   dtype=torch.long,
                                   device=scores.device)

        loss = self.loss_fn(scores, target_label)
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving rerank model checkpoint to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.model.tokenizer.save_pretrained(output_dir)


def main():
    config = CrossEncoderConfig()

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path,
        num_labels=config.sample_num,
    )

    train_dataset = dataset.CrossEncoderTrainDataset(config)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    num_train_steps = int(
        len(train_dataset) / config.batch_size * config.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_proportion * num_train_steps,
        num_training_steps=num_train_steps,
    )

    training_args = TrainingArguments(
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.batch_size //
        config.per_device_train_batch_size,
        gradient_checkpointing=config.gradient_checkpoint,
        num_train_epochs=config.epochs,
        output_dir=config.model_out_dir,
        remove_unused_columns=False,
        logging_steps=config.report,
        fp16=config.dtype == "float16",
        report_to="none",
    )
    trainer = RerankTrainer(
        config=config,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=train_dataset._collate_fn,
    )
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.train()
    model.save_pretrained(config.model_out_dir)


if __name__ == '__main__':
    main()
    '''
    1/6260 [00:25<44:07:56, 25.38s/it
    17.055Gi/23.988Gi
    '''
