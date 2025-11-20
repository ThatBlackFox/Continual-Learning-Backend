import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import RobertaForSequenceClassification

class CILHead(nn.Module):
    """Head for sentence-level multi-label classification (3 independent binary outputs)."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if getattr(config, "classifier_dropout", None) is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, 3)  # 3 independent outputs

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # [CLS] token
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)   # [batch, 3]
        return x

class CILChemBERTa(nn.Module):
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1", num_labels=2):
        super().__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.classifier = CILHead(self.model.config)

    def forward(self, input_ids, attention_mask, labels=None):
        
        # Forward through the base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        loss = None
        if labels is not None:
            if labels.shape[1] == 4:
                selected_idx = labels[:, 3].long()   # [B]

                # pick the logits to train: [B]
                selected_logits = outputs.logits.gather(1, selected_idx.unsqueeze(1)).squeeze(1)

                # pick the corresponding 0/1 targets: [B]
                selected_targets = labels.gather(1, selected_idx.unsqueeze(1)).squeeze(1).float()

                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(selected_logits, selected_targets)
            else:
                raise NotImplementedError("Please provide task ID when providing labels")
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=outputs.logits
        )

if __name__ == "__main__":
    model = CILChemBERTa()