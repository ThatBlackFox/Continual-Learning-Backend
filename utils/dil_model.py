import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import RobertaForSequenceClassification

class DILChemBERTa(nn.Module):
    def __init__(self, num_tasks=3, model_name="seyonec/ChemBERTa-zinc-base-v1", num_labels=2):
        super().__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        hidden_size = self.model.config.hidden_size
        
        # Single learned embedding per task
        self.task_emb = nn.Embedding(num_tasks, hidden_size)

    def forward(self, input_ids, attention_mask, task_ids, labels=None):
        # Lookup and prepend task token embedding
        task_token = self.task_emb(task_ids).unsqueeze(1)                # [B,1,H]
        inputs_embeds = self.model.roberta.embeddings(input_ids)         # [B,L,H]
        inputs_embeds = torch.cat([task_token, inputs_embeds], dim=1)    # prepend task token
        
        # Adjust attention mask
        task_mask = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype).to(attention_mask.device)
        attention_mask = torch.cat([task_mask, attention_mask], dim=1)
        
        # Forward through the base model
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(outputs.logits.view(-1, self.model.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=outputs.logits
        )