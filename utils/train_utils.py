import torch
from transformers import RobertaTokenizer
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from typing import Literal
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datasets = {
    "Sweet": "data/sweet.csv",
    "Bitter": "data/bitter.csv",
    "BBBP": "data/BBBP.csv"
}



def select_samples(encoded_inputs, indices):
    return {'input_ids': encoded_inputs['input_ids'][indices],
            'attention_mask': encoded_inputs['attention_mask'][indices]}

def load_data(dataset: Literal["Sweet", "Bitter", "BBBP"], task_id: int = -1, expand: int = -1):

    #Load data
    df1 = pd.read_csv(datasets[dataset])
    inputs = df1['SMILES'].tolist()
    labels = df1['Label'].tolist()
    labels = [int(label) for label in labels]
    if expand != -1 and task_id == -1:
        for i in range(len(labels)):
            labels[i] = ([(labels[i] if idx == task_id else 0) for idx, _ in enumerate([0,0,0])])
    
    elif expand !=-1 and task_id != -1:
        for i in range(len(labels)):
            labels[i] = ([(labels[i] if idx == task_id else (task_id if idx == 3 else 0)) for idx, _ in enumerate([0,0,0,0])])
    

    # Tokenization using ChemBERTa tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
    max_len = 50
    encoded_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')

    train_indices, test_indices, train_labels, test_labels = train_test_split(range(len(inputs)), labels, test_size=0.2, random_state=42)
    val_indices, test_indices, val_labels, test_labels = train_test_split(test_indices, test_labels, test_size=0.5, random_state=42)
    

    if expand != -1 and task_id !=-1:
        train_data = TensorDataset(select_samples(encoded_inputs, train_indices)['input_ids'], select_samples(encoded_inputs, train_indices)['attention_mask'], torch.tensor(train_labels, dtype=torch.float))
        val_data = TensorDataset(select_samples(encoded_inputs, val_indices)['input_ids'], select_samples(encoded_inputs, val_indices)['attention_mask'], torch.tensor(val_labels, dtype=torch.float))
        test_data = TensorDataset(select_samples(encoded_inputs, test_indices)['input_ids'], select_samples(encoded_inputs, test_indices)['attention_mask'], torch.tensor(test_labels, dtype=torch.float))
    
    elif expand != -1:
        train_data = TensorDataset(select_samples(encoded_inputs, train_indices)['input_ids'], select_samples(encoded_inputs, train_indices)['attention_mask'], torch.tensor(train_labels, dtype=torch.float))
        val_data = TensorDataset(select_samples(encoded_inputs, val_indices)['input_ids'], select_samples(encoded_inputs, val_indices)['attention_mask'], torch.tensor(val_labels, dtype=torch.float))
        test_data = TensorDataset(select_samples(encoded_inputs, test_indices)['input_ids'], select_samples(encoded_inputs, test_indices)['attention_mask'], torch.tensor(test_labels, dtype=torch.float))
    
    elif task_id != -1:
        train_data = TensorDataset(select_samples(encoded_inputs, train_indices)['input_ids'], select_samples(encoded_inputs, train_indices)['attention_mask'], torch.tensor(train_labels), torch.full((len(train_labels),), task_id, dtype=torch.long))
        val_data = TensorDataset(select_samples(encoded_inputs, val_indices)['input_ids'], select_samples(encoded_inputs, val_indices)['attention_mask'], torch.tensor(val_labels), torch.full((len(val_labels),), task_id, dtype=torch.long))
        test_data = TensorDataset(select_samples(encoded_inputs, test_indices)['input_ids'], select_samples(encoded_inputs, test_indices)['attention_mask'], torch.tensor(test_labels), torch.full((len(test_labels),), task_id, dtype=torch.long))
    
    else:
        train_data = TensorDataset(select_samples(encoded_inputs, train_indices)['input_ids'], select_samples(encoded_inputs, train_indices)['attention_mask'], torch.tensor(train_labels))
        val_data = TensorDataset(select_samples(encoded_inputs, val_indices)['input_ids'], select_samples(encoded_inputs, val_indices)['attention_mask'], torch.tensor(val_labels))
        test_data = TensorDataset(select_samples(encoded_inputs, test_indices)['input_ids'], select_samples(encoded_inputs, test_indices)['attention_mask'], torch.tensor(test_labels))

    return train_data, val_data, test_data

# Function to add to buffer
def add_to_buffer(replay_buffer, buffer_size, batch_size, input_ids, attention_mask, labels):
    if len(replay_buffer['input_ids']) >= buffer_size:
        del replay_buffer['input_ids'][:batch_size]
        del replay_buffer['attention_mask'][:batch_size]
        del replay_buffer['labels'][:batch_size]
    replay_buffer['input_ids'].extend(input_ids.cpu().tolist())
    replay_buffer['attention_mask'].extend(attention_mask.cpu().tolist())
    replay_buffer['labels'].extend(labels.cpu().tolist())

def sample_from_buffer(replay_buffer, batch_size):
    indices = random.sample(range(len(replay_buffer['input_ids'])), batch_size)
    return {'input_ids': torch.tensor([replay_buffer['input_ids'][i] for i in indices]),
            'attention_mask': torch.tensor([replay_buffer['attention_mask'][i] for i in indices]),
            'labels': torch.tensor([replay_buffer['labels'][i] for i in indices])}

# Function to compute EWC loss
def compute_ewc_loss(model, prev_task_means, prev_task_fisher):
    ewc_loss = 0
    for name, param in model.named_parameters():
        if name in prev_task_means:
            fisher = prev_task_fisher[name]
            mean = prev_task_means[name]
            ewc_loss += (fisher * (param - mean) ** 2).sum()
    return ewc_loss

# Pareto gradient computation
def compute_pareto_gradients(grads1, grads2):
    dot_product = sum([(g1 * g2).sum() for g1, g2 in zip(grads1, grads2)])
    alpha = torch.clamp(dot_product / (sum([(g1 ** 2).sum() for g1 in grads1])), min=0.0, max=1.0)
    blended_grads = [(1 - alpha) * g1 + alpha * g2 for g1, g2 in zip(grads1, grads2)]
    return blended_grads

# Function to compute Fisher Information Matrix (FIM)
def compute_fisher_information(model, data_loader):
    fisher_info = {}
    for name, param in model.named_parameters():
        fisher_info[name] = torch.zeros_like(param)
    model.eval()
    for batch in data_loader:
        input_ids, attention_mask, labels = batch
        output = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels.to(device))
        loss = output.loss
        model.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
            fisher_info[name] += param.grad ** 2
    for name in fisher_info:
        fisher_info[name] /= len(data_loader)
    return fisher_info