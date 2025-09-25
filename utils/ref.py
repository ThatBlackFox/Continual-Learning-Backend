import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import random

# Load your dataset
df1 = pd.read_csv('/content/drive/MyDrive/ewc data/explbitter.csv')
inputs = df1['SMILES'].tolist()
labels = df1['Label'].tolist()
labels = [int(label) for label in labels]

# Tokenization using ChemBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
max_len = 50
encoded_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')

# Split the dataset
train_indices, test_indices, train_labels, test_labels = train_test_split(range(len(inputs)), labels, test_size=0.2, random_state=42)
val_indices, test_indices, val_labels, test_labels = train_test_split(test_indices, test_labels, test_size=0.5, random_state=42)

def select_samples(indices):
    return {'input_ids': encoded_inputs['input_ids'][indices],
            'attention_mask': encoded_inputs['attention_mask'][indices]}

# Create TensorDatasets
train_data = TensorDataset(select_samples(train_indices)['input_ids'], select_samples(train_indices)['attention_mask'], torch.tensor(train_labels))
val_data = TensorDataset(select_samples(val_indices)['input_ids'], select_samples(val_indices)['attention_mask'], torch.tensor(val_labels))
test_data = TensorDataset(select_samples(test_indices)['input_ids'], select_samples(test_indices)['attention_mask'], torch.tensor(test_labels))

# Create DataLoaders
batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# EWC-related variables
ewc_lambda = 0.4  # Regularization strength for oEWC
previous_task_means = {}
previous_task_fisher = {}
ewc_importance = {}

# ER buffer
buffer_size = 1000
replay_buffer = {'input_ids': [], 'attention_mask': [], 'labels': []}

# Refresh Learning settings
refresh_frequency = 1  # Refresh after every 'refresh_frequency' epochs
refresh_steps = 5  # Number of refresh steps per refresh cycle

# Function to add to buffer
def add_to_buffer(input_ids, attention_mask, labels):
    if len(replay_buffer['input_ids']) >= buffer_size:
        del replay_buffer['input_ids'][:batch_size]
        del replay_buffer['attention_mask'][:batch_size]
        del replay_buffer['labels'][:batch_size]
    replay_buffer['input_ids'].extend(input_ids.cpu().tolist())
    replay_buffer['attention_mask'].extend(attention_mask.cpu().tolist())
    replay_buffer['labels'].extend(labels.cpu().tolist())

# Function to sample from buffer
def sample_from_buffer(batch_size):
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

# Load the ChemBERTa model
model = RobertaForSequenceClassification.from_pretrained('seyonec/ChemBERTa-zinc-base-v1', num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 3

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
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        model.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
            fisher_info[name] += param.grad ** 2
    for name in fisher_info:
        fisher_info[name] /= len(data_loader)
    return fisher_info

# Variables for tracking accuracy
anytime_accuracies = []
initial_accuracies = {}
final_accuracies = {}

# Main training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        # Current task loss
        input_ids, attention_mask, labels = batch
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss

        # oEWC loss (if previous tasks exist)
        if len(previous_task_means) > 0:
            ewc_loss = compute_ewc_loss(model, previous_task_means, previous_task_fisher)
            total_loss = loss + ewc_lambda * ewc_loss
        else:
            total_loss = loss

        # ER loss
        if len(replay_buffer['input_ids']) >= batch_size:
            replay_batch = sample_from_buffer(batch_size)
            replay_output = model(input_ids=replay_batch['input_ids'], attention_mask=replay_batch['attention_mask'], labels=replay_batch['labels'])
            replay_loss = replay_output.loss

            # Compute gradients for both losses (current task and EWC)
            total_loss.backward(retain_graph=True)
            current_task_grads = [param.grad.clone() for param in model.parameters()]

            # Compute gradients for replay loss
            replay_loss.backward()
            replay_grads = [param.grad.clone() for param in model.parameters()]

            # Pareto optimization: blend gradients
            blended_grads = compute_pareto_gradients(current_task_grads, replay_grads)

            # Apply blended gradients
            for param, grad in zip(model.parameters(), blended_grads):
                param.grad = grad
        else:
            total_loss.backward()  # Only backpropagate current task loss if no replay buffer samples are available

        optimizer.step()
        total_loss += loss.item()

        # Add current batch to replay buffer
        add_to_buffer(input_ids, attention_mask, labels)

    avg_train_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss}')

    # Calculate Fisher Information for oEWC after the task
    current_fisher = compute_fisher_information(model, train_loader)

    # Store task-specific Fisher information and model parameters
    for name, param in model.named_parameters():
        previous_task_means[name] = param.detach().clone()
        previous_task_fisher[name] = current_fisher[name]

    # Refresh Learning: Retrain on old data from buffer
    if (epoch + 1) % refresh_frequency == 0 and len(replay_buffer['input_ids']) > 0:
        print(f"Performing Refresh Learning after Epoch {epoch + 1}")
        model.train()
        for _ in range(refresh_steps):
            replay_batch = sample_from_buffer(batch_size)
            optimizer.zero_grad()

            replay_output = model(input_ids=replay_batch['input_ids'], attention_mask=replay_batch['attention_mask'], labels=replay_batch['labels'])
            replay_loss = replay_output.loss
            replay_loss.backward()
            optimizer.step()

    # Validation after each epoch
    model.eval()
    val_preds = []
    val_true = []
    for batch in val_loader:
        input_ids, attention_mask, labels = batch
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        val_preds.extend(preds)
        val_true.extend(labels.cpu().tolist())

    val_accuracy = accuracy_score(val_true, val_preds)
    anytime_accuracies.append(val_accuracy)

    print(f'Epoch {epoch + 1} Validation Accuracy: {val_accuracy}')

# Calculate anytime average accuracy
avg_anytime_accuracy = sum(anytime_accuracies) / len(anytime_accuracies)
print(f'Anytime Average Accuracy: {avg_anytime_accuracy}')

# Test on the initial task
test_preds = []
test_true = []
for batch in test_loader:
    input_ids, attention_mask, labels = batch
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits
    preds = torch.argmax(logits, dim=1).cpu().tolist()
    test_preds.extend(preds)
    test_true.extend(labels.cpu().tolist())

test_accuracy = accuracy_score(test_true, test_preds)
initial_accuracies['Task1'] = test_accuracy
print(f'Initial Test Accuracy: {test_accuracy}')

# After further tasks, test again on the first task for forgetting measure
final_accuracies['Task1'] = test_accuracy

# Calculate forgetting measure
forgetting_measure = initial_accuracies['Task1'] - final_accuracies['Task1']
print(f'Forgetting Measure for Task1: {forgetting_measure}')