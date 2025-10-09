import torch
from transformers import RobertaForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from typing import Literal
import logging
from utils.train_utils import *
import requests

def train_loop(dataset: Literal["Sweet", "Bitter", "BBBP"], batch_size: int = 16, ewc_lambda = 0.4, buffer_size = 1000, epochs = 3, lr=2e-5, refresh_frequency = 1, refresh_steps = 5):
    train_data, val_data, test_data = load_data(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    #yield (device)
    requests.post(url="http://localhost:8000/add-msg", json={"message":f'Using device: {device}\n'})
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    previous_task_means = {}
    previous_task_fisher = {}

    replay_buffer = {'input_ids': [], 'attention_mask': [], 'labels': []}

    model = RobertaForSequenceClassification.from_pretrained('seyonec/ChemBERTa-zinc-base-v1', num_labels=2)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Variables for tracking accuracy
    anytime_accuracies = []
    initial_accuracies = {}
    final_accuracies = {}

    # Main training loop
    for epoch in range(epochs):
        print("In loop")
        #yield ("In loop")
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            # Current task loss
            
            input_ids, attention_mask, labels = batch
            output = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels.to(device))
            loss = output.loss

            # oEWC loss (if previous tasks exist)
            if len(previous_task_means) > 0:
                ewc_loss = compute_ewc_loss(model, previous_task_means, previous_task_fisher)
                total_loss = loss + ewc_lambda * ewc_loss
            else:
                total_loss = loss

            # ER loss
            if len(replay_buffer['input_ids']) >= batch_size:
                replay_batch = sample_from_buffer(replay_buffer, batch_size)
                replay_output = model(input_ids=replay_batch['input_ids'].to(device), attention_mask=replay_batch['attention_mask'].to(device), labels=replay_batch['labels'].to(device))
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
            add_to_buffer(replay_buffer, buffer_size, batch_size, input_ids, attention_mask, labels)

        avg_train_loss = total_loss / len(train_loader)
        requests.post(url="http://localhost:8000/add-msg", json={"message":f'Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss}\n'})

        print(f'Epocgh {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss}')
        #yield (f'Epocgh {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss}')

        # Calculate Fisher Information for oEWC after the task
        current_fisher = compute_fisher_information(model, train_loader)

        # Store task-specific Fisher information and model parameters
        for name, param in model.named_parameters():
            previous_task_means[name] = param.detach().clone()
            previous_task_fisher[name] = current_fisher[name]

        # Refresh Learning: Retrain on old data from buffer
        if (epoch + 1) % refresh_frequency == 0 and len(replay_buffer['input_ids']) > 0:
            requests.post(url="http://localhost:8000/add-msg", json={"message":f"Performing Refresh Learning after Epoch {epoch + 1}\n"})
    
            print(f"Performing Refresh Learning after Epoch {epoch + 1}")
            #yield (f"Performing Refresh Learning after Epoch {epoch + 1}")
            model.train()
            for _ in range(refresh_steps):
                replay_batch = sample_from_buffer(replay_buffer, batch_size)
                optimizer.zero_grad()

                replay_output = model(input_ids=replay_batch['input_ids'].to(device), attention_mask=replay_batch['attention_mask'].to(device), labels=replay_batch['labels'].to(device))
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
                output = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            logits = output.logits
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            val_preds.extend(preds)
            val_true.extend(labels.cpu().tolist())

        val_accuracy = accuracy_score(val_true, val_preds)
        anytime_accuracies.append(val_accuracy)

        requests.post(url="http://localhost:8000/add-msg", json={"message":f'Epoch {epoch + 1} Validation Accuracy: {val_accuracy}\n'})

        print(f'Epoch {epoch + 1} Validation Accuracy: {val_accuracy}')
        #yield (f'Epoch {epoch + 1} Validation Accuracy: {val_accuracy}')

    # Calculate anytime average accuracy
    avg_anytime_accuracy = sum(anytime_accuracies) / len(anytime_accuracies)
    requests.post(url="http://localhost:8000/add-msg", json={"message":f'Anytime Average Accuracy: {avg_anytime_accuracy}\n'})
    print(f'Anytime Average Accuracy: {avg_anytime_accuracy}')
    #yield (f'Anytime Average Accuracy: {avg_anytime_accuracy}')

    # Test on the initial task
    test_preds = []
    test_true = []
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        with torch.no_grad():
            output = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        logits = output.logits
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        test_preds.extend(preds)
        test_true.extend(labels.cpu().tolist())

    test_accuracy = accuracy_score(test_true, test_preds)
    initial_accuracies['Task1'] = test_accuracy
    requests.post(url="http://localhost:8000/add-msg", json={"message":f'Initial Test Accuracy: {test_accuracy}\n'})
    print(f'Initial Test Accuracy: {test_accuracy}')
    #yield (f'Initial Test Accuracy: {test_accuracy}')

    # After further tasks, test again on the first task for forgetting measure
    final_accuracies['Task1'] = test_accuracy

    # Calculate forgetting measure
    forgetting_measure = initial_accuracies['Task1'] - final_accuracies['Task1']
    requests.post(url="http://localhost:8000/add-msg", json={"message":f'Forgetting Measure for Task1: {forgetting_measure}\n'})
    print(f'Forgetting Measure for Task1: {forgetting_measure}')
    #yield (f'Forgetting Measure for Task1: {forgetting_measure}')