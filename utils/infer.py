import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification


def inference(smiles_input:str, dataset:str):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load model and tokenizer ===
    model_path = f"./models/TIL/{dataset}/"  # <-- change this to your actual model path
    tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    try:
        model = RobertaForSequenceClassification.from_pretrained(pretrained_model_name_or_path = model_path, local_files_only=True)
    except Exception as e:
        print(e)
        return {"classification":"Model not found. Please ensure the model has been trained before inference."}
    model.to(device)
    model.eval()

    # === Single input example ===
    # smiles_input = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Example: Aspirin

    # Tokenize input
    encoded_input = tokenizer(
        smiles_input,
        padding='max_length',
        truncation=True,
        max_length=50,
        return_tensors='pt'
    ).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**encoded_input)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predicted_label = torch.argmax(probs, dim=1).item()

    # === Print results ===
    print(f"Input SMILES: {smiles_input}")
    print(f"Predicted probabilities: {probs.cpu().numpy().flatten()}")
    print(f"Predicted label: {predicted_label}")
    return {"classification":str(predicted_label),"model":f"TIL_ChemBERTa"}