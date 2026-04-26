import torch
import json
from transformers import BertTokenizerFast, BertForSequenceClassification

MODEL_PATH = "model"

# Load label names
with open(f"{MODEL_PATH}/labels.json", "r") as f:
    label_names = json.load(f)

# Load tokenizer & model
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()

    return {
        "label": label_names[pred_id],
        "confidence": float(probs[0][pred_id]),
        "all_probs": {
            label_names[i]: float(probs[0][i])
            for i in range(len(label_names))
        }
    }