# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
tokenizer = AutoTokenizer.from_pretrained("vietdata/vietnamese-content-cls")
model = AutoModelForSequenceClassification.from_pretrained("vietdata/vietnamese-content-cls")
import argparse


def process_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        sentences = [line.strip() for line in file.readlines()]

    # Tokenize input sentences
    tokenized_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**tokenized_inputs)

    # Get predicted logits
    logits = outputs.logits

    # Get predicted class probabilities using softmax
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Get predicted class indices
    predicted_class_indices = torch.argmax(probs, dim=-1)

    # Get class labels from the model config
    class_labels = model.config.id2label

    predicted_classes = [class_labels[idx.item()] for idx in predicted_class_indices]

    # Initialize a dictionary to count predicted classes
    class_counts = {label: 0 for label in set(class_labels.values())}

    # Update counts
    for predicted_class in predicted_classes:
        class_counts[predicted_class] += 1

    print("Class Counts:")
    for label, count in class_counts.items():
        print(f"{label}: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a file and count predicted class occurrences.")
    parser.add_argument("--path", type=str, help="Path to the input file", required=True)
    
    args = parser.parse_args()
    process_file(args.path)