from datasets import load_dataset
from transformers import TrOCRProcessor
import config

# Load the dataset split (adjust as needed)
print(f"Loading and preprocessing dataset: {config.DATASET_NAME} ({config.TRAIN_SPLIT}) with config '{config.DATASET_CONFIG}'...")

# Load dataset
dataset = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, split=config.TRAIN_SPLIT)

# Initialize processor
processor = TrOCRProcessor.from_pretrained(config.MODEL_NAME)

def preprocess_data(examples):
    """
    Preprocess the dataset: extract features from images and tokenize LaTeX formulas.
    """
    images = [image.convert("RGB") for image in examples["image"]]
    pixel_values = processor.feature_extractor(images=images, return_tensors="pt").pixel_values
    labels = processor.tokenizer(
        examples["latex_formula"],
        padding="max_length",
        truncation=True,
        max_length=config.MAX_LENGTH,
        return_tensors="pt"
    ).input_ids
    return {"pixel_values": pixel_values, "labels": labels}

# Preprocess the dataset
dataset = dataset.map(preprocess_data, batched=True)

# Save the preprocessed dataset to disk
save_path = "preprocessed_train_dataset"
dataset.save_to_disk(save_path)
print(f"Preprocessed dataset saved to {save_path}")
