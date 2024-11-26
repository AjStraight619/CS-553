from datasets import load_dataset
from transformers import TrOCRProcessor
import config
import os

def preprocess_and_save_dataset(start_index, end_index, save_dir):
    """
    Preprocess and save a dataset chunk to a specific directory.
    Args:
        start_index (int): Start index of the dataset split.
        end_index (int): End index of the dataset split.
        save_dir (str): Directory to save the preprocessed dataset chunk.
    """
    split = f"train[{start_index}:{end_index}]"
    save_path = os.path.join(save_dir, f"chunk_{start_index}_{end_index}")

    print(f"Loading and preprocessing dataset: {config.DATASET_NAME} ({split}) with config '{config.DATASET_CONFIG}'...")
    dataset = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, split=split)

    # Initialize processor
    processor = TrOCRProcessor.from_pretrained(config.MODEL_NAME)

    def preprocess_data(examples):
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
    print(f"Preprocessing dataset chunk {start_index}-{end_index}...")
    dataset = dataset.map(preprocess_data, batched=True)

    # Save the dataset chunk
    os.makedirs(save_dir, exist_ok=True)
    dataset.save_to_disk(save_path)
    print(f"Preprocessed dataset chunk saved to {save_path}")


if __name__ == "__main__":
    preprocess_and_save_dataset(config.START_INDEX, config.END_INDEX, config.SAVE_DIR)
