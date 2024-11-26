from transformers import VisionEncoderDecoderModel, TrOCRProcessor, AdamW, get_scheduler
from tqdm import tqdm
import torch
import preprocess
import config

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
print("Loading datasets...")
train_dataloader, processor = preprocess.load_and_preprocess_data(config.TRAIN_SPLIT)
val_dataloader, _ = preprocess.load_and_preprocess_data(config.VALIDATION_SPLIT)

# Initialize model
print("Initializing model...")
model = VisionEncoderDecoderModel.from_pretrained(config.MODEL_NAME)
model.to(device)

# Set special tokens for the decoder
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * config.EPOCHS
)

def train():
    """
    Train the model for the specified number of epochs.
    """
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0

        print(f"Starting Epoch {epoch + 1}/{config.EPOCHS}...")
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}: Training Loss: {avg_loss:.4f}")

        # Validate after each epoch
        validate()

    # Save the model and processor after training
    save_model()

def validate():
    """
    Validate the model on the validation dataset.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    print("Validating...")
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()

            # Generate predictions
            predictions = model.generate(pixel_values)
            predicted_texts = processor.batch_decode(predictions, skip_special_tokens=True)
            true_texts = processor.batch_decode(labels, skip_special_tokens=True)

            # Compare predictions to ground truth
            for pred, true in zip(predicted_texts, true_texts):
                if pred.strip() == true.strip():
                    correct += 1
                total += 1

    avg_loss = total_loss / len(val_dataloader)
    accuracy = correct / total if total > 0 else 0.0
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

def save_model():
    """
    Save the trained model and processor.
    """
    print("Saving the trained model and processor...")
    model.save_pretrained(config.OUTPUT_DIR)
    processor.save_pretrained(config.OUTPUT_DIR)
    print(f"Model and processor saved to {config.OUTPUT_DIR}")

if __name__ == "__main__":
    print("Starting training process...")
    train()
