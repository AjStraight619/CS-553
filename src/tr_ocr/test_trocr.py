import torch
import preprocess
import config
from transformers import TrOCRForConditionalGeneration

# Load test dataset
test_dataloader, processor = preprocess.load_and_preprocess_data(config.TEST_SPLIT)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrOCRForConditionalGeneration.from_pretrained(config.OUTPUT_DIR)
model.to(device)

def test():
    """
    Test the model on the test dataset.
    """
    model.eval()
    correct = 0
    total = 0

    print("Testing...")
    with torch.no_grad():
        for batch in test_dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            predictions = model.generate(pixel_values)
            predicted_texts = processor.batch_decode(predictions, skip_special_tokens=True)
            true_texts = processor.batch_decode(labels, skip_special_tokens=True)

            # Compare predictions to ground truth
            for pred, true in zip(predicted_texts, true_texts):
                if pred.strip() == true.strip():
                    correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    test()