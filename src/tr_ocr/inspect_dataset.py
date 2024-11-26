from datasets import load_dataset
import config

# Inspect dataset structure
def inspect_dataset():
    """
    Load and inspect the dataset structure and a few examples.
    """
    print(f"Loading the dataset: {config.DATASET_NAME} ({config.DATASET_SUBSET}) with 'cleaned_formulas' config...")
    dataset = load_dataset(config.DATASET_NAME, "cleaned_formulas", split=config.DATASET_SUBSET)

    # Print dataset structure
    print(f"Dataset columns: {dataset.column_names}")
    print(f"First example: {dataset[0]}")

# Run inspection
if __name__ == "__main__":
    inspect_dataset()