import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# We will reuse the FEN parsing logic from the utils file
from chess_utils import fen_to_tensor, tensor_to_fen, VOCAB_SIZE
from visualize import display_board_from_tensor

# --- Configuration ---
CSV_FILE = 'positions.csv'
OUTPUT_FILE = 'chess_positions.pt'
NUM_SAMPLES = 50000  # Number of random positions to include in the dataset

class ChessDataset(Dataset):
    """
    A PyTorch Dataset for loading pre-processed chess board tensors.
    """
    def __init__(self, tensor_file=OUTPUT_FILE):
        """
        Args:
            tensor_file (str): Path to the .pt file containing the board tensors.
        """
        if not os.path.exists(tensor_file):
            raise FileNotFoundError(
                f"Dataset file not found: {tensor_file}. "
                f"Please run create_chess_dataset.py first to generate it."
            )
        print(f"Loading dataset from {tensor_file}...")
        self.data = torch.load(tensor_file)
        print("Dataset loaded successfully.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def create_and_save_dataset():
    """
    Reads the FENs from the CSV, converts them to tensors, and saves them to a file.
    """
    print(f"--- Starting Dataset Creation ---")
    try:
        # Use low_memory=False to handle the DtypeWarning we saw earlier
        df = pd.read_csv(CSV_FILE, low_memory=False)
        print(f"Successfully loaded '{CSV_FILE}' with {len(df)} total positions.")
    except FileNotFoundError:
        print(f"ERROR: The file '{CSV_FILE}' was not found.")
        return

    # Ensure we don't request more samples than available
    num_to_sample = min(NUM_SAMPLES, len(df))
    print(f"Taking a random sample of {num_to_sample} positions.")

    # Take a random sample from the dataframe
    if num_to_sample < len(df):
        df_sample = df.sample(n=num_to_sample, random_state=23)
    else:
        df_sample = df

    # List to hold all the tensor representations
    board_tensors = []

    print("Converting FEN strings to tensors...")
    # Iterate through the 'fen' column and convert each to a tensor
    for fen_string in tqdm(df_sample['fen'], desc="Processing FENs"):
        try:
            board_tensor = fen_to_tensor(fen_string)
            board_tensors.append(board_tensor)
        except Exception as e:
            print(f"\nSkipping invalid or malformed FEN: '{fen_string}'. Error: {e}")

    # Stack all individual tensors into a single large tensor
    # The shape will be (NUM_SAMPLES, 64)
    final_dataset_tensor = torch.stack(board_tensors).long()

    print(f"\nConversion complete. Final tensor shape: {final_dataset_tensor.shape}")

    # Save the tensor to the output file
    torch.save(final_dataset_tensor, OUTPUT_FILE)
    print(f"Dataset successfully saved to '{OUTPUT_FILE}'.")
    print(f"--- Dataset Creation Finished ---")

    return final_dataset_tensor

if __name__ == "__main__":
    # 1. Generate and save the dataset file
    dataset_tensor = create_and_save_dataset()

    # 2. (Optional) Demonstrate how to use the ChessDataset class
    if dataset_tensor is not None and len(dataset_tensor) > 0:
        print("\n--- Demonstrating Dataset Usage ---")
        # This is how you would use it in your training script
        chess_dataset = ChessDataset(tensor_file=OUTPUT_FILE)

        # Create a DataLoader to batch and shuffle the data
        data_loader = DataLoader(chess_dataset, batch_size=4, shuffle=True)

        # Get one batch of data
        first_batch = next(iter(data_loader))

        print(f"Shape of one batch from DataLoader: {first_batch.shape}")
        print(f"Data type of the batch: {first_batch.dtype}")
        print(f"Vocabulary size (number of unique piece states): {VOCAB_SIZE}")

        # Visualize the first board in the batch to verify correctness
        print("\nVisualizing the first board from the first batch...")
        first_board_tensor = first_batch[0]

        # You will need cairosvg installed: pip install cairosvg
        display_board_from_tensor(first_board_tensor)

        reconstructed_fen = tensor_to_fen(first_board_tensor)
        print(f"Reconstructed FEN for the visualized board: {reconstructed_fen}")
        print("------------------------------------")
