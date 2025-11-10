import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import kagglehub
from kagglehub import KaggleDatasetAdapter
from io import StringIO
import chess.pgn

# --- Local Imports from updated chess_utils ---
from chess_utils import (
    sc_fen_to_tensor, sc_tensor_to_fen,
    SC_VOCAB_SIZE, SC_SEQUENCE_LENGTH, SC_ABSORBING_STATE_INT,
    data_pgn_to_fens # The PGN parsing utility
)
from visualize import display_sc_board_from_tensor # For demonstration

# --- Configuration ---
KAGGLE_DATASET_REF = "farhadzamani/chess-com-2022" # Kaggle dataset for PGNs
PGN_FILE_NAME = "gm_games_2022.csv"
OUTPUT_FILE = 'chess_positions_sc.pt'
TARGET_FENS = 5000000 # Number of FEN positions to aim for
MAX_GAMES_TO_PROCESS = None # Set to an int like 10000 to limit games processed, None for all

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
                f"Please run this script first to generate it."
            )
        print(f"Loading dataset from {tensor_file}...")
        self.data = torch.load(tensor_file)
        print("Dataset loaded successfully.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def create_fen_dataset_from_pgns(df: pd.DataFrame, target_fens: int, max_games: int | None) -> list[str]:
    """
    Extracts FEN positions from PGN games in a DataFrame.
    """
    print(f"Extracting FENs from {len(df)} games...")
    print(f"Targeting {target_fens:,} FEN positions.")

    all_fens = []
    games_processed = 0

    # Shuffle the dataframe to get random games
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    for idx, row in tqdm(df_shuffled.iterrows(), total=len(df_shuffled), desc="Processing PGNs"):
        if len(all_fens) >= target_fens:
            break
        if max_games and games_processed >= max_games:
            break

        fens = data_pgn_to_fens(row['pgn'], max_positions=3, skip_early_moves=8, max_moves=70)
        if fens:
            all_fens.extend(fens)
        games_processed += 1

    print(f"\nFEN extraction complete!")
    print(f"Games processed: {games_processed:,}")
    print(f"Total FENs collected: {len(all_fens):,}")
    return all_fens

def create_and_save_sc_dataset():
    """
    Reads PGNs, converts them to Square-Centric tensors, and saves them to a file.
    """
    print("--- Starting Square-Centric (SC) Dataset Creation ---")

    # Download/load the Kaggle dataset
    print(f"Loading PGN data from Kaggle dataset: {KAGGLE_DATASET_REF}/{PGN_FILE_NAME}...")
    try:
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            KAGGLE_DATASET_REF,
            PGN_FILE_NAME,
        )
        # Filter for standard chess, blitz, and non-null PGNs
        df = df[
            (df['rules'] == 'chess') &
            (df['time_class'] == 'blitz') &
            (df['pgn'].notnull())
        ].copy()
        print(f"Successfully loaded and filtered {len(df)} games from '{PGN_FILE_NAME}'.")
    except Exception as e:
        print(f"ERROR: Could not load PGN dataset from Kaggle. Make sure you have Kaggle API configured. Error: {e}")
        return

    # 1. Extract FEN strings
    fen_list = create_fen_dataset_from_pgns(df, TARGET_FENS, MAX_GAMES_TO_PROCESS)

    # 2. Convert FEN strings to SC tensors
    print("\nConverting FEN strings to Square-Centric tensors...")
    board_tensors = []
    failed_conversions = 0

    for fen_string in tqdm(fen_list, desc="Processing FENs to SC Tensors"):
        try:
            # We only care about the piece placement for the tensor,
            # so we normalize the rest of the FEN for `sc_fen_to_tensor`
            board_fen_placement = fen_string.split(' ')[0] + " w KQkq - 0 1"
            board_tensor = sc_fen_to_tensor(board_fen_placement)
            board_tensors.append(board_tensor)
        except Exception as e:
            failed_conversions += 1
            # print(f"\nSkipping invalid or malformed FEN: '{fen_string}'. Error: {e}") # Enable for debug

    # Stack all individual tensors into a single large tensor
    final_dataset_tensor = torch.stack(board_tensors).long()

    print(f"\nConversion complete. Final tensor shape: {final_dataset_tensor.shape}")
    print(f"Failed/skipped FEN conversions: {failed_conversions:,}")

    # Save the tensor to the output file
    torch.save(final_dataset_tensor, OUTPUT_FILE)
    print(f"SC dataset successfully saved to '{OUTPUT_FILE}'.")
    print(f"--- SC Dataset Creation Finished ---")

    return final_dataset_tensor

if __name__ == "__main__":
    # 1. Generate and save the dataset file
    dataset_tensor = create_and_save_sc_dataset()

    # 2. (Optional) Demonstrate how to use the ChessDataset class
    if dataset_tensor is not None and len(dataset_tensor) > 0:
        print("\n--- Demonstrating SC Dataset Usage ---")
        chess_dataset = ChessDataset(tensor_file=OUTPUT_FILE)
        data_loader = DataLoader(chess_dataset, batch_size=4, shuffle=True)
        first_batch = next(iter(data_loader))

        print(f"Shape of one batch from DataLoader: {first_batch.shape}")
        print(f"Data type of the batch: {first_batch.dtype}")
        print(f"Vocabulary size (SC): {SC_VOCAB_SIZE}")
        print(f"Sequence Length (SC): {SC_SEQUENCE_LENGTH}")

        print("\nVisualizing the first board from the first batch (SC)...")
        first_board_tensor = first_batch[0]
        display_sc_board_from_tensor(first_board_tensor, save_path='sample_sc_dataset_board.png', show_image=False)

        reconstructed_fen = sc_tensor_to_fen(first_board_tensor)
        print(f"Reconstructed FEN for the visualized board: {reconstructed_fen}")
        print("------------------------------------")
