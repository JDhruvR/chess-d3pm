import torch

# Define the vocabulary for a single square (token)
# This mapping is crucial and must be used consistently.
PIECE_TO_INT = {
    'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
    'P': 7, 'N': 8, 'B': 9, 'R': 10, 'Q': 11, 'K': 12
}
INT_TO_PIECE = {v: k for k, v in PIECE_TO_INT.items()}
ABSORBING_STATE_INT = 13
VOCAB_SIZE = 14  # 12 pieces + 1 empty + 1 absorbing

def fen_to_tensor(fen_string: str) -> torch.Tensor:
    """
    Parses a FEN string and converts it into a 1D tensor of 64 tokens.
    Each token represents a square on the board, with an integer value
    corresponding to the piece on it (0 for empty).

    Args:
        fen_string (str): The FEN string for a board position.

    Returns:
        torch.Tensor: A 1D tensor of shape (64,) with integer piece representations.
    """
    # The board tensor, initialized to 0 (empty)
    board_tensor = torch.zeros(64, dtype=torch.long)

    # The first part of FEN is the piece placement
    piece_placement = fen_string.split(' ')[0]

    rank_index = 7  # Start from rank 8 (index 7)
    file_index = 0  # Start from file 'a' (index 0)

    for char in piece_placement:
        if char == '/':
            rank_index -= 1
            file_index = 0
        elif char.isdigit():
            file_index += int(char)
        else:
            square_index = rank_index * 8 + file_index
            board_tensor[square_index] = PIECE_TO_INT[char]
            file_index += 1

    return board_tensor

def tensor_to_fen(board_tensor: torch.Tensor, active_color='w', castling='KQkq', en_passant='-', halfmove_clock=0, fullmove_number=1) -> str:
    """
    Converts a 1D tensor of 64 tokens back into a FEN string.

    Note: This function only reconstructs the piece placement part of the FEN.
    Other game state information (turn, castling, etc.) must be provided.

    Args:
        board_tensor (torch.Tensor): A 1D tensor of shape (64,) representing the board.
        active_color (str): 'w' or 'b'.
        castling (str): Castling availability (e.g., 'KQkq', '-', 'Kq').
        en_passant (str): En passant target square (e.g., 'e3', '-').
        halfmove_clock (int): Halfmove clock value.
        fullmove_number (int): Fullmove number.

    Returns:
        str: The full FEN string for the position.
    """
    if board_tensor.shape != (64,):
        raise ValueError("Input tensor must have shape (64,)")

    fen_parts = []
    for rank_index in range(7, -1, -1):  # Iterate from rank 8 down to 1
        rank_fen = ""
        empty_count = 0
        for file_index in range(8):  # Iterate from file 'a' to 'h'
            square_index = rank_index * 8 + file_index
            piece_int = board_tensor[square_index].item()

            if piece_int == 0:
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_fen += str(empty_count)
                    empty_count = 0
                rank_fen += INT_TO_PIECE[piece_int]

        if empty_count > 0:
            rank_fen += str(empty_count)

        fen_parts.append(rank_fen)

    piece_placement = "/".join(fen_parts)

    # Combine all parts of the FEN string
    full_fen = f"{piece_placement} {active_color} {castling} {en_passant} {halfmove_clock} {fullmove_number}"

    return full_fen

def mask_half(fen_str: str, ratio: float):
    """
    Masks approximately half of the squares on a chessboard represented by a FEN string,
    replacing them with the absorbing state (13).

    Args:
        fen_str (str): The FEN string representing the chessboard.
        ratio (float): The ratio of squares to mask (between 0 and 1). E.g 0.6 masks 60% of the squares.

    Returns:
        torch.Tensor: A tensor representing the masked board.
    """
    assert 0.0 <= ratio <= 1.0, "Ratio must be between 0 and 1."

    board = fen_to_tensor(fen_str).float()
    # Create a random tensor and threshold it to create a boolean mask
    mask = torch.rand_like(board) < ratio  # True for ~ratio% of elements

    # Use the mask to replace values with the absorbing state (13)
    masked_board = torch.where(mask, board, torch.tensor(13.0))

    return masked_board.long()

if __name__ == "__main__":
    fen_str = "r2q1Bk1/p3pp1p/1pnp2p1/2p5/P3P3/1P1P1N2/2P2KnP/R2Q3R b - - 0 14"
    print(mask_half(fen_str, 0.5))
