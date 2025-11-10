import torch
import chess.pgn
from io import StringIO

from collections import defaultdict

def data_pgn_to_fens(pgn_string: str, max_positions: int = 5, skip_early_moves: int = 10, max_moves: int = 80) -> list[str]:
    """
    Convert a PGN string to a list of FEN positions.
    """
    try:
        pgn_io = StringIO(pgn_string)
        game = chess.pgn.read_game(pgn_io)

        if game is None:
            return []

        board = game.board()
        fens = []
        move_count = 0

        for move in game.mainline_moves():
            move_count += 1
            board.push(move)

            if move_count < skip_early_moves or move_count > max_moves:
                continue

            if len(fens) < max_positions:
                interval = max(1, (max_moves - skip_early_moves) // max_positions)
                if (move_count - skip_early_moves) % interval == 0:
                    fens.append(board.fen())
        return fens

    except Exception as e:
        # print(f"Error processing PGN: {e}") # Suppress for cleaner output in production
        return []

# --- Square-Centric (SC) Representation Constants ---
SC_PIECE_TO_INT = {
    'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
    'P': 7, 'N': 8, 'B': 9, 'R': 10, 'Q': 11, 'K': 12
}
SC_INT_TO_PIECE = {v: k for k, v in SC_PIECE_TO_INT.items()}
SC_EMPTY_SQUARE_INT = 0 # Explicitly define 0 for empty in SC
SC_ABSORBING_STATE_INT = 13
SC_VOCAB_SIZE = 14  # 12 pieces + 1 empty (0) + 1 absorbing (13)
SC_SEQUENCE_LENGTH = 64

# --- Piece-Centric (PC) Representation Constants ---
PC_SEQUENCE_LENGTH = 32 # Max 32 pieces on a board
PC_OFF_BOARD_INT = 0    # Value for a piece that is off the board
PC_ABSORBING_STATE_INT = 65 # Absorbing state for piece-centric
PC_VOCAB_SIZE = 66      # 64 squares (1-64) + 1 off-board (0) + 1 absorbing (65)

# Defines the canonical order of pieces for the PC representation.
# Used for both fen_to_tensor and tensor_to_fen.
PIECE_ORDER = [
    chess.Piece(chess.KING, chess.WHITE),
    chess.Piece(chess.QUEEN, chess.WHITE),
    chess.Piece(chess.ROOK, chess.WHITE),
    chess.Piece(chess.ROOK, chess.WHITE),
    chess.Piece(chess.KNIGHT, chess.WHITE),
    chess.Piece(chess.KNIGHT, chess.WHITE),
    chess.Piece(chess.BISHOP, chess.WHITE),
    chess.Piece(chess.BISHOP, chess.WHITE),
    chess.Piece(chess.PAWN, chess.WHITE),
    chess.Piece(chess.PAWN, chess.WHITE),
    chess.Piece(chess.PAWN, chess.WHITE),
    chess.Piece(chess.PAWN, chess.WHITE),
    chess.Piece(chess.PAWN, chess.WHITE),
    chess.Piece(chess.PAWN, chess.WHITE),
    chess.Piece(chess.PAWN, chess.WHITE),
    chess.Piece(chess.PAWN, chess.WHITE),
    chess.Piece(chess.KING, chess.BLACK),
    chess.Piece(chess.QUEEN, chess.BLACK),
    chess.Piece(chess.ROOK, chess.BLACK),
    chess.Piece(chess.ROOK, chess.BLACK),
    chess.Piece(chess.KNIGHT, chess.BLACK),
    chess.Piece(chess.KNIGHT, chess.BLACK),
    chess.Piece(chess.BISHOP, chess.BLACK),
    chess.Piece(chess.BISHOP, chess.BLACK),
    chess.Piece(chess.PAWN, chess.BLACK),
    chess.Piece(chess.PAWN, chess.BLACK),
    chess.Piece(chess.PAWN, chess.BLACK),
    chess.Piece(chess.PAWN, chess.BLACK),
    chess.Piece(chess.PAWN, chess.BLACK),
    chess.Piece(chess.PAWN, chess.BLACK),
    chess.Piece(chess.PAWN, chess.BLACK),
    chess.Piece(chess.PAWN, chess.BLACK),
]

def sc_fen_to_tensor(fen_string: str) -> torch.Tensor:
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
            board_tensor[square_index] = SC_PIECE_TO_INT[char]
            file_index += 1

    return board_tensor

def sc_tensor_to_fen(board_tensor: torch.Tensor, active_color: str ='w', castling: str ='KQkq', en_passant: str ='-', halfmove_clock: int =0, fullmove_number: int=1) -> str:
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
    if board_tensor.shape != (SC_SEQUENCE_LENGTH,):
        raise ValueError(f"Input tensor must have shape ({SC_SEQUENCE_LENGTH},)")

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
                rank_fen += SC_INT_TO_PIECE[piece_int]

        if empty_count > 0:
            rank_fen += str(empty_count)

        fen_parts.append(rank_fen)

    piece_placement = "/".join(fen_parts)

    # Combine all parts of the FEN string
    full_fen = f"{piece_placement} {active_color} {castling} {en_passant} {halfmove_clock} {fullmove_number}"

    return full_fen

def sc_mask_board(fen_str: str, ratio: float, absorbing_state_idx: int = SC_ABSORBING_STATE_INT) -> torch.Tensor:
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

    board = sc_fen_to_tensor(fen_str).float()
    # Create a random tensor and threshold it to create a boolean mask
    mask = torch.rand_like(board) < ratio  # True for ~ratio% of elements

    # Use the mask to replace values with the absorbing state (13)
    masked_board = torch.where(mask, board, torch.tensor(float(absorbing_state_idx)))

    return masked_board.long()

def pc_fen_to_tensor(fen_string: str) -> torch.Tensor | None:
    """
    Parses a FEN string and converts it into a 1D tensor of PC_SEQUENCE_LENGTH tokens.
    Each token represents a piece from PIECE_ORDER. The token's value is the square it's on (1-64),
    0 if the piece is off board, or PC_ABSORBING_STATE_INT if masked/unknown.

    Returns None if the board state implies promotions or too many pieces,
    which cannot be represented cleanly in this fixed-piece format.
    """
    board_tensor = torch.full((PC_SEQUENCE_LENGTH,), PC_OFF_BOARD_INT, dtype=torch.long)
    try:
        board = chess.Board(fen_string)
    except (ValueError, IndexError): # Catch malformed FENs
        return None

    found_pieces_on_board = defaultdict(list) # Maps chess.Piece objects to a list of their squares (+1 for 1-64)

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        # Only consider pieces that are on the board
        # For pawns, we ignore those on the first/last rank as they imply promotion
        if piece and (piece.piece_type != chess.PAWN or chess.square_rank(sq) not in [0, 7]):
            found_pieces_on_board[piece].append(sq + 1) # Store 1-64 values

    # Validate piece counts: check if any piece type count exceeds the initial setup
    # This helps filter out boards with promoted pawns that don't fit our fixed PIECE_ORDER
    for piece_type, locations in found_pieces_on_board.items():
        if len(locations) > PIECE_ORDER.count(piece_type):
            return None # Board has more of a piece type than start, likely due to promotion

    # Sort locations for canonical representation
    for piece_type in found_pieces_on_board:
        found_pieces_on_board[piece_type].sort()

    piece_counters = defaultdict(int) # To track instances of each piece type (e.g., White Rook 1, White Rook 2)
    for i, piece_to_find in enumerate(PIECE_ORDER):
        instance_index = piece_counters[piece_to_find]
        if instance_index < len(found_pieces_on_board[piece_to_find]):
            square_location = found_pieces_on_board[piece_to_find][instance_index]
            board_tensor[i] = square_location
        # If the piece_to_find isn't found (e.g., it was captured), its position remains PC_OFF_BOARD_INT (0)
        piece_counters[piece_to_find] += 1

    return board_tensor

def pc_tensor_to_fen(board_tensor: torch.Tensor, active_color: str ='w', castling: str ='KQkq', en_passant: str ='-', halfmove_clock: int =0, fullmove_number: int=1) -> str:
    """
    Converts a 1D piece-centric tensor (PC_SEQUENCE_LENGTH tokens) back into a FEN string.
    """
    if board_tensor.shape != (PC_SEQUENCE_LENGTH,):
        raise ValueError(f"Input tensor must have shape ({PC_SEQUENCE_LENGTH},)")

    board = chess.Board(fen=None) # Start with an empty board

    for i, square_val_tensor in enumerate(board_tensor):
        square_val = square_val_tensor.item()
        if 1 <= square_val <= 64: # If the value is a valid square
            if i < len(PIECE_ORDER): # Ensure we don't go out of bounds of PIECE_ORDER
                piece_to_place = PIECE_ORDER[i]
                square_index = square_val - 1 # Convert 1-64 to 0-63
                board.set_piece_at(square_index, piece_to_place)
            else:
                # This case should ideally not happen if PC_SEQUENCE_LENGTH matches len(PIECE_ORDER)
                print(f"Warning: Tensor index {i} out of bounds for PIECE_ORDER.")
        elif square_val == PC_OFF_BOARD_INT:
            # Piece is off-board, do nothing. It's simply not placed on the board.
            pass
        elif square_val == PC_ABSORBING_STATE_INT:
            # Absorbing state means it's unknown/masked. For FEN reconstruction,
            # we treat this as if the piece is off-board (i.e., not placed).
            pass
        else:
            print(f"Warning: Invalid square value {square_val} encountered at index {i}.")

    piece_placement = board.board_fen()
    full_fen = f"{piece_placement} {active_color} {castling} {en_passant} {halfmove_clock} {fullmove_number}"
    return full_fen

def pc_mask_board(board_tensor: torch.Tensor, ratio: float, absorbing_state_idx: int = PC_ABSORBING_STATE_INT) -> torch.Tensor:
    """
    Masks a ratio of pieces on a piece-centric board tensor, replacing their square value
    with the absorbing state.

    Args:
        board_tensor (torch.Tensor): The 1D piece-centric tensor of shape (PC_SEQUENCE_LENGTH,).
        ratio (float): The ratio of pieces to mask (between 0 and 1).
        absorbing_state_idx (int): The integer representing the absorbing state.

    Returns:
        torch.Tensor: A tensor representing the masked piece-centric board.
    """
    assert 0.0 <= ratio <= 1.0, "Ratio must be between 0 and 1."
    if board_tensor.shape != (PC_SEQUENCE_LENGTH,):
        raise ValueError(f"Input tensor must have shape ({PC_SEQUENCE_LENGTH},)")

    # Create a random mask: True for elements to be masked
    mask = torch.rand_like(board_tensor.float()) < ratio

    # Apply the mask: replace values with absorbing_state_idx where mask is True
    masked_board = torch.where(mask, torch.tensor(float(absorbing_state_idx), device=board_tensor.device), board_tensor.float())

    return masked_board.long()

if __name__ == "__main__":
    TEST_FEN_STR = "r2q1Bk1/p3pp1p/1pnp2p1/2p5/P3P3/1P1P1N2/2P2KnP/R2Q3R b - - 0 14"

    # Extract game state components for accurate FEN reconstruction
    fen_parts = TEST_FEN_STR.split(' ')
    piece_placement_only = fen_parts[0]
    active_color_orig = fen_parts[1]
    castling_orig = fen_parts[2]
    en_passant_orig = fen_parts[3]
    halfmove_clock_orig = int(fen_parts[4])
    fullmove_number_orig = int(fen_parts[5])

    print("--- Testing Square-Centric (SC) Functions ---")
    sc_tensor = sc_fen_to_tensor(TEST_FEN_STR)

    if sc_tensor is not None:
        print(f"SC FEN to Tensor (first 16 tokens): {sc_tensor[0:16].tolist()}...")
        re_sc_fen = sc_tensor_to_fen(
            sc_tensor,
            active_color=active_color_orig,
            castling=castling_orig,
            en_passant=en_passant_orig,
            halfmove_clock=halfmove_clock_orig,
            fullmove_number=fullmove_number_orig
        )
        print(f"SC Tensor to FEN: {re_sc_fen}")
        assert TEST_FEN_STR == re_sc_fen, "SC FEN roundtrip failed!"
        print("SC FEN roundtrip successful.")

        masked_sc_tensor = sc_mask_board(TEST_FEN_STR, 0.5)
        print(f"Masked SC Tensor (first 16 tokens): {masked_sc_tensor[0:16].tolist()}...")
        # For display, replace absorbing state with 0 (empty square)
        display_masked_sc_tensor = torch.where(masked_sc_tensor == SC_ABSORBING_STATE_INT, SC_EMPTY_SQUARE_INT, masked_sc_tensor)
        print(f"Masked SC Board FEN (for display): {sc_tensor_to_fen(display_masked_sc_tensor).split(' ')[0]}")
    else:
        print("SC FEN to Tensor failed.")
    print("-" * 40)

    print("\n--- Testing Piece-Centric (PC) Functions ---")
    pc_tensor = pc_fen_to_tensor(TEST_FEN_STR)

    if pc_tensor is not None:
        print(f"PC FEN to Tensor: {pc_tensor.tolist()}")
        re_pc_fen = pc_tensor_to_fen(
            pc_tensor,
            active_color=active_color_orig,
            castling=castling_orig,
            en_passant=en_passant_orig,
            halfmove_clock=halfmove_clock_orig,
            fullmove_number=fullmove_number_orig
        )
        print(f"PC Tensor to FEN: {re_pc_fen}")

        assert TEST_FEN_STR.split(' ')[0] == re_pc_fen.split(' ')[0], "PC FEN roundtrip (piece placement) failed!"
        print("PC FEN roundtrip (piece placement) successful.")

        masked_pc_tensor = pc_mask_board(pc_tensor, 0.5)
        print(f"Masked PC Tensor: {masked_pc_tensor.tolist()}")
        # For display, replace absorbing state with OFF_BOARD_INT (0)
        display_masked_pc_tensor = torch.where(masked_pc_tensor == PC_ABSORBING_STATE_INT, PC_OFF_BOARD_INT, masked_pc_tensor)
        print(f"Masked PC Board FEN (for display): {pc_tensor_to_fen(display_masked_pc_tensor).split(' ')[0]}")
    else:
        print(f"PC FEN to Tensor failed for '{TEST_FEN_STR}'. This FEN might imply promotions.")
    print("-" * 40)
    # [PGN testing NOT DONE]
