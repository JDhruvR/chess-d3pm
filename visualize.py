import chess
import chess.svg
from PIL import Image
from io import BytesIO
import torch
import os

from chess_utils import (
    sc_tensor_to_fen, sc_fen_to_tensor,
    pc_tensor_to_fen, pc_fen_to_tensor,
)

def display_sc_board_from_tensor(board_tensor: torch.Tensor, size=400, save_path=None, show_image=False):
    """
    Generates a visual representation of the board from a tensor.

    Args:
        board_tensor (torch.Tensor): The 1D tensor of shape (64,) representing the board.
        size (int): The size of the output image in pixels.
        save_path (str, optional): Path to save the image file (e.g., 'board.png').
                                   If None, the image is not saved. Defaults to None.
        show_image (bool, optional): If True, attempts to open the image in the default
                                     system viewer. Defaults to False.
    """
    # Convert the tensor to a FEN string. We use default values for game state
    # as they don't affect the visual piece placement.
    fen = sc_tensor_to_fen(board_tensor)

    # Create a board object from the FEN
    board = chess.Board(fen)

    # Generate an SVG image of the board
    svg_data = chess.svg.board(board=board, size=size)

    # Convert SVG to a PNG and handle it
    try:
        from cairosvg import svg2png
        png_data = svg2png(bytestring=svg_data.encode('utf-8'))
        img = Image.open(BytesIO(png_data))

        if save_path:
            # Ensure the directory exists before saving
            output_dir = os.path.dirname(save_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            img.save(save_path)
            print(f"Board image saved to: {save_path}")

        if show_image:
            img.show() # This will open the image in your default image viewer

        return img
    except ImportError:
        print("CairoSVG not found. Cannot display or save image.")
        print("Board FEN:", fen)
        # The SVG data can still be useful for debugging
        if save_path and save_path.endswith(".svg"):
             with open(save_path, "w") as f:
                f.write(svg_data)
             print(f"Saved board as SVG to: {save_path}")
        else:
            print("Board SVG data:\n", svg_data)
        return None

def display_pc_board_from_tensor(board_tensor: torch.Tensor, size=400, save_path=None, show_image=False):
    """
    Generates a visual representation of the board from a piece-centric tensor.

    Args:
        board_tensor (torch.Tensor): The 1D tensor of shape (32,) representing the board.
        size (int): The size of the output image in pixels.
        save_path (str, optional): Path to save the image file (e.g., 'board.png').
                                    If None, the image is not saved. Defaults to None.
        show_image (bool, optional): If True, attempts to open the image in the default
                                        system viewer. Defaults to False.
    """
    fen = pc_tensor_to_fen(board_tensor)

    board = chess.Board(fen)

    svg_data = chess.svg.board(board=board, size=size)

    try:
        from cairosvg import svg2png
        png_data = svg2png(bytestring=svg_data.encode('utf-8'))
        img = Image.open(BytesIO(png_data))

        if save_path:
            output_dir = os.path.dirname(save_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            img.save(save_path)
            print(f"Piece-centric board image saved to: {save_path}")

        if show_image:
            img.show()

        return img
    except ImportError:
        print("CairoSVG not found. Cannot display or save image for piece-centric board.")
        print("Piece-centric Board FEN:", fen)
        if save_path and save_path.endswith(".svg"):
                with open(save_path, "w") as f:
                    f.write(svg_data)
                print(f"Saved piece-centric board as SVG to: {save_path}")
        else:
            print("Piece-centric Board SVG data:\n", svg_data)
        return None


if __name__ == "__main__":
    # --- Demonstrate Square-Centric (SC) Visualization ---
    print("--- Demonstrating Square-Centric (SC) Visualization ---")
    sc_fen_str = "r2q1Bk1/p3pp1p/1pnp2p1/2p5/P3P3/1P1P1N2/2P2KnP/R2Q3R b - - 0 14"

    # 1. Convert SC FEN to tensor
    sc_board_tensor = sc_fen_to_tensor(sc_fen_str)

    if sc_board_tensor is not None: #
        # 2. Reconstruct FEN (using default game state values for simplicity)
        reconstructed_sc_fen = sc_tensor_to_fen(sc_board_tensor)
        print(f"Original SC FEN: {sc_fen_str.split(' ')[0]}")
        print(f"Reconstructed SC FEN: {reconstructed_sc_fen.split(' ')[0]}")

        # 3. Visualize SC board
        display_sc_board_from_tensor(sc_board_tensor, save_path='sample_sc_board.png', show_image=True)

        # Demonstrate masking
        from chess_utils import SC_ABSORBING_STATE_INT # Assuming this exists or is defined directly
        print("\n--- Demonstrating SC Masking ---")
        masked_sc_tensor = sc_board_tensor.clone()
        # Mask a few squares by setting their value to the absorbing state
        masked_sc_tensor[0] = SC_ABSORBING_STATE_INT # Mask a8
        masked_sc_tensor[9] = SC_ABSORBING_STATE_INT # Mask b7

        # For visualization, absorbing state tokens should appear as empty squares (0)
        viz_masked_sc_tensor = torch.where(masked_sc_tensor == SC_ABSORBING_STATE_INT, 0, masked_sc_tensor)
        display_sc_board_from_tensor(viz_masked_sc_tensor, save_path='sample_sc_masked_board.png', show_image=False)
        print("SC Visualization SUCCESS, including masking demo.")
    else:
        print("Skipped SC visualization: sc_fen_to_tensor returned None (this case is less likely for SC).")

    # --- Demonstrate Piece-Centric (PC) Visualization (Requires pc_fen_to_tensor and PIECE_ORDER) ---
    try:
        from chess_utils import pc_fen_to_tensor, PIECE_ORDER, PC_ABSORBING_STATE_INT, PC_VOCAB_SIZE, PC_OFF_BOARD_INT, pc_tensor_to_fen

        print("\n--- Demonstrating Piece-Centric (PC) Visualization ---")
        pc_fen_str = "r2q1Bk1/p3pp1p/1pnp2p1/2p5/P3P3/1P1P1N2/2P2KnP/R2Q3R b - - 0 14" # Initial board

        # 1. Convert PC FEN to tensor
        pc_board_tensor = pc_fen_to_tensor(pc_fen_str)
        if pc_board_tensor is not None:
            # 2. Reconstruct FEN (using default game state values for simplicity)
            reconstructed_pc_fen = pc_tensor_to_fen(pc_board_tensor)
            print(f"Original PC FEN: {pc_fen_str.split(' ')[0]}")
            print(f"Reconstructed PC FEN: {reconstructed_pc_fen.split(' ')[0]}")

            # 3. Visualize PC board
            display_pc_board_from_tensor(pc_board_tensor, save_path='sample_pc_board.png', show_image=False)

            # Demonstrate masking
            print("\n--- Demonstrating PC Masking ---")
            masked_pc_tensor = pc_board_tensor.clone()
            # Mask a few pieces by setting their square to the absorbing state
            masked_pc_tensor[0] = PC_ABSORBING_STATE_INT # Mask White King
            masked_pc_tensor[8] = PC_ABSORBING_STATE_INT # Mask a White Pawn

            # For visualization, absorbing state tokens should appear as empty squares (0)
            viz_masked_pc_tensor = torch.where(masked_pc_tensor == PC_ABSORBING_STATE_INT, PC_OFF_BOARD_INT, masked_pc_tensor)
            display_pc_board_from_tensor(viz_masked_pc_tensor, save_path='sample_pc_masked_board.png', show_image=False)
            print("PC Visualization SUCCESS, including masking demo.")
        else:
            print("Skipped PC visualization: pc_fen_to_tensor returned None (e.g., due to promotions).")

    except ImportError:
        print("\nSkipping Piece-Centric Visualization demonstration: chess_utils not fully updated for PC mode yet.")
    except Exception as e:
        print(f"\nAn error occurred during PC Visualization demo: {e}")

    print("\nAll visualization demos finished.")
