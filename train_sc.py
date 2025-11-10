import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import wandb

# --- Local Imports ---
from create_sc_dataset import ChessDataset, OUTPUT_FILE # SC-specific dataset
from dit import ChessDiT
from chess_d3pm import D3PM
from chess_utils import (
    SC_VOCAB_SIZE, SC_ABSORBING_STATE_INT, SC_SEQUENCE_LENGTH, SC_EMPTY_SQUARE_INT,
    sc_tensor_to_fen, sc_mask_board, sc_fen_to_tensor
)
from visualize import display_sc_board_from_tensor # SC-specific visualization

# --- Configuration ---
# Dataloader
BATCH_SIZE = 128
NUM_WORKERS = 2 # Reduced from 4 for Colab compatibility

# Model
MODEL_DIM = 256
MODEL_DEPTH = 2
MODEL_HEADS = 4

# Diffusion
NUM_TIMESTEPS = 16

# Training
LEARNING_RATE = 9e-5
NUM_EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRECISION = "16-mixed" if DEVICE == "cuda" else "32"

# Logging & Sampling
PROJECT_NAME = "chess-d3pm-sc" # Differentiate project name for SC
SAMPLE_EVERY_N_STEPS = 500
NUM_SAMPLES_TO_GENERATE = 2
PARTIAL_GENERATE_FEN = "r2q1Bk1/p3pp1p/1pnp2p1/2p5/P3P3/1P1P1N2/2P2KnP/R2Q3R b - - 0 14"
PARTIAL_GENERATE_MASK_RATIO = 0.7


class D3PMLightning(pl.LightningModule):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.save_hyperparameters()

        x0_model = ChessDiT(
            vocab_size=SC_VOCAB_SIZE,
            sequence_length=SC_SEQUENCE_LENGTH,
            hidden_size=MODEL_DIM,
            depth=MODEL_DEPTH,
            num_heads=MODEL_HEADS,
        )

        self.d3pm = D3PM(
            x0_model=x0_model,
            n_T=NUM_TIMESTEPS,
            num_classes=SC_VOCAB_SIZE,
            hybrid_loss_coeff=0.0,
        )

    def training_step(self, batch, batch_idx):
        clean_boards = batch
        loss, info = self.d3pm(clean_boards)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ce_loss", info["ce_loss"], on_step=True, logger=True)
        self.log("vb_loss", info["vb_loss"], on_step=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.d3pm.x0_model.parameters(),
            lr=self.hparams.learning_rate
        )
        return optimizer

    @torch.no_grad()
    def sample(self, num_samples: int) -> torch.Tensor:
        initial_noise = torch.full(
            (num_samples, SC_SEQUENCE_LENGTH),
            SC_ABSORBING_STATE_INT,
            device=self.device,
            dtype=torch.long
        )
        return self.d3pm.sample(initial_state=initial_noise)

    @torch.no_grad()
    def partial_sample(self, initial_state: torch.Tensor) -> torch.Tensor:
        return self.d3pm.sample(initial_state=initial_state, partial_generate=True)

class SamplingCallback(pl.Callback):
    def __init__(self, sample_every_n_steps: int, num_samples: int, partial_generate_fen: str, partial_generate_mask_ratio: float):
        super().__init__()
        self.sample_every_n_steps = sample_every_n_steps
        self.num_samples = num_samples
        self.partial_generate_fen = partial_generate_fen
        self.partial_generate_mask_ratio = partial_generate_mask_ratio

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.logger or not wandb:
            return

        global_step = trainer.global_step
        if (global_step + 1) % self.sample_every_n_steps == 0:
            pl_module.eval()

            # --- Full Generation ---
            print(f"\n--- Sampling Full Boards at step {global_step+1} ---")
            generated_boards = pl_module.sample(self.num_samples)

            for i in range(min(self.num_samples, 2)): # Log up to 2 full samples
                board_tensor = generated_boards[i].cpu()
                # For visualization, absorbing state tokens should appear as empty squares (0)
                viz_board_tensor = torch.where(board_tensor == SC_ABSORBING_STATE_INT, SC_EMPTY_SQUARE_INT, board_tensor)
                img = display_sc_board_from_tensor(viz_board_tensor, show_image=False)

                if img:
                    trainer.logger.experiment.log({
                        f"generated_full_sample_{i+1}": wandb.Image(img, caption=f"Step {global_step+1}")
                    })
                reconstructed_fen = sc_tensor_to_fen(board_tensor)
                trainer.logger.experiment.log({
                    f"generated_full_fen_{i+1}": reconstructed_fen
                })
                print(f"Full Sample {i+1} FEN: {reconstructed_fen}")

            # --- Partial Generation ---
            print(f"\n--- Sampling Partial Board at step {global_step+1} ---")

            # The SC masking function already takes a FEN string
            masked_sc_tensor = sc_mask_board(self.partial_generate_fen, ratio=self.partial_generate_mask_ratio)
            initial_state_for_model = masked_sc_tensor.unsqueeze(0).to(pl_module.device)

            completed_board_tensor = pl_module.partial_sample(initial_state=initial_state_for_model)
            final_tensor_squeezed = completed_board_tensor.squeeze(0).cpu()

            # Original masked for visualization
            viz_masked_tensor = torch.where(masked_sc_tensor == SC_ABSORBING_STATE_INT, SC_EMPTY_SQUARE_INT, masked_sc_tensor)
            masked_img = display_sc_board_from_tensor(viz_masked_tensor, show_image=False)
            if masked_img:
                trainer.logger.experiment.log({
                    "partial_gen_masked_input": wandb.Image(masked_img, caption=f"Step {global_step+1} - Input Masked")
                })
            print(f"Partial Gen Input FEN: {sc_tensor_to_fen(masked_sc_tensor)}")

            # Completed board for visualization
            viz_completed_tensor = torch.where(final_tensor_squeezed == SC_ABSORBING_STATE_INT, SC_EMPTY_SQUARE_INT, final_tensor_squeezed)
            completed_img = display_sc_board_from_tensor(viz_completed_tensor, show_image=False)
            if completed_img:
                trainer.logger.experiment.log({
                    "partial_gen_completed_output": wandb.Image(completed_img, caption=f"Step {global_step+1} - Output Completed")
                })
            reconstructed_partial_fen = sc_tensor_to_fen(final_tensor_squeezed)
            trainer.logger.experiment.log({
                "partial_gen_completed_fen": reconstructed_partial_fen
            })
            print(f"Partial Gen Output FEN: {reconstructed_partial_fen}")

            print("--- End Sampling ---")
            pl_module.train()

def main():
    if not os.path.exists(OUTPUT_FILE):
        raise FileNotFoundError(
            f"Dataset file not found: {OUTPUT_FILE}. "
            f"Please run create_sc_dataset.py first to generate it."
        )
    dataset = ChessDataset(tensor_file=OUTPUT_FILE)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = D3PMLightning(learning_rate=LEARNING_RATE)

    wandb_logger = WandbLogger(project=PROJECT_NAME, log_model="all")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints_sc/",
        filename="chess-d3pm-sc-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        monitor="train_loss",
        mode="min",
    )

    sampling_callback = SamplingCallback(
        sample_every_n_steps=SAMPLE_EVERY_N_STEPS,
        num_samples=NUM_SAMPLES_TO_GENERATE,
        partial_generate_fen=PARTIAL_GENERATE_FEN,
        partial_generate_mask_ratio=PARTIAL_GENERATE_MASK_RATIO
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        accelerator=DEVICE,
        precision=PRECISION,
        max_epochs=NUM_EPOCHS,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, sampling_callback, lr_monitor],
        log_every_n_steps=10,
    )

    print("--- Starting Training (Square-Centric) ---")
    trainer.fit(model, dataloader)
    print("--- Training Finished (Square-Centric) ---")

    wandb.finish()

if __name__ == "__main__":
    main()
