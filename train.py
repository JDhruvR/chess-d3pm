import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# --- Local Imports ---
# Data related
from create_chess_dataset import ChessDataset, OUTPUT_FILE
# Model and Diffusion Engine
from dit import ChessDiT
from chess_d3pm import D3PM
# Utilities and constants
from chess_utils import VOCAB_SIZE, ABSORBING_STATE_INT, tensor_to_fen
from visualize import display_board_from_tensor

import wandb

# --- Configuration ---
# Dataloader
BATCH_SIZE = 128
NUM_WORKERS = 4

# Model
MODEL_DIM = 128
MODEL_DEPTH = 8
MODEL_HEADS = 8

# Diffusion
NUM_TIMESTEPS = 10

# Training
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRECISION = "16-mixed" if DEVICE == "cuda" else "32"

# Logging & Sampling
PROJECT_NAME = "chess-d3pm"
SAMPLE_EVERY_N_STEPS = 500
NUM_SAMPLES_TO_GENERATE = 4


class D3PMLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for our D3PM model.
    This class handles the training, optimization, and logging.
    """
    def __init__(self, learning_rate: float):
        super().__init__()
        self.save_hyperparameters() # Saves learning_rate, etc. to the checkpoint

        # 1. Create the x0-prediction model (the Transformer)
        x0_model = ChessDiT(
            vocab_size=VOCAB_SIZE,
            hidden_size=MODEL_DIM,
            depth=MODEL_DEPTH,
            num_heads=MODEL_HEADS,
        )

        # 2. Create the D3PM diffusion engine
        self.d3pm = D3PM(
            x0_model=x0_model,
            n_T=NUM_TIMESTEPS,
            num_classes=VOCAB_SIZE,
            hybrid_loss_coeff=0.0, # Using only CE loss for simplicity
        )

    def training_step(self, batch, batch_idx):
        # The batch is a tensor of clean chess boards from our dataset
        clean_boards = batch

        # The forward pass of our D3PM module does everything:
        # - picks a random t
        # - corrupts the input
        # - runs the model
        # - calculates the loss
        loss, info = self.d3pm(clean_boards)

        # Log the loss and its components
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ce_loss", info["ce_loss"], on_step=True, logger=True)
        self.log("vb_loss", info["vb_loss"], on_step=True, logger=True)

        return loss

    def configure_optimizers(self):
        # The optimizer should only target the parameters of the x0_model
        optimizer = torch.optim.AdamW(
            self.d3pm.x0_model.parameters(),
            lr=self.hparams.learning_rate
        )
        return optimizer

    @torch.no_grad()
    def sample(self, num_samples: int) -> torch.Tensor:
        """ Helper function to generate samples from noise. """
        # Start with a board full of the absorbing state token
        initial_noise = torch.full(
            (num_samples, 64),
            ABSORBING_STATE_INT,
            device=self.device,
            dtype=torch.long
        )
        # Use the D3PM engine's sample method
        return self.d3pm.sample(initial_noise)

class SamplingCallback(pl.Callback):
    """
    A PyTorch Lightning Callback to periodically generate and log board samples.
    """
    def __init__(self, sample_every_n_steps: int, num_samples: int):
        super().__init__()
        self.sample_every_n_steps = sample_every_n_steps
        self.num_samples = num_samples

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """ Called after each training step. """
        # Ensure we have a logger and wandb is available
        if not trainer.logger or not wandb:
            return

        # Check if it's time to sample
        global_step = trainer.global_step
        if (global_step + 1) % self.sample_every_n_steps == 0:
            print(f"\n--- Sampling at step {global_step+1} ---")
            pl_module.eval() # Set model to evaluation mode

            # Generate samples
            generated_boards = pl_module.sample(self.num_samples)

            # Visualize the first generated board
            first_board_tensor = generated_boards[0].cpu()
            img = display_board_from_tensor(first_board_tensor, show_image=False)

            if img:
                # Log the image to wandb
                trainer.logger.experiment.log({
                    "generated_sample": wandb.Image(img, caption=f"Step {global_step+1}")
                })

            # Also log the FEN string for text-based inspection
            reconstructed_fen = tensor_to_fen(first_board_tensor)
            trainer.logger.experiment.log({
                "generated_fen": reconstructed_fen
            })
            print(f"Sampled FEN: {reconstructed_fen}")
            print("--- End Sampling ---")

            pl_module.train() # Set model back to training mode


def main():
    # --- 1. Setup Data ---
    if not os.path.exists(OUTPUT_FILE):
        raise FileNotFoundError(
            f"Dataset file not found: {OUTPUT_FILE}. "
            f"Please run create_chess_dataset.py first to generate it."
        )
    dataset = ChessDataset(tensor_file=OUTPUT_FILE)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # --- 2. Setup Model ---
    model = D3PMLightning(learning_rate=LEARNING_RATE)

    # --- 3. Setup Logging & Callbacks ---
    wandb_logger = WandbLogger(project=PROJECT_NAME, log_model="all")
    # wandb_logger.watch(model, log="all") # Optional: log gradients

    # Callback to save the model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="chess-d3pm-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        monitor="train_loss",
        mode="min",
    )

    # Callback for sampling
    sampling_callback = SamplingCallback(
        sample_every_n_steps=SAMPLE_EVERY_N_STEPS,
        num_samples=NUM_SAMPLES_TO_GENERATE
    )

    # Callback for monitoring learning rate
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # --- 4. Setup Trainer and Start Training ---
    trainer = pl.Trainer(
        accelerator=DEVICE,
        precision=PRECISION,
        max_epochs=NUM_EPOCHS,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, sampling_callback, lr_monitor],
        log_every_n_steps=10,
    )

    print("--- Starting Training ---")
    trainer.fit(model, dataloader)
    print("--- Training Finished ---")


if __name__ == "__main__":
    main()
