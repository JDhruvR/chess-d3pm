"""
This file contains the core D3PM (Structured Denoising Diffusion Models in
Discrete State-Spaces) engine, adapted for an unconditional generation task
using an absorbing state.

This class is designed to be generic and can be paired with any suitable
x0-prediction model (like a Transformer) for training and sampling.
"""

import torch
import torch.nn as nn


class D3PM(nn.Module):
    """
    The core D3PM engine for discrete state spaces using an absorbing state.
    """
    def __init__(self, x0_model: nn.Module, n_T: int, num_classes: int, hybrid_loss_coeff=0.0,) -> None:
        """
        Args:
            x0_model (nn.Module): The neural network that predicts the original data x_0 from a noisy input x_t.
            n_T (int): The total number of diffusion timesteps.
            num_classes (int): The total number of discrete states in the vocabulary, INCLUDING the absorbing state.
                               For chess, this will be 12 pieces + 1 empty + 1 absorbing = 14.
            hybrid_loss_coeff (float, optional): The coefficient for the variational bound loss.
                                                 Defaults to 0.0, which means only CrossEntropyLoss is used.
        """
        super(D3PM, self).__init__()
        self.x0_model = x0_model
        self.n_T = n_T
        self.num_classes = num_classes
        self.hybrid_loss_coeff = hybrid_loss_coeff
        self.eps = 1e-6

        # --- Set up the noise schedule and transition matrices for an absorbing state ---

        # Cosine noise schedule
        steps = torch.arange(n_T + 1, dtype=torch.float64) / n_T
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        self.beta_t = torch.minimum(
            1 - alpha_bar[1:] / alpha_bar[:-1], torch.ones_like(alpha_bar[1:]) * 0.999
        )

        q_onestep_mats = []
        # The absorbing state is assumed to be the last class index (num_classes - 1)
        absorbing_state_idx = self.num_classes - 1

        for beta in self.beta_t:
            # Create a one-step transition matrix for the absorbing state diffusion process
            mat = torch.eye(self.num_classes, dtype=torch.float64) * (1 - beta)
            mat[:, absorbing_state_idx] += beta
            q_onestep_mats.append(mat)

        q_one_step_mats = torch.stack(q_onestep_mats, dim=0)

        # This will be used for q_posterior_logits
        q_one_step_transposed = q_one_step_mats.transpose(1, 2)

        # Calculate the cumulative transition matrices q(x_t | x_0) by matrix multiplication
        q_mat_t = q_one_step_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, self.n_T):
            q_mat_t = q_mat_t @ q_one_step_mats[idx]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)

        # Register buffers so they are moved to the correct device with the model
        self.register_buffer("q_one_step_transposed", q_one_step_transposed)
        self.register_buffer("q_mats", q_mats)

        assert self.q_mats.shape == (self.n_T, self.num_classes, self.num_classes)

    def _at(self, a: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Helper function to select rows from a based on t and x."""
        bs = t.shape[0]
        t_broadcast = t.reshape(bs, *([1] * (x.dim() - 1)))
        # out[i, j, k, l, m] = a[t[i], x[i, j, k, l], m]
        return a[t_broadcast - 1, x, :]

    def q_posterior_logits(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Calculates the logits of the posterior distribution q(x_{t-1} | x_t, x_0).

        Args:
            x_0 (torch.Tensor): The original clean data. Can be integer tensor or logits.
            x_t (torch.Tensor): The noisy data at timestep t.
            t (torch.Tensor): The current timestep.
        """
        # If x_0 is integer, convert it to one-hot logits
        if x_0.dtype in [torch.int64, torch.int32]:
            x_0_logits = torch.log(
                torch.nn.functional.one_hot(x_0, self.num_classes) + self.eps
            )
        else:
            x_0_logits = x_0.clone()

        # Equation (3) from the D3PM paper, simplified for our case
        fact1 = self._at(self.q_one_step_transposed, t, x_t)

        # We need q_mats for t-2, so handle the t=1 case
        safe_t = torch.max(t, torch.ones_like(t) * 2)
        qmats2 = self.q_mats[safe_t - 2].to(dtype=torch.float32)

        softmaxed_x0 = torch.softmax(x_0_logits, dim=-1)
        fact2 = torch.einsum("b...c,bcd->b...d", softmaxed_x0, qmats2)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        # For t=1, the posterior is just the distribution over x_0
        t_broadcast = t.reshape(t.shape[0], *([1] * (x_t.dim())))
        return torch.where(t_broadcast == 1, x_0_logits, out)

    def vb(self, dist1: torch.Tensor, dist2: torch.Tensor) -> torch.Tensor:
        """Calculates the KL-divergence for the variational bound loss."""
        dist1_flat = dist1.flatten(start_dim=0, end_dim=-2)
        dist2_flat = dist2.flatten(start_dim=0, end_dim=-2)

        kl_div = torch.softmax(dist1_flat + self.eps, dim=-1) * (
            torch.log_softmax(dist1_flat + self.eps, dim=-1)
            - torch.log_softmax(dist2_flat + self.eps, dim=-1)
        )
        return kl_div.sum(dim=-1).mean()

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        The forward process q(x_t | x_0). Corrupts the clean input x_0 to a noisy x_t.
        """
        logits = torch.log(self._at(self.q_mats, t, x_0) + self.eps)

        # Use Gumbel-Max trick for sampling from the categorical distribution
        gumbel_noise = -torch.log(-torch.log(torch.clip(noise, self.eps, 1.0)))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def model_predict(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        """
        Calls the underlying x0_model to predict the logits of the original data x_0.
        """
        return self.x0_model(x_t, t, cond)

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> tuple[torch.Tensor, dict]:
        """
        The main training step.
        1. Samples a timestep t.
        2. Corrupts the input x to x_t.
        3. Predicts the original x_0 using the model.
        4. Calculates the loss.
        """
        # Sample a random timestep for each item in the batch
        t = torch.randint(1, self.n_T + 1, (x.shape[0],), device=x.device)

        # Create x_t by running the forward diffusion process
        noise = torch.rand((*x.shape, self.num_classes), device=x.device)
        x_t = self.q_sample(x, t, noise)

        # Get the model's prediction of the original x_0's logits
        predicted_x0_logits = self.model_predict(x_t, t, cond)

        # --- Calculate Loss ---
        # 1. The primary loss: Cross-Entropy between predicted x_0 and true x_0
        ce_loss = torch.nn.functional.cross_entropy(
            predicted_x0_logits.permute(0, 2, 1), # Shape: (B, Vocab, SeqLen)
            x                                      # Shape: (B, SeqLen)
        )

        # 2. The auxiliary variational bound loss (optional)
        vb_loss = torch.tensor(0.0)
        if self.hybrid_loss_coeff > 0:
            true_q_posterior = self.q_posterior_logits(x, x_t, t)
            pred_q_posterior = self.q_posterior_logits(predicted_x0_logits, x_t, t)
            vb_loss = self.vb(true_q_posterior, pred_q_posterior)

        total_loss = ce_loss + self.hybrid_loss_coeff * vb_loss

        return total_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        """
        The reverse process step p(x_{t-1} | x_t). Samples x_{t-1} given x_t.
        """
        predicted_x0_logits = self.model_predict(x_t, t, cond)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t)

        # Use Gumbel-Max trick for sampling
        noise = torch.rand_like(pred_q_posterior_logits)
        gumbel_noise = -torch.log(-torch.log(torch.clip(noise, self.eps, 1.0)))

        # Don't add noise at the last step (t=1)
        not_first_step = (t != 1).float().reshape(x_t.shape[0], *([1] * (x_t.dim())))
        sample = torch.argmax(
            pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1
        )
        return sample

    @torch.no_grad()
    def sample(self, initial_noise: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        """
        Generates a full sample from noise by iterating through the reverse process.

        Args:
            initial_noise (torch.Tensor): A tensor of shape (B, SeqLen) filled with the absorbing state index.
            cond (torch.Tensor, optional): Conditioning information, if any. Defaults to None.
        """
        x = initial_noise
        for i in reversed(range(1, self.n_T + 1)):
            t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
            x = self.p_sample(x, t, cond)
        return x

    @torch.no_grad()
    def sample_with_history(self, initial_noise: torch.Tensor, cond: torch.Tensor = None, stride: int = 10) -> list[torch.Tensor]:
        """
        Generates a full sample and saves intermediate steps.
        """
        x = initial_noise
        history = []
        for i in reversed(range(1, self.n_T + 1)):
            t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
            x = self.p_sample(x, t, cond)
            if (i - 1) % stride == 0 or i == 1:
                history.append(x.cpu())
        return history
