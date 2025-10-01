# test.py
import torch
import torch.nn as nn
from unittest.mock import MagicMock

# Import the class from your project
from chess_d3pm import D3PM
from chess_utils import VOCAB_SIZE, ABSORBING_STATE_INT

def test_p_sample_logic():
    print("--- Running Test for p_sample with Absorb + Top-5 Logic ---")

    # 1. SETUP
    dummy_x0_model = nn.Linear(1, 1)
    d3pm_instance = D3PM(dummy_x0_model, n_T=10, num_classes=VOCAB_SIZE)
    d3pm_instance = d3pm_instance.to("cpu")

    x_t = torch.full((1, 16), 0, dtype=torch.long)
    x_t[0, 5] = 1  # Unmasked token (Pawn)
    x_t[0, 10] = ABSORBING_STATE_INT
    x_t[0, 11] = ABSORBING_STATE_INT

    # 2. MOCKING with CORRECTED LOGITS
    mock_full_logits = torch.full((1, 16, VOCAB_SIZE), -50.0) # Start with low values

    # For token 10, a standard top-5
    mock_full_logits[0, 10, 2] = 10.0 # Knight
    mock_full_logits[0, 10, 3] = 9.0  # Bishop
    mock_full_logits[0, 10, 4] = 8.0  # Rook
    mock_full_logits[0, 10, 5] = 7.0  # Queen
    mock_full_logits[0, 10, 6] = 6.0  # King
    mock_full_logits[0, 10, 7] = -10.0 # 6th is low

    # For token 11, the CORRECTED trick case
    # The "trick" token (index 8) is now the 6th highest logit.
    # It should be filtered out by top-5.
    mock_full_logits[0, 11, 5] = 20.0 # Queen (top-1)
    mock_full_logits[0, 11, 4] = 19.0 # Rook  (top-2)
    mock_full_logits[0, 11, 3] = 18.0 # Bishop(top-3)
    mock_full_logits[0, 11, 2] = 17.0 # Knight(top-4)
    mock_full_logits[0, 11, 1] = 16.0 # Pawn  (top-5)
    mock_full_logits[0, 11, 8] = 15.0 # A different Knight (THIS IS THE TRICK 6th value)

    d3pm_instance.model_predict = MagicMock(return_value=None)
    d3pm_instance.q_posterior_logits = MagicMock(return_value=mock_full_logits)

    # 3. EXECUTION
    unmasked_tokens_preserved = True
    token_10_in_top5 = True
    token_11_not_the_trick = True

    final_x_next = None
    for i in range(20): # Run more times for certainty
        t = torch.full((1,), 2, dtype=torch.long)
        x_next = d3pm_instance.p_sample(
            x_t, t, cond=None, use_absorbing=True, sampling_strategy="top5"
        )
        final_x_next = x_next

        if x_next[0, 5] != 1:
            unmasked_tokens_preserved = False

        top_5_indices_10 = torch.tensor([2, 3, 4, 5, 6])
        if x_next[0, 10] not in top_5_indices_10:
            token_10_in_top5 = False

        if x_next[0, 11] == 8: # The trick token is index 8
            token_11_not_the_trick = False

    # 4. ASSERTION
    print(f"Original x_t:      {x_t[0].tolist()}")
    print(f"Final x_{{t-1}}:       {final_x_next[0].tolist()} (from last run)")
    print("-" * 20)
    print("Verification Results:")

    assert unmasked_tokens_preserved, "FAIL: An unmasked token was changed!"
    print("✅ PASS: Unmasked tokens were preserved across all runs.")

    assert token_10_in_top5, "FAIL: Sampled token at index 10 was not from its top-5 logits!"
    print("✅ PASS: Sampled token at index 10 was always from its top-5 logits.")

    assert token_11_not_the_trick, "FAIL: Top-5 filtering failed! The trick token was selected."
    print("✅ PASS: Top-5 filtering correctly ignored the 6th-best logit.")
    print("\n--- Test SUCCEEDED ---")


if __name__ == "__main__":
    test_p_sample_logic()
