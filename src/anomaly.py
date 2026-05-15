import torch
import numpy as np

def calculate_reconstruction_error(model, sequences):

    model.eval()

    with torch.no_grad():

        reconstructed = model(sequences)

        errors = torch.mean(
            (sequences - reconstructed) ** 2,
            dim=(1, 2)
        )

    return errors.numpy()