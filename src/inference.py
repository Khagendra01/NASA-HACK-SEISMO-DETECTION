import torch
import numpy as np
import scipy.signal as sg

from preprocessing import preprocess_signal


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def predict_arrival(
    model,
    data,
    fs,
    lowcut,
    highcut,
    window_size_sec,
    threshold=0.25
):

    processed = preprocess_signal(
        data,
        fs,
        lowcut,
        highcut,
        window_size_sec
    )

    x = torch.tensor(
        processed[np.newaxis, :, :],
        dtype=torch.float32
    ).to(DEVICE)

    model.eval()

    with torch.no_grad():
        pred = model(x).cpu().numpy().reshape(-1)

    binary = np.where(pred > threshold, 1, 0)

    idxs = np.where(binary == 1)[0]

    return pred, idxs