import pytest
import torch
import numpy as np
from src.model import SeismicUNet
from src.inference import predict_single, predict
from torch.utils.data import DataLoader, TensorDataset


def test_predict_single():
    batch_size = 1
    length = 256

    waveform = torch.randn(batch_size, 1, length)
    model = SeismicUNet(1, 16)

    output = predict_single(model, waveform)

    assert output.shape == (length,)
    assert np.all(output >= 0) and np.all(output <= 1)


def test_predict_with_dataloader():
    batch_size = 4
    length = 128

    waveforms = torch.randn(batch_size, 1, length)
    labels = torch.zeros(batch_size, length)

    dataset = TensorDataset(waveforms, labels)
    dataloader = DataLoader(dataset, batch_size=2)

    model = SeismicUNet(1, 16)

    predictions = predict(model, dataloader)

    assert predictions.shape == (batch_size, length)


def test_load_model():
    length = 128
    checkpoint_path = 'outputs/checkpoints/test_model.pth'

    model = SeismicUNet(1, 16)
    torch.save(model.state_dict(), checkpoint_path)

    loaded_model = SeismicUNet(1, 16)
    loaded_model.load_state_dict(torch.load(checkpoint_path))

    x = torch.randn(2, 1, length)
    original_out = model(x)
    loaded_out = loaded_model(x)

    assert torch.allclose(original_out, loaded_out)