import pytest
import torch
from src.model import SeismicUNet, SEBlock, ResBlock


def test_se_block():
    batch_size = 4
    channels = 32
    length = 100

    x = torch.randn(batch_size, channels, length)
    se = SEBlock(channels)

    out = se(x)

    assert out.shape == x.shape


def test_res_block():
    batch_size = 4
    in_channels = 32
    out_channels = 64
    length = 100

    x = torch.randn(batch_size, in_channels, length)
    res = ResBlock(in_channels, out_channels)

    out = res(x)

    assert out.shape[0] == batch_size
    assert out.shape[1] == out_channels


def test_seismic_unet():
    batch_size = 4
    in_channels = 1
    length = 256
    base_filters = 32

    x = torch.randn(batch_size, in_channels, length)
    model = SeismicUNet(in_channels, base_filters)

    out = model(x)

    assert out.shape == (batch_size, length)
    assert torch.all(out >= 0)
    assert torch.all(out <= 1)


def test_seismic_unet_output_range():
    batch_size = 2
    length = 128

    x = torch.randn(batch_size, 1, length)
    model = SeismicUNet(1, 16)

    out = model(x)

    assert out.min() >= 0
    assert out.max() <= 1