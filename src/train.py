import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


set_seed(42)


def train(
    model,
    train_loader,
    val_loader,
    epochs,
    learning_rate,
    checkpoint_path
):

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5
    )

    best_loss = float('inf')

    for epoch in range(epochs):

        model.train()

        train_loss = 0

        for x, y in train_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            pred = model(x).squeeze(1)

            loss = criterion(pred, y)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        val_loss = validate(model, val_loader, criterion)

        scheduler.step(val_loss)

        print(
            f'Epoch {epoch+1} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Val Loss: {val_loss:.4f}'
        )

        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(
                model.state_dict(),
                checkpoint_path
            )



def validate(model, loader, criterion):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x).squeeze(1)

            loss = criterion(pred, y)

            total_loss += loss.item()

    return total_loss / len(loader)