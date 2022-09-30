from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ser.train import train1
from ser.train import validation1
from ser.model import Net
from ser.data import setup_dataloaders
from ser.transforms import transforms1
import typer

main = typer.Typer()



@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
       2, "-e", "--epochs", help = "Number of epochs" 
    ),
    batch_size: int = typer.Option(
        1000, "-b", "--batch_size", help = "Batch size"
    ),
    learning_rate: int = typer.Option(
        0.01, "-l", "--learning_rate", help = "Learning rate"
    )
):
    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_dataloader, validation_dataloader = setup_dataloaders(batch_size)

    for epoch in range(epochs):
        mean_loss = train1(device, model, optimizer, training_dataloader)
        validation1(validation_dataloader, device,model)
        print(
            f"Train Epoch: {epoch} "
            f"| Loss: {mean_loss:.4f}"
        )


@main.command()
def infer():
    print("This is where the inference code will go")
