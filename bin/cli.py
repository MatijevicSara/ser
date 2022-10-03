from datetime import datetime
from pathlib import Path

import typer
import torch
import git

from ser.train import train as run_train
from ser.constants import RESULTS_DIR
from ser.data import train_dataloader, val_dataloader, test_dataloader
from ser.params import Params, save_params, load_params
from ser.transforms import transforms, normalize
from ser.inference import generate_ascii_art
from ser.inference import infer1
from ser.inference import pixel_to_char
from ser.inference import setup_inference


main = typer.Typer()


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        5, "-e", "--epochs", help="Number of epochs to run for."
    ),
    batch_size: int = typer.Option(
        1000, "-b", "--batch-size", help="Batch size for dataloader."
    ),
    learning_rate: float = typer.Option(
        0.01, "-l", "--learning-rate", help="Learning rate for the model."
    ),
):
    """Run the training algorithm."""
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    # wraps the passed in parameters
    params = Params(name, epochs, batch_size, learning_rate, sha)

    # setup device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup run
    fmt = "%Y-%m-%dT%H-%M"
    timestamp = datetime.strftime(datetime.utcnow(), fmt)
    run_path = RESULTS_DIR / name / timestamp
    run_path.mkdir(parents=True, exist_ok=True)

    # Save parameters for the run
    save_params(run_path, params)

    # Train!
    run_train(
        run_path,
        params,
        train_dataloader(params.batch_size, transforms(normalize)),
        val_dataloader(params.batch_size, transforms(normalize)),
        device,
    )


@main.command()
def infer(
    label: int = typer.Option(
        6, "-l",  "--label", help = "Specify which label "
    ),
    run_path: Path = typer.Option(
        ..., "-r", "--run_path", help = "Specify the path to the experiment"
    ),
):

    # TODO load the parameters from the run_path so we can print them out!
    params = load_params(run_path)


    # select image to run inference for
    images, labels = setup_inference(test_dataloader, label)


    # load the model
    model = torch.load(run_path / "model.pt")

    # run inference
    infer1(model,images, params, label)

