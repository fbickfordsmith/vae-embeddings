from argparse import ArgumentParser, Namespace
from pathlib import Path
from subprocess import check_call
from typing import Tuple, Union

import numpy as np
import torch
from numpy.lib.npyio import NpzFile
from numpy.random import Generator
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor

from src.models.encoders import EncoderBurgess
from src.utils.modelIO import load_model


def get_config() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--use_gpu", type=bool, default=True)

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_dir", type=str, default="models")

    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--encoder", type=str, default="vae")

    parser.add_argument("--batch_size", type=int, default=4_096)

    return parser.parse_args()


def get_device(use_gpu: bool = True, use_deterministic_ops: bool = False) -> str:
    """
    References:
        https://pytorch.org/docs/stable/notes/mps.html
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
    elif use_gpu and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if use_deterministic_ops:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

    return device


@torch.inference_mode()
def encode(loader: DataLoader, encoder: EncoderBurgess, device: str) -> np.ndarray:
    embeddings = []

    for images_i, _ in loader:
        images_i = images_i.to(device)
        embeddings_i, _ = encoder(images_i)
        embeddings += [embeddings_i.cpu().numpy()]

    return np.concatenate(embeddings)


def split_indices_using_class_labels(
    labels: np.ndarray,
    test_size: Union[dict, float, int],
    rng: Generator,
    balance_test_set: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    sklearn.model_selection.train_test_split() doesn't produce exact stratification.
    """
    if isinstance(test_size, float) and balance_test_set:
        class_counts = np.bincount(labels)
        test_size = int(test_size * min(class_counts))
        test_size = max(test_size, 1)

    train_inds, test_inds = [], []

    for _class in np.unique(labels):
        class_inds = np.flatnonzero(labels == _class)
        class_inds = rng.permutation(class_inds)

        if isinstance(test_size, dict):
            class_test_size = test_size[_class]
        else:
            class_test_size = test_size

        if isinstance(class_test_size, float):
            class_test_size = int(class_test_size * len(class_inds))
            class_test_size = max(class_test_size, 1)

        train_inds += [class_inds[:-class_test_size]]
        test_inds += [class_inds[-class_test_size:]]

    train_inds = np.concatenate(train_inds)
    train_inds = rng.permutation(train_inds)

    test_inds = np.concatenate(test_inds)
    test_inds = rng.permutation(test_inds)

    return train_inds, test_inds


class DSprites(Dataset):
    """
    DSprites dataset with one of the six underlying latents as the class label. The dataset is split
    into training and test sets. The split depends on three parameters: the latent used as the class
    label, the fraction of the dataset used for testing, and the random seed.

    References:
        https://github.com/google-deepmind/dsprites-dataset
    """

    filename = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    target_names = {"color": 0, "shape": 1, "scale": 2, "orientation": 3, "pos_x": 4, "pos_y": 5}

    def __init__(
        self,
        data_dir: Union[Path, str],
        train: bool = True,
        test_fraction: float = 0.15,
        target_latent: str = "shape",
        seed: int = 0,
    ) -> None:
        data_dir = Path(data_dir)

        if not (data_dir / self.filename).exists():
            self.download(data_dir)

        npz_file = np.load(data_dir / self.filename)
        inputs, labels = self.process_npz(npz_file, target_latent)

        rng = np.random.default_rng(seed=seed)
        train_inds, test_inds = split_indices_using_class_labels(labels, test_fraction, rng)

        if train:
            self.data = inputs[train_inds]
            self.targets = labels[train_inds]
        else:
            self.data = inputs[test_inds]
            self.targets = labels[test_inds]

        self.data = torch.tensor(self.data)
        self.targets = torch.tensor(self.targets)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.data)

    def download(self, data_dir: Path) -> None:
        url = f"https://github.com/google-deepmind/dsprites-dataset/raw/master/{self.filename}"
        data_dir.mkdir(parents=True, exist_ok=True)
        check_call(["curl", "--location", "--output", data_dir / self.filename, url])

    def process_npz(self, npz_file: NpzFile, target_latent: str) -> Tuple[np.ndarray, np.ndarray]:
        inputs = npz_file["imgs"]  # [N, 64, 64], np.uint8, values in {0, 1}
        inputs = inputs[:, None, :, :]  # [N, 1, 64, 64]
        inputs = inputs.astype(np.float32)  # [N, 1, 64, 64]

        labels = npz_file["latents_classes"]  # [N, 6], np.int64
        labels = labels[:, self.target_names[target_latent]]  # [N,]

        return inputs, labels  # [N, 1, 64, 64], [N,]


def main(cfg: Namespace) -> None:
    device = get_device(cfg.use_gpu)

    data_dir = Path(cfg.data_dir) / cfg.dataset
    model_dir = Path(cfg.model_dir) / cfg.dataset / cfg.encoder

    encoder = load_model(model_dir, is_gpu=(device in {"cuda", "mps"})).encoder
    encoder.eval()

    for subset in ("train", "test"):
        if cfg.dataset == "mnist":
            dataset = MNIST(
                root=data_dir,
                train=(subset == "train"),
                download=True,
                transform=Compose([Resize(32), ToTensor()]),
            )
        else:
            dataset = DSprites(data_dir=data_dir, train=(subset == "train"))

        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

        embeddings = encode(loader, encoder, device)

        np.save(data_dir / f"embeddings_{cfg.encoder}_{subset}.npy", embeddings, allow_pickle=False)
        np.save(data_dir / f"labels_{subset}.npy", dataset.targets, allow_pickle=False)


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
