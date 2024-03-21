import os
import pickle

DATASETS = ["roxford5k", "rparis6k", "revisitop1m"]


def config_gnd(dataset, dir_main):
    """Configures ground truth data for the specified dataset.

    Args:
        dataset: Name of the dataset (should be one of 'roxford5k', 'rparis6k', or 'revisitop1m').
        dir_main: Main data directory.

    Returns:
        A configuration dictionary.
    """

    dataset = dataset.lower()
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}!")

    cfg = {}  # Initialize the configuration dictionary

    if dataset in ("roxford5k", "rparis6k"):
        gnd_fname = os.path.join(dir_main, dataset, f"gnd_{dataset}.pkl")
        with open(gnd_fname, "rb") as f:
            cfg = pickle.load(f)
        cfg["gnd_fname"] = gnd_fname
        cfg["ext"] = ".jpg"
        cfg["qext"] = ".jpg"

    elif dataset == "revisitop1m":
        cfg["imlist_fname"] = os.path.join(dir_main, dataset, f"{dataset}.txt")
        cfg["imlist"] = read_imlist(cfg["imlist_fname"])
        cfg["qimlist"] = []
        cfg["ext"] = ""
        cfg["qext"] = ""

    cfg["dir_data"] = os.path.join(dir_main, dataset)
    cfg["dir_images"] = os.path.join(cfg["dir_data"], "jpg")
    cfg["n"] = len(cfg["imlist"])
    cfg["nq"] = len(cfg["qimlist"])
    cfg["im_fname"] = config_imname
    cfg["qim_fname"] = config_qimname
    cfg["dataset"] = dataset
    return cfg


def config_imname(cfg, i):
    """Constructs the image filename based on the configuration."""
    return os.path.join(cfg["dir_images"], cfg["imlist"][i] + cfg["ext"])


def config_qimname(cfg, i):
    """Constructs the query image filename based on the configuration."""
    return os.path.join(cfg["dir_images"], cfg["qimlist"][i] + cfg["qext"])


def read_imlist(imlist_fn):
    """Reads the image list from the specified file."""
    with open(imlist_fn, "r") as file:
        imlist = file.read().splitlines()
    return imlist
