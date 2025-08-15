import logging
from torch.utils.tensorboard import SummaryWriter
import os

def setup_logging(experiment_name):
    """
    Creates an experiment folder, sets up file logging, and returns a TensorBoard SummaryWriter.
    All logs are stored inside experiments/<experiment_name>/
    """
    log_dir = os.path.join("experiments", experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(log_dir, "train.log"),
        filemode='w',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    writer = SummaryWriter(log_dir=log_dir)
    return writer