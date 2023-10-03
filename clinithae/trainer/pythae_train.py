import os
from typing import List
from clinicadl.utils.maps_manager.maps_manager import MapsManager
from logging import getLogger
import json
import shutil
import subprocess
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader

from clinicadl.utils.caps_dataset.data import (
    get_transforms,
    load_data_test,
    return_dataset,
)
from clinicadl.utils.cmdline_utils import check_gpu
from clinicadl.utils.early_stopping import EarlyStopping
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
    ClinicaDLDataLeakageError,
    MAPSError,
)
from clinicadl.utils.logger import setup_logging
from clinicadl.utils.maps_manager.logwriter import LogWriter
from clinicadl.utils.maps_manager.maps_manager_utils import (
    add_default_values,
    change_path_to_str,
    change_str_to_path,
    read_json,
)
from clinicadl.utils.metric_module import RetainBest
from clinicadl.utils.network.network import Network
from clinicadl.utils.seed import get_seed, pl_worker_init_function, seed_everything
logger = getLogger("clinicadl-pythae.pythae_train")

def train_pythae(maps_manager : MapsManager, split_list: List[int] = None):
    """
    Train using Pythae procedure
    only works for single splits
    """
    from pythae.pipelines import TrainingPipeline

    from clinithae.dataset.pythae_dataset import PythaeCAPS

    train_transforms, all_transforms = get_transforms(
        normalize=maps_manager.normalize,
        data_augmentation=maps_manager.data_augmentation,
        size_reduction=maps_manager.size_reduction,
        size_reduction_factor=maps_manager.size_reduction_factor,
    )

    split_manager = maps_manager._init_split_manager(split_list)
    for split in split_manager.split_iterator():
        logger.info(f"Training split {split}")

        model_dir = maps_manager.maps_path / f"split-{split}"/ "best-loss"
        if not model_dir.is_dir():
            model_dir.mkdir(parents=True)

        seed_everything(maps_manager.seed, maps_manager.deterministic, maps_manager.compensation)

        split_df_dict = split_manager[split]
        train_dataset = PythaeCAPS(
            maps_manager.caps_directory,
            split_df_dict["train"],
            maps_manager.preprocessing_dict,
            train_transformations=train_transforms,
            all_transformations=all_transforms,
        )
        eval_dataset = PythaeCAPS(
            maps_manager.caps_directory,
            split_df_dict["validation"],
            maps_manager.preprocessing_dict,
            train_transformations=train_transforms,
            all_transformations=all_transforms,
        )

        # Import the model

        clinicadl_model, _ = maps_manager._init_model(
            split=split,
            gpu=True,
        )
        print(clinicadl_model)
        print(_)
        model = clinicadl_model.model
        config = clinicadl_model.get_trainer_config(
            output_dir=model_dir,
            num_epochs=maps_manager.epochs,
            learning_rate=maps_manager.learning_rate,
            batch_size=maps_manager.batch_size,
        )
        # Create Pythae Training Pipeline
        pipeline = TrainingPipeline(training_config=config, model=model)
        # Create Pythae Training Pipeline
        pipeline = TrainingPipeline(training_config=config, model=model)

        # Launch training
        pipeline(
            train_data=train_dataset,  # must be torch.Tensor or np.array
            eval_data=eval_dataset,  # must be torch.Tensor or np.array
        )
        # Move saved model to the correct path in the MAPS
        src = model_dir / "*_training_*/final_model/model.pt"
        os.system(f"mv {src} {model_dir}")
