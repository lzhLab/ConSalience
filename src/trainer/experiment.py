# src/trainer/experiment.py
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models import build_model
from src.trainer.base_trainer import BaseTrainer
from src.utils.io import load_dataset
from src.salience.build_salience import build_salience_generator

class Experiment:
    """
    Build dataset, dataloader, model, salience generator, and trainer from config.

    Baseline:
        Dataset/DataLoader input: (B, 1, D, H, W)
        model.in_channels: 1

    With salience:
        Dataset/DataLoader input: (B, 1, D, H, W)
        SalienceGenerator output: (B, 3, D, H, W)
        model.in_channels: 3
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.seed = cfg.get("seed", 2024)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._setup_seed(self.seed)
        self._validate_cfg()
        self._build_data()
        self._build_model()
        self._build_salience_generator()
        self._build_trainer()

    def _setup_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _validate_cfg(self):
        required_top_keys = ["model", "data", "train"]
        for key in required_top_keys:
            if key not in self.cfg:
                raise KeyError(f"Missing required config section: '{key}'")

        data_cfg = self.cfg["data"]
        train_cfg = self.cfg["train"]
        model_cfg = self.cfg["model"]

        for key in ("dataset", "split_path", "root_dir"):
            if key not in data_cfg:
                raise KeyError(f"Missing config key: data.{key}")

        if "batch_size" not in train_cfg:
            raise KeyError("Missing config key: train.batch_size")

        if train_cfg["batch_size"] != 1:
            print(
                "[Warning] This Experiment currently uses batch_size=1 because "
                "3D volumes may have different depth D. "
                f"Current train.batch_size={train_cfg['batch_size']} will be overridden to 1."
            )
            train_cfg["batch_size"] = 1

        use_salience = bool(data_cfg.get("use_salience", False))
        in_channels = int(model_cfg.get("in_channels", 1))

        if use_salience and in_channels != 3:
            raise ValueError(
                "When data.use_salience=true, model.in_channels must be 3. "
                f"Got model.in_channels={in_channels}. "
                "Use scripts/train_salience.py or override model.in_channels=3."
            )

        if not use_salience and in_channels != 1:
            print(
                "[Warning] Baseline usually uses model.in_channels=1. "
                f"Current model.in_channels={in_channels}."
            )

    def _build_data(self):
        data_cfg = self.cfg["data"]
        train_cfg = self.cfg["train"]

        dataset_name = data_cfg["dataset"]
        split_path = data_cfg.get("split_path", "data/preprocessed/splits.json")
        root_dir = data_cfg.get("root_dir", "data/preprocessed")

        train_set, val_set, test_set = load_dataset(
            dataset_name=dataset_name,
            split_path=split_path,
            root_dir=root_dir,
        )

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        num_workers = train_cfg.get("num_workers", 4)
        pin_memory = torch.cuda.is_available()

        self.train_loader = DataLoader(
            train_set,
            batch_size=1,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        self.val_loader = DataLoader(
            val_set,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        self.test_loader = DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        print("=== Data Loaded ===")
        print(f"Dataset: {dataset_name}")
        print(f"Root dir: {root_dir}")
        print(f"Split path: {split_path}")
        print(f"Train cases: {len(train_set)}")
        print(f"Val cases: {len(val_set)}")
        print(f"Test cases: {len(test_set)}")
        print("Batch size: 1")

    def _build_model(self):
        model_cfg = self.cfg["model"]

        self.model = build_model(model_cfg).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("=== Model Built ===")
        print(f"Model: {model_cfg.get('name', 'Unknown')}")
        print(f"Input channels: {model_cfg.get('in_channels', 'Unknown')}")
        print(f"Device: {self.device}")
        print(f"Total parameters: {n_params:,}")
        print(f"Trainable parameters: {n_trainable:,}")
    
    def _build_salience_generator(self):
        data_cfg = self.cfg["data"]
    
        self.use_salience = bool(data_cfg.get("use_salience", False))
        self.salience_gen = None
    
        if not self.use_salience:
            print("=== Salience Disabled ===")
            return
    
        self.salience_gen, plugin_name, kwargs = build_salience_generator(self.cfg)
    
        print("=== Salience Enabled ===")
        print(f"Plugin: {plugin_name}")
        print(f"delta_theta: {kwargs.get('delta_theta')}")
        print(f"delta_sigma: {kwargs.get('delta_sigma')}")
        print(f"K: {kwargs.get('K')}")
    
        if "alpha" in kwargs:
            print(f"alpha: {kwargs.get('alpha')}")
    
        print("Model input will be converted from (B, 1, D, H, W) to (B, 3, D, H, W).")


    def _build_trainer(self):
        self.trainer = BaseTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device,
            cfg=self.cfg,
            salience_gen=self.salience_gen,
        )

    def run(self):
        print("=== Start Training ===")
        self.trainer.train()
        print("=== Training Finished ===")

