import abc
import pathlib
import torch
from torch.utils.data import DataLoader
from common import settings
import random
import numpy as np

class BaseModel(abc.ABC):
    def __init__(self):
        random.seed(settings.SEED)
        np.random.seed(settings.SEED)
        torch.manual_seed(settings.SEED)
        torch.cuda.manual_seed(settings.SEED)
        torch.cuda.manual_seed_all(settings.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    @abc.abstractmethod
    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None, **kwargs):
        """Обучает модель на тренировочных данных."""
        pass

    @abc.abstractmethod
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Возвращает маску (индексы классов) для входного тензора изображений."""
        pass

    @abc.abstractmethod
    def save(self, path: str):
        """Сохраняет состояние модели в файл."""
        pass

    @abc.abstractmethod
    def load(self, path: str):
        """Загружает состояние модели из файла."""
        pass

    @staticmethod
    @abc.abstractmethod
    def get_model_save_path() -> pathlib.Path:
        """Где сохраняется обученная модель"""
        pass

    @staticmethod
    @abc.abstractmethod
    def get_metrics_save_path() -> pathlib.Path:
        """Куда сохранятся метрики"""
        pass
