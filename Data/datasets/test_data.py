import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from Models.interpretable_diffusion.model_utils import (
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)


class TestDataset(Dataset):
    def __init__(self, csv_path, window=24, var_num=None, auto_norm=True,
                 output_dir=None, missing_ratio=0.0, predict_length=0,
                 sampling_steps=50, step_size=0.01, coefficient=0.1):
        self.window = window
        self.auto_norm = auto_norm
        self.output_dir = output_dir
        self.missing_ratio = missing_ratio
        self.predict_length = predict_length
        self.sampling_steps = sampling_steps
        self.step_size = step_size
        self.coefficient = coefficient

        df = pd.read_csv(csv_path)
        data = df.values.astype(np.float32)

        if var_num:
            data = data[:, :var_num]
        self.data_raw = data
        self.var_num = data.shape[1]

        # 初始化 scaler，并拟合数据
        self.scaler = MinMaxScaler()
        data = self.scaler.fit_transform(data)

        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)

        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True) + 1e-6

        self.windows = []
        for i in range(len(data) - window):
            self.windows.append(data[i:i+window])
        self.windows = np.stack(self.windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx]  # (window, features)
        if self.missing_ratio > 0:
            mask = np.ones_like(x)
            num_mask = int(np.prod(x.shape) * self.missing_ratio)
            indices = np.random.choice(x.size, num_mask, replace=False)
            mask.flat[indices] = 0
            return torch.tensor(x, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)
        else:
            return torch.tensor(x, dtype=torch.float32)

    def normalize(self, x):  # x shape: (B, T, C)
        x_flat = x.reshape(-1, self.var_num)
        x_scaled = self.scaler.transform(x_flat)
        if self.auto_norm:
            x_scaled = normalize_to_neg_one_to_one(x_scaled)
        return x_scaled.reshape(x.shape)

    def unnormalize(self, x):  # x shape: (B, T, C)
        x_flat = x.reshape(-1, self.var_num)
        if self.auto_norm:
            x_flat = unnormalize_to_zero_to_one(x_flat)
        x_inv = self.scaler.inverse_transform(x_flat)
        return x_inv.reshape(x.shape)
