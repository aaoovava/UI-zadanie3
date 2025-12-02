from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.dataset_flat import DatasetFlatAugmented, DatasetFlat
from dataset.dataset_sequential import DatasetSequential, DatasetSequentialAugmented
from dataset.model_data import ModelData
from loaders.pred_prices_provider import PredPricesProvider
from loaders.real_prices_provider import RealPricesProvider
from timeseries.TimeseriesInterval import TimeseriesInterval

from models.gru_classifier import GRUClassifier
from models.lstm_classifier import LSTMClassifier

@dataclass
class AgentConfig:
    device: str = "cpu"
    learning_rate: float = 0.006
    batch_size: int = 64
    epochs: int = 150
    train_days: int = 7

    hidden_size: int = 48
    num_layers: int = 2
    dropout: float = 0.3

class Agent:
    def __init__(
            self,
            config: AgentConfig,
            real_prices_provider: RealPricesProvider,
            pred_prices_provider: PredPricesProvider
    ):
        self.config = config
        self.real_prices_provider = real_prices_provider
        self.pred_prices_provider = pred_prices_provider

        self.device = torch.device(config.device)

        models_count = len (self.pred_prices_provider.get_models_defs())
        input_size = models_count + 1

        self.model = GRUClassifier(
            input_size = input_size,
            hidden_size = self.config.hidden_size,
            num_layers = self.config.num_layers,
            dropout = self.config.dropout
        ).to(self.device)


    def train(self, interval: TimeseriesInterval, symbols: List[str]):
        dataset = self.prepare_dataset(symbols, interval, augmentation=True)
        dataloader = DataLoader(dataset, batch_size = self.config.batch_size, shuffle = True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        self.model.train()
        print("Start training on", len(dataset), "samples for", self.config.epochs, "episodes")

        losses_history = []

        for epoch in range (self.config.epochs):
            total_loss = 0
            correct = 0
            total = 0

            for batch_x, batch_y in dataloader:
                inputs = batch_x.to(self.device).float()
                targets = batch_y.to(self.device).float().unsqueeze(1)

                optimizer.zero_grad()
                logits = self.model(inputs)
                loss = criterion(logits, targets)

                loss.backward()
                optimizer.step()

                losses_history.append(loss.item())

                total_loss += loss.item()
                probs = torch.sigmoid(logits)
                predicted = (probs > 0.5).float()
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                acc = correct / total * 100

                print("Epoch:", epoch + 1, "Loss:", avg_loss, "Accuracy(%):", acc)

        return losses_history


    def test(self, interval: TimeseriesInterval, symbols: List[str]):
        dataset = self.prepare_dataset(symbols, interval, augmentation=False)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                inputs = batch_x.to(self.device).float()
                targets = batch_y.to(self.device).float().unsqueeze(1)

                logits = self.model(inputs)
                probs = torch.sigmoid(logits)
                predicted = (probs > 0.5).float()

                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        acc = 100 * correct / total
        print(f"Test Results for {symbols}: Accuracy = {acc:.2f}%")
        print("-" * 30)

        return acc

    def prepare_dataset(self, symbols: List[str], timeseries_interval: TimeseriesInterval, augmentation=False):
        model_data = self.__get_model_state(
            timeseries_interval,
            symbols=symbols
        )

        # choose between sequential and flat representation
        if augmentation:
            # dataset = DatasetFlatAugmented(model_data, seq_len=self.config.train_days)
            dataset = DatasetSequentialAugmented(model_data, seq_len=self.config.train_days)
        else:
            # dataset = DatasetFlat(model_data, seq_len=self.config.train_days)
            dataset = DatasetSequential(model_data, seq_len=self.config.train_days)

        return dataset

    # DO NOT MODIFY!
    def __get_model_state(self, timeseries_interval: TimeseriesInterval, symbols: List[str], hist_items_cnt: int = 1):
        symbols_cnt = len(symbols)
        timeseries_cnt = timeseries_interval.get_steps_cnt()
        models_defs = self.pred_prices_provider.get_models_defs()

        seq_features = []

        open_price = np.ndarray((symbols_cnt, timeseries_cnt), dtype=np.float32)
        close_price = np.ndarray((symbols_cnt, timeseries_cnt), dtype=np.float32)

        for symbol_idx, symbol in enumerate(symbols):
            real_prices_hist = self.real_prices_provider.get_prices_np(symbol=symbol, date=timeseries_interval.get_date_to(), hist_cnt=timeseries_cnt + 2)
            real_return_hist = (real_prices_hist[1:] - real_prices_hist[:-1]) / real_prices_hist[:-1]

            for i in range(timeseries_cnt):
                real_return = [real_return_hist[i]]
                close_price[symbol_idx][i] = real_prices_hist[i + 2]
                open_price[symbol_idx][i] = real_prices_hist[i + 1]

                # --- Predictions ---
                date_to = timeseries_interval.get_next_timeseries_date(i).get_date()
                pred_features = []
                for model_idx, model_id in enumerate(models_defs):
                    pred_prices_hist = self.pred_prices_provider.get_prices_np(models_defs[model_id], symbol, date=date_to, hist_cnt=hist_items_cnt + 1)
                    pred_return_hist = (pred_prices_hist[1:] - pred_prices_hist[:-1]) / pred_prices_hist[:-1]
                    pred_features.extend(pred_return_hist.tolist())

                # Combine features: [real_return_hist + all_pred_returns]
                day_features = np.concatenate([real_return, pred_features])
                seq_features.append(day_features)

        seq_features = np.array(seq_features, dtype=np.float32)  # shape: (days_cnt, feature_dim)

        return ModelData(
            days_cnt=timeseries_cnt,
            symbols_cnt=symbols_cnt,
            state=seq_features,
            open_price=open_price.flatten(),
            close_price=close_price.flatten(),
        )
