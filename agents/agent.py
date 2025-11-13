from dataclasses import dataclass
from typing import List

import numpy as np

from dataset.dataset_flat import DatasetFlatAugmented, DatasetFlat
from dataset.dataset_sequential import DatasetSequential, DatasetSequentialAugmented
from dataset.model_data import ModelData
from loaders.pred_prices_provider import PredPricesProvider
from loaders.real_prices_provider import RealPricesProvider
from timeseries.TimeseriesInterval import TimeseriesInterval


@dataclass
class AgentConfig:
    device: str = "cpu"
    learning_rate: float = 1e-3
    batch_size: int = 32
    train_episodes: int = 200
    train_days: int = 7
    # ...

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
        self.model = None # here comes your model

    def train(self, interval: TimeseriesInterval, symbols: List[str]):
        raise NotImplementedError()

    def test(self, interval: TimeseriesInterval, symbols: List[str]):
        raise NotImplementedError()

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
