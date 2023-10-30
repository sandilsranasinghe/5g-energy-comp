from typing import Callable
import logging

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ModelEvalWrapper:
    y_col = "energy"

    def __init__(
        self,
        model_getter: Callable,
        model_trainer: Callable,
        model_predictor: Callable,
        name=None,
        kfolds=5,
    ):
        self.model_getter = model_getter
        self.name = name
        self.best_model = None
        self.kfolds = kfolds

        # model_trainer takes in model and X and y dataframes
        self.model_trainer = model_trainer
        # model_predictor takes in model and X dataframe
        self.model_predictor = model_predictor

    def predict(self, df):
        X = df.drop(self.y_col, axis=1)
        return self.model_predictor(self.best_model, X)

    def evaluate(self, model, df):
        y_df = df[self.y_col]
        X = df.drop(self.y_col, axis=1)
        y_pred = self.model_predictor(model, X)

        mae = mean_absolute_error(y_df, y_pred)
        mape = mean_absolute_percentage_error(y_df, y_pred)
        mse = mean_squared_error(y_df, y_pred)

        return mae, mape, mse

    def train_and_eval(self, train_df):
        # seed for reproducibility
        np.random.seed(42)
        kf = KFold(n_splits=self.kfolds, shuffle=True)
        mae, mape, mse = [], [], []
        best_fold = 0
        for fold, (train_index, test_index) in enumerate(kf.split(train_df)):
            logging.info(f"Fold {fold+1}/{self.kfolds}")

            train, test = train_df.iloc[train_index], train_df.iloc[test_index]
            model = self.model_getter()
            X = train.drop(self.y_col, axis=1)
            y = train[self.y_col]
            self.model_trainer(model, X, y)

            train_mae, train_mape, train_mse = self.evaluate(model, train)
            test_mae, test_mape, test_mse = self.evaluate(model, test)

            logging.info(
                f"Train MAE: {train_mae:.4f}, MAPE: {train_mape:.4f}, MSE: {train_mse:.4f}"
            )
            logging.info(
                f"Test MAE: {test_mae:.4f}, MAPE: {test_mape:.4f}, MSE: {test_mse:.4f}"
            )

            mae.append(test_mae)
            mape.append(test_mape)
            mse.append(test_mse)

            if self.best_model is not None and (mae[-1] * 0.6 + mape[-1] * 0.4) < (
                min(
                    (mae[i] * 0.6 + mape[i] * 0.4) for i in range(len(mae)) if i != fold
                )
            ):
                self.best_model = model
                best_fold = fold
            elif self.best_model is None:
                self.best_model = model

        logging.info(f"\nBest model from fold {best_fold+1}")
        logging.info(
            f"Best model mae: {mae[best_fold]:.4f}, mape: {mape[best_fold]:.4f} mse: {mse[best_fold]:.4f}\n"
        )

        logging.info(
            "Average mae: {:.4f}, mape: {:.4f}, mse: {:.4f}".format(
                sum(mae) / len(mae), sum(mape) / len(mape), sum(mse) / len(mse)
            )
        )
        logging.info(
            "Std mae: {:.4f}, mape: {:.4f}, mse: {:.4f}".format(
                np.std(mae), np.std(mape), np.std(mse)
            )
        )

    def compare_predictions_with(self, df, y_comparison):
        y_pred = self.predict(df)

        plt.plot(y_pred, y_comparison, "o")
        plt.xlabel("y_pred")
        plt.ylabel("y_comparison")
        plt.show()
