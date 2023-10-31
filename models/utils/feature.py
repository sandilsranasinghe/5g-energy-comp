import numpy as np
import pandas as pd


def apply_feature_engineering(train_df, predict_df, funcs):
    for func in funcs:
        train_df, predict_df = func(train_df, predict_df)
    return train_df, predict_df


def add_hours(train_df, predict_df):
    train_df["hour"] = train_df["time"].dt.hour
    predict_df["hour"] = predict_df["time"].dt.hour
    return train_df, predict_df


def add_hours_trig_cyclic(train_df, predict_df):
    train_df["hour_sin"] = np.sin(2 * np.pi * train_df["time"].dt.hour / 24)
    train_df["hour_cos"] = np.cos(2 * np.pi * train_df["time"].dt.hour / 24)
    predict_df["hour_sin"] = np.sin(2 * np.pi * predict_df["time"].dt.hour / 24)
    predict_df["hour_cos"] = np.cos(2 * np.pi * predict_df["time"].dt.hour / 24)
    return train_df, predict_df


def get_dummies(columns):
    def _get_dummies(train_df, predict_df):
        # we need to combine the dfs and then get dummies and then split them again
        combined_df = pd.concat([train_df, predict_df])
        combined_df = pd.get_dummies(combined_df, columns=columns)
        train_df = combined_df.iloc[: len(train_df)]
        predict_df = combined_df.iloc[len(train_df) :]
        # train_df = pd.get_dummies(train_df, columns=columns)
        # predict_df = pd.get_dummies(predict_df, columns=columns)
        return train_df, predict_df

    return _get_dummies


def fill_na_zero(columns):
    def _fill_na(train_df, predict_df):
        if columns == "all":
            train_df = train_df.fillna(0)
            predict_df = predict_df.fillna(0)
        else:
            train_df[columns] = train_df[columns].fillna(0)
            predict_df[columns] = predict_df[columns].fillna(0)
        return train_df, predict_df

    return _fill_na


def drop_non_input_cols(train_df, predict_df):
    train_df = train_df.drop(columns=["time", "bs"])
    predict_df = predict_df.drop(columns=["time", "bs"])
    return train_df, predict_df


def normalize(columns):
    def _normalize(train_df, predict_df):
        combined_df = pd.concat([train_df, predict_df])
        if columns == "all":
            combined_df = (combined_df - combined_df.min()) / (
                combined_df.max() - combined_df.min()
            )
        else:
            combined_df[columns] = (
                combined_df[columns] - combined_df[columns].min()
            ) / (combined_df[columns].max() - combined_df[columns].min())
        train_df = combined_df.iloc[: len(train_df)]
        predict_df = combined_df.iloc[len(train_df) :]
        return train_df, predict_df

    return _normalize


def standardize(columns):
    def _standardize(train_df, predict_df):
        combined_df = pd.concat([train_df, predict_df])
        if columns == "all":
            combined_df = (combined_df - combined_df.mean()) / combined_df.std()
        else:
            combined_df[columns] = (
                combined_df[columns] - combined_df[columns].mean()
            ) / (combined_df[columns].std())
        train_df = combined_df.iloc[: len(train_df)]
        predict_df = combined_df.iloc[len(train_df) :]
        return train_df, predict_df

    return _standardize

def feature_enginning(train1, valid_df, test_df):

    train1['hour'] = train1['time'].dt.hour
    valid_df['hour'] = valid_df['time'].dt.hour
    test_df['hour'] = test_df['time'].dt.hour

    train1['split'] = 'train'
    valid_df['split'] = 'valid'
    test_df['split'] = 'test'

    df = pd.concat([train1, valid_df, test_df])
    df['bs_en'] = df['bs'].apply(lambda x: int(x.strip('B_')))

   
    df = pd.get_dummies(df, columns=['rutype',  'mode', 'hour'])

    df.sort_values(['time', 'bs'], inplace=True)


    train1 = df[df['split'] =='train']
    valid_df = df[df['split'] =='valid']
    test_df = df[df['split'] =='test']

    return train1, valid_df, test_df

