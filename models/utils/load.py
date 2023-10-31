from typing import Tuple

import pandas as pd


def _rename_columns(df):
    df.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)
    return df


def load_data(path):
    # station and cell level data
    bs_info = pd.read_csv(path + "/BSinfo.csv")
    cl_data = pd.read_csv(path + "/CLdata.csv")

    # train and to_predict data
    train_data = pd.read_csv(path + "/ECdata.csv")
    to_predict = pd.read_csv(path + "/to_predict.csv")
    to_predict = to_predict.drop(columns=["w"])
    ec_df = pd.read_csv(path + "/ECdata.csv")
    
    # rename columns to lower case without spaces for convenience
    bs_info = _rename_columns(bs_info)
    cl_data = _rename_columns(cl_data)
    train_data = _rename_columns(train_data)
    to_predict = _rename_columns(to_predict)
    ec_df = _rename_columns(ec_df)

    # convert time columns to datetime
    train_data["time"] = pd.to_datetime(train_data["time"])
    to_predict["time"] = pd.to_datetime(to_predict["time"])
    cl_data["time"] = pd.to_datetime(cl_data["time"])
    ec_df['time'] = pd.to_datetime(ec_df['time'])


    # create a single row for each base station with multiple sets of columns for each cell
    info_df = cl_data.merge(bs_info, on=["bs", "cellname"], how="left")
    info_df = info_df.pivot(
        index=["time", "bs"],
        columns=["cellname"],
        values=[
            "load",
            "esmode1",
            "esmode2",
            "esmode3",
            "esmode4",
            "esmode5",
            "esmode6",
            "frequency",
            "bandwidth",
            "antennas",
            "txpower",
        ],
    ).reset_index()

    # flatten the multi-index column names
    info_df.columns = [
        "_".join([_col.lower() for _col in col]).strip() if col[1] else col[0]
        for col in info_df.columns.values
    ]
    info_df = info_df.merge(
        bs_info.groupby("bs")[["rutype", "mode"]].first().reset_index(),
        on="bs",
        how="left",
    )

    train_df = train_data.merge(info_df, on=["time", "bs"], how="left")
    predict_df = to_predict.merge(info_df, on=["time", "bs"], how="left")

    df = info_df.merge(ec_df, on=['time', 'bs'], how='left')
    df['split'] = df['energy'].isna().apply(lambda x: 'test' if x == True else 'train')

    train_1 = df[df['split'] =='train']
    test_data = df[df['split'] =='test']

    return train_df, predict_df,train_1, test_data


def load_submission(path):
    return pd.read_csv(path)