from enum import Enum

import pandas as pd

from utils.mobility_data_manager import DataManager


class Data(Enum):
    ALL = 1
    SMALL = 2
    LAST_3000 = 3
    LAST_1000 = 4
    FIRST_1000 = 5
    FIRST_3000 = 6


def load_data(type: Data):
    match type:
        case Data.ALL:
            csv_file = "../Data/cityA_groundtruthdata.csv/cityA_groundtruthdata.csv"
            manager = DataManager(
                csv_path=csv_file,
            )
            df = manager.load_csv()
        case Data.SMALL:
            df = pd.read_csv(
                "../Data/cityA_groundtruthdata.csv/cityA_groundtruthdata_small.csv"
            )
        case Data.LAST_1000:
            df = pd.read_csv(
                "../Data/cityA_groundtruthdata.csv/cityA_groundtruthdata_last_1000_users.csv"
            )
        case Data.LAST_3000:
            df = pd.read_csv(
                "../Data/cityA_groundtruthdata.csv/cityA_groundtruthdata_last_3000_users.csv"
            )
        case Data.FIRST_1000:
            df = pd.read_csv(
                "../Data/cityA_groundtruthdata.csv/cityA_groundtruthdata_first_1000_users.csv"
            )
        case Data.FIRST_3000:
            df = pd.read_csv(
                "../Data/cityA_groundtruthdata.csv/cityA_groundtruthdata_first_3000_users.csv"
            )
        case _:
            df = pd.read_csv(
                "../Data/cityA_groundtruthdata.csv/cityA_groundtruthdata_small.csv"
            )
    return df
