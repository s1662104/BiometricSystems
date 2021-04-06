import numpy as np
import pandas as pd


def recognize():
    # DA MODIFICARE
    csv = pd.read_csv("dataset_user.csv", index_col=[0])
    return csv.iloc[0]


def main():
    pass


if __name__ == '__main__':
    main()
