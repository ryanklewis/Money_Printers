import pandas as pd
from pathlib import Path


def load_data(base_path, output_path, auction=False, normalization="Zscore", days_train=7, rename_cols=True):
    # check for valid arguments
    if normalization not in ("Zscore", "MinMax", "DecPre"):
        raise Exception("invalid normalization")
    if days_train < 1 or days_train > 9:
        raise Exception("invalid number of training days")

    # generate correct base path
    auction_word = "Auction" if auction else "NoAuction"
    base_path = base_path.joinpath(auction_word)
    normalization_path_dict = {"Zscore": 1, "MinMax": 2, "DecPre": 3}
    base_path = base_path.joinpath(
        f"{normalization_path_dict[normalization]}.{auction_word}_{normalization}")

    # generate train and test paths, clear files
    train_path = output_path.joinpath(
        f"Train_{auction_word}_{normalization}.csv")
    test_path = output_path.joinpath(
        f"Test_{auction_word}_{normalization}.csv")
    with open(train_path, "w") as f:
        pass
    with open(test_path, "w") as f:
        pass

    # load data, each day is done individually so that the normalization is consistent
    for i in range(10):
        if i == 0:
            input_filepath = base_path.joinpath(
                f"{auction_word}_{normalization}_Training", f"Train_Dst_{auction_word}_{normalization}_CF_{i+1}.txt")
        else:
            input_filepath = base_path.joinpath(
                f"{auction_word}_{normalization}_Testing", f"Test_Dst_{auction_word}_{normalization}_CF_{i}.txt")

        data = pd.read_csv(
            input_filepath, sep=r'\s{2,3}', header=None, engine="python")
        data = data.T

        # rename columns
        if rename_cols:
            for j in range(10):
                data = data.rename(columns={
                    4*j: f"ask_price_{j}",
                    4*j+1: f"ask_volume_{j}",
                    4*j+2: f"bid_price_{j}",
                    4*j+3: f"bid_volume_{j}",

                    40+2*j: f"spread_{j}",
                    40+2*j+1: f"mid_price_{j}",

                    144: "label_1",
                    145: "label_2",
                    146: "label_3",
                    147: "label_5",
                    148: "label_10"
                })

        # write to csv
        if i < days_train:
            data.to_csv(train_path, mode='a', index=False, header=(i == 0))
        else:
            data.to_csv(test_path, mode='a', index=False,
                        header=(i == days_train))

        print(f"{data.shape[0]} records processed for day {i+1}")


load_data(Path("C:\\Users\\joshr\\Downloads\\7643-dataset\\published\\BenchmarkDatasets\\BenchmarkDatasets\\BenchmarkDatasets"),
          Path("data"))
