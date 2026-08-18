import cupy as cp
import polars as pl

def gen():
    temp = pl.read_csv("Data/BTC_1sec.csv").drop(["", "system_time"]).to_numpy()

    data = cp.empty_like(temp)
    data.set(temp)

    window = 2

    momentum = 5

    featArr = cp.empty((data.shape[0] - window - momentum, 40), order="F")

    # The bid-ask spread
    featArr[:, 0] = data[momentum:-window, 1]

    # Accounting for the buy and sell orders executed
    featArr[:, cp.arange(1, 3)] = data[momentum:-window, cp.arange(2, 4)]

    # Volume weighted Mid Price - Mid Price
    featArr[:, 3] = ((data[momentum:-window, 19]/(data[momentum:-window, 0] + data[momentum:-window, 4]))*(data[momentum:-window, 0] + data[momentum:-window, 79]) + (data[momentum:-window, 94]/(data[momentum:-window, 0] + data[momentum:-window, 79]))*(data[momentum:-window, 0] + data[momentum:-window, 4]))/((data[momentum:-window, 19]/(data[momentum:-window, 0] + data[momentum:-window, 4])) + (data[momentum:-window, 94]/(data[momentum:-window, 0] + data[momentum:-window, 79]))) - data[momentum:-window, 0]

    # Order Flow Imbalance (OFI)
    featArr[:, cp.arange(4, 19)] = ((data[momentum:-window, 0:1] + data[momentum:-window, cp.arange(4, 19)])*(data[momentum:-window, cp.arange(49, 64)] - data[momentum:-window, cp.arange(34, 49)] - data[momentum:-window, cp.arange(64, 79)]) - (data[momentum:-window, 0:1] + data[momentum:-window, cp.arange(79, 94)])*(data[momentum:-window, cp.arange(124, 139)] - data[momentum:-window, cp.arange(109, 124)] - data[momentum:-window, cp.arange(139, 154)]))

    # Aggregating OFI for shallow part of the order book
    featArr[:, 19] = featArr[:, cp.arange(4, 7)].sum(axis=1)
    featArr[:, 20] = featArr[:, cp.arange(4, 9)].sum(axis=1)

    # Spoofing
    featArr[:, cp.arange(21, 36)] = ((data[momentum:-window, cp.arange(34, 49)]/(data[momentum:-window, cp.arange(19, 34)] + 1e-6)) - (data[momentum:-window, cp.arange(109, 124)]/(data[momentum:-window, cp.arange(94, 109)] + 1e-6)))

    # Average spoofing across first five levels
    featArr[:, 36] = featArr[:, cp.arange(21, 26)].mean(axis=1)

    # Spread change
    featArr[:, 37] = data[momentum:-window, 1] - data[:-(window + momentum), 1]

    # Rolling sums of buys and sells
    buy_cumsum  = cp.cumsum(data[:, 2])
    sell_cumsum = cp.cumsum(data[:, 3])
    featArr[:, 38] = buy_cumsum[momentum:-window] - buy_cumsum[:-window-momentum]
    featArr[:, 39] = sell_cumsum[momentum:-window] - sell_cumsum[:-window-momentum]

    # Separating 80% of data for training
    train_size = ((data.shape[0] - window - momentum)*4)//5

    training_featArr = featArr[:train_size]
    training_objective = data[momentum + window:train_size + momentum + window, 0] - data[momentum + 1:train_size + momentum + 1, 0]

    # Extracting the testing data

    tester_featArr   = featArr[train_size:]
    tester_objective = data[momentum + train_size + window:, 0] - data[train_size + momentum + 1: -window + 1, 0]

    test_mid_price  = data[momentum + train_size:-window, 0]
    test_best_ask   = test_mid_price + data[momentum + train_size:-window, 4]
    test_best_bid   = test_mid_price + data[momentum + train_size:-window, 79]
    triggered_buys  = data[momentum + train_size:, 2]
    triggered_sells = data[momentum + train_size:, 3]

    del data, featArr
    cp.get_default_memory_pool().free_all_blocks()

    return training_featArr, training_objective, tester_featArr, tester_objective, test_mid_price, test_best_ask, test_best_bid, window, momentum, train_size, triggered_buys, triggered_sells

gen()