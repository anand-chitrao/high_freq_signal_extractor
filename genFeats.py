import cupy as cp
import polars as pl

def gen():
    temp = pl.read_csv("Data/BTC_1sec.csv").drop(["", "system_time"]).to_numpy()

    data = cp.empty_like(temp)
    data.set(temp)

    # 1 column for Volume weighted Mid Price - Mid Price, 15 columns for OFI, 15 columns for Spoofing
    featArr = cp.empty((data.shape[0] - 2, 31), order="F")

    # Volume weighted Mid Price - Mid Price
    featArr[:, 0] = ((data[:-2, 19]/(data[:-2, 0] + data[:-2, 4]))*(data[:-2, 0] + data[:-2, 79]) + (data[:-2, 94]/(data[:-2, 0] + data[:-2, 79]))*(data[:-2, 0] + data[:-2, 4]))/((data[:-2, 19]/(data[:-2, 0] + data[:-2, 4])) + (data[:-2, 94]/(data[:-2, 0] + data[:-2, 79]))) - data[:-2, 0]

    # OFI computations
    featArr[:, cp.arange(1, 16)] = ((data[:-2, 0:1] + data[:-2, cp.arange(4, 19)])*(data[:-2, cp.arange(49, 64)] - data[:-2, cp.arange(34, 49)] -  data[:-2, cp.arange(139, 154)]) - (data[:-2, 0:1] + data[:-2, cp.arange(79, 94)])*(data[:-2, cp.arange(124, 139)] - data[:-2, cp.arange(109, 124)] - data[:-2, cp.arange(64, 79)]))  * cp.exp(-0.5*cp.arange(0, 15))

    # Spoofing computations
    featArr[:, cp.arange(16, 31)] = ((data[:-2, cp.arange(34, 49)]/(data[:-2, cp.arange(19, 34)] + 1e-6)) - (data[:-2, cp.arange(109, 124)]/(data[:-2, cp.arange(94, 109)] + 1e-6))) * cp.exp(-0.5*cp.arange(0, 15))

    # Separating 80% of data for training
    train_size = ((data.shape[0] - 2)*4)//5
    training_featArr = featArr[:train_size]
    training_objective = data[2:train_size + 2, 0] - data[1:train_size + 1, 0]
    tester_featArr = featArr[train_size:]
    tester_objective = data[train_size + 2:, 0] - data[train_size + 1: -1, 0]

    del data, featArr
    cp.get_default_memory_pool().free_all_blocks()

    return training_featArr, training_objective, tester_featArr, tester_objective

gen()