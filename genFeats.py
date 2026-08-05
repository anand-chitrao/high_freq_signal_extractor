import cupy as cp
import polars as pl
import time

def gen():
    temp = pl.read_csv("Data/BTC_1sec.csv").drop(["", "system_time"]).to_numpy()

    data = cp.empty_like(temp)
    data.set(temp)

    del temp
    cp.get_default_memory_pool().free_all_blocks()

    # time.sleep(5)

    # 1 column for Mid-Price, 15 columns for OFI, 15 columns for Spoofing
    featArr = cp.empty((data.shape[0], 31), order="F")

    # Mid-Price
    featArr[:, 0] = data[:, 0]

    # OFI computations
    featArr[:, cp.arange(1, 16)] = (data[:, 0:1] + data[:, cp.arange(4, 19)])*(data[:, cp.arange(49, 64)] - data[:, cp.arange(34, 49)] -  data[:, cp.arange(139, 154)]) - (data[:, 0:1] + data[:, cp.arange(79, 94)])*(data[:, cp.arange(124, 139)] - data[:, cp.arange(109, 124)] - data[:, cp.arange(64, 79)])

    # Spoofing computations
    featArr[:, cp.arange(16, 31)] = (data[:, cp.arange(34, 49)]/data[:, cp.arange(19, 34)]) - (data[:, cp.arange(109, 124)]/data[:, cp.arange(94, 109)])

    del data
    cp.get_default_memory_pool().free_all_blocks()

    time.sleep(5)
    return featArr

gen()