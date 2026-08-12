import cupy as cp
import linReg as lR
import XGB as xgb

def xgb_trading_profits():
    tester_objective, preds, test_mid_price, test_best_ask, test_best_bid = xgb.XGB()
    profits_buy  = cp.where(preds[:-2] >  0.00000*test_mid_price[:-2], test_best_bid[2:] - test_best_bid[1:-1] - 0.00000*test_mid_price[:-2], 0)
    profits_sell = cp.where(preds[:-2] < -0.00000*test_mid_price[:-2], test_best_ask[1:-1] - test_best_ask[2:] - 0.00000*test_mid_price[:-2], 0)

    trade_number = (cp.where(profits_buy != 0, 1, 0) + cp.where(profits_sell != 0, 1, 0)).sum()

    positive_trades = (cp.where(profits_buy > 0, 1, 0) + cp.where(profits_sell > 0, 1, 0)).sum()

    # print(trade_number)
    print(f"XGBoost made {positive_trades} successful trades and {trade_number} total trades")

    profits = profits_buy.sum() + profits_sell.sum()

    del tester_objective, preds, test_mid_price, test_best_ask, test_best_bid, profits_buy, profits_sell
    cp.get_default_memory_pool().free_all_blocks()

    return profits

def OLS_trading_profits():
    tester_featArr, tester_objective, V, β, γ, test_mid_price, test_best_ask, test_best_bid = lR.linReg()
    preds = (γ + tester_featArr @ V @ β.reshape(-1, 1)).flatten()
    profits_buy  = cp.where(preds[:-2] >  0.00000*test_mid_price[:-2], test_best_bid[2:] - test_best_bid[1:-1] - 0.00000*test_mid_price[:-2], 0)
    profits_sell = cp.where(preds[:-2] < -0.00000*test_mid_price[:-2], test_best_ask[1:-1] - test_best_ask[2:] - 0.00000*test_mid_price[:-2], 0)

    trade_number = (cp.where(profits_buy != 0, 1, 0) + cp.where(profits_sell != 0, 1, 0)).sum()

    positive_trades = (cp.where(profits_buy > 0, 1, 0) + cp.where(profits_sell > 0, 1, 0)).sum()
    
    # print(trade_number)
    print(f"OLS made {positive_trades} successful trades and {trade_number} total trades")
    
    profits = profits_buy.sum() + profits_sell.sum()
    
    return profits
    

print(f"The profits gained using the XGBoost model = {xgb_trading_profits()} and those gained using the OLS model = {OLS_trading_profits()}")