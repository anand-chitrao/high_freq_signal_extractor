import cupy as cp
import linReg as lR
import XGB as xgb

def xgb_trading_profits():
    tester_objective, preds, test_mid_price, test_best_ask, test_best_bid, window, momentum, train_size, triggered_buys, triggered_sells = xgb.XGB()

    # threshold = 0.00015*test_mid_price[:-window]
    threshold = 0.01

    keep_bid       = preds[:-window] > -threshold
    keep_ask       = preds[:-window] <  threshold
    bid_executed   = (test_best_bid[window:] < test_best_bid[1:-window+1]) | (triggered_sells[window+1:-window+1] > 0) | (triggered_sells[window+2:] > 0)
    ask_executed   = (test_best_ask[window:] > test_best_ask[1:-window+1]) | (triggered_buys[window+1:-window+1] > 0) | (triggered_buys[window+2:] > 0)

    executed_buys  = cp.where(keep_bid & bid_executed,  test_mid_price[window:] - test_best_bid[1:-window+1], 0)
    executed_sells = cp.where(keep_ask & ask_executed, -test_mid_price[window:] + test_best_ask[1:-window+1], 0)

    profit = (executed_buys + executed_sells).sum()

    buys   = (keep_bid & bid_executed).sum()
    sells  = (keep_ask & ask_executed).sum()

    print(f"XGB generated a profit of {profit} in {buys} buys and {sells} sells amongst {keep_bid.shape[0]} opportunities.")

def ols_trading_profits():
    tester_featArr, tester_objective, V, β, γ, test_mid_price, test_best_ask, test_best_bid, window = lR.linReg()

    preds = (γ + tester_featArr @ V @ β.reshape(-1, 1)).flatten()

    threshold = 0.0001*test_mid_price[:-window]

    keep_bid       = preds[:-window] > -threshold
    keep_ask       = preds[:-window] <  threshold
    bid_executed   = test_best_bid[window:] < test_best_bid[1:-window+1]
    ask_executed   = test_best_ask[window:] > test_best_ask[1:-window+1]

    executed_buys  = cp.where(keep_bid & bid_executed,  test_mid_price[window:] - test_best_bid[1:-window+1], 0)
    executed_sells = cp.where(keep_ask & ask_executed, -test_mid_price[window:] + test_best_ask[1:-window+1], 0)

    profit = (executed_buys + executed_sells).sum()
    
    buys   = (keep_bid & bid_executed).sum()
    sells  = (keep_ask & ask_executed).sum()
    
    print(f"OLS generated a profit of {profit} in {buys} buys and {sells} sells amongst {keep_bid.shape[0]} opportunities.")

xgb_trading_profits()
# ols_trading_profits()
