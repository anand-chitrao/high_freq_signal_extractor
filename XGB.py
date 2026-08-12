import cupy as cp
import xgboost as xgb
import genFeats as gF

def XGB():
    training_featArr, training_objective, tester_featArr, tester_objective, test_mid_price, test_best_ask, test_best_bid = gF.gen()

    train_data = xgb.QuantileDMatrix(training_featArr, label=training_objective)
    test_data  = xgb.QuantileDMatrix(tester_featArr, label=tester_objective, ref=train_data)

    params = {
        "device": "cuda",
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.1,
        "max_depth": 4,
        "seed": 42,
    }

    # evals = [(train_data, "train"), (test_data, "test")]

    # model_xgb = xgb.train(params, train_data, num_boost_round=100, evals=evals, verbose_eval=100)
    model_xgb = xgb.train(params, train_data, num_boost_round=100)

    preds = cp.array(model_xgb.predict(test_data))

    # print(err(tester_objective, preds))

    return tester_objective, preds, test_mid_price, test_best_ask, test_best_bid

def err(objective, preds):
    tss = cp.sum(objective**2)
    rss = cp.sum((preds - objective)**2)
    return 1 - (rss/tss)

# tester_objective, preds, test_mid_price, test_best_ask, test_best_bid = XGB()
# When you are importing .py files in a different .py file, they seem to get executed
# print(err(tester_objective, preds))

