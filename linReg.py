import cupy as cp
import genFeats as gF

# Extracting principal components
def PCA(featArr):
    N = featArr.shape[0]
    feat_cov = cp.cov(featArr, rowvar=False)
    eigVal, eigVec = cp.linalg.eigh(feat_cov)
    eigVal = eigVal[::-1]
    tot_eigVal = cp.sum(eigVal)
    trunc_point = cp.where(cp.cumsum(eigVal)/tot_eigVal > 0.95)[0]
    eigVal = eigVal[:trunc_point[0] + 1]
    eigVec = eigVec[:, ::-1]
    eigVec = eigVec[:, :trunc_point[0] + 1]
    
    return eigVec

# The OLS module
def OLS(featArr, objective, V):
    # Projecting the features onto the principal components
    featArr = featArr @ V
    feat_cov = cp.var(featArr, axis=0)

    # Getting the OLS coefficients using the covariance matrix trick
    meanObj   = objective.mean()
    meanFeat  = featArr.mean(axis = 0)
    meanCross = (objective.reshape(-1, 1) * featArr).mean(axis = 0)
    b = meanCross - meanObj*meanFeat

    β = b/feat_cov
    γ = meanObj - β @ meanFeat

    return β, γ

def linReg():
    # Getting the data and separating it into the training set and the testing set in a ratio of 80:20
    training_featArr, training_objective, tester_featArr, tester_objective, test_mid_price, test_best_ask, test_best_bid = gF.gen()

    V = PCA(training_featArr)

    β, γ = OLS(training_featArr, training_objective, V)

    return tester_featArr, tester_objective, V, β, γ, test_mid_price, test_best_ask, test_best_bid
    # return training_featArr, training_objective, V, β, γ

def err(X, Y, β, γ):
    meanY = Y.mean()
    tss = cp.sum((Y - meanY)**2)
    rss = cp.sum((Y.reshape(-1, 1) - γ - X @ β.reshape(-1, 1))**2)
    R2 = 1 - (rss/tss)
    return R2

# tester_featArr, tester_objective, V, β, γ, test_mid_price, test_best_ask, test_best_bid = linReg()

# R2 = err(tester_featArr @ V, tester_objective, β, γ)
# print(R2)