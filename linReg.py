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
    training_featArr, training_objective, tester_featArr, tester_objective = gF.gen()

    V = PCA(training_featArr)

    β, γ = OLS(training_featArr, training_objective, V)

    return tester_featArr, tester_objective, V, β, γ
    # return training_featArr, training_objective, V, β, γ

linReg()