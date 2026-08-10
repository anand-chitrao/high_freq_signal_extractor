import cupy as cp
import linReg as lR

def err(X, Y, β, γ):
    meanY = Y.mean()
    tss = cp.sum((Y - meanY)**2)
    rss = cp.sum((Y.reshape(-1, 1) - γ - X @ β.reshape(-1, 1))**2)
    R2 = 1 - (rss/tss)
    return R2

tester_featArr, tester_objective, V, β, γ = lR.linReg()

R2 = err(tester_featArr @ V, tester_objective, β, γ)
print(R2)
