import cupy as cp
import genFeats as gF

def coeffs(featArr, objective):
    λ = 3
    D = cp.exp(λ*cp.arange(0, 15, dtype=cp.float64))
    Λ = cp.zeros((31, 31), dtype=cp.float64)
    Λ[0, 0] = cp.exp(10)
    Λ[1:16, 1:16] = D
    Λ[16:31, 16:31] = D

    featArr = (featArr - featArr.mean(axis=1).reshape(-1, 1))/(featArr.std(axis=1).reshape(-1, 1) + 1e-6)

    β = cp.linalg.solve(featArr.T @ featArr + Λ, featArr.T @ objective)

    return β

f, o = gF.gen()

β = coeffs(f, o)
print(β)
