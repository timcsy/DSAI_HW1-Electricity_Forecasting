import tensorly as tl
import tensorly.tenalg as ta
from tensorly import random
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def parafac_gd(X, mask=None, Ls=None, rank=None, penalty_weight=1e-3, learning_rate=1e-3, n_iter_max=None, tol=1e-5):
    # gradient descent method with regularization term
    # n_iter_max is usually 10000
    if Ls is None:
        Ls = [penalty_weight * tl.eye(X.shape[k]) for k in range(X.ndim)] # graph Laplacian and regularization term
    if mask is None:
        mask = X != 0
    if rank is None:
        rank = min(X.shape)
    
    A = [tl.tensor(0.1 * np.random.rand(X.shape[i], rank)) for i in range(X.ndim)] # Feature map

    X0_prev = tl.zeros_like(tl.unfold(X, 0))
    iteration = 0
    while True:
        iteration += 1

        E = X.copy()
        for r in range(rank):
            P = tl.tensor(1)
            for a in A:
                P = ta.tensor_dot(P, a[:, r])
            E -= P
        E *= mask
        
        # Update A[k]
        for k in range(len(A)):
            A[k] += learning_rate * (tl.unfold(E, k) @ ta.khatri_rao([A[i] for i in reversed(list(set(range(len(A))) ^ {k}))]) - Ls[k] @ A[k])
        
        X0_hat = A[0] @ ta.khatri_rao([A[i] for i in reversed(range(1, len(A)))]).T
        local_error = tl.norm(X0_hat - X0_prev) / (tl.norm(X0_hat) + tol)
        X0_prev = X0_hat
        if local_error < tol:
            break
        if n_iter_max is not None:
            if iteration >= n_iter_max:
                break

    result = tl.zeros_like(X)
    for r in range(rank):
        P = tl.tensor(1)
        for a in A:
            P = ta.tensor_dot(P, a[:, r])
        result += P
    
    RMSE = tl.norm((tl.unfold(X, 0) - X0_prev) * tl.unfold(mask, 0))

    print('Iterations:', iteration)
    print('RMSE:', RMSE)
    return result, A

def parafac_als(X, mask=None, rank=None, penalty_weight=1e-3, n_iter_max=10000, tol=1e-5):
    # ALS with regularization term
    if mask is None:
        mask = X != 0
    if rank is None:
        rank = min(X.shape)

    A = [0.1 * random.random_tensor((X.shape[i], rank)) for i in range(X.ndim)] # Feature map

    X0_prev = tl.zeros_like(tl.unfold(X, 0))
    iteration = 0
    while True:
        iteration += 1

        # Update A[k]
        for k in range(len(A)):
            ids = list(reversed(list(set(range(len(A))) ^ {k}))) # ndim-1 ~ 0 without k
            for pk, Mk in enumerate(tl.unfold(mask, k)):
                ATAI = penalty_weight * tl.eye(rank)
                ATX = tl.zeros([rank, 1])
                for index, observed in enumerate(Mk):
                    p = []

                    for i in reversed(ids):
                        p.append(index % mask.shape[i])
                        index = int(index / mask.shape[i])
                    p.insert(k, pk)
                    p = tuple(p) # index of the observed value X_p

                    if observed:
                        ATA = tl.ones([rank, rank])
                        for i in ids:
                            ATA *= ta.outer([A[i][p[i]], A[i][p[i]]])
                        if len(A) > 0:
                            ATAI += ATA
                        ATX += ta.khatri_rao([A[i][p[i]].reshape(1, -1) for i in ids]).T * X[p]
                C, is_low = cho_factor(ATAI)
                A[k][pk] = cho_solve((C, is_low), ATX, overwrite_b=True).T

        X0_hat = A[0] @ ta.khatri_rao([A[i] for i in reversed(range(1, len(A)))]).T
        local_error = tl.norm(X0_hat - X0_prev) / (tl.norm(X0_hat) + tol)
        X0_prev = X0_hat
        if local_error < tol:
            break
        if n_iter_max is not None:
            if iteration >= n_iter_max:
                break

    result = tl.zeros_like(X)
    for r in range(rank):
        P = tl.tensor(1)
        for a in A:
            P = ta.tensor_dot(P, a[:, r])
        result += P
    
    RMSE = tl.norm((tl.unfold(X, 0) - X0_prev) * tl.unfold(mask, 0))

    print('Iterations:', iteration)
    print('RMSE:', RMSE)
    return result, A

def grals_multiplication(Proj_k, W, H, L):
    K = tl.zeros_like(H) # nk x R matrix
    for i, j in Proj_k:
        K[i] += ta.inner(W[j], H[i]) * W[j]
    MS = (K + L @ H) # nk x R matrix
    return MS

def conjugate_gradient(B, Proj_k, W, H, L, tol=1e-5, n_iter_max=None):
    R = B - grals_multiplication(Proj_k, W, H, L) # vec(R) = vec(B) - Mvec(S.T)
    P = R
    if tl.norm(R) / tl.norm(B) < tol:
        return H
    for iteration in range(H.shape[0] * H.shape[1]):
        MP = grals_multiplication(Proj_k, W, P, L) # vec(R) = vec(B) - Mvec(P.T)
        a = tl.sum(R * R) / tl.sum(P * MP)
        H += a * P
        R_next = R - a * MP
        if tl.norm(R_next) / tl.norm(B) < tol:
            return H
        if n_iter_max is not None:
            if iteration >= n_iter_max:
                return H
        b = tl.sum(R_next * R_next) / tl.sum(R * R)
        P = R_next + b * P
        R = R_next
    return H


def parafac_grals(X, mask=None, Ls=None, rank=None, A=None, penalty_weight=1e-3, n_iter_max=10000, tol=1e-5):
    # Graph Regularized Alternating Least Squares (GRALS)
    if Ls is None:
        Ls = [penalty_weight * tl.eye(X.shape[k]) for k in range(X.ndim)] # graph Laplacian and regularization term
    if mask is None:
        mask = X != 0
    if rank is None:
        rank = min(X.shape)
    if A is None:
        A = [0.1 * random.random_tensor((X.shape[i], rank)) for i in range(X.ndim)] # Feature map
    
    rN = [list(reversed(list(set(range(len(A))) ^ {k}))) for k in range(len(A))] # ndim-1 ~ 0 without k
    Proj = [[] for _ in range(len(A))] # Proj[k]: tuple[ik, j], j is the index of W
    for p in tl.tensor(mask.nonzero()).T:
        for k in range(len(A)):
            j = 0
            for i in rN[k]:
                j *= mask.shape[i]
                j += p[i]
            Proj[k].append((p[k], j))

    X0_prev = tl.zeros_like(tl.unfold(X, 0))
    iteration = 0
    while True:
        iteration += 1

        # Update A[k]
        for k in range(len(A)):
            Y = tl.unfold(X * mask, k)
            W = ta.khatri_rao([A[i] for i in rN[k]])
            B = Y @ W # nk x R matrix
            H = A[k]
            A[k] = conjugate_gradient(B, Proj[k], W, H, Ls[k], tol=tol)

        X0_hat = A[0] @ ta.khatri_rao([A[i] for i in reversed(range(1, len(A)))]).T
        local_error = tl.norm(X0_hat - X0_prev) / (tl.norm(X0_hat) + tol)
        X0_prev = X0_hat
        if local_error < tol:
            break
        if n_iter_max is not None:
            if iteration >= n_iter_max:
                break
    
    result = tl.zeros_like(X)
    for r in range(rank):
        P = tl.tensor(1)
        for a in A:
            P = ta.tensor_dot(P, a[:, r])
        result += P
    
    RMSE = tl.norm((tl.unfold(X, 0) - X0_prev) * tl.unfold(mask, 0))

    print('Iterations:', iteration)
    print('RMSE:', RMSE)
    return result, A

def regurization_AR(L, W, T, r, penalty_weight=1e-3):
    m = max(L) # Also the length of W[:,r]
    L_bar = set(L) | {0}
    delta = {}
    for d in range(1, m + 1):
        delta[d] = set()
        for l in L_bar:
            if l - d in L_bar:
                delta[d].add(l)
    
    G = tl.zeros([T, T])
    for t in range(T):
        for d in range(1, m + 1):
            if len(delta[d]) > 0:
                for l in delta[d]:
                    if m <= t + l < T:
                        if l == d:
                            G[t, t+d] += W[d-1, r]
                        else:
                            G[t, t+d] += -W[l-1, r] * W[1-d-1, r]
    G = (G + G.T) / 2

    LapG = tl.diag([G[t].sum() for t in range(T)])
    LapG -= G
    for t in range(T):
        D_y = 0
        D_n = 0
        for l in L_bar:
            if m <= t + l < T:
                D_y += -1 if l == 0 else W[l-1, r]
            else:
                D_n += -1 if l == 0 else W[l-1, r]
        LapG[t, t] += (D_n + D_y) * D_y + penalty_weight
    
    return LapG

def trmf(X, mask=None, Ls=None, rank=None, A=None, penalty_weight=1e-3, n_iter_max=10000, tol=1e-5):
    # Graph Regularized Alternating Least Squares (GRALS)
    if Ls is None:
        Ls = [penalty_weight * tl.eye(X.shape[k]) for k in range(X.ndim)] # graph Laplacian and regularization term
    if mask is None:
        mask = X != 0
    if rank is None:
        rank = min(X.shape)
    if A is None:
        A = [0.1 * random.random_tensor((X.shape[i], rank)) for i in range(X.ndim)] # Feature map
    
    rN = [list(reversed(list(set(range(len(A))) ^ {k}))) for k in range(len(A))] # ndim-1 ~ 0 without k
    Proj = [[] for _ in range(len(A))] # Proj[k]: tuple[ik, j], j is the index of W
    for p in tl.tensor(mask.nonzero()).T:
        for k in range(len(A)):
            j = 0
            for i in rN[k]:
                j *= mask.shape[i]
                j += p[i]
            Proj[k].append((p[k], j))

    X0_prev = tl.zeros_like(tl.unfold(X, 0))
    iteration = 0
    while True:
        iteration += 1

        # Update A[k]
        for k in range(len(A)):
            Y = tl.unfold(X * mask, k)
            W = ta.khatri_rao([A[i] for i in rN[k]])
            B = Y @ W # nk x R matrix
            H = A[k]
            A[k] = conjugate_gradient(B, Proj[k], W, H, Ls[k], tol=tol)

        X0_hat = A[0] @ ta.khatri_rao([A[i] for i in reversed(range(1, len(A)))]).T
        local_error = tl.norm(X0_hat - X0_prev) / (tl.norm(X0_hat) + tol)
        X0_prev = X0_hat
        if local_error < tol:
            break
        if n_iter_max is not None:
            if iteration >= n_iter_max:
                break
    
    result = tl.zeros_like(X)
    for r in range(rank):
        P = tl.tensor(1)
        for a in A:
            P = ta.tensor_dot(P, a[:, r])
        result += P
    
    RMSE = tl.norm((tl.unfold(X, 0) - X0_prev) * tl.unfold(mask, 0))

    print('Iterations:', iteration)
    print('RMSE:', RMSE)
    return result, A