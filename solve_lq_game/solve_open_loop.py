import numpy as np
from collections import deque

def solve_open_loop(As, Bs, Qs, ls, Rs, rs, x0, calc_deriv_cost):
    horizon = len(As) - 1
    num_players = len(Bs)

    x_dim = As[0].shape[0]
    u_dims = [Bis[0].shape[1] for Bis in Bs]

    lambdas = deque()
    ms = [deque([lis[-1]]) for lis in ls]
    Ms = [deque([Qis[-1]]) for Qis in Qs]

    us = [deque() for ii in range(num_players)]
    xs = deque([x0])
    ps = [deque() for ii in range(num_players)]

    for k in range(horizon, -1, -1):
        A = As[k]
        B = [Bis[k] for Bis in Bs]
        Q = [Qis[k] for Qis in Qs]
        l = [lis[k] for lis in ls]
        R = [[Rijs[k] for Rijs in Ris] for Ris in Rs]
        r = [ris[k] for ris in rs]

        m = [mis[0] for mis in ms]
        M = [Mis[0] for Mis in Ms]

        lda = np.sum(
            [
                B[jj] @ np.linalg.inv(R[jj][jj]) @ B[jj].T @ M
                for jj in range(num_players)
            ]
        )
        lambdas.appendleft(lda + np.eye(lda.shape[0]))

        lda_inv = np.linalg.inv(lambdas[0])

        for ii in range(num_players):
            ms[ii].appendleft(
                A.T @ (
                    m[ii] - M[ii] @ lda_inv * np.sum(
                        [
                            B[jj] @ np.linalg.inv(R[jj][jj]) @ (B[jj].T @ m[ii] + r[jj][jj:jj+1])
                            for jj in range(num_players)
                        ]
                    )
                ) + l
            )

            Ms[ii].appendleft(
                Q[ii] + A.T @ M[ii] @ lda_inv @ A
            )
    
    for k in range(horizon):
        A = As[k]
        B = [Bis[k] for Bis in Bs]
        Q = [Qis[k] for Qis in Qs]
        l = [lis[k] for lis in ls]
        R = [[Rijs[k] for Rijs in Ris] for Ris in Rs]
        r = [ris[k] for ris in rs]

        m = [mis[k+1] for mis in ms]
        M = [Mis[k+1] for Mis in Ms]
        lda_inv = np.linalg.inv(lambdas[k])

        for ii in range(num_players):
            xs.append(
                lda_inv @ (
                    A @ xs[-1] - np.sum(
                        [
                            B[jj] @ np.linalg.inv(R[jj][jj]) @ (B[jj].T @ m[ii] + r[jj][jj:jj+1])
                            for jj in range(num_players)
                        ]
                    )
                )
            )

            us[ii].append(
                -np.linalg.inv(R[ii][ii]) @ (B[ii].T @ (M[ii] @ xs[-1] + m[ii]) + r[ii][ii:ii+1])
            )

            ps[ii].append(
                A.T @ (M[ii] @ xs[-1] + m[ii])
            )
    
    return us, xs, ps

