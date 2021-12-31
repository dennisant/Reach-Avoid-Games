"""
BSD 3-Clause License

Copyright (c) 2019, HJ Reachability Group
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author(s): David Fridovich-Keil ( dfk@eecs.berkeley.edu )
"""
################################################################################
#
# Function to solve time-varying finite horizon LQ game.
# Please refer to Corollary 6.1 on pp. 279 of Dynamic Noncooperative Game Theory
# (second edition) by Tamer Basar and Geert Jan Olsder.
#
################################################################################

import numpy as np
from collections import deque

def solve_lq_game(As, Bs, Qs, ls, Rs, rs, calc_deriv_cost):
    """
    Solve a time-varying, finite horizon LQ game (finds closed-loop Nash
    feedback strategies for both players).
    Assumes that dynamics are given by
           ``` dx_{k+1} = A_k dx_k + \sum_i Bs[i]_k du[i]_k ```

    NOTE: Bs, Qs, ls, R1s, R2s are all lists of lists of matrices.
    NOTE: all indices of inner lists correspond to the "current time" k except
    for those of Q1 and Q2, which correspond to the "next time" k+1. That is,
    the kth entry of Q1s is the state cost corresponding to time step k+1. This
    makes sense because there is no point assigning any state cost to the
    initial state x_0.
    Returns Ps, alphas

    :param As: A matrices
    :type As: [np.array]
    :param Bs: list of list of B matrices (1 list for each player)
    :type Bs: [[np.array]]
    :param Qs: list of list of quadratic state costs (1 list for each player)
    :type Qs: [[np.array]]
    :param ls: list of list of linear state costs (1 list for each player)
    :type ls: [[np.array]]
    :param Rs: list of list of lists of quadratic control costs. Each player
        (outer list) has a list of lists of costs due to other players' control,
        i.e. Rs[i][j][k] is R_{ij}(k) from Basar and Olsder.
    :type Rs: [[[np.array]]]
    :return: gain matrices P[i]_k and feedforward term alpha[i]_k for each
        player, i.e. (u[i]_k = uref[i]_k -P[i]_k dx_k - alpha[i]_k).
        Returned as a list of lists P, alpha (1 list for each player)
    :rtype: [[np.array]], [[np.array]]
    """
    # Unpack horizon and number of players.
    horizon = len(As) - 1
    num_players = len(Bs)
    print("horizon is: ", horizon)

    # Cache dimensions of state and controls for each player.
    x_dim = As[0].shape[0]
    u_dims = [Bis[0].shape[1] for Bis in Bs]

    # Note: notation and variable naming closely follows that introduced in
    # the "Preliminary Notation for Corollary 6.1" section, which may be found
    # on pp. 279 of Basar and Olsder.
    # NOTE: we will assume that `c` from Basar and Olsder is always `0`.

    # Recursive computation of all intermediate and final variables.
    # Use deques for efficient prepending.
    Zs = [deque([Qis[-1]]) for Qis in Qs]
    zetas = [deque([lis[-1]]) for lis in ls]
    Fs = deque()
    Ps = [deque() for ii in range(num_players)]
    betas = deque()
    alphas = [deque() for ii in range(num_players)]
    for k in range(horizon, -1, -1):
        # Unpack all relevant variables.
        A = As[k]
        B = [Bis[k] for Bis in Bs]
        Q = [Qis[k] for Qis in Qs]
        l = [lis[k] for lis in ls]
        R = [[Rijs[k] for Rijs in Ris] for Ris in Rs]
        #r = [[rijs[k] for rijs in ris] for ris in rs]
        r = [ris[k] for ris in rs]
        #print("Q is: ", Q)

        Z = [Zis[0] for Zis in Zs]
        zeta = [zetais[0] for zetais in zetas]
        #print("horizon is: ", horizon)
        #print("k is: ", k)
        #print("Z[0] is: ", Z[0])

        # Compute Ps given previously computed Zs.
        # Refer to equation 6.17a in Basar and Olsder.
        # This will involve solving a system of matrix linear equations of the
        # form [S1s; S2s; ...] * [P1; P2; ...] = [Y1; Y2; ...].
        S_rows = []
        for ii in range(num_players):
            Sis = []
            for jj in range(num_players):
                if jj == ii:
                    Sis.append(R[ii][ii] + B[ii].T @ Z[ii] @ B[ii])
                else:
                    Sis.append(B[ii].T @ Z[ii] @ B[jj])

            S_rows.append(np.concatenate(Sis, axis=1))

        S = np.concatenate(S_rows, axis=0)
        Y = np.concatenate(
            [B[ii].T @ Z[ii] @ A for ii in range(num_players)], axis=0)

        P, _, _, _ = np.linalg.lstsq(a=S, b=Y, rcond=None)
        P_split = np.split(P, np.cumsum(u_dims[:-1]), axis=0)
        for ii in range(num_players):
            Ps[ii].appendleft(P_split[ii])

        # Compute F_k = A_k - B1_k P1_k - B2_k P2_k.
        # This is eq. 6.17c from Basar and Olsder.
        F = A - sum([B[ii] @ P_split[ii] for ii in range(num_players)])
        Fs.appendleft(F)

        # Update Zs to be the next step earlier in time (now they
        # correspond to time k+1).
        # if calc_deriv_cost == True:
        #     for ii in range(num_players):
        #         Z[ii].appendleft(Q[ii])
        # else:
        #     for ii in range(num_players):
        #         #print("Z[ii] is: ", Z[ii])
        #         Zs[ii].appendleft(F.T @ Z[ii] @ F + Q[ii] + sum(
        #             [P_split[jj].T @ R[ii][jj] @ P_split[jj]
        #              for jj in range(num_players)]))
                
        if calc_deriv_cost[k] == 'True':
            Z[ii] = Q[ii]
            Zs[ii].appendleft(F.T @ Z[ii] @ F + Q[ii] + sum(
                [P_split[jj].T @ R[ii][jj] @ P_split[jj]
                  for jj in range(num_players)]))
            print("Q[ii] in solve_lq_game is: ", Q[ii])
        
        else:
            Zs[ii].appendleft(F.T @ Z[ii] @ F + Q[ii] + sum(
                [P_split[jj].T @ R[ii][jj] @ P_split[jj]
                  for jj in range(num_players)]))

        # Compute alphas using previously computed zetas.
        # Refer to equation 6.17d in Basar and Olsder.
        # This will involve solving a system of linear matrix equations of the
        # form [S1s; S2s; ...] * [alpha1; alpha2; ..] = [Y1; Y2; ...].
        # In fact, this is the same S matrix as before (just a different Y).
        Y = np.concatenate(
            [B[ii].T @ zeta[ii] for ii in range(num_players)], axis=0)

        alpha, _, _, _ = np.linalg.lstsq(a=S, b=Y, rcond=None)
        alpha_split = np.split(alpha, np.cumsum(u_dims[:-1]), axis=0)
        for ii in range(num_players):
            alphas[ii].appendleft(alpha_split[ii])

        # Compute beta_k = -B1_k alpha1 - B2_k alpha2_k.
        # This is eq. 6.17f in Basar and Olsder (with `c = 0`).
        # NOTE: in eq. 6.17f, the Bis are transposed, but I believe that this
        # is a typo.
        beta = -sum([B[ii] @ alpha_split[ii] for ii in range(num_players)])
        betas.appendleft(beta)

        # Update zetas to be the next step earlier in time (now they
        # correspond to time k+1). This is Remark 6.3 in Basar and Olsder.
        if calc_deriv_cost == True:
            for ii in range(num_players):
                zetas[ii].appendleft(l[ii])
        else:
            for ii in range(num_players):
                # print("r is: ", r)
                # print("r[0] is: ", r[0])
                # print("r[0].shape is: ", r[0].shape)
                # print("r[0][1:2] is: ", r[0][1:2])
                # print("r[0][1:2].shape is: ", r[0][1:2].shape)
                # print("P_split[0] is: ", P_split[0])
                # print("r[0][0][0] is: ", r[ii][0][jj])
                # print("Turning r[0][0][0] into np.array is: ", np.array(r[ii][0][jj]))
                # print(r[ii][0][0].shape)
                # print(P_split[0].shape)
                # print("R is: ", R)
                #print("P_split[jj].T @ R[ii][jj].shape is: ", (P_split[jj].T @ R[ii][jj]).shape)
                # print("r[0][1] is: ", r[0][1])
                # print("r[1] is:")
                # zetas[ii].appendleft(F.T @ (zeta[ii] + Z[ii] @ beta) + l[ii] + sum(
                #     [P_split[jj].T @ R[ii][jj] @ alpha_split[jj] - P_split[jj].T @ r[ii][jj:jj+1].T
                #      for jj in range(num_players)]))
                
                if calc_deriv_cost[k] == 'True':
                    zeta[ii] = l[ii]
                    zetas[ii].appendleft(F.T @ (zeta[ii] + Z[ii] @ beta) + l[ii] + sum(
                        [P_split[jj].T @ R[ii][jj] @ alpha_split[jj] - P_split[jj].T @ r[ii][jj:jj+1].T
                          for jj in range(num_players)]))
                    print("l[ii] in solve_lq_game is: ", l[ii])
                    
                else:
                    zetas[ii].appendleft(F.T @ (zeta[ii] + Z[ii] @ beta) + l[ii] + sum(
                        [P_split[jj].T @ R[ii][jj] @ alpha_split[jj] - P_split[jj].T @ r[ii][jj:jj+1].T
                          for jj in range(num_players)]))



    # - P_split[jj].T @ r[ii][0][jj].T
    #print("P is: ", Ps)
    #print(" ")
    #print(" ")
    #print("alpha is: ", alphas)
    #print("length of zeta[0] is: ", len(zeta[0]))
    
    # Return P1s, P2s, alpha1s, alpha2s
    return [list(Pis) for Pis in Ps], [list(alphais) for alphais in alphas]
