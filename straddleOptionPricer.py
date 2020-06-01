import numpy as np
from scipy.sparse import spdiags
from scipy.linalg import lu_factor, lu_solve
from scipy.stats import norm
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from math import exp, floor, sqrt

#######################################################################################################
# Description: This script is part of Assignment 2 for CS 476: Numerical Computation for Financial
# Modeling (W2020). In this question, one has to implement American option pricer using both constant 
# and variable timestepping method given underlying asset price vector.
# 
# Output: Tables are generated in Discussion.txt, with various number of underlying asset 
# price nodes and simulation steps.
# 
# Side effect: Delta-Price plots are generated for both constant/variable timestepping, using the
# finest grid (last row in output table)
#######################################################################################################

alpha, r, T, K, S_0 = 0.8, 0.02, 1, 10, 10
num_rows = 5
Large = 10**6
tol = 1/Large
CONST_DELTA = "Constant Timestepping"
VAR_DELTA = "Variable Timestepping"

og_S = np.concatenate([np.arange(0, 0.5*K, 0.1*K),
                    np.arange(0.45*K, 0.81*K, 0.05*K),
                    np.arange(0.82*K, 0.91*K, 0.02*K),
                    np.arange(0.91*K, 1.10*K, 0.01*K),
                    np.arange(1.12*K, 1.21*K, 0.02*K),
                    np.arange(1.25*K, 1.61*K, 0.05*K),
                    np.arange(1.7*K, 2.01*K, 0.1*K),
                    np.array([2.2*K, 2.4*K, 2.8*K, 3.6*K, 5*K, 7.5*K, 10*K])])
S = np.copy(og_S)
S = np.insert(S, 4, sqrt(K)) # kink in option price, shouldn't affect price grid too much

# Generate number of steps/nodes
num_steps = [25*2**i for i in range(num_rows)]
num_nodes = [2**i*len(S)-2**i+1 for i in range(num_rows)]

def strad_quad_payoff(S, K):
    return np.array([max(K-(s**2), (s**2)-K) for s in S])

def sigma(S):
    global alpha
    return alpha/sqrt(S)

def generate_postive_coeff(S):
    global sigma, r
    res = [(0, 0)]  # a_0, b_0 is invalid, insert as filler
    
    # prioritize central difference first, then forward, then backward
    for i in range(1, len(S)-1):
        a_central = (sigma(S[i])**2 * S[i]**2) / ((S[i] - S[i-1]) * (S[i+1] - S[i-1])) \
            - (r * S[i]) / (S[i+1] - S[i-1])
        b_central = (sigma(S[i])**2 * S[i]**2) / ((S[i+1] - S[i]) * (S[i+1] - S[i-1])) \
            + (r * S[i]) / (S[i+1] - S[i-1])
        if a_central >= 0 and b_central >= 0:
            res.append((a_central, b_central))
        else:
            a_forward = (sigma(S[i])**2 * S[i]**2) / \
                ((S[i] - S[i-1]) * (S[i+1] - S[i-1]))
            b_forward = (sigma(S[i])**2 * S[i]**2) / ((S[i+1] - S[i]) * (S[i+1] - S[i-1])) \
                + (r * S[i]) / (S[i+1] - S[i])
            if a_forward >= 0 and b_forward >= 0:
                res.append((a_forward, b_forward))
            else:
                a_backward = (sigma(S[i])**2 * S[i]**2) / ((S[i] - S[i-1]) * (S[i+1] - S[i-1])) \
                    - (r * S[i]) / (S[i] - S[i-1])
                b_backward = (sigma(S[i])**2 * S[i]**2) / \
                    ((S[i+1] - S[i]) * (S[i+1] - S[i-1]))
            res.append((a_backward, b_backward))
    
    res.append((0, 0)) # filler
    return res

def generate_S(S):
    ans = [S[0]]
    # Using initial S arrays to 
    # generate new S nodes (avg of the 2 closest nodes)
    for i in range(1, len(S)):
        ans.append(S[i-1] + (S[i]-S[i-1])/2)
        ans.append(S[i])
    return ans

def generate_M(S, deltaT):
    global r
    coeffs = generate_postive_coeff(S)
    m = len(S)
    # 2 O's at the beginning/end as filler for upper/lower diagonal
    data = np.array(
        [
            [-deltaT*a_i for a_i, _ in coeffs[1:-1]] + 2*[0.0],  # lower diag
            [deltaT*(a_i + b_i + r)
             for a_i, b_i in coeffs[:-1]] + [0.0],  # main diag
            2*[0.0] + [-deltaT*b_i for _, b_i in coeffs[1:-1]]  # upper diag
        ])
    diags = np.array([-1, 0, 1])
    return spdiags(data, diags, m, m).toarray()

def generate_P(V, V_star):
    global Large
    m = len(V)
    data = np.array([[Large if V[i] < V_star[i] else 0 for i in range(m)]])  # main diag
    diags = np.array([0])
    return spdiags(data, diags, m, m).toarray()

optvals = {}
optvals.setdefault(CONST_DELTA, [])
optvals.setdefault(VAR_DELTA, [])
V_final = {}

# Constant timestepping
for nnodes, nsteps in zip(num_nodes, num_steps):
    deltT = T/nsteps
    while len(S) != nnodes:
        S = generate_S(S)
    V_n = V_star = strad_quad_payoff(S, K)
    V_n_k = V_n
    # we use this to retrieve V(S=10,t=0)
    # set atol = 0 and rtol = 0 to find the exact node S=10
    strk_idx = np.where(np.isclose(S, 10., rtol=10**(-10), atol=0))[0][0]
    M = generate_M(S, deltT)
    
    # C-N Rannacher
    for i in range(nsteps):
        while True:
            P = generate_P(V_n_k, V_star)
            V_n_k_1 = lu_solve(lu_factor(np.identity(len(S)) + M + P), V_n + np.dot(P, V_star)) if i < 2 else\
                    lu_solve(lu_factor(np.identity(len(S)) + 1/2*M + P),
                    np.dot(np.identity(len(S)) - 1/2*M, V_n) + np.dot(P, V_star))
            # check whether or not we should stop calculating V^{n+1}
            if max([abs(V_n_k_1[i] - V_n_k[i])/max(1, abs(V_n_k_1[i])) for i in range(len(S))]) < tol: break
            V_n_k = V_n_k_1
        V_n = V_n_k = V_n_k_1
    
    # Save results delta-asset plot
    if nsteps == num_steps[-1]:
        V_final[CONST_DELTA] = V_n_k_1

    # Add V(S=10,t=0) to generate table
    optvals[CONST_DELTA].append(V_n_k_1[strk_idx])

# Reset S
S = np.copy(og_S)
S = np.insert(S, 4, sqrt(K)) # kink in option price, shouldn't affect price grid too much
# Variable timestepping
deltTs = [T/(25*4**i) for i in range(num_rows)]
dnorms = [0.1/(2**i) for i in range(num_rows)]
num_steps_VAR = []
for deltT_start, dnorm, nnodes in zip(deltTs, dnorms, num_nodes):
    deltT = deltT_start
    while len(S) != nnodes:
        S = generate_S(S)
    V_n = V_n_k = V_star = strad_quad_payoff(S, K)
    # we use this to retrieve V(S=10,t=0)
    # set atol = 0 and rtol = 0 to find the exact node S=10
    strk_idx = np.where(np.isclose(S, 10., rtol=10**(-10), atol=0))[0][0]
    nsteps = 0
    t = 0

    # C-N Rannacher
    while t < T:
        M = generate_M(S, deltT)
        while True:
            P = generate_P(V_n_k, V_star)
            V_n_k_1 = lu_solve(lu_factor(np.identity(len(S)) + M + P), V_n + np.dot(P, V_star)) if nsteps < 2 else\
                    lu_solve(lu_factor(np.identity(len(S)) + 1/2*M + P),
                    np.dot(np.identity(len(S)) - 1/2*M, V_n) + np.dot(P, V_star))
            # check whether or not we should stop calculating V^{n+1}
            if max([abs(V_n_k_1[i] - V_n_k[i])/max(1, abs(V_n_k_1[i])) for i in range(len(S))]) < tol: break
            V_n_k = V_n_k_1
        nsteps += 1
        t += deltT
        relChanges = [abs(V_n_k_1[i] - V_n[i])/max(1, abs(V_n_k_1[i]), abs(V_n[i])) for i in range(len(S))]
        maxRelChange = max(relChanges)
        deltT = min((dnorm/maxRelChange)*deltT, T-t)
        V_n = V_n_k = V_n_k_1
    num_steps_VAR.append(nsteps)
    
    # Save results for delta-asset plot
    if nnodes == num_nodes[-1]:
        V_final[VAR_DELTA] = V_n_k_1

    optvals[VAR_DELTA].append(V_n_k_1[strk_idx])

change = {}
change.setdefault(CONST_DELTA, [""])
change.setdefault(VAR_DELTA, [""])
for mode, V in optvals.items():
  for i in range(1,len(V)):
    change[mode].append(abs(V[i]-V[i-1]))

ratio = {}
ratio.setdefault(CONST_DELTA, ["", ""])
ratio.setdefault(VAR_DELTA, ["", ""])
for mode, deltas in change.items():
  for i in range(2, len(deltas)):
    ratio[mode].append(deltas[i-1]/deltas[i])

# Generate convergence table for constant + variable timestepping
table = PrettyTable(["Nodes", "Num Steps", "Value", "Change", "Ratio"])
table.title = CONST_DELTA
for i in range(len(num_steps)):
    table.add_row([num_nodes[i], num_steps[i], optvals[CONST_DELTA][i],
                change[CONST_DELTA][i], ratio[CONST_DELTA][i]])
print(table)
table.clear_rows()

table.title = VAR_DELTA
for i in range(len(num_nodes)):
    table.add_row([num_nodes[i], num_steps_VAR[i], optvals[VAR_DELTA][i],
                change[VAR_DELTA][i], ratio[VAR_DELTA][i]])
print(table)
table.clear_rows()

# Show plots of the delta for range S = [5, 15], for your solution on the finest grid
# for CN-Rannacher timestepping.
for step_type, V in V_final.items():
    SDs = S[np.where(np.isclose(S, 5))[0][0]:np.where(np.isclose(S, 15))[0][0]+2]
    VDs = V[np.where(np.isclose(S, 5))[0][0]:np.where(np.isclose(S, 15))[0][0]+2]
    Ds = [(VDs[i]-VDs[i-1])/(SDs[i]-SDs[i-1]) for i in range(1,len(SDs))]
    plt.title(f'Option Delta vs. Asset Price with CN-Rannacher using {step_type}')
    plt.xlabel("Asset Price ($)")
    plt.ylabel("Option Delta")
    plt.plot(SDs[:-1], Ds)
    plt.show()
