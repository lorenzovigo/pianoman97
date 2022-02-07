import time
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import ldl
from numpy.linalg import norm, solve, cholesky, norm, cond
from tqdm import tqdm

script_dir = os.path.dirname(__file__)

## Utils
def f(G, g, x):
    return np.matmul(x.T, np.matmul(G, x))/2 + np.matmul(g, x)


# Given code:
def Newton_step(lamb0,dlamb,s0,ds):
    alp=1
    idx_lamb0=np.array(np.where(dlamb<0))
    if idx_lamb0.size>0:
        alp = min(alp,np.min(-lamb0[idx_lamb0]/dlamb[idx_lamb0]))
    
    idx_s0=np.array(np.where(ds<0))
    if idx_s0.size>0:
        alp = min(alp,np.min(-s0[idx_s0]/ds[idx_s0]))
    
    return alp


# Recover dimensions implicitly from the matrices passed
def dims(G, A, C):
    n = G.shape[0]
    p = A.shape[1]
    m = C.shape[1]
    N = n + p + 2*m

    return n, p, m, N


# Read external files
def load_matrix(path, n, m, symmetric = False):
    with open(path, "r") as file:
        matrix = file.readlines()

    ret_matrix = np.zeros((n,m))
    for line in matrix:
        row, column, val = line.strip().split()

        ret_matrix[int(row) - 1, int(column) - 1] = float(val)
        if symmetric == True:
            ret_matrix[int(column) - 1, int(row) - 1] = float(val)
    return ret_matrix


def load_vector(path, n):
    with open(path, "r") as file:
        vector = file.readlines()

    ret_vector = np.zeros(n)
    for line in vector:
        idx, value = line.strip().split()
        ret_vector[int(idx) - 1] = float(value)
    return ret_vector





# Generates the KKT Matrix
def KKT(A, C, G, z, m, n, p):
    # Recover dimensions implicitly from the matrices passed
    n, p, m, N = dims(G, A, C)
    mat = np.zeros((N, N))

    # Input G in the matrix
    mat[:n, :n] = G

    # Input A in the matrix
    mat[:n, n : n+p] = -A
    mat[n : n+p, :n] = -A.T

    # Input C in the matrix
    mat[:n, n+p : n+p+m] = -C
    mat[n+p : n+p+m, :n] = -C.T

    # Right bottom corner
    mat[-2*m:-m, -m:] = np.eye(m) # I in the KKT matrix
    mat[-m:, -2*m:-m] = np.diag(z[-m:]) # S in the KKT matrix, s is the last component of z
    mat[-m:, -m:] = np.diag(z[-2*m:-m]) # /\ in the KKT matrix, lambda is the second to last component in z
    return mat

# Generates the KKT Matrix required in C4, Strategy 1
def KKT_C4S1(A, C, G, z, m, n, p):
    N = n + m + p
    mat = np.zeros((N, N))

    # Input G in the matrix
    mat[:n, :n] = G

    # Input A in the matrix
    mat[:n, n:-m] = -A
    mat[n:-m, :n] = -A.T

    # Input C in the matrix
    mat[:n, -m:] = -C
    mat[-m:, :n] = -C.T

    # Right bottom corner. Retr
    mat[-m:, -m:] = - np.diag(z[-m:] / z[-2*m:-m]) # S / /\. Remember that s is last component in z, and lambda second to last
    return mat

# Generates the KKT Matrix required in C4, Strategy 2
def KKT_C4S2(A, C, G, z, m, n, p):
    # We are computing hatted G, given in the strategy
    # / S * /\. Remember that s is last component in z, and lambda second to last
    return G + np.matmul(C / z[-m:] * z[-2*m:-m], C.T)





# Characterizes the function F that builds the right-hand vector
def F(A, b, C, d, G, g, z, m, n, p):
    # Recover dimensions implicitly from the matrices passed
    n, p, m, N = dims(G, A, C)

    # Retrieve the information kept in z
    x = z[:n]
    gamma = z[n : n+p]
    landa = z[-2*m : -m]
    s = z[-m:]

    # Compute the values given by the equations that define F
    F = np.zeros(N)
    F[:n] = np.matmul(G, x) + g - np.matmul(A, gamma) - np.matmul(C, landa)
    F[n : n+p] = b - np.matmul(A.T, x)
    F[-2 * m:-m] = s + d - np.matmul(C.T, x)
    F[-m:] = s * landa

    return F

# Modification needed for C4,
def F_C4(A, b, C, d, G, g, z, m, n, p):
    N = n + 2 * m + p

    # Retrieve the information kept in z
    x = z[:n]
    # gamma = z[n : n + p] (unused as p = 0, unused)
    landa = z[-2*m : -m]
    s = z[-m:]

    # Compute the values given by the equations that define F
    F = np.zeros(N)
    F[:n] = np.matmul(G, x) + g - np.matmul(C, landa)
    F[n : n + p] = b - np.matmul(A.T, x)
    F[-2*m:-m] = s + d - np.matmul(C.T, x)
    F[-m:] = s * landa

    return F





# Numpy LinAlg system solving
def C2_system_solver(KKT_mat, rhv, C = None, n = None, m = None, p = None, landa = None, s = None):
    return solve(KKT_mat, rhv)

# LDLT system solving
def C4S1_system_solver(KKT_mat, rhv, C = None, n = None, m = None, p = None, landa = None, s = None):
    # Generate the rhv = (r_1, r_2, r_3)
    new_rhv = np.zeros(n + m + p)  
    new_rhv[:n+p] = -rhv[:n+p]
    new_rhv[-m:] = -(rhv[-2*m : -m] - rhv[-m:] / landa)

    # LDLT decomposition
    L, D, perm = ldl(KKT_mat) #L can be used as L and L.T

    # Solving the system (delta_x, delta_lambda)
    y = solve(L, new_rhv)
    z = solve(D, y)
    delta_x_lambda = solve(L.T, z)

    # Constructing whole delta_z
    delta_lambda = delta_x_lambda[-m:]
    delta_s = (-rhv[-m:] - s * delta_lambda) / landa
    return np.concatenate([delta_x_lambda, delta_s])

# Cholesky solving
def C4S2_system_solver(KKK_mat, rhv, C = None, n = None, m = None, p = None, landa = None, s = None):
    # Generate the rhv = (r_1, r_2, r_3)
    r1 = -rhv[:n]
    r2 = -rhv[-(m + m):-m]
    r3 = -rhv[-m:]
    new_rhv = np.zeros(n)  
    new_rhv = rhv[:n] - np.matmul(-C / s, rhv[-m:] - rhv[-2*m : -m] * landa)

    # Cholesky factorization
    L = cholesky(KKK_mat)

    # Solve the systems
    y = solve(L, new_rhv)

    # Constructing whole delta_z
    delta_z = np.zeros(n + 2 * m)
    delta_z[:n] = solve(L.T, y)
    delta_z[-2*m : -m] = (-r3 + landa * r2) / s - np.matmul(C.T, delta_z[:n]) * landa / s
    delta_z[-m:] = -r2 + np.matmul(C.T, delta_z[:n])

    return -delta_z






# Follows the algorithm given in the project
def KKT_solver_step(KKT_mat, rhv, A, b, C, d, G, g, z, system_solver, kkt_gen, rhv_gen, crit_flag=False, eps=1e-16, A_flag=False, sign=1):
    # Retrieve dimensions from available matrices
    n, p, m, N = dims(G, A, C)

    # Retrieve the information kept in z
    # x = z[:n] (unused)
    # gamma = z[n : n + p] (unused)
    landa = z[-2*m : -m]
    s = z[-m:]

    # 1. Predictor substep
    delta_z = system_solver(KKT_mat, rhv, C=C, n=n, m=m, p=p, landa=landa, s=s)
        
    # Retrieve the information available in delta_z
    # delta_x = delta_z[: n] # (unused)
    # delta_gamma = delta_z[n : n + p] # (unused)
    delta_lambda = delta_z[-2 * m : -m]
    delta_s = delta_z[-m:]

    # 2. Step-size correction substep. Use given code.
    alpha = Newton_step(landa, delta_lambda, s, delta_s)

    # 3. Several computations
    mu = np.matmul(s, landa) / m
    mu_tilde = np.matmul(s + alpha * delta_s, landa + alpha * delta_lambda) / m
    sigma = (mu_tilde / mu)**3

    # 4. Corrector substep
    # Modify the last positions of rhv as told in step 4. Take into account
    rhv[-m:] = rhv[-m:] + sign * delta_s * delta_lambda - sign * sigma * mu
    delta_z = system_solver(KKT_mat, rhv, C=C, n=n, m=m, p=p, landa=landa, s=s)

    # Retrieve the information available in delta_z
    # delta_x = delta_z[: n] # (unused)
    # delta_gamma = delta_z[n : n + p] # (unused)
    # delta_lambda = delta_z[-2 * m:-m]
    delta_lambda = delta_z[-2 * m : -m]
    delta_s = delta_z[-m:]
    
    # 5. Carry out step-size correction once again.
    alpha = Newton_step(landa, delta_lambda, s, delta_s)

    # 6. Update z, KKT Matrix and right-hand vector
    z = z + 0.95 * alpha * delta_z
    KKT_mat = kkt_gen(A, C, G, z, m, n, p)
    rhv = sign * rhv_gen(A, b, C, d, G, g, z, m, n, p)

    # Stopping criteria. Retrieve information from rhv.
    r_L = rhv[:n]
    r_A = rhv[n : n + p]
    r_C = rhv[-2*m : -m]
    # If any of these conditions is satisfied, we should stop. In C2, A is always zero, so we should not consider that stopping criterion.
    if A_flag:
        crit_flag = norm(r_L) < eps or norm(r_C) < eps or abs(mu) < eps
    else: 
        crit_flag = norm(r_L) < eps or norm(r_A) < eps or norm(r_C) < eps or abs(mu) < eps

    return KKT_mat, rhv, z, crit_flag





def standard_initial_matrices(n, p):
    # Inital scalar variables
    m = 2*n # Given in the exercise
    N = n + p + 2*m # Dimension of KKT Matrix (given in KKT System desc.)

    # Initial vector/matrix variables
    A = np.zeros((n, p)) # A is in R^{n x p} in the problem description
    b = np.zeros(p) # b is in R^{p} in the problem description.
    C = np.concatenate((np.eye(n), -np.eye(n)), axis = 1) # Given in the exercise
    d = np.array([-10 for i in range(m)]) # d is in R^m in the problem description, given in the exercise
    G = np.eye(n) # Given in the exercise
    g = np.squeeze(np.random.normal(0, 1, size=(n, 1))) # Given by the exercise: Normal Distribution
    z = np.zeros(N) # Given in KKT systems. It includes x (R^n), gamma (R^p) and lambda and s (both R^m), remember N = n + p + 2*m

    return A, b, C, d, G, g, z


def load_matrices(n, p):
    m = 2*n # Given in the exercise
    N = n + int(p) + 2*m # Dimensuon of KKT Matrix (given in KKT System desc.)

    # Initial vector/matrix variables
    A = load_matrix(os.path.join(script_dir, "./" + str(n) + "/A.dad"), n, int(p)) # A is in R^{n x p} in the problem description
    b = load_vector(os.path.join(script_dir, "./" + str(n) + "/b.dad"), int(p)) # b is in R^{p} in the problem description.
    C = load_matrix(os.path.join(script_dir, "./" + str(n) + "/C.dad"), n, m) # Given in the exercise
    d = load_vector(os.path.join(script_dir, "./" + str(n) + "/d.dad"), m) # d is in R^m in the problem description, given in the exercise
    G = load_matrix(os.path.join(script_dir, "./" + str(n) + "/G_uppercase.dad"), n, n, True) # Given in the exercise
    g = load_vector(os.path.join(script_dir, "./" + str(n) + "/g_lowercase.dad"), n) # Given by the exercise: Normal Distribution
    z = np.zeros(N) # Given in KKT systems. It includes x (R^n), gamma (R^p) and lambda and s (both R^m), remember N = n + p + 2*m

    return A, b, C, d, G, g, z


def problem_starter(A, b, C, d, G, g, z, n, p, system_solver, kkt_gen, rhv_gen, sign=1, diff=True):
    # Inital scalar variables
    m = 2*n # Given in the exercise

    # Given:
    # x_0 in z initially will be (0, ..., 0)
    # And both s_0 and lambda_0 will be (1, ..., 1) in z
    z[-2*m:] = 1
    KKT_mat = kkt_gen(A, C, G, z, m, n, p) # generate needed KKT matrix
    rhv = sign * rhv_gen(A, b, C, d, G, g, z, m, n, p) # right hand vector

    num_iter = 100 # max number iterations given by the problem
    i = 0 # current iteration
    flag = False # if one of the stopping criteria is fulfilled

    while i < num_iter and not flag:
        KKT_mat, rhv, z, flag = KKT_solver_step(KKT_mat, rhv, A, b, C, d, G, g, z, system_solver, kkt_gen, rhv_gen, crit_flag=flag, A_flag=diff, sign=sign)
        i += 1

    if diff:
        # The solution should be x = -g. x is stored in z[x] and g is not modified.
        # If solution is correct, something close to 0 must be returned.
        # We will return solution, precision, number of iterations and condition number
        sol = z[:n] + g
        precision = norm(sol)
        cond_numb = cond(KKT_mat)
        return sol, precision, i, cond_numb
    else: 
        # We will compute the value of the function with the found minimum and check the result with the given one in the project.
        if n == 100:
            value = 1.15907181e4
        else:
            value = 1.08751157e6
        return f(G, g, z[:n])





## Exercises
def c1(lamb0, dlamb, s0, ds): # Just using the code given in the project tasks
    return Newton_step(lamb0, dlamb, s0, ds)


def c2(n):
    # Inital scalar variables
    p = 0 # A and b will be "ignored" as A = 0.

    # Initial vector/matrix variables
    A, b, C, d, G, g, z = standard_initial_matrices(n, p)

    return problem_starter(A, b, C, d, G, g, z, n, p, C2_system_solver, KKT, F, sign=-1)
sol, precision, i, cond_numb = c2(3)
print("C2 - Solution (should return 0):", sol)


def c3():
    ns = [i for i in range(2, 100)]
    times = []
    # For each dimension
    for n in tqdm(ns, desc="C3 - Executing C2 for several dimensions"):
        start_time = time.time()

        try:
            # Run the algorithm and add the time if it works
            c2(n)
            times.append(time.time() - start_time)
        except:
            # Add 0 if there is any issue (first dimensions)
            times.append(0)

    # Plot and save to file
    plt.plot(ns, times, color="blue")
 
    plt.xlabel('dimension')
    plt.ylabel('seconds')
    plt.title('C3 - C2 execution times per dimension')
 
    plt.savefig("lvigo_c3.png")
    print("C3 - Solution can be checked in external file: c3.png")     
    return None
c3()


def c4_strat1(n):
    # Inital scalar variables
    p = 0 # A and b will be "ignored" as A = 0.

    # Initial vector/matrix variables
    A, b, C, d, G, g, z = standard_initial_matrices(n, p)

    return problem_starter(A, b, C, d, G, g, z, n, p, C4S1_system_solver, KKT_C4S1, F_C4)
sol, precision, i, cond_numb = c4_strat1(3)
print("C4, Strat 1 - Solution (should return 0):", sol)


def c4_strat2(n):
    # Inital scalar variables
    p = 0 # A and b will be "ignored" as A = 0.

    # Initial vector/matrix variables
    A, b, C, d, G, g, z = standard_initial_matrices(n, p)

    return problem_starter(A, b, C, d, G, g, z, n, p, C4S2_system_solver, KKT_C4S2, F_C4)
sol, precision, i, cond_numb = c4_strat2(3)
print("C4, Strat 2 - Solution (should return 0):", sol)


# General case, using the implementation in c2. n should be 100 or 1000 (problems 1 and 2, respectively)
def c5(n):
    # Inital scalar variables
    p = int(n / 2) #

    # Initial vector/matrix variables
    A, b, C, d, G, g, z = load_matrices(n, p)

    return problem_starter(A, b, C, d, G, g, z, n, p, C2_system_solver, KKT, F, sign=-1, diff=False)
print("C5 - Solution, Problem 1 (should return 11590.7181):", c5(100))
print("C5 - Solution, Problem 2 (should return 1087511.57):", c5(1000))


# General case, using the implementation in c4_strat1. n should be 100 or 1000 (problems 1 and 2, respectively)
def c6(n):
    # Inital scalar variables
    p = int(n / 2) #

    # Initial vector/matrix variables
    A, b, C, d, G, g, z = load_matrices(n, p)

    return problem_starter(A, b, C, d, G, g, z, n, p, C4S1_system_solver, KKT_C4S1, F_C4, diff=False)
print("C6 - Solution, Problem 1 (should return 11590.7181):", c6(100))
print("C6 - Solution, Problem 2 (should return 1087511.57):", c6(1000))



def exercise_analysis(exercise, exercise_name):
    ns = [i for i in range(1, 100)]
    times = []
    precisions = []
    iters = []
    cond_numbs = []

    # For each dimension
    for n in tqdm(ns, desc="Exercise Analysis - Executing " + exercise_name + " for several dimensions"):
        start_time = time.time()

        try:
            # Run the algorithm and add the time if it works
            sol, precision, i, cond_numb = exercise(n)
            
            times.append(time.time() - start_time)
            precisions.append(precision)
            iters.append(i)
            cond_numbs.append(cond_numb)
        except:
            # Add 0 if there is any issue (first dimensions)
            times.append(0)
            precisions.append(0)
            iters.append(0)
            cond_numbs.append(0)


    # Plot and save to file 
    plt.plot(ns, times, color="blue")
 
    plt.xlabel('dimension')
    plt.ylabel('seconds')
    plt.title('Analysis - ' + exercise_name + ' execution times per dimension')
 
    filename = "lvigo_analysis_" + exercise_name + "_time.png"
    plt.savefig(filename)
    print("Exercise Analysis- Solution can be checked in external file: " + filename)     
    plt.clf()


    plt.plot(ns, precisions, color="green")
 
    plt.xlabel('dimension')
    plt.ylabel('precision')
    plt.title('Analysis - ' + exercise_name + ' execution precisions per dimension')
 
    filename = "lvigo_analysis_" + exercise_name + "_precisions.png"
    plt.savefig(filename)
    print("Exercise Analysis- Solution can be checked in external file: " + filename)     
    plt.clf()



    plt.plot(ns, iters, color="red")
 
    plt.xlabel('dimension')
    plt.ylabel('iterations')
    plt.title('Analysis - ' + exercise_name + ' execution iterations per dimension')
 
    filename = "lvigo_analysis_" + exercise_name + "_iterations.png"
    plt.savefig(filename)
    print("Exercise Analysis- Solution can be checked in external file: " + filename)   
    plt.clf()



    plt.plot(ns, cond_numbs, color="yellow")
 
    plt.xlabel('dimension')
    plt.ylabel('condition number')
    plt.title('Analysis - ' + exercise_name + ' execution condition numbers per dimension')
 
    filename = "lvigo_analysis_" + exercise_name + "_cond.png"
    plt.savefig(filename)
    print("Exercise Analysis- Solution can be checked in external file: " + filename)  

    return None
#exercise_analysis(c4_strat1, "C4S1")
