import numpy as np
from numpy.linalg import norm
from scipy.sparse import diags, csr_matrix, csc_matrix
from scipy.io import mmread
from time import time
import matplotlib.pyplot as plt

def build_diag(G):
    # Perform the intermediate sum of the pages
    out_degrees = G.sum(axis=0)

    # Invert the values different from zeros
    d_ii = np.divide(1, out_degrees, out=np.zeros_like(out_degrees), where=out_degrees!=0)

    return diags(np.squeeze(np.asarray(d_ii)))

def compute_PR(mat, m=0.15, tol=1e-15, store=True):
    if store:
        n = mat.shape[0]

        # Declare z vector. Default value is 1/n, change where needed.
        z = np.ones(n)/n
        z[np.unique(mat.indices)] = m/n

        # Declare e vector.
        e = np.ones(n)
        
        # Initialize values.
        x_k = np.zeros(n)
        x_k1 = np.ones(n) / n

        # Iterate with stopping criteria.
        while norm(x_k1 - x_k, np.inf) > tol:
            # Update values.
            x_k = x_k1
            x_k1 = (1-m)*mat.dot(x_k) + e*(z.dot(x_k))
            
        return x_k1 / np.sum(x_k1)
    
    else:
        n = mat.shape[0]

        # Initialize arrays that will contain sets and lengths respectively.
        L = []
        n_j = []

        # We will use index pointer in order to get
        indptr = mat.indptr
        for i in range(0, n):
            # Get the indices of all the data corresponding to row i
            # In this case, data represents the links.
            L_i = mat.indices[indptr[i] : indptr[i + 1]]

            # Append the set and the length to the corresponding array.
            L.append(L_i)
            n_j.append(len(L_i))

        # Initialize values.
        x = np.zeros(n)
        xc = np.ones(n) / n

        # Iterate with stopping criteria.
        while norm(x - xc, np.inf) > tol:
            # Code included in the project statement
            xc=x
            x=np.zeros(n)
            for j in range (0,n):
                if (n_j[j]==0):
                    x=x+xc[j]/n
                else:
                    for i in L[j]:
                        x[i]=x[i]+xc[j]/n_j[j]
            x=(1-m)*x+m/n

        return x / np.sum(x)


G = mmread("p2p-Gnutella30.mtx")
D = build_diag(G)
A = csr_matrix(G.dot(D))

# Store experiments
start = time()
x_store = compute_PR(A)
end = time()

print("PR vector using store", x_store)
print("· Execution time:", end - start)

# No store experiments
start = time()
x_storent = compute_PR(csc_matrix(G), store=False)
end = time()

print("PR vector without store", x_storent)
print("· Execution time:", end - start)
print("Norm comparison", norm(x_store - x_storent, 2))

'''Graphs for report'''
'''
times_store = []
times_storent = []
precisions = []
m_values = np.linspace(0.05, 0.95, num=19)

for m in m_values:
    print("Iteration", m)
    start = time()
    x_store = compute_PR(A, m=m)
    end = time()
    times_store.append(end - start)

    start = time()
    x_storent = compute_PR(csc_matrix(G), m=m, store=False)
    end = time()
    times_storent.append(end - start)

    precisions.append(norm(x_store - x_storent, 2))

plt.plot(m_values, times_store, color="green")

plt.xlabel('dampling')
plt.ylabel('seconds')
plt.title('Execution times for store Page Rank')

plt.savefig("lvigo_store.jpg")
plt.close()

plt.plot(m_values, times_storent, color="red")

plt.xlabel('dampling')
plt.ylabel('seconds')
plt.title('Execution times for no-store Page Rank')

plt.savefig("lvigo_storent.jpg")
plt.close()

plt.plot(m_values, precisions, color="blue")

plt.xlabel('dampling')
plt.ylabel('difference')
plt.title('Precision between store and no-store Page Rank implementations')

plt.savefig("lvigo_precision.jpg")
plt.close()
'''