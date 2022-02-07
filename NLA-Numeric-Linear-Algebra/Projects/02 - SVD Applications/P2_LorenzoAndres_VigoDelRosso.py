from numpy import dot, genfromtxt, zeros, concatenate, transpose, eye, uint8, rint, sqrt, std, reshape, sum, vstack, diag, min, max
from numpy.core.fromnumeric import argmin
from numpy.linalg import pinv, norm, matrix_rank, solve, svd
from imageio import imread, imsave
from tqdm import tqdm
from pandas import read_csv, DataFrame
import scipy.linalg as scla
import matplotlib.pyplot as plt

def ex1_svd(A, b):
    # in x=A^{+}b, A^{+} is the pseudoinverse of A
    pseudo = pinv(A)

    # Compute x=A^{+}b
    return dot(pseudo, b)

def ex1_qr(A, b):
    # Use QR decomposition of A and compute y
    Q, R, P = scla.qr(A, pivoting=True)
    y = dot(Q.T, b)
    
    # We need to separate y in two pieces and select part of R. If A is full-rank, these lines will have no effect.
    r = matrix_rank(A)
    n = A.shape[1]
    y1, y2 = y[:r], y[r:]
    R1 = R[:r, :r]

    # In both cases we will get x through solve_traingular
    # If A is rank deficient (A.shape[1] > r), we need to concatenate zeros, as we are taking v = 0, and solving R1*u=c.
    x = concatenate((scla.solve_triangular(R1, y1)[:r], zeros(n - r)))

    # If pivoting is needed, apply it to x as specified in P^t*x = (u, v)
    return solve(transpose(eye(n)[:,P]), x)


def ex1_datafile(degree):
    data = genfromtxt("dades.txt", delimiter="   ")
    points, b = data[:, 0], data[:, 1]

    # Form A from the points given in dataset, for a given degree
    A = vstack([points ** d for d in range(degree)]).T

    return A, b

def ex1_datafile2(degree):
    # Read matrices and return
    data = genfromtxt('dades_regressio.csv', delimiter=',')
    A, b = data[:, :-1], data[:, -1]
    return A, b

def ex1(generator, solver, degree=0):
    A, b = generator(degree)
    return A, b, solver(A, b)






def ex2(source, rate, target):
    # Load image
    img = imread(source)

    # Compression with the given rate and relative error computation in frobenius norm
    compressed_img = zeros(img.shape)
    error = 0

    # We can do this applying along axis or with a for loop:
    for channel in range(3):
        img_channel = img[:, :, channel]

        # Use SVD decomposition of the channel
        u, s, vh = svd(img_channel)

        # Compression
        compressed_channel = dot(u[:, :rate], dot(diag(s[:rate]), vh[:rate, :]))

        # Assign compressed channel to compressed image
        compressed_img[:, :, channel] = compressed_channel

        # Compute error for this channel and add it to general error
        error += norm(img_channel - compressed_channel, ord=2) / norm(img_channel, ord=2)

    error = 100 * error / 3.0

    # Correct ranges (it's rounded in order to avoid lossy conversion warning)
    compressed_img = rint(255*(compressed_img - min(compressed_img))/(max(compressed_img) - min(compressed_img)))

    # Save result
    imsave(target + str(rate) + "_" + str(error) + ".jpg", compressed_img.astype(uint8))

    return error





def ex3_example():
    # Load the dataset
    X = genfromtxt('example.dat', delimiter=' ')
    return X.T

def ex3_RCsGoff():
    # Load the dataset
    X = genfromtxt('RCsGoff.csv', delimiter=',')

    # Get rid of unnecessary variables
    return X[1:, 1:].T




def covariance_matrix(X):
    # Compute Y and then obtain its SVD decomposition
    n = X.shape[0]
    Y = 1 / sqrt(n - 1) * X.T

    U, S, Vt = svd(Y, full_matrices=False)
    return U, S, Vt


def correlation_matrix(X):
    # Standarize data to compute the correlation matrix through the covariance matrix method
    X = X.T / std(X, axis=1)
    return covariance_matrix(X.T)


def ex3(dataset_loader, matrix_generator):
    # Load dataset
    X = dataset_loader()

    # Correct the mean (as we are imposing zero mean datasets)
    X = X - X.mean(axis = 0)

    # Generate either covariance or correlation matrix
    U, S, Vt = matrix_generator(X)

    # Accumulated total variance
    variance = S**2 / sum(S**2)

    # Standard deviation of each of the principal components (eigenvectors of Cx, found in V)
    standard_dev = std(Vt, axis=0)

    # PCA coordinates
    PCA_coord = dot(Vt, X).T

    return variance, standard_dev, PCA_coord, S


def file_generator(pca, var, target):
    # Load the indices
    df = read_csv("RCsGoff.csv").drop('gene', axis=1)
    indices = df.columns
    column_names = ["PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","PC18","PC19","PC20","Variance"]

    to_file = DataFrame(data=concatenate((pca[:20, :].T, reshape(var, (20,1))), axis=1), index=indices, columns=column_names)
    to_file.index.name = "Sample"
    to_file.to_csv(target)


def scree_plot(S, target):
    plt.plot(range(len(S)), S, color="red")
 
    plt.xlabel('Principal Components')
    plt.ylabel('eigval')
    plt.title("Scree Plot for " + target)
 
    plt.savefig("lvigo_scree_plot_" + target + ".jpg")
    plt.close()

def rule_34(var):
    total_var = sum(var)
    new_var = []
    i = 0
    
    while sum(new_var) < 3*total_var/4:
        new_var.append(var[i])
        i += 1

    return len(new_var)
        
    





## Exercise 1 executions
# datafile
svd_errors = []
for degree in range(2, 10):
    A, b, x_svd = ex1(ex1_datafile, ex1_svd, degree=degree)
    A, b, x_qr = ex1(ex1_datafile, ex1_qr, degree=degree)
    svd_errors.append(norm(A.dot(x_svd)-b))

min_svd_error_pos = argmin(svd_errors)
best_degree = min_svd_error_pos+2
print("Ex 1 (datafile) - Best degree:", best_degree)
A, b, x_svd = ex1(ex1_datafile, ex1_svd, degree=best_degree)
A, b, x_qr = ex1(ex1_datafile, ex1_qr, degree=best_degree)
print("SVD (datafile):", x_svd)
print("Solution norm:", norm(x_svd))
print("Error:", norm(A.dot(x_svd)-b))
print("QR (datafile):", x_qr)
print("Solution norm:", norm(x_qr))
print("Error:", norm(A.dot(x_qr)-b))


# datafile2
A, b, x_svd = ex1(ex1_datafile2, ex1_svd)
A, b, x_qr = ex1(ex1_datafile2, ex1_qr)
print("Ex 1 - SVD (datafile2):", x_svd)
print("Solution norm:", norm(x_svd))
print("Error:", norm(A.dot(x_svd)-b))
print("Ex 1 - QR (datafile2):", x_qr)
print("Solution norm:", norm(x_qr))
print("Error:", norm(A.dot(x_qr)-b))





## Exercise 2 executions
print("Ex 2")
sources = ["face.jpg", "eye.jpg", "landscape.jpg"]
targets = ["face_compressed_", "eye_compressed_", "landscape_compressed_"]
for img in tqdm(range(len(sources)), desc="Images to be compressed"):
    errors = []
    for rate in tqdm(range(5,105,5), desc="Rates of compression for current image"):
        errors.append(ex2(sources[img], rate, targets[img]))

    plt.plot(range(5,105,5), errors, color="red")
 
    plt.xlabel('rate')
    plt.ylabel('error')
    plt.title('Ex 2 - Errors for image ' + sources[img])
 
    plt.savefig("lvigo_ex2_error_" + sources[img])
    plt.close()





## Exercise 3 executions
var, std_dev, pca, S = ex3(ex3_example, covariance_matrix)
print("Ex 3 (example.dat, covariance)")
print("Accumulated total variance in each principal component:", var)
print("Standard deviation of each principal component:", std_dev)
print("PCA coordinates of original dataset:", pca)
scree_plot(S, "example_cov")
print("Kaiser Rule:", len([s for s in S if s > 1]))
print("3/4 rule:", rule_34(var))

var, std_dev, pca, S = ex3(ex3_example, correlation_matrix)
print("Ex 3 (example.dat, correlation)")
print("Accumulated total variance in each principal component:", var)
print("Standard deviation of each principal component:", std_dev)
print("PCA coordinates of original dataset:", pca)
scree_plot(S, "example_corr")
print("Kaiser Rule: ", len([s for s in S if s > 1]))
print("3/4 rule:", rule_34(var))

var, std_dev, pca, S = ex3(ex3_RCsGoff, covariance_matrix)
print("Ex 3 (RCsGoff.csv, covariance)")
print("Accumulated total variance in each principal component:", var)
print("Standard deviation of each principal component:", std_dev)
print("PCA coordinates of original dataset:", pca)
file_generator(pca, var, "rcsgoff_covariance.txt")
scree_plot(S, "rcsgoff_cov")
print("Kaiser Rule: ", len([s for s in S if s > 1]))
print("3/4 rule:", rule_34(var))

var, std_dev, pca, S = ex3(ex3_RCsGoff, correlation_matrix)
print("Ex 3 (RCsGoff.csv, correlation)")
print("Accumulated total variance in each principal component:", var)
print("Standard deviation of each principal component:", std_dev)
print("PCA coordinates of original dataset:", pca)
file_generator(pca, var, "rcsgoff_correlation.txt")
scree_plot(S, "rcsgoff_corr")
print("Kaiser Rule: ", len([s for s in S if s > 1]))
print("3/4 rule:", rule_34(var))
