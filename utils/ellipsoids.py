# packages and setup
import cvxpy as cvx
import matplotlib.pyplot as plt
import torch

pastelBlue = "#0072B2"
pastelRed = "#F5615C"
pastelGreen = "#009E73"
pastelPurple = "#8770FE"

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def plot_ellipsoid(sigma, mu, ax, color='b', n_points=100):
    ''' 
    Plot a 3D ellipsoid.
    Args:
        sigma:      [D x D] covariance matrix tensor
        mu:         [D] mean tensor
        ax:         matplotlib axis object
        color:      ellipsoid color
        n_points:   number of points to generate to define ellipsoid surface
    Return:
        ax:         matplotlib axis object with 3D ellipsoid plot
    '''
    eigvals, eigvecs = torch.linalg.eigh(sigma)
    idx = eigvals.argsort(descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # compute radii of the ellipsoid
    radii = torch.sqrt(eigvals)
    # generate points on a unit sphere
    u = torch.linspace(0, 2*torch.pi, n_points)
    v = torch.linspace(0, torch.pi, n_points)
    x = torch.outer(torch.cos(u), torch.sin(v))
    y = torch.outer(torch.sin(u), torch.sin(v))
    z = torch.outer(torch.ones(u.shape), torch.cos(v))
    # transform points to the shape of the ellipsoid
    points = torch.vstack([x.flatten(), y.flatten(), z.flatten()]).T @ \
        torch.diag(radii)
    points = points @ eigvecs.T + mu
    points = points.reshape((n_points, n_points, 3))
    # plot the ellipsoid
    ax.plot_surface(points[:,:,0], points[:,:,1], points[:,:,2], 
                    color=color, alpha = 0.4)
    return ax


def min_enclosing_ellipsoid(X):
    ''' 
    Find the minimum-volume ellipsoid containing a set of points.
    Args:
        X:      [Ns x D] tensor of data
    Returns:
        sigma:  [D x D] covariance matrix tensor
        mu:     [D] mean tensor
    '''
    n, d = X.shape
    A = cvx.Variable((d, d), PSD=True)
    b = cvx.Variable((d))
    constraints = [cvx.norm((X @ A)[i] + b) <= 1 for i in torch.arange(0, n)]
    objective = cvx.Minimize(-cvx.log_det(A))
    problem = cvx.Problem(objective, constraints)
    try:
        problem.solve(solver='SCS', verbose=False)
    except:
        return None, None
    
    A = torch.tensor(A.value, dtype=torch.float32)
    b = torch.tensor(b.value, dtype=torch.float32)
    sigma = torch.linalg.inv(A.T @ A)
    mu = -(sigma @ A.T) @ b

    return sigma, mu


def outside_ellipsoid(X, mu, sigma):
    ''' 
    Compute Mahalanobis distance using Cholesky factorization. 
    Args:
        X:      [Ns x D] tensor of data
        mu:     [D] mean tensor
        sigma:  [D x D] covariance matrix tensor
    Return:
        mahalanobis_dist: the Mahalanobis distance between the data points and
            the ellipsoid
    '''
    chol_factor = torch.linalg.cholesky(sigma)
    chol_inv = torch.linalg.solve(chol_factor, torch.eye(sigma.shape[-1]))
    mahalanobis_dist = \
        torch.norm(chol_inv @ (X - mu)[..., None], dim=1).squeeze() -  1.
    
    return mahalanobis_dist