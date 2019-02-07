import numpy as np
from scipy.optimize import minimize
import cmocean
import cmocean.cm as cmo
import matplotlib.pyplot as plt
import logging
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams["figure.dpi"] = 100


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%y-%m-%d %H:%M:%S'
)


# Compute the friction coefficient beta^2
def beta_squared(p):
    res = p[0]
    for k in range(1, N//2+1):
        res += p[2*k-1]*np.sin(2*np.pi*k*x_grid/length)
        res += p[2*k]*np.cos(2*np.pi*k*x_grid/length)
    return res


# Define indexing functions
def node_index(i, j):
    return j*(ny-1)+i


def u_index(i, j):
    n = 3*(nx-1)*(ny-1)
    return (3*node_index(i, j)) % n


def v_index(i, j):
    n = 3*(nx-1)*(ny-1)
    return (3*node_index(i, j) + 1) % n


def p_index(i, j):
    n = 3*(nx-1)*(ny-1)
    return (3*node_index(i, j) + 2) % n


# Print info of matrix construction
def print_info(node, index):
    if verbose:
        if index % 3 == 0:
            print("node {:<2d} -> index {}: Adding x-Stokes equation".format(node, 3*node))
        if index % 3 == 1:
            print("node {:<2d} -> index {}: Adding y-Stokes equation".format(node, 3*node+1))
        if index % 3 == 2:
            print("node {:<2d} -> index {}: Adding continuity equation".format(node, 3*node+2))


# Construct coefficient matrix A and right-hand side vector b
def setup(beta2, mu):
    # Define a helper function
    def mu_value(i, j):
        return mu[i % (ny-1), j % (nx-1)]

    # Construct A and b
    A = np.zeros((3*(nx-1)*(ny-1), 3*(nx-1)*(ny-1)))
    b = np.zeros(3*(nx-1)*(ny-1))
    for j in range(nx-1):
        for i in range(ny-1):
            k = node_index(i, j)
            if i == 0:
                # Obtain viscosity values
                mu1x = mu_value(i, j)
                mu2x = mu_value(i, j-1)
                mu4x = (mu_value(i, j-1)+mu_value(i, j)+mu_value(i+1, j)+mu_value(i+1, j-1))/4
                mu2y = mu_value(i, j)

                # Add x-Stokes equations to surface nodes
                print_info(k, 3*k)
                A[u_index(i, j), u_index(i, j+1)] = 2*mu1x/hx**2
                A[u_index(i, j), u_index(i, j)] -= 2*mu1x/hx**2
                A[u_index(i, j), u_index(i, j)] -= 2*mu2x/hx**2
                A[u_index(i, j), u_index(i, j-1)] = 2*mu2x/hx**2
                A[u_index(i, j), p_index(i, j)] = -1/hx
                A[u_index(i, j), p_index(i, j-1)] = 1/hx
                A[u_index(i, j), u_index(i, j)] -= 1*mu4x/hy**2
                A[u_index(i, j), u_index(i+1, j)] = 1*mu4x/hy**2
                A[u_index(i, j), v_index(i+1, j)] = -1*mu4x/(hx*hy)
                A[u_index(i, j), v_index(i+1, j-1)] = 1*mu4x/(hx*hy)
                b[u_index(i, j)] = -rho*g*np.sin(alpha)

                # Add y-Stokes equations to surface nodes
                print_info(k, 3*k+1)
                A[v_index(i, j), v_index(i, j)] -= 2*mu2y/hy**2
                A[v_index(i, j), v_index(i+1, j)] = 2*mu2y/hy**2
                A[v_index(i, j), p_index(i, j)] = 1/hy
                b[v_index(i, j)] = rho*g*np.cos(alpha)

                # Add continuity equations to surface nodes
                print_info(k, 3*k+2)
                A[p_index(i, j), u_index(i, j+1)] = 1/hx
                A[p_index(i, j), u_index(i, j)] = -1/hx
                A[p_index(i, j), v_index(i, j)] = 1/hy
                A[p_index(i, j), v_index(i+1, j)] = -1/hy
                b[p_index(i, j)] = 0

            if i in range(1, ny-2):
                # Obtain viscosity values
                mu1x = mu_value(i, j)
                mu2x = mu_value(i, j-1)
                mu3x = (mu_value(i-1, j-1)+mu_value(i-1, j)+mu_value(i, j)+mu_value(i, j-1))/4
                mu4x = (mu_value(i, j-1)+mu_value(i, j)+mu_value(i+1, j)+mu_value(i+1, j-1))/4
                mu1y = mu_value(i-1, j)
                mu2y = mu_value(i, j)
                mu3y = (mu_value(i-1, j)+mu_value(i-1, j+1)+mu_value(i, j+1)+mu_value(i, j))/4
                mu4y = (mu_value(i-1, j-1)+mu_value(i-1, j)+mu_value(i, j)+mu_value(i, j-1))/4

                # Add x-Stokes equations to inner nodes
                print_info(k, 3*k)
                A[u_index(i, j), u_index(i, j+1)] = 2*mu1x/hx**2
                A[u_index(i, j), u_index(i, j)] -= 2*mu1x/hx**2
                A[u_index(i, j), u_index(i, j)] -= 2*mu2x/hx**2
                A[u_index(i, j), u_index(i, j-1)] = 2*mu2x/hx**2
                A[u_index(i, j), p_index(i, j)] = -1/hx
                A[u_index(i, j), p_index(i, j-1)] = 1/hx
                A[u_index(i, j), u_index(i-1, j)] = 1*mu3x/hy**2
                A[u_index(i, j), u_index(i, j)] -= 1*mu3x/hy**2
                A[u_index(i, j), v_index(i, j)] = 1*mu3x/(hx*hy)
                A[u_index(i, j), v_index(i, j-1)] = -1*mu3x/(hx*hy)
                A[u_index(i, j), u_index(i, j)] -= 1*mu4x/hy**2
                A[u_index(i, j), u_index(i+1, j)] = 1*mu4x/hy**2
                A[u_index(i, j), v_index(i+1, j)] = -1*mu4x/(hx*hy)
                A[u_index(i, j), v_index(i+1, j-1)] = 1*mu4x/(hx*hy)
                b[u_index(i, j)] = -rho*g*np.sin(alpha)

                # Add y-Stokes equations to inner nodes
                print_info(k, 3*k+1)
                A[v_index(i, j), v_index(i-1, j)] = 2*mu1y/hy**2
                A[v_index(i, j), v_index(i, j)] -= 2*mu1y/hy**2
                A[v_index(i, j), v_index(i, j)] -= 2*mu2y/hy**2
                A[v_index(i, j), v_index(i+1, j)] += 2*mu2y/hy**2
                A[v_index(i, j), p_index(i-1, j)] = -1/hy
                A[v_index(i, j), p_index(i, j)] = 1/hy
                A[v_index(i, j), u_index(i-1, j+1)] = 1*mu3y/(hx*hy)
                A[v_index(i, j), u_index(i, j+1)] = -1*mu3y/(hx*hy)
                A[v_index(i, j), v_index(i, j+1)] = 1*mu3y/hx**2
                A[v_index(i, j), v_index(i, j)] -= 1*mu3y/hx**2
                A[v_index(i, j), u_index(i-1, j)] = -1*mu4y/(hx*hy)
                A[v_index(i, j), u_index(i, j)] = 1*mu4y/(hx*hy)
                A[v_index(i, j), v_index(i, j)] -= 1*mu4y/hx**2
                A[v_index(i, j), v_index(i, j-1)] = 1*mu4y/hx**2
                b[v_index(i, j)] = rho*g*np.cos(alpha)

                # Add continuity equations to inner nodes
                print_info(k, 3*k+2)
                A[p_index(i, j), u_index(i, j+1)] = 1/hx
                A[p_index(i, j), u_index(i, j)] = -1/hx
                A[p_index(i, j), v_index(i, j)] = 1/hy
                A[p_index(i, j), v_index(i+1, j)] = -1/hy
                b[p_index(i, j)] = 0

            if i == ny-2:
                # Obtain viscosity values
                mu1x = mu_value(i, j)
                mu2x = mu_value(i, j-1)
                mu3x = (mu_value(i-1, j-1)+mu_value(i-1, j)+mu_value(i, j)+mu_value(i, j-1))/4
                mu1y = mu_value(i-1, j)
                mu2y = mu_value(i, j)
                mu3y = (mu_value(i-1, j)+mu_value(i-1, j+1)+mu_value(i, j+1)+mu_value(i, j))/4
                mu4y = (mu_value(i-1, j-1)+mu_value(i-1, j)+mu_value(i, j)+mu_value(i, j-1))/4

                # Add x-Stokes equations to base nodes
                print_info(k, 3*k)
                A[u_index(i, j), u_index(i, j+1)] = 2*mu1x/hx**2
                A[u_index(i, j), u_index(i, j)] -= 2*mu1x/hx**2
                A[u_index(i, j), u_index(i, j)] -= 2*mu2x/hx**2
                A[u_index(i, j), u_index(i, j-1)] = 2*mu2x/hx**2
                A[u_index(i, j), p_index(i, j)] = -1/hx
                A[u_index(i, j), p_index(i, j-1)] = 1/hx
                A[u_index(i, j), u_index(i-1, j)] = 1*mu3x/hy**2
                A[u_index(i, j), u_index(i, j)] -= 1*mu3x/hy**2
                A[u_index(i, j), v_index(i, j)] = 1*mu3x/(hx*hy)
                A[u_index(i, j), v_index(i, j-1)] = -1*mu3x/(hx*hy)
                A[u_index(i, j), u_index(i, j)] -= 1/hy*beta2[j]
                b[u_index(i, j)] = -rho*g*np.sin(alpha)

                # Add y-Stokes equations to base nodes
                print_info(k, 3*k+1)
                A[v_index(i, j), v_index(i-1, j)] = 2*mu1y/hy**2
                A[v_index(i, j), v_index(i, j)] -= 2*mu1y/hy**2
                A[v_index(i, j), v_index(i, j)] -= 2*mu2y/hy**2
                A[v_index(i, j), p_index(i-1, j)] = -1/hy
                A[v_index(i, j), p_index(i, j)] = 1/hy
                A[v_index(i, j), u_index(i-1, j+1)] = 1*mu3y/(hx*hy)
                A[v_index(i, j), u_index(i, j+1)] = -1*mu3y/(hx*hy)
                A[v_index(i, j), v_index(i, j+1)] = 1*mu3y/hx**2
                A[v_index(i, j), v_index(i, j)] -= 1*mu3y/hx**2
                A[v_index(i, j), u_index(i-1, j)] = -1*mu4y/(hx*hy)
                A[v_index(i, j), u_index(i, j)] = 1*mu4y/(hx*hy)
                A[v_index(i, j), v_index(i, j)] -= 1*mu4y/hx**2
                A[v_index(i, j), v_index(i, j-1)] = 1*mu4y/hx**2
                b[v_index(i, j)] = rho*g*np.cos(alpha)

                # Add continuity equations to base nodes
                print_info(k, 3*k+2)
                A[p_index(i, j), u_index(i, j+1)] = 1/hx
                A[p_index(i, j), u_index(i, j)] = -1/hx
                A[p_index(i, j), v_index(i, j)] = 1/hy
                b[p_index(i, j)] = 0
    return A, b


# Compute the viscosity
def viscosity(x):
    # Define helper functions
    def u_value(i, j):
        return x[u_index(i, j)]

    def v_value(i, j):
        return x[v_index(i, j)]

    eps2 = np.zeros((ny-1, nx-1))
    for j in range(nx-1):
        for i in range(ny-1):

            # Surface nodes
            if i == 0:
                dudx = (u_value(i, j+1)-u_value(i, j-1))/hx
                dvdy = (v_value(i, j)-v_value(i+1, j))/hy
                dudy = (u_value(i, j+1)-u_value(i, j)-u_value(i+1, j+1)+u_value(i+1, j))/(2*hx*hy)
                dvdx = (v_value(i, j+1)-v_value(i+1, j+1)-v_value(i, j-1)+v_value(i+1, j-1))/(2*hx*hy)

            # Inner nodes
            if i in range(1, ny-2):
                dudx = (u_value(i, j+1)-u_value(i, j-1))/hx
                dvdy = (v_value(i, j)-v_value(i+1, j))/hy
                dudy = (u_value(i-1, j+1)-u_value(i-1, j)-u_value(i+1, j+1)+u_value(i+1, j))/(2*hx*hy)
                dvdx = (v_value(i, j+1)-v_value(i+1, j+1)-v_value(i, j-1)+v_value(i+1, j-1))/(2*hx*hy)

            # Base nodes
            if i == ny-2:
                dudx = (u_value(i, j+1)-u_value(i, j-1))/hx
                dvdy = v_value(i, j)/hy
                dudy = (u_value(i-1, j+1)-u_value(i-1, j)-u_value(i, j+1)+u_value(i, j))/(2*hx*hy)
                dvdx = (v_value(i, j+1)-v_value(i, j-1))/(2*hx*hy)

            # Compute effective strain rate
            eps2[i, j] = 0.5*dudx**2 + 0.5*(dvdy)**2 + 0.25*(dudy+dvdx)**2

    # Compute effective viscosity
    mu = 0.5*A_glen**(-1/n_glen)*eps2**((1-n_glen)/(2*n_glen))

    return mu, eps2


# Solve Ax=b using p
def solve(p):
    num = 1
    nummax = 20
    mu = np.ones((ny-1, nx-1))
    beta2 = beta_squared(p)
    while num <= nummax:
        A, b = setup(beta2, mu)
        x = np.linalg.solve(A, b)
        mu, eps2 = viscosity(x)
        num += 1
    return A, b, x


# Define the error function
def error(p):  
    A, b, x = solve(p)
    u_surface = np.zeros(3*(ny-1)*(nx-1))
    u_surface[u_indices_top] = x[u_indices_top]
    coeff = np.ones(nx-1)
    i = np.arange(nx-1)
    coeff[i%2==0] = 2
    coeff[i%2!=0] = 4
    coeff[0] = 1
    coeff *= hx/3
    integral = np.zeros(3*(ny-1)*(nx-1))
    integral[u_indices_top] = coeff
    error = integral.dot((u_surface-u_desired)**2)
    return error


# Define the derivative of error function wrt p
def dgdp(p):
    # Solve equations with current params
    A, b, x = solve(p)
    u_surface = np.zeros(3*(ny-1)*(nx-1))
    u_surface[u_indices_top] = x[u_indices_top]

    # Compute df/dx
    dfdx = A

    # Compute df/dp
    dfdp = np.zeros((3*(ny-1)*(nx-1), N+1))
    for j in range(nx-1):
        i = ny-2  # base index
        for k in range(N+1):
            if k == 0:
                dfdp[u_index(i, j), k] = -1/hy*x[u_index(i, j)]
            elif k % 2 != 0:
                dfdp[u_index(i, j), k] = -1/hy*x[u_index(i, j)]*np.sin(2*np.pi*k*x_grid[j]/length)
            elif k % 2 == 0:
                dfdp[u_index(i, j), k] = -1/hy*x[u_index(i, j)]*np.cos(2*np.pi*k*x_grid[j]/length)

    # Compute dg/dx using Simpson's rule
    coeff = np.ones(nx-1)
    i = np.arange(nx-1)
    coeff[i % 2 == 0] = 2
    coeff[i % 2 != 0] = 4
    coeff[0] = 1
    coeff *= 2*hx/3
    dgdx = np.zeros(3*(ny-1)*(nx-1))
    dgdx[u_indices_top] = coeff
    dgdx = np.multiply(dgdx, u_surface-u_desired)

    # Solve adjoint equation
    lmbd = np.linalg.solve(dfdx.T, dgdx)

    # Compute dg/dp
    dgdp = -lmbd.dot(dfdp)

    return dgdp


# Store values for plots
def keep(p):
    A, b, x = solve(p)
    u_surface = np.zeros(3*(ny-1)*(nx-1))
    u_surface[u_indices_top] = x[u_indices_top]
    coeff = np.ones(nx-1)
    i = np.arange(nx-1)
    coeff[i % 2 == 0] = 2
    coeff[i % 2 != 0] = 4
    coeff[0] = 1
    coeff *= hx/3
    integral = np.zeros(3*(ny-1)*(nx-1))
    integral[u_indices_top] = coeff
    err = integral.dot((u_surface-u_desired)**2)
    params.append(p)
    errors.append(err)
    surface_velocities.append(x[u_indices_top])
    logging.info("Current error: g(u)={}".format(err))


# Define plotting functions
def plot_fields(u, v, p):
    fig, axs = plt.subplots(3, 1, sharex=True)
    vmin = np.min((u, v))
    vmax = np.max((u, v))
    for ax, values, title in ((axs[0], u, r'$\mathbf{u}}$'), 
                              (axs[1], v, r'$\mathbf{v}$')):
        im0 = ax.pcolor(x, y, values, vmin=vmin, vmax=vmax, cmap=cmo.deep)
        ax.set_title(title)
        ax.set_ylabel("y", rotation=0)
        ax.set_xlim([0, length])
        ax.set_ylim([0, height])
    fig.colorbar(im0, ax=axs[:2], label="m/a", aspect=2*10)
    ax = axs[2]
    im1 = ax.pcolor(x, y, p, vmin=0, cmap=cmo.deep)
    ax.set_title(r'$\mathbf{p}$')
    ax.set_ylabel("y", rotation=0)
    ax.set_xlabel("x")
    ax.set_xlim([0, length])
    ax.set_ylim([0, height])
    fig.colorbar(im1, ax=[axs[2]], label="MPa", aspect=8.7)


def plot_beta2(params):
    col = np.arange(len(params), 0, -1)/len(params)
    plt.plot(x_grid, beta_squared(p_initial), "C1", label="initial")
    plt.plot(x_grid, beta_squared(p_desired), "C2", label="desired")
    for i, p in enumerate(params):
        plt.plot(x_grid, beta_squared(p), color=(col[i], col[i], col[i]), alpha=0.5)
    plt.xlim([0, length])
    legend = plt.legend()
    plt.xticks(np.arange(0, 20001, step=5000))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\beta^2$')


def plot_params(params):
    iterations = np.arange(len(params))
    params = np.asarray(params)
    for i in range(len(params)):
        plt.plot(iterations, params[:, i], 'C0')
    plt.xticks(np.arange(0, len(params), step=5))
    plt.xlabel('iterations')
    plt.ylabel(r'$p_k$')
    plt.xlim([0, len(params)])
    plt.grid(linestyle='--')


def plot_errors(errors):
    iterations = np.arange(len(errors))
    plt.semilogy(iterations, errors, 'C0')
    plt.xticks(np.arange(0, len(errors), step=5))
    plt.ylim([10**-1, 10**6])
    plt.xlim([0,len(errors)])
    plt.xlabel('iterations')
    plt.ylabel(r'$g(u)$')
    plt.grid(linestyle='--')


def plot_surface_velocities(velocities):
    col = np.arange(len(velocities), 0, -1)/len(velocities)
    x = np.linspace(hx/2, length-hx/2, nx-1)
    plt.plot(x, velocities[0], "C1", label="initial")
    plt.plot(x, u_desired[u_indices_top], "C2", label="desired")
    for i, u in enumerate(velocities):
        plt.plot(x, u, color=(col[i],col[i],col[i]), alpha=0.5)
    plt.xlim([0, length])
    legend = plt.legend()
    plt.xticks(np.arange(0, 20001, step=5000))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u$')


if __name__ == "__main__":
    # Set parameters
    alpha = 0.1/180*np.pi  # inclination of the plane
    height = 1.0e3         # height of the ice sheet (m)
    length = 2.0e4         # length of the domain (m)
    nx = 33                # number of grid points in x-direction, must be odd
    ny = 16                # number of grid points in y-direction, >3
    rho = 910.0            # density of ice, kg m^-3
    mu0 = 1.0              # initial viscosity
    g = 9.81               # gravitational acceleration (m s^-2)
    A_glen = 1.0e-16       # Glen flow law ice softness (Pa^-n a^-1)
    n_glen = 3             # Glen flow law exponent
    N = 30                 # number of coefficients in the trigonometric expansion of beta^2
    verbose = False

    # Set up the grid
    hx = length/(nx-1)
    hy = height/(ny-1)
    x_grid = np.linspace(0, length, nx)
    y_grid = np.linspace(0, height, ny)

    # Initialize coefficients
    p_initial = np.zeros(N+1)
    p_initial[0] = 1000.0
    p_desired = np.zeros(N+1)
    p_desired[0] = 1000.0
    p_desired[1] = 1000.0

    # Define indices
    node_indices = np.arange((nx-1)*(ny-1)).reshape((ny-1, nx-1), order="F")
    u_indices_top = 3*node_indices[0, :]
    u_indices_base = 3*node_indices[-1, :]
    u_indices = 3*np.arange((nx-1)*(ny-1))
    v_indices = u_indices+1
    p_indices = u_indices+2

    # Solve Ax=b and obtain the desired surface velocities
    A, b, x = solve(p_desired)
    u_desired = np.zeros(3*(ny-1)*(nx-1))
    u_desired[u_indices_top] = x[u_indices_top]

    # Solve Ax=b and obtain the initial surface velocities
    A, b, x = solve(p_initial)

    # Solve Ax=b and obtain the surface velocities
    params = [p_initial]
    errors = [error(p_initial)]
    surface_velocities = [x[u_indices_top]]

    # Optimize p
    optimum = minimize(
        error,
        p_initial,
        method='BFGS',
        jac=dgdp,
        callback=keep,
        options={'disp': True, 'maxiter': 30}
    )  

    # Solve Ax=b with the newest params p
    p = params[-1]
    A, b, x = solve(p)

    # Extract values of u, v, and p from the solution
    u_values = np.take(x, u_indices)
    v_values = np.take(x, v_indices)
    p_values = np.take(x, p_indices)

    # Compute the averages at the center of each cell
    U_values = np.zeros((ny-1, nx))
    U_values[:, :-1] = u_values.reshape(ny-1, nx-1, order="F")
    U_values[:, -1] = U_values[:, 0]  # Add first column to last one
    U_values = (U_values[:, 0:-1]+U_values[:,1:])/2

    u_surface = U_values[0,:]

    # Compute average of v at the center of each cell (ignore ghost points, i.e. first row)
    V_values = np.zeros((ny, nx-1))
    V_values[0:-1, :] = v_values.reshape(ny-1, nx-1, order="F")
    V_values = (V_values[0:-1, :]+V_values[1:, :])/2

    # Obtain pressure at the center of each cell
    P_values = p_values.reshape(ny-1, nx-1, order="F")

    # Express pressure in MPa
    P_values *= 1e-6

    # Create a meshgrid for plotting
    x = np.linspace(hx/2, length-hx/2, nx-1)
    y = np.linspace(hy/2, height-hy/2, ny-1)

    # Create and save plots
    fig1 = plt.figure(1)
    plot_fields(np.flipud(U_values), np.flipud(V_values), np.flipud(P_values))
    plt.savefig("figures/fields.pdf")

    plt.clf()
    fig2 = plt.figure(2)
    plt.subplot(221)
    plot_beta2(params)
    plt.subplot(222)
    plot_params(params)
    plt.subplot(223)
    plot_surface_velocities(surface_velocities)
    plt.subplot(224)
    plot_errors(errors)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig("figures/results.pdf")

