# Contains functions to simulate various aspects of the temperature distribution.
import numpy as np

def simulate_dirichlet_sink(u0, b0, D, T, Nt_points, L, Nx_points, a, u_inf = 0):
    """
    Simulates the heat distribution over time of a material over time with one fixed temperature endpoint, one endpoint with a time dependent temperature, and a sink term, using a backward Euler scheme.
    
    Args:
        u0 (1d array): The initial condition. Should be a vector of length Nx_points
        b0 (function of t): The bottom boundary condition U(0,t) = b0(t). Can set to constant by passing e.g. lambda t: 150
        D (float): Thermal diffusivity coefficient of the liquid
        T (float): End time of simulation (seconds)
        Nt_points (int): Number of time points to simulate between 0 and T
        L (float): Height of can (metres)
        Nx_points (int): Number of to discretise x from 0 to L 
        a (float): The heat transfer coefficient for the sink term. Set to 0 for no sink term.
        u_inf (float): The ambient (air) temperature, for the sink term. If no sink term, you can leave this blank        
    Returns:
        U (matrix): A Nx_points by Nt_points matrix, where each column is the simulated heat distribution at a time t.

    """
    dx = L/(Nx_points - 1)
    dt = T/(Nt_points - 1)
    C = D*dt/(dx**2)

    U = np.zeros((Nx_points,Nt_points)) # this is where we'll put results
    U[:,0] = u0 # initial condition

    # This defines the backward Euler timestep AU_{t+1} = U_t. Definition of A is given as below in lectures for dirichlet conditions
    A = np.zeros((Nx_points, Nx_points))
    for i in range(1, Nx_points-1):
        A[i,i-1] = -C
        A[i,i+1] = -C
        A[i,i] = 1 + 2*C    
    # implement the (constant-in-time) Dirichlet conditions (i.e. the end points never change temp, U(0, t+dt) = U(0, t), same at x=1)
    A[0,0] = 1
    A[Nx_points-1,Nx_points-1] = 1

    # Run simulation
    for n in range(1, Nt_points):
        # update u by solving the matrix system AU_{t+1} = U_t
        u_old = U[:,n-1]
        u_new = np.linalg.solve(A,u_old) 
        u_new[1:Nx_points-1] -= a*(u_old[1:Nx_points-1] - u_inf) # This is the sink term - it doesn't affect the endpoints

        u_new[0] = b0(n * T/Nt_points) # enforce bottom boundary condition
        
        U[:,n] = u_new # store heat distribution in the results matrix 

    return U

def create_numeric_avtemp_fn(U, dx, time_to_index):
    """Returns the numeric function describing the average temperature at times t"""
    def avg_temp_numeric(t):
        """t is in seconds. Must be an np.array"""
        Ut = U[:, time_to_index(t)]

        # trapezoid rule
        avtemp = np.sum(dx * 0.5 * (Ut[:-1, :] + Ut[1:, :]), axis = 0) # sum along columns, i.e. sum along all x
        return avtemp
    
    return avg_temp_numeric

def time_to_temp(avg_temps, temp, index_to_time):
    for i, avg_temp in enumerate(avg_temps):
        if avg_temp > temp:
            return index_to_time(i)
        
    print(f"None of the given average temperatures is above {temp}.")
    return None


# EXPERIMENTAL -------------------------------------------------------------------------------------------------
# some taken from https://levelup.gitconnected.com/solving-2d-heat-equation-numerically-using-python-3334004aa01a

def simulate_2d_corner_heat(u0, bT, bB, D, T, Nt_points, Lx, Ly, Nx_points, Ny_points):
    """
    Simulates the heat distribution over time of a material over time with one fixed temperature endpoint, one endpoint with a time dependent temperature, and a sink term, using a backward Euler scheme.
    
    Args:
        u0 (2d array): The initial condition. Should be a vector of length Nx_points
        bT (float): The top temperature
        bB (float): The temp of the bottom side
        D (float): Thermal diffusivity coefficient of the liquid
        T (float): End time of simulation (seconds)
        Nt_points (int): Number of time points to simulate between 0 and T
        Lx (float): Width of space in x dir (metres)
        Ly (float): Width of space in y dir (metres)
        Nx_points (int): Number of to discretise x from 0 to Lx
        Ny_points (int): Number of to discretise y from 0 to Ly
        a (float): The heat transfer coefficient for the sink term. Set to 0 for no sink term.
        u_inf (float): The ambient (air) temperature, for the sink term. If no sink term, you can leave this blank        
    Returns:
        U (matrix): A Nx_points by Ny_points by Nt_points matrix, where each U[:, :, t] is the simulated heat distribution at a time t.

    """
    dx = Lx/(Nx_points - 1)
    dy = Ly/(Ny_points - 1)
    dt = T/(Nt_points - 1)
    Cx = D*dt/(dx**2)
    Cy = D*dt/(dy**2)
    if Cx > 0.5:
        print(f"Warning: {Cx=} is greater than 0.5. This may cause instability. Try using a smaller timestep.")
    if Cy > 0.5:
        print(f"Warning: {Cy=} is greater than 0.5. This may cause instability. Try using a smaller timestep.")

    U = np.zeros((Nx_points, Ny_points, Nt_points)) # this is where we'll put results
    U[:, :, 0] = u0 # initial condition

    # BCs:
    U[:, 0, 1:] = bT # top
    U[:, Ny_points-1, 1:] = bB # bottom

    # Run simulation
    for s in range(1, Nt_points):
        for m in range(1,Nx_points - 1):
            for n in range(1, Ny_points - 1):
                U[m,n,s] = U[m,n,s-1] + Cx*U[m+1,n,s-1] -2*Cx*U[m,n,s-1] + Cx*U[m-1,n,s-1] + Cy*U[m,n+1,s-1] -2*Cy*U[m,n,s-1] + Cy*U[m,n-1,s-1]

        U[0,1:-1,s] = U[1,1:-1,s] # L and R bdary conditions
        U[-1,1:-1,s] = U[-2,1:-1,s]

    return U
