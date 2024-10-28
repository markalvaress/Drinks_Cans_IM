# Contains functions to simulate various aspects of the temperature distribution.
import numpy as np

def simulate_dirichlet_sink(u0, b0, D, T, Nt_points, L, Nx_points, a, u_inf = 0, diam = 0):
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
        diam (float): The diameter of the can, for the sink term. If no sink term, you can leave this blank.     
    Returns:
        U (matrix): A Nx_points by Nt_points matrix, where each column is the simulated heat distribution at a time t.

    """
    dx = L/(Nx_points - 1)
    dt = T/(Nt_points - 1)
    C = D*dt/(dx**2)
    unit_area = dx*np.pi*diam # for use in the sink term. If we're not using sink term, this will be 0 but won't be used.

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
        u_new[1:Nx_points-1] -= unit_area*a*(u_old[1:Nx_points-1] - u_inf) # This is the sink term - it doesn't affect the endpoints

        u_new[0] = b0(n * T/Nt_points) # enforce bottom boundary condition
        
        U[:,n] = u_new # store heat distribution in the results matrix 

    return U

def create_analytic_toy_fn(b0, bL, L, D, n_sum_terms = 200):
    """Create a function that returns the value of the analytic solution to the toy problem with the given parameters."""
    
    def U_analyt(x, t):
        """x can be a 1-d array"""
        if t == 0: # initial condition
            sol = bL * np.ones(len(x))
            sol[0] = b0
        else:
            u_d = b0 - bL

            sol = b0 - u_d*(x / L)
            for n in range(1,n_sum_terms):
                sol -= 2*u_d/(n*np.pi) * np.sin(n*np.pi*x/L) * np.exp(-(n*np.pi/L)**2 * D * t)

        return sol
    
    return U_analyt

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

def simulate_sink_with_fancy_bcs(u0, b0, D, T, Nt_points, L, Nx_points, a, h, kW, u_inf = 0, diam = 0, insulated = False):
    """
    Simulates the heat distribution over time of a material over time with one Newton cooling boundary condition, one endpoint with a time dependent temperature, and a sink term, using a backward Euler scheme.
    
    Args:
        b0 (function of t): The bottom boundary condition U(0,t) = b0(t). Can set to constant by passing e.g. lambda t: 150
        u0 (1d array): The initial condition. Should be a vector of length Nx_points
        D (float): Thermal diffusivity coefficient of the liquid
        T (float): End time of simulation (seconds)
        Nt_points (int): Number of time points to simulate between 0 and T
        L (float): Height of can (metres)
        Nx_points (int): Number of to discretise x from 0 to L 
        a (float): The value of the heat transfer coefficent for the sink term
        h (float): The value of the heat transfer coefficent for the top boundary condition
        kW (float): The thermal conductivity of the liquid within the can.
        u_inf (float): The ambient (air) temperature, for the sink term and fancy boundary conditions.
        insulated (bool): If true then the Newton cooling boundary condition is off and the top boundary condition is Neuman boundary condition =0
    Returns:
        U (matrix): A Nx_points by Nt_points matrix, where each column is the simulated heat distribution at a time t.

    """
    dx = L/(Nx_points - 1)
    dt = T/(Nt_points - 1)
    C = D*dt/(dx**2)
    unit_area = dx*np.pi*diam # for use in the sink term. If we're not using sink term, this will be 0 but won't be used.

    U = np.zeros((Nx_points,Nt_points)) # this is where we'll put results
    U[:,0] = u0 # initial condition

    # This defines the backward Euler timestep AU_{t+1} = U_t. Definition of A is given as below in lectures for dirichlet conditions
    A = np.zeros((Nx_points, Nx_points))
    for i in range(1, Nx_points-1):
        A[i,i-1] = -C
        A[i,i+1] = -C
        A[i,i] = 1 + 2*C    
    
    # implement the time dependent bottom boundary condition
    A[0,0] = 1

    if insulated == False:
        # Implement the Newton cooling boundary condition at the top
        A[Nx_points-1,Nx_points-1] = 1+2*C + (C*2*dx*h)/kW
        A[Nx_points-1,Nx_points-2] = -2*C

        # Create the vector that updates u_old to account for the Newton cooling boundary condition
        newtcool = np.zeros(Nx_points)
        # Set up the rest of the Newton cooling boundary condition
        newtcool[-1] = (C*2*dx*h*u_inf)/kW

    else: 
        newtcool = 0
        A[Nx_points-1,Nx_points-1] = 1+2*C 
        A[Nx_points-1,Nx_points-2] = -2*C

    # Run simulation
    for n in range(1, Nt_points):
        # update u by solving the matrix system AU_{t+1} = U_t
        u_old = U[:,n-1] + newtcool
        u_new = np.linalg.solve(A,u_old) 
        u_new[1:Nx_points-1] -= unit_area*a*(u_old[1:Nx_points-1] - u_inf) # This is the sink term - it doesn't affect the endpoints

        u_new[0] = b0(n * T/Nt_points) # enforce bottom boundary condition
        
        U[:,n] = u_new # store heat distribution in the results matrix 

    return U