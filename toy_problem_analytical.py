# Contains functions that build analytic solutions to the toy problem based on various parameters.
import numpy as np

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

def create_analytic_avg_temp(b0, bL, L, D, n_summation = 200):
    """Returns the analytic function describing the average temperature at times t"""
    def U_avg_analyt(t):
        """t is in seconds. Must be an np.array"""
        u_d = b0 - bL
        sol = b0 - u_d/2

        for k in range(n_summation):
            sol -= 4*u_d/np.pi**2 * np.exp(-((2*k + 1)*np.pi/L)**2 * D * t) / (2*k + 1)**2

        return sol

    return U_avg_analyt