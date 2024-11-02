import numpy as np

def create_index_to_time_fn(T, Nt_points):
    def index_to_time(i):
        t = i * T/(Nt_points - 1)
        return t

    return index_to_time

def create_time_to_index_fn(T, Nt_points):
    def time_to_index(t):
        """This gives you the closest index to your time, so you can input a time that is not 
        of the form n * dt and you'll still get an integer index"""
        i = t * (Nt_points - 1)/T
        i = np.array(i.round(0), dtype = int)
        return i

    return time_to_index

def sink_HTC(L, r1, r2, kS, hA, hB):
    """
    Calculates the heat transfer coefficent between the water, through the side of the can, into the surrounding air.

    Args:
        L (float): Height of can (metres)
        r1 (float): Radius of the liquid in the can (up to the interior surface of the can) (metres)
        r2 (float): Radius of the liquid in the can and the can (up to the exterior surface of the can) (metres)
        kS (float): Thermal conductivity of stainless steel (W/mC)
        hA (float): Heat transfer coefficient of water (W/m^2C)
        hB (float): Heat transfer coefficiemt of air (W/m^2C)

    Returns:
        a (float): The values of the heat transfer coefficent for the sink term
    """

    a = (2*np.pi)/(1/(hA*r1) + 1/(hB*r2) + np.log(r2/r1)/kS)

    return a

def boundary_condition_HTC(W, r1, kS, hA, hB):
    """
    Calculates the heat transfer coefficent between the water, through the top of the can, into the surrounding air.

    Args:
        W (float): Width of the material at the top of the can (metres)
        r1 (float): Radius of the liquid in the can (up to the interior surface of the can) (metres)
        kS (float): Thermal conductivity of stainless steel (W/mC)
        hA (float): Heat transfer coefficient of water (W/m^2C)
        hB (float): Heat transfer coefficiemt of air (W/m^2C)

    Returns:
        h (float): The value of the heat transfer coefficent for the top boundary condition
    """
    # Calculate the area at the top of the can
    A = np.pi*r1**2

    h = 1/(1/(hA*A) + W/(kS*A) + 1/(hB*A))

    return h