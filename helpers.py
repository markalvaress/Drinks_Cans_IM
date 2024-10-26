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

