import matplotlib.pyplot as plt
import simulate as sim
import numpy as np
import matplotlib.animation as animation

bT = 20
bB = 150
D = 0.1
T = 10
Nt_points = 1000
Lx = 1
Ly = 1
Nx_points = 20
Ny_points = 20
u0 = 20*np.ones((Nx_points, Ny_points))

# run simulation
U = sim.simulate_2d_heat(u0, bT, bB, D, T, Nt_points, Lx, Ly, Nx_points, Ny_points)

delta_t = T/(Nt_points - 1)

def plotheatmap(u_k, k):
    # Clear the current plot figure
    plt.clf()

    plt.title(f"Temperature at t = {k*delta_t:.3f} unit time")
    plt.xlabel("x")
    plt.ylabel("y")

    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=20, vmax=150)
    plt.colorbar()

    return plt

def animate(k):
    plotheatmap(U[:,:,k], k)

# animate results and save
anim = animation.FuncAnimation(plt.figure(), animate, interval=T, frames=Nt_points//5, repeat=False)
anim.save("heat_equation_solution.gif")
