from itertools import count
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from plasma_shift import cal_shift
from DxDz import cal_approx_DxDz as DxDz
from DxDz import cal_newton_DxDz as newton
from DxDz import cal_exact
from signal_strength import cal_signal
from geometry_TT1 import cross
import geometry_TT1

iteration = [0]
#actual value
R_sim, Z_sim = [0.00], [0.00]
I = [0]

#calculated value
R0_cal, Z0_cal = [R_sim[0]], [Z_sim[0]] #-30 deg
R0_err, Z0_err = [0], [0]
R1_cal, Z1_cal = [R_sim[0]], [Z_sim[0]] #0 deg
R1_err, Z1_err = [0], [0]
R2_cal, Z2_cal = [R_sim[0]], [Z_sim[0]] #30 deg
R2_err, Z2_err = [0], [0]

index = count()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5))

def animate(i):
    iteration.append(next(index))

    #actual shift value
    if abs(R_sim[-1]) <= 0.12:
        R_shift = R_sim[-1] + np.random.choice([-0.001, 0.001], p = [0.8,0.2])
        Z_shift = Z_sim[-1] + np.random.choice([-0.001, 0.001], p = [0.5,0.5])

    else:
        R_shift = R_sim[-1] + np.random.choice([-0.001, 0.001], p = [0.5,0.5])
        Z_shift = Z_sim[-1] + np.random.choice([-0.001, 0.001], p = [0.5,0.5])

    R_sim.append(R_shift)
    Z_sim.append(Z_shift)

    def helper(cross_index, est_horizontal, est_vertical, order=2):
        signal = cal_signal(R_shift, Z_shift, cross[cross_index])
        shift = cal_shift(DxDz_method=newton, taylor_order=order, signal=signal,
                          est_horizontal_shift=est_horizontal, est_vertical_shift=est_vertical,
                          coil_angle=cross[cross_index],
                          beta_horizontal_range=np.linspace(-0.01,0.01,101), alpha_vertical_range=np.linspace(-0.01,0.01,101))
        R, Z = shift[0, 0], shift[1, 0]
        Re, Ze = shift[0,1], shift[1,1]
        return R, Z, Re, Ze

    #calculated shift value

    #cross 0
    R0, Z0, R0e,Z0e = helper(0,R0_cal[-1],Z0_cal[-1])
    R0_cal.append(R0)
    R0_err.append(R0e)
    Z0_cal.append(Z0)
    Z0_err.append(Z0e)

    #cross 1
    R1, Z1, R1e,Z1e = helper(1, R1_cal[-1], Z1_cal[-1])
    R1_cal.append(R1)
    R1_err.append(R1e)
    Z1_cal.append(Z1)
    Z1_err.append(Z1e)

    #cross 2
    R2, Z2,R2e,Z2e = helper(2, R2_cal[-1],Z2_cal[-1])
    R2_cal.append(R2)
    R2_err.append(R2e)
    Z2_cal.append(Z2)
    Z2_err.append(Z2e)


    #plot of R shift
    ax1.cla()
    ax1.plot(iteration, R_sim,color = "red", label = "R_shift")
    ax1.errorbar(iteration, R0_cal,yerr = R0_err, color = "blue", label = "R0_cal")
    ax1.errorbar(iteration, R1_cal,yerr = R1_err, color = "green", label = "R1_cal")
    ax1.errorbar(iteration, R2_cal,yerr = R2_err, color = "purple", label = "R2_cal")
    ax1.set_xlim(iteration[-1] - 50, iteration[-1] + 50)
    ax1.set_ylim(-0.2,0.2)
    ax1.set_xlabel("iteration [1]")
    ax1.set_ylabel("R shift [m]")
    ax1.legend()
    ax1.grid()

    #plot of Z shift
    ax2.cla()
    ax2.plot(iteration, Z_sim, color = "red", label = "Z_shift")
    ax2.errorbar(iteration, Z0_cal,Z0_err ,color = "blue", label = "Z0_cal")
    ax2.errorbar(iteration, Z1_cal,Z1_err, color = "green", label = "Z1_cal")
    ax2.errorbar(iteration, Z2_cal,Z2_err, color = "purple", label = "Z2_cal")
    ax2.set_ylim(-0.2,0.2)
    ax2.set_xlim(iteration[-1] - 50, iteration[-1] + 50)
    ax2.set_xlabel("iteration [1]")
    ax2.set_ylabel("Z shift [m]")
    ax2.legend()
    ax2.grid()

ani = FuncAnimation(plt.gcf(), animate, interval = 1)
plt.tight_layout()
plt.show()

### test plasma_shift function ###
R_val, Z_val = 0.01,0.01
signal = cal_signal(R_val,Z_val, cross[0])
shift = cal_shift(DxDz_method=newton, taylor_order=6,signal = signal,
                         est_horizontal_shift=R_val, est_vertical_shift=Z_val,coil_angle = cross[0])

R_cal,Z_cal = shift[0,0],shift[1,0]

R_error = abs(R_val-R_cal)/R_val * 100
Z_error = abs(Z_val-Z_cal)/Z_val * 100

print(f"""
R shift = {R_val:.2f}, R cal = {R_cal:.2f}, error = {R_error:.2f} %
Z shift = {Z_val:.2f}, Z cal = {Z_cal:.2f}, error = {Z_error:.2f} %
R err = {R0_err}
""")