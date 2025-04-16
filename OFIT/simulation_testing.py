### test ###
from transformation import *
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
import pickle
import time

#define geometry
R0 = 0.65  #major radius of plasma
r = 0.2   #minor radius of plasma
camera_location = (0.98,0,-0.036) #location of the camera in cartesian coordinate (u,v,w)

# create projection image
ue,ve = simulate_projection(R0,r,camera_location)

edge1, edge2 = cluster(ue,ve)
u1,v1 = edge1[:,0], edge1[:,1]
u2,v2 = edge2[:,0], edge2[:,1]

start = time.time()
# transform to poloidal plane
R1, Z1, param1, _, u1_pass, v1_pass = poloidal_transformation(u1,v1,camera_location,RANSAC_epsilon=0.0001)
R2, Z2, param2, _, u2_pass, v2_pass = poloidal_transformation(u2,v2,camera_location,RANSAC_epsilon=0.0001)

R_com, Z_com = np.append(R1,R2), np.append(Z1,Z2)
circle_param,_,_,_ = RANSAC_circle(R_com,Z_com,sample_size = 3, n = 100)

end = time.time()

print("sample size: ", len(ue))
print("runtime: ", end - start)
fig, (ax_projection,ax_poloidal) = plt.subplots(1,2,figsize = (10,5))

horizontal_parabola = lambda y,a,b,c: a*y**2 + b*y + c
ax_projection.plot(ue,ve,".-")
ys = np.linspace(-0.6,0.8,500)
ax_projection.plot(horizontal_parabola(ys,*param1),ys)
ax_projection.plot(horizontal_parabola(ys,*param2),ys)
ax_projection.plot(u1_pass,v1_pass,".")
ax_projection.plot(u2_pass,v2_pass,".")
ax_projection.set_xlabel("u [m]")
ax_projection.set_ylabel("v [m]")
ax_projection.set_title("TT1 projection plane")
ax_projection.grid()

ax_poloidal.plot(R1,Z1,".")
ax_poloidal.plot(R2,Z2,".")
circle = patches.Circle((circle_param[0],circle_param[1]), radius = circle_param[2], fill = False)
ax_poloidal.add_patch(circle)
ax_poloidal.set_xlabel("R [m]")
ax_poloidal.set_ylabel("Z [m]")
ax_poloidal.set_title("poloidal plane")
ax_poloidal.set_aspect("equal")
ax_poloidal.grid()
ax_poloidal.set_xlim(0,2)
ax_poloidal.set_ylim(-1,1)

plt.show()