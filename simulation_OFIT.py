### test ###
from methods_OFIT.transformation import *
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
import pickle
import time
from tqdm import tqdm

#define geometry
R0 = 0.65  #major radius of plasma
r = 0.2   #minor radius of plasma
camera_location = (0.98,0,-0.036) #location of the camera in cartesian coordinate (u,v,w)

time_arr, R_err, r_err = [[] for _ in range(3)]

def simulate_projection(R0: float, r: float, camera_location: tuple[float, float, float], n_samples: int = 1000) -> tuple[NDArray, NDArray]:  # create projection image from simulated shell
    """
    Function to create image on the projection plane given major and minor radius of plasma

    :param R0: plasma major radius in meters
    :param r: plasma minor radius in meters
    :param camera_location: location of camera in cartesian coordinate
    :param n_samples: number of edge points
    :return: 2 numpy arrays of plasma edge in projection plane (u,v)
    """
    wc, uc, vc = camera_location  # extract location of camera

    theta_poloidal = np.linspace(0, 2 * np.pi, n_samples)  # angle on poloidal plane [0,2*pi)

    # plasma edge in tokamak coordinate
    def mk_Re(theta):
        return R0 + r * np.cos(theta)

    def mk_Ze(theta):
        return r * np.sin(theta)

    # equation for solving for toroidal angle of plasma
    def phi_proj(theta):
        Re, Ze = mk_Re(theta), mk_Ze(theta)

        return np.arcsin( (Re - (Ze - vc) * -np.tan(theta))/np.sqrt(uc**2 + wc**2) ) - np.arctan( uc / wc )

    # convert to projection coordinate
    def cal_ue_ve(phi, theta):
        Re, Ze = mk_Re(theta), mk_Ze(theta)

        # calculate for ue in projection plane
        ue = wc * (Re * np.cos(phi) - uc) / (wc - Re * np.sin(phi)) + uc

        # calculate for ve in projection plane
        ve = wc * (Ze - vc) / (wc - Re * np.sin(phi)) + vc
        return ue, ve

    # suppress runtime warning
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="The iteration is not making good progress")

    # calculate phi from each theta
    phi_e = [phi_proj(theta_val) for theta_val in theta_poloidal]

    # convert to projection plane
    projection_coordinate = np.array([cal_ue_ve(phi, theta) for phi, theta in zip(phi_e, theta_poloidal)])
    ue, ve = projection_coordinate[:, 0], projection_coordinate[:, 1]

    return ue, ve

def cluster(u: NDArray, v: NDArray) -> tuple[NDArray, NDArray]:
    """
    cluster given data into two edges (cluster) by AgglomerativeClustering with n_samples = 2

    :param u: edge data along u dimension
    :param v: edge data along v dimension
    :return: clustered data into two edges
    """
    model = AgglomerativeClustering(n_clusters=2)
    Edge = np.column_stack((u, v))
    labels = model.fit_predict(Edge)

    clusters = np.unique(labels)
    clusters_arr = {cluster: Edge[labels == cluster] for cluster in clusters}

    return clusters_arr[0], clusters_arr[1]  # return the two clusters

for _ in tqdm(range(1000)):

    R_fluctuation = np.random.uniform(-0.001,0.001)
    r_fluctuation = np.random.uniform(-0.001,0.001)

    R0 = R0 + R_fluctuation
    r = r + r_fluctuation

    # create projection image
    ue,ve = simulate_projection(R0,r,camera_location)

    ue, ve = ue[~np.isnan(ue)], ve[~np.isnan(ve)]

    edge1, edge2 = cluster(ue,ve)
    u1,v1 = edge1[:,0], edge1[:,1]
    u2,v2 = edge2[:,0], edge2[:,1]

    start = time.time()
    # transform to poloidal plane
    R1, Z1, param1, _, u1_pass, v1_pass = poloidal_transformation(u1,v1,camera_location,RANSAC_epsilon=0.0001)
    R2, Z2, param2, _, u2_pass, v2_pass = poloidal_transformation(u2,v2,camera_location,RANSAC_epsilon=0.0001)

    R_com, Z_com = np.append(R1,R2), np.append(Z1,Z2)
    (R_cal,Z_cal,r_cal),_,_,_ = RANSAC_circle(R_com,Z_com,sample_size = 3, n = 100)

    end = time.time()

    time_arr.append(end - start)
    R_err.append(abs(R0 - R_cal))
    r_err.append(abs(r - r_cal))

    # print("sample size: ", len(ue))
    # print("runtime: ", end - start)
    # fig, (ax_projection,ax_poloidal) = plt.subplots(1,2,figsize = (10,5))

    # horizontal_parabola = lambda y,a,b,c: a*y**2 + b*y + c
    # ax_projection.plot(ue,ve,".-")
    # ys = np.linspace(-0.6,0.8,500)
    # ax_projection.plot(horizontal_parabola(ys,*param1),ys)
    # ax_projection.plot(horizontal_parabola(ys,*param2),ys)
    # ax_projection.plot(u1_pass,v1_pass,".")
    # ax_projection.plot(u2_pass,v2_pass,".")
    # ax_projection.set_xlabel("u [m]")
    # ax_projection.set_ylabel("v [m]")
    # ax_projection.set_title("TT1 projection plane")
    # ax_projection.grid()

    # ax_poloidal.plot(R1,Z1,".")
    # ax_poloidal.plot(R2,Z2,".")
    # circle = patches.Circle((circle_param[0],circle_param[1]), radius = circle_param[2], fill = False)
    # ax_poloidal.add_patch(circle)
    # ax_poloidal.set_xlabel("R [m]")
    # ax_poloidal.set_ylabel("Z [m]")
    # ax_poloidal.set_title("poloidal plane")
    # ax_poloidal.set_aspect("equal")
    # ax_poloidal.grid()
    # ax_poloidal.set_xlim(0,2)
    # ax_poloidal.set_ylim(-1,1)

    # plt.show()

print(np.mean(time_arr))
print(np.mean(R_err))
print(np.mean(r_err))