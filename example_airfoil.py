import numpy as np
from lattice_boltzmann import LBMCalculator2D
from naca_airfoil import get_airfoil_mask
import matplotlib.pyplot as plt
import matplotlib

nx = 1200
ny = 400
tau = 0.6
delta_t_images = 150

ux = 0.1
airfoil_size = 250


def build_obstacle_mask(angle):
    obstacle_mask = get_airfoil_mask(nx, ny, 80, ny // 2, angle, airfoil_size)
    obstacle_mask[0, :] = 1
    obstacle_mask[-1, :] = 1
    return obstacle_mask


def get_angle(time):
    t = np.array([2, 6, 11, 14, 17]) * 10_000
    min_angle = 4.0
    max_angle = 20.0
    if time < t[0]:
        return min_angle
    if time < t[1]:
        return (max_angle - min_angle) * (time - t[0]) / (t[1] - t[0]) + min_angle
    if time < t[2]:
        return max_angle
    if time < t[3]:
        return (max_angle - min_angle) * (1 - (time - t[2]) / (t[3] - t[2])) + min_angle
    if time < t[4]:
        return min_angle
    return None


lbm = LBMCalculator2D(nx, ny, tau)
lbm.add_obstacle(build_obstacle_mask(0.0))

inlet_mask = np.zeros([ny, nx], dtype=bool)
inlet_mask[1:-1, 0] = 1
# inlet_mask[1:-1,-1] = 1
lbm.add_fixed_boundary(inlet_mask, lbm.rho0, ux, 0.0)
lbm.initialize()

plt.close("all")
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
cmap = matplotlib.cm.get_cmap("bwr").copy()
cmap.set_bad(color="gray")
fig.suptitle(f"t = {0:5d} steps")
ax.set_title("$\\nabla \\times u$")
ax.axis("equal")
ax.set_axis_off()
fig.tight_layout()
handle_vort = ax.imshow(np.zeros([ny, nx]), cmap=cmap, clim=np.array([-1, 1]) * 0.02)

t = 0
while True:
    t += delta_t_images
    print(f"{t}")
    angle = get_angle(t)
    if angle is None:
        break

    lbm.is_obstacle = build_obstacle_mask(angle)
    lbm.run(delta_t_images)
    handle_vort.set_data(lbm.vorticity)
    fig.suptitle(f"t = {t/1000:.1f}k steps, angle of attack = {angle:.0f} deg")
    plt.savefig(f"data/tmp/{t:06d}.png", bbox_inches="tight")
    # plt.pause(0.001)
