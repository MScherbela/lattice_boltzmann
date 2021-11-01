import matplotlib.cm
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from functools import partial
import imageio

class LBMCalculator2D:
    n_directions = 9
    c_vecs = np.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=int)
    weights = np.array([16, 4, 4, 4, 4, 1, 1, 1, 1], dtype=float) / 36

    def __init__(self, nx, ny, tau=1.0):
        self.tau = tau
        self.nx = nx
        self.ny = ny
        self.rho0 = 1.0
        self.is_obstacle = np.zeros([ny, nx], dtype=bool)
        self.initial_random_amplitude = 0.01
        self.fixed_boundaries = []

    @classmethod
    def build_from_file(cls, fname, velocity_maps=None):
        velocity_maps = velocity_maps or []
        data = imageio.imread(fname)[...,:3]
        lbm = cls(data.shape[1], data.shape[0])

        is_wall = np.all(data == np.array([0,0,0], dtype=int), axis=-1)
        lbm.add_obstacle(is_wall)

        for color, ux, uy in velocity_maps:
            mask = np.all(data == np.array(color, dtype=int), axis=-1)
            lbm.add_fixed_boundary(mask, 1.0, ux, uy)
        return lbm

    def initialize(self, ux0=0.0, uy0=0.0):
        v0_distr = self.build_equalized_v_distribution(self.rho0, ux0, uy0)
        self.f = v0_distr[None, None, :] + self.rho0 * self.initial_random_amplitude * np.random.uniform(-1.0, 1.0, [self.ny, self.nx, self.n_directions])
        self.step_func = self._build_step_function()

    @partial(jax.jit, static_argnums=[0,1])
    def _run_steps(self, n_steps, f, obstacle_mask):
        def _loop_body(i, f):
            return self.step_func(f, obstacle_mask)
        return jax.lax.fori_loop(0, n_steps, _loop_body, f)

    def run(self, n_steps):
        self.f = self._run_steps(n_steps, self.f, self.is_obstacle)
        self.rho, self.u = self.get_observables(self.f)
        self.vorticity = self.calculate_vorticity(self.u)

        self.rho = np.where(self.is_obstacle, np.nan, self.rho)
        self.vorticity = np.where(self.is_obstacle, np.nan, self.vorticity)
        self.u = np.where(self.is_obstacle[...,None], np.nan, self.u)
        return self.f

    def add_obstacle_cylinder(self, x, y, r):
        yy, xx = np.arange(self.ny)[:, None], np.arange(self.nx)[None, :]
        self.is_obstacle = self.is_obstacle | ((xx - x) ** 2 + (yy - y) ** 2 < (r ** 2))

    def add_obstacle(self, mask):
        self.is_obstacle = self.is_obstacle | mask


    def add_fixed_boundary(self, mask, rho, ux, uy):
        v_distr = self.build_equalized_v_distribution(rho, ux, uy)
        self.fixed_boundaries.append([np.array(mask, dtype=bool)[...,None], v_distr])

    @staticmethod
    def flip_flow_direction(f):
        f = f[...,[0,2,1,4,3,8,7,6,5]]
        return f

    @classmethod
    def flow(cls, f):
        f_new = []
        for i,(cx,cy) in enumerate(cls.c_vecs):
            f_new.append(jnp.roll(jnp.roll(f[...,i], cy, axis=0), cx, axis=1))
        return jnp.stack(f_new, axis=-1)

    @classmethod
    def get_equilibrium_distribution(cls, f):
        rho, u = cls.get_observables(f)
        c_u = u @ cls.c_vecs.T # [batch x ndims]
        u_u = np.sum(u**2, axis=-1, keepdims=True)  # [batch x 1]
        f_eq = rho[..., None] * cls.weights * (1+3*c_u + 4.5*c_u**2 - 1.5 * u_u)
        return f_eq

    @classmethod
    def get_observables(cls, f):
        rho = jnp.sum(f, axis=-1)
        u = (f @ cls.c_vecs) / rho[..., None]
        return rho, u

    @staticmethod
    def calculate_vorticity(u):
        return jnp.gradient(u[...,1], axis=1) - jnp.gradient(u[...,0], axis=0)

    def build_equalized_v_distribution(self, rho, ux, uy):
        v_distr_init = np.ones(self.n_directions)
        u_scaling = self.n_directions / (1 - ux - uy)

        if ux > 0:
            v_distr_init[3] += ux * u_scaling
        else:
            v_distr_init[4] += ux * u_scaling

        if uy > 0:
            v_distr_init[1] += uy * u_scaling
        else:
            v_distr_init[2] += uy * u_scaling

        v_distr_init *= rho / np.sum(v_distr_init)
        return self.get_equilibrium_distribution(v_distr_init)


    def _build_step_function(self):
        @jax.jit
        def step_func(f, obstacle_mask):
            f = self.flow(f)
            f_eq = self.get_equilibrium_distribution(f)
            f += (f_eq - f) / self.tau

            # Apply reflecting boundaries
            f = jnp.where(obstacle_mask[..., None], self.flip_flow_direction(f), f)

            # Apply fixed boundaries at in-/outlets
            for mask, v_distr in self.fixed_boundaries:
                f = jnp.where(mask, v_distr, f)
            return f
        return step_func

def run_animation_loop(lbm, n_t, delta_t):
    nx, ny = lbm.nx, lbm.ny

    fig, axes = plt.subplots(2,2, figsize=(15,5))
    cmap = matplotlib.cm.get_cmap("bwr").copy()
    cmap.set_bad(color='gray')
    handle_rho = axes[0][0].imshow(np.ones([ny, nx]) * lbm.rho0, cmap=cmap, clim=np.array([-1,1]) * 0.2 * lbm.rho0)
    handle_vort = axes[0][1].imshow(np.zeros([ny, nx]), cmap=cmap, clim=np.array([-1,1])*0.02)
    handle_ux = axes[1][0].imshow(np.zeros([ny, nx]), cmap=cmap, clim=np.array([-1,1]) * 0.5)
    handle_uy = axes[1][1].imshow(np.zeros([ny, nx]), cmap=cmap, clim=np.array([-1,1]) * 0.5)
    axes_flat = list(axes[0]) + list(axes[1])
    for ax, title in zip(axes_flat, ["$\\rho$", "$\\nabla \\times u$", "$u_x$", "$u_y$"]):
        ax.set_title(title)
        ax.axis("equal")
    fig.tight_layout()

    for t in range(n_t//delta_t):
        lbm.run(delta_t)
        print(t * delta_t)
        handle_rho.set_data(lbm.rho - lbm.rho0)
        handle_vort.set_data(lbm.vorticity)
        handle_ux.set_data(lbm.u[...,0])
        handle_uy.set_data(lbm.u[...,1])
        plt.pause(0.001)






