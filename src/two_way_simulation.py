from src.constants import Classification, State, Water, Ice, Simulation
from src.solvers import PressureSolver, HeatSolver
from src.solvers import StaggeredSolver
from typing import override

import taichi.math as tm
import taichi as ti


@ti.data_oriented
class TwoWay_MLSMPM(StaggeredSolver):
    def __init__(self, max_particles: int, n_dimensions: int, n_grid: int, vol_0: float):
        super().__init__(max_particles, n_dimensions, n_grid, vol_0)

        # Properties on MAC-faces:
        self.classification_x = ti.field(dtype=ti.i32, shape=(self.wx + 1, self.wy, self.wz), offset=self.w_offset)
        self.classification_y = ti.field(dtype=ti.i32, shape=(self.wx, self.wy + 1, self.wz), offset=self.w_offset)
        self.classification_z = ti.field(dtype=ti.i32, shape=(self.wx, self.wy, self.wz + 1), offset=self.w_offset)
        self.conductivity_x = ti.field(dtype=ti.f32, shape=(self.wx + 1, self.wy, self.wz), offset=self.w_offset)
        self.conductivity_y = ti.field(dtype=ti.f32, shape=(self.wx, self.wy + 1, self.wz), offset=self.w_offset)
        self.conductivity_z = ti.field(dtype=ti.f32, shape=(self.wx, self.wy, self.wz + 1), offset=self.w_offset)

        # Properties on MAC-cells:
        self.inv_lambda_c = ti.field(dtype=ti.f64, shape=(self.wx, self.wy, self.wz), offset=self.w_offset)
        self.capacity_c = ti.field(dtype=ti.f32, shape=(self.wx, self.wy, self.wz), offset=self.w_offset)
        self.JE_c = ti.field(dtype=ti.f32, shape=(self.wx, self.wy, self.wz), offset=self.w_offset)
        self.JP_c = ti.field(dtype=ti.f32, shape=(self.wx, self.wy, self.wz), offset=self.w_offset)

        # Properties on particles:
        self.conductivity_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.capacity_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.theta_c_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.theta_s_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.lambda_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.zeta_p = ti.field(dtype=ti.i32, shape=max_particles)
        self.FE_p = ti.Matrix.field(self.d, self.d, dtype=ti.f32, shape=max_particles)
        self.JE_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.JP_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.mu_p = ti.field(dtype=ti.f32, shape=max_particles)
        self.B_p = ti.Matrix.field(self.d, self.d, dtype=ti.f32, shape=max_particles)

        # Fields needed for the latent heat and phase change.
        self.latent_heat_p = ti.field(dtype=ti.f32, shape=max_particles)  # U_p

        # Poisson solvers for pressure and heat.
        self.pressure_solver = PressureSolver(self)
        self.heat_solver = HeatSolver(self)

        self.cubic_neighbors = (4,) * self.d
        self.quadratic_neighbors = (3,) * self.d

    @ti.func
    @override
    def left_hand_offset(self, i: ti.i32, j: ti.i32, k: ti.i32) -> ti.f32:  # pyright: ignore
        # lambda_c approaches infinity for incompressible materials, this way we end up with the
        # usual pressure equation for cells where a lot of incompressible material has accumulated.
        return (self.JP_c[i, j, k] / (self.dt[None] * self.JE_c[i, j, k])) * self.inv_lambda_c[i, j, k]

    @ti.func
    @override
    def right_hand_offset(self, i: ti.i32, j: ti.i32, k: ti.i32) -> ti.f32:  # pyright: ignore
        # JE_c approaches 1 for incompressible materials, this way we end up with the usual
        # pressure equation for cells where a lot of incompressible material has accumulated.
        return (1 - self.JE_c[i, j, k]) / (self.dt[None] * self.JE_c[i, j, k])

    @ti.func
    @override
    def add_particle(self, index: ti.i32, position: ti.template(), geometry: ti.template()):  # pyright: ignore
        self.change_particle_material(index, geometry.material)
        self.velocity_p[index] = geometry.velocity
        self.position_p[index] = position
        self.state_p[index] = State.Active
        self.B_p[index] = ti.Matrix.zero(ti.f32, self.d, self.d)

    @ti.func
    def change_particle_material(self, p: ti.i32, material: ti.template()):  # pyright: ignore
        self.conductivity_p[p] = material.Conductivity
        self.latent_heat_p[p] = material.LatentHeat
        self.temperature_p[p] = 0.0
        self.capacity_p[p] = material.Capacity
        self.theta_c_p[p] = material.Theta_c
        self.theta_s_p[p] = material.Theta_s
        self.lambda_p[p] = material.Lambda
        self.color_p[p] = material.Color
        self.phase_p[p] = material.Phase
        self.mass_p[p] = self.vol_0_p * material.Density
        self.zeta_p[p] = material.Zeta
        self.mu_p[p] = material.Mu
        self.FE_p[p] = ti.Matrix.identity(ti.f32, self.d)
        self.JP_p[p] = 1.0
        self.JE_p[p] = 1.0

    @ti.kernel
    def reset_grids(self):
        for i, j, k in self.mass_x:
            self.conductivity_x[i, j, k] = 0
            self.velocity_x[i, j, k] = 0
            self.volume_x[i, j, k] = 0
            self.mass_x[i, j, k] = 0

        for i, j, k in self.mass_y:
            self.conductivity_y[i, j, k] = 0
            self.velocity_y[i, j, k] = 0
            self.volume_y[i, j, k] = 0
            self.mass_y[i, j, k] = 0

        for i, j, k in self.mass_z:
            self.conductivity_z[i, j, k] = 0
            self.velocity_z[i, j, k] = 0
            self.volume_z[i, j, k] = 0
            self.mass_z[i, j, k] = 0

        for i, j, k in self.mass_c:
            self.temperature_c[i, j, k] = 0
            self.inv_lambda_c[i, j, k] = 0
            self.capacity_c[i, j, k] = 0
            self.mass_c[i, j, k] = 0
            self.JE_c[i, j, k] = 0
            self.JP_c[i, j, k] = 0

    @ti.kernel
    def particle_to_grid(self):
        for p in ti.ndrange(self.n_particles[None]):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            # Compute D^(-1), which equals constant scaling for quadratic/cubic kernels.
            D_inv = 3 * self.inv_dx * self.inv_dx  # Cubic interpolation

            # Now we can convert B_p to C_p with C = B @ (D^(-1))
            C_p = D_inv * self.B_p[p]

            # Evolve deformation gradient:
            self.FE_p[p] += (self.dt[None] * C_p) @ self.FE_p[p]  # pyright: ignore

            # Remove the deviatoric component from the deformation gradient:
            if self.phase_p[p] == Water.Phase:
                # NOTE: 3D:
                self.FE_p[p] = (self.JE_p[p] ** (1 / self.d)) * ti.Matrix.identity(ti.f32, self.d)

            # Clamp singular values to apply plasticity:
            U, sigma, V = ti.svd(self.FE_p[p])
            self.JE_p[p] = 1.0
            for d in ti.ndrange(self.d):
                singular_value = ti.f32(sigma[d, d])
                clamped = ti.f32(sigma[d, d])
                if self.phase_p[p] == Ice.Phase:
                    # Clamp singular values to [1 - theta_c, 1 + theta_s]
                    clamped = max(clamped, 1 - self.theta_c_p[p])
                    clamped = min(clamped, 1 + self.theta_s_p[p])
                self.JP_p[p] *= singular_value / clamped
                self.JE_p[p] *= clamped
                sigma[d, d] = clamped

            # Reconstruct elastic deformation gradient after plasticity
            self.FE_p[p] = U @ sigma @ V.transpose()

            # Apply ice strain hardening by adjusting Lame parameters:
            la, mu = self.lambda_p[p], self.mu_p[p]
            if self.phase_p[p] == Ice.Phase:
                hardening = ti.max(0.1, ti.min(20, ti.exp(self.zeta_p[p] * (1.0 - self.JP_p[p]))))
                la, mu = la * hardening, mu * hardening

            # Eliminate dilational component explicitly [Jiang 2014, Eqn. 8], then
            # compute deviatoric Piola-Kirchhoff stress P(F) [Jiang 2016, Eqn. 52]:
            FE_deviatoric = self.FE_p[p] * (self.JE_p[p] ** (1 / self.d))
            # FE_deviatoric = self.FE_p[p] * (self.JE_p[p] ** (-1 / self.d)) # FIXME: this should be right?!
            U_deviatoric, _, V_deviatoric = ti.svd(FE_deviatoric)
            piola_kirchhoff = FE_deviatoric - (U_deviatoric @ V_deviatoric.transpose())
            piola_kirchhoff = (2 * mu * piola_kirchhoff) @ self.FE_p[p].transpose()  # pyright: ignore

            # Cauchy stress times dt and D_inv:
            cauchy_stress = -self.dt[None] * self.vol_0_p * D_inv * piola_kirchhoff

            # APIC momentum + MLS-MPM stress contribution [Hu et al. 2018, Eqn. 29].
            # TODO: add z fields
            affine = cauchy_stress + self.mass_p[p] * C_p
            affine_x = affine @ ti.Vector([1, 0, 0])  # pyright: ignore
            affine_y = affine @ ti.Vector([0, 1, 0])  # pyright: ignore
            affine_z = affine @ ti.Vector([0, 0, 1])  # pyright: ignore

            # Lower left corner of the interpolation grid:
            base_x = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([1.0, 1.5, 1.5])), dtype=ti.i32)
            base_y = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([1.5, 1.0, 1.5])), dtype=ti.i32)
            base_z = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([1.5, 1.5, 1.0])), dtype=ti.i32)
            base_c = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([2.0, 2.0, 2.0])), dtype=ti.i32)

            # Distance between lower left corner and particle position:
            dist_x = self.position_p[p] * self.inv_dx - ti.cast(base_x, ti.f32) - ti.Vector([0.0, 0.5, 0.5])
            dist_y = self.position_p[p] * self.inv_dx - ti.cast(base_y, ti.f32) - ti.Vector([0.5, 0.0, 0.5])
            dist_z = self.position_p[p] * self.inv_dx - ti.cast(base_z, ti.f32) - ti.Vector([0.5, 0.5, 0.0])
            dist_c = self.position_p[p] * self.inv_dx - ti.cast(base_c, ti.f32) - ti.Vector([0.5, 0.5, 0.5])

            # Cubic kernels:
            w_x = self.compute_cubic_kernel(dist_x)
            w_y = self.compute_cubic_kernel(dist_y)
            w_z = self.compute_cubic_kernel(dist_z)
            w_c = self.compute_cubic_kernel(dist_c)

            velocity_x, velocity_y, velocity_z = self.velocity_p[p][0], self.velocity_p[p][1], self.velocity_p[p][2]
            mass_p = self.mass_p[p]
            for offset in ti.grouped(ti.ndrange(*self.cubic_neighbors)):
                # FIXME: this is not working without static outer loop on macOS?!
                weight_c, weight_x, weight_y, weight_z = 1.0, 1.0, 1.0, 1.0
                for i in ti.ndrange(self.d):
                    weight_c *= w_c[offset[i], i]
                    weight_x *= w_x[offset[i], i]
                    weight_y *= w_y[offset[i], i]
                    weight_z *= w_z[offset[i], i]

                # offset = ti.Vector([i, j, k])
                # weight_c = w_c[i][0] * w_c[j][1] * w_c[k][2]
                # weight_x = w_x[i][0] * w_x[j][1] * w_x[k][2]
                # weight_y = w_y[i][0] * w_y[j][1] * w_y[k][2]
                # weight_z = w_z[i][0] * w_z[j][1] * w_z[k][2]

                dpos_x = ti.cast(offset - dist_x, ti.f32) * self.dx
                dpos_y = ti.cast(offset - dist_y, ti.f32) * self.dx
                dpos_z = ti.cast(offset - dist_z, ti.f32) * self.dx

                # Rasterize to cell centers:
                self.temperature_c[base_c + offset] += weight_c * mass_p * self.temperature_p[p]
                self.inv_lambda_c[base_c + offset] += weight_c * (mass_p / la)
                self.capacity_c[base_c + offset] += weight_c * mass_p * self.capacity_p[p]
                self.mass_c[base_c + offset] += weight_c * mass_p
                self.JE_c[base_c + offset] += weight_c * mass_p * self.JE_p[p]
                self.JP_c[base_c + offset] += weight_c * mass_p * self.JP_p[p]

                # Rasterize to cell faces:
                self.conductivity_x[base_x + offset] += weight_x * mass_p * self.conductivity_p[p]
                self.conductivity_y[base_y + offset] += weight_y * mass_p * self.conductivity_p[p]
                self.conductivity_z[base_z + offset] += weight_z * mass_p * self.conductivity_p[p]
                self.velocity_x[base_x + offset] += weight_x * (mass_p * velocity_x + affine_x @ dpos_x)
                self.velocity_y[base_y + offset] += weight_y * (mass_p * velocity_y + affine_y @ dpos_y)
                self.velocity_z[base_z + offset] += weight_z * (mass_p * velocity_z + affine_z @ dpos_z)
                self.mass_x[base_x + offset] += weight_x * mass_p
                self.mass_y[base_y + offset] += weight_y * mass_p
                self.mass_z[base_z + offset] += weight_z * mass_p

    @ti.kernel
    def momentum_to_velocity(self):
        for i, j, k in self.mass_x:
            if (mass_x := self.mass_x[i, j, k]) > 0:
                self.velocity_x[i, j, k] /= mass_x
                # Everything outside the visible grid belongs to the simulation boundary.
                # We enforce a free-slip boundary condition:
                if (i >= self.n_grid and self.velocity_x[i, j, k] > 0) or (i <= 0 and self.velocity_x[i, j, k] < 0):
                    self.velocity_x[i, j, k] = 0

        for i, j, k in self.mass_y:
            if (mass_y := self.mass_y[i, j, k]) > 0:
                self.velocity_y[i, j, k] /= mass_y
                self.velocity_y[i, j, k] += self.gravity[None] * self.dt[None]
                # Everything outside the visible grid belongs to the simulation boundary.
                # We enforce a free-slip boundary condition:
                if (j >= self.n_grid and self.velocity_y[i, j, k] > 0) or (j <= 0 and self.velocity_y[i, j, k] < 0):
                    self.velocity_y[i, j, k] = 0

        for i, j, k in self.mass_z:
            if (mass_z := self.mass_z[i, j, k]) > 0:
                self.velocity_z[i, j, k] /= mass_z
                # Everything outside the visible grid belongs to the simulation boundary.
                # We enforce a free-slip boundary condition:
                if (k >= self.n_grid and self.velocity_z[i, j, k] > 0) or (k <= 0 and self.velocity_z[i, j, k] < 0):
                    self.velocity_z[i, j, k] = 0

        for i, j, k in self.mass_c:
            if (mass_c := self.mass_c[i, j, k]) > 0:
                self.temperature_c[i, j, k] /= mass_c
                self.inv_lambda_c[i, j, k] /= mass_c
                self.capacity_c[i, j, k] /= mass_c
                self.JE_c[i, j, k] /= mass_c
                self.JP_c[i, j, k] /= mass_c

    @ti.kernel
    def classify_cells(self):
        # A face is colliding if the level set computed by any collision object is negative at the face center.
        # NOTE: collision objects are not implemented, we only care about the simulation boundary right now.

        for i, j, k in self.classification_x:
            # The simulation boundary is always colliding:
            x_face_is_colliding = i >= self.n_grid or i <= 0
            x_face_is_colliding |= j >= self.n_grid or j < 0
            x_face_is_colliding |= k >= self.n_grid or k < 0
            if x_face_is_colliding:
                self.classification_x[i, j, k] = Classification.Colliding
                continue

            # All remaining faces are reset, we only care about colliding faces.
            self.classification_x[i, j, k] = Classification.Unknown

        for i, j, k in self.classification_y:
            # The simulation boundary is always colliding.
            y_face_is_colliding = i >= self.n_grid or i < 0
            y_face_is_colliding |= j >= self.n_grid or j <= 0
            y_face_is_colliding |= k >= self.n_grid or k < 0
            if y_face_is_colliding:
                self.classification_y[i, j, k] = Classification.Colliding
                continue

            # All remaining faces are reset, we only care about colliding faces.
            self.classification_y[i, j, k] = Classification.Unknown

        for i, j, k in self.classification_z:
            # The simulation boundary is always colliding.
            z_face_is_colliding = i >= self.n_grid or i < 0
            z_face_is_colliding |= j >= self.n_grid or j < 0
            z_face_is_colliding |= k >= self.n_grid or k <= 0
            if z_face_is_colliding:
                self.classification_z[i, j, k] = Classification.Colliding
                continue

            # All remaining faces are reset, we only care about colliding faces.
            self.classification_z[i, j, k] = Classification.Unknown

        for i, j, k in self.classification_c:
            # A cell is colliding if all of its surrounding faces are colliding:
            is_colliding = self.classification_x[i + 1, j, k] == Classification.Colliding
            is_colliding &= self.classification_x[i, j, k] == Classification.Colliding
            is_colliding &= self.classification_y[i, j + 1, k] == Classification.Colliding
            is_colliding &= self.classification_y[i, j, k] == Classification.Colliding
            is_colliding &= self.classification_z[i, j, k + 1] == Classification.Colliding
            is_colliding &= self.classification_z[i, j, k] == Classification.Colliding

            if is_colliding:
                self.classification_c[i, j, k] = Classification.Colliding

                # The boundary temperature is recorded for boundary (colliding) cells:
                self.temperature_c[i, j, k] = self.boundary_temperature[None]
                continue

            # A cell is interior if the cell and all of its surrounding faces have mass.
            cell_is_interior = self.mass_c[i, j, k] > 0
            cell_is_interior &= self.mass_x[i, j, k] > 0 and self.mass_x[i + 1, j, k] > 0
            cell_is_interior &= self.mass_y[i, j, k] > 0 and self.mass_y[i, j + 1, k] > 0
            cell_is_interior &= self.mass_z[i, j, k] > 0 and self.mass_z[i, j, k + 1] > 0

            if cell_is_interior:
                self.classification_c[i, j, k] = Classification.Interior
                continue

            # All remaining cells are empty.
            self.classification_c[i, j, k] = Classification.Empty

            # If the free surface is being enforced as a Dirichlet temperature condition,
            # the ambient air temperature is recorded for empty cells.
            self.temperature_c[i, j, k] = self.ambient_temperature[None]

    @ti.kernel
    def compute_volumes(self):
        # FIXME: this seems to be wrong, the paper has a sum over CDFs
        control_volume = 0.5 * (self.dx ** self.d)
        for i, j, k in self.classification_c:
            if self.classification_c[i, j, k] == Classification.Interior:
                self.volume_x[i + 1, j, k] += control_volume
                self.volume_x[i, j, k] += control_volume
                self.volume_y[i, j + 1, k] += control_volume
                self.volume_y[i, j, k] += control_volume
                self.volume_z[i, j, k + 1] += control_volume
                self.volume_z[i, j, k] += control_volume

    @ti.kernel
    def grid_to_particle(self):
        for p in ti.ndrange(self.n_particles[None]):
            # We ignore uninitialized particles:
            if self.state_p[p] == State.Hidden:
                continue

            # Lower left corner of the interpolation grid:
            base_x = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([0.5, 1.0, 1.0])), dtype=ti.i32)
            base_y = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([1.0, 0.5, 1.0])), dtype=ti.i32)
            base_z = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([1.0, 1.0, 0.5])), dtype=ti.i32)
            base_c = ti.floor((self.position_p[p] * self.inv_dx - ti.Vector([1.0, 1.0, 1.0])), dtype=ti.i32)

            # Distance between lower left corner and particle position:
            dist_x = self.position_p[p] * self.inv_dx - ti.cast(base_x, ti.f32) - ti.Vector([0.0, 0.5, 0.5])
            dist_y = self.position_p[p] * self.inv_dx - ti.cast(base_y, ti.f32) - ti.Vector([0.5, 0.0, 0.5])
            dist_z = self.position_p[p] * self.inv_dx - ti.cast(base_z, ti.f32) - ti.Vector([0.5, 0.5, 0.0])
            dist_c = self.position_p[p] * self.inv_dx - ti.cast(base_c, ti.f32) - ti.Vector([0.5, 0.5, 0.5])

            # Quadratic kernels:
            w_c = self.compute_quadratic_kernel(dist_c)
            w_x = self.compute_quadratic_kernel(dist_x)
            w_y = self.compute_quadratic_kernel(dist_y)
            w_z = self.compute_quadratic_kernel(dist_z)

            temperature = 0.0
            velocity = ti.Vector.zero(ti.f32, self.d)
            b_x = ti.Vector.zero(ti.f32, self.d)
            b_y = ti.Vector.zero(ti.f32, self.d)
            b_z = ti.Vector.zero(ti.f32, self.d)
            for offset in ti.grouped(ti.ndrange(*self.quadratic_neighbors)):
                weight_c, weight_x, weight_y, weight_z = 1.0, 1.0, 1.0, 1.0
                for i in ti.ndrange(self.d):
                    weight_c *= w_c[offset[i], i]
                    weight_x *= w_x[offset[i], i]
                    weight_y *= w_y[offset[i], i]
                    weight_z *= w_z[offset[i], i]

                temperature += weight_c * self.temperature_c[base_c + offset]
                velocity_x = weight_x * self.velocity_x[base_x + offset]
                velocity_y = weight_y * self.velocity_y[base_y + offset]
                velocity_z = weight_z * self.velocity_z[base_z + offset]
                velocity += [velocity_x, velocity_y, velocity_z]
                x_dpos = (ti.cast(offset, ti.f32) - dist_x) * self.dx
                y_dpos = (ti.cast(offset, ti.f32) - dist_y) * self.dx
                z_dpos = (ti.cast(offset, ti.f32) - dist_z) * self.dx
                b_x += velocity_x * x_dpos
                b_y += velocity_y * y_dpos
                b_z += velocity_z * z_dpos

            self.B_p[p] = ti.Matrix([[b_x[0], b_y[0], b_z[0]], [b_x[1], b_y[1], b_z[1]], [b_x[2], b_y[2], b_z[2]]])
            self.position_p[p] += self.dt[None] * velocity
            self.velocity_p[p] = velocity

            # Initially, we allow each particle to freely change its temperature according to the heat equation.
            # But whenever the freezing point is reached, any additional temperature change is multiplied by
            # conductivity and mass and added to the buffer, with the particle temperature kept unchanged.
            if (self.phase_p[p] == Ice.Phase) and (temperature >= 0):
                # Ice reached the melting point, additional temperature change is added to heat buffer.
                difference = temperature - self.temperature_p[p]
                self.latent_heat_p[p] += self.conductivity_p[p] * self.mass_p[p] * difference

                # If the heat buffer is full the particle changes its phase to water,
                # everything is then reset according to the new phase.
                if self.latent_heat_p[p] >= Water.LatentHeat:
                    self.change_particle_material(p, Water)

            elif (self.phase_p[p] == Water.Phase) and (temperature < 0):
                # Water particle reached the freezing point, additional temperature change is added to heat buffer.
                difference = temperature - self.temperature_p[p]
                self.latent_heat_p[p] += self.conductivity_p[p] * self.mass_p[p] * difference

                # If the heat buffer is empty the particle changes its phase to ice,
                # everything is then reset according to the new phase.
                if self.latent_heat_p[p] <= Ice.LatentHeat:
                    self.change_particle_material(p, Ice)

            else:
                # Freely change temperature according to heat equation, but clamp temperature for realism.
                self.temperature_p[p] = tm.clamp(temperature, Simulation.MinTemperature, Simulation.MaxTemperature)

    @override
    def substep(self):
        self.reset_grids()
        self.particle_to_grid()
        self.momentum_to_velocity()
        self.classify_cells()
        self.compute_volumes()
        self.pressure_solver.solve()
        self.heat_solver.solve()
        self.grid_to_particle()
