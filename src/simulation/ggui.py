from src.constants import ColorRGB, State, Simulation
from src.configurations import Configuration
from src.samplers import PoissonDiskSampler
from src.simulation import BaseSimulation
from src.solvers import CollocatedSolver

from typing import Callable

import taichi as ti


class DrawingOption:
    """
    This holds name, state and a callable for drawing a chosen foreground/background.
    """

    def __init__(self, name: str, is_active: bool, call_draw: Callable) -> None:
        self.is_active = is_active
        self.draw = call_draw
        self.name = name


@ti.data_oriented
class GGUI_Simulation(BaseSimulation):
    frame_counter = 0
    def __init__(
        self,
        configurations: list[Configuration],
        sampler: PoissonDiskSampler,
        res: tuple[int, int],
        solver: CollocatedSolver,
        prefix: str,
        name: str,
        initial_configuration: int = 0,
        radius=0.0015,
    ) -> None:
        super().__init__(
            initial_configuration=initial_configuration,
            configurations=configurations,
            radius=radius,
            prefix=prefix,
            sampler=sampler,
            solver=solver,
            name=name,
        )

        # GGUI.
        self.window = ti.ui.Window(name, res, fps_limit=self.fps)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()

        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self.camera.position(0.8, 0.5, 1.65)
        self.camera.lookat(0.5, 0.3, 0.5)
        # self.camera.up(0, 1, 0)
        self.camera.fov(55)
        # self.camera.projection_mode(ti.ui.ProjectionMode.Perspective)
        self.scene.set_camera(self.camera)

        # Fields that hold certain colors, must be update in each draw call.
        self.temperature_colors_p = ti.Vector.field(3, dtype=ti.f32, shape=self.solver.max_particles)
        # TODO: also move the phase colors here, then only update the phase colors when drawing the phase?!

        # Scratch field, will be updated with the visible cells of the offset fields of the solver.
        self.scratch_field = ti.field(dtype=ti.f32, shape=(self.solver.n_grid, self.solver.n_grid))

        # Construct a vector field as a heat map:
        self.heat_map_length = len(ColorRGB.HeatMap)
        self.heat_map = ti.Vector.field(3, dtype=ti.f32, shape=self.heat_map_length)
        for i, color in enumerate(ColorRGB.HeatMap):
            self.heat_map[i] = color

        # Values to control the drawing of the temperature:
        # TODO: these should be moved somewhere else
        self.should_normalize_temperature = False

        # Foreground Options:
        self.foreground_options = [
            # DrawingOption("Temperature", False, self.draw_temperature_p),
            # DrawingOption("Background", False, lambda: None),
            DrawingOption("Phase", True, self.draw_phase_p),
        ]

        # Background Options:
        self.background_options = [
            # DrawingOption("Classification", False, lambda: self.show_contour(self.solver.classification_c)),
            # DrawingOption("Temperature", False, lambda: self.show_contour(self.solver.temperature_c)),
            DrawingOption("Background", True, lambda: self.canvas.set_background_color(ColorRGB.Background)),
        ]

    def show_configurations(self) -> None:
        """
        Show all possible configurations inside own subwindow, choosing one will
        load that configuration and reset the solver.
        """
        prev_configuration_id = self.configuration_id
        with self.gui.sub_window("Configurations", 0.01, 0.01, 0.65, 0.64) as subwindow:
            for i in range(len(self.configurations)):
                name = self.configurations[i].name
                if subwindow.checkbox(name, self.configuration_id == i):
                    self.configuration_id = i
            if self.configuration_id != prev_configuration_id:
                _id = self.configuration_id
                configuration = self.configurations[_id]
                self.load_configuration(configuration)
                self.is_paused = True

    def show_foreground_options(self) -> None:
        """
        Show the foreground drawing options as checkboxes inside own subwindow.
        """
        with self.gui.sub_window("Foreground", 0.67, 0.01, 0.32, 0.16) as subwindow:
            for option in self.foreground_options:
                if subwindow.checkbox(option.name, option.is_active):
                    for _option in self.foreground_options:
                        _option.is_active = False
                    option.is_active = True

    def show_background_options(self) -> None:
        """
        Show the background drawing options as checkboxes inside own subwindow.
        """
        with self.gui.sub_window("Background", 0.67, 0.18, 0.32, 0.16) as subwindow:
            for option in self.background_options:
                if subwindow.checkbox(option.name, option.is_active):
                    for _option in self.background_options:
                        _option.is_active = False
                    option.is_active = True

    def show_parameters(self) -> None:
        """
        Show all parameters in the subwindow, the user can then adjust these values
        with sliders which will update the correspoding value in the solver.
        """
        with self.gui.sub_window("Parameters", 0.01, 0.66, 0.98, 0.33) as subwindow:
            self.solver.ambient_temperature[None] = subwindow.slider_int(
                text="Ambient Temperature",
                old_value=int(self.solver.ambient_temperature[None]),  # pyright: ignore
                minimum=-273,
                maximum=273,
            )
            self.solver.boundary_temperature[None] = subwindow.slider_int(
                text="Boundary Temperature",
                old_value=int(self.solver.boundary_temperature[None]),  # pyright: ignore
                minimum=-273,
                maximum=273,
            )
            self.solver.gravity[None] = subwindow.slider_float(
                text="Gravity",
                old_value=float(self.solver.gravity[None]),  # pyright: ignore
                minimum=0,
                maximum=-9.81,
            )
            # NOTE: dt needs to be scaled, otherwise the precision of slider_float is not enough
            self.solver.dt[None] = (
                subwindow.slider_float(
                    text="10 * dt",
                    old_value=self.solver.dt[None] * 10,  # pyright: ignore
                    minimum=1e-3,
                    maximum=3e-2,
                )
                / 10
            )

    def show_buttons(self) -> None:
        """
        Show a set of buttons in the subwindow, this mainly holds functions to control the simulation.
        """
        with self.gui.sub_window("Settings", 0.67, 0.35, 0.32, 0.3) as subwindow:
            if subwindow.button(" Stop recording  " if self.should_write_to_disk else " Start recording "):
                # This button toggles between saving frames and not saving frames.
                self.should_write_to_disk = not self.should_write_to_disk
                if self.should_write_to_disk:
                    self.dump_frames()
                else:
                    self.create_video()
            if subwindow.button(" Reset Particles "):
                self.reset()
            if subwindow.button(" Start Simulation"):
                self.is_paused = False

            self.should_create_video = subwindow.checkbox("Create Video", self.should_create_video)
            self.should_create_gif = subwindow.checkbox("Create GIF", self.should_create_gif)

    def show_settings(self) -> None:
        """
        Show settings in a GGUI subwindow, this should be called once per generated frames
        and will only show these settings if the simulation is paused at the moment.
        """
        if not self.is_paused or not self.should_show_settings:
            self.is_showing_settings = False
            return  # don't bother

        self.is_showing_settings = True
        self.show_foreground_options()
        self.show_background_options()
        self.show_configurations()
        self.show_parameters()
        self.show_buttons()

    def handle_events(self) -> None:
        """
        Handle key presses arising from window events.
        """
        if self.window.get_event(ti.ui.PRESS):
            if self.window.event.key == "r":
                self.reset()
            elif self.window.event.key in ["h"]:
                self.should_show_settings = not self.should_show_settings
            elif self.window.event.key in [ti.GUI.BACKSPACE, "s"]:
                self.should_write_to_disk = not self.should_write_to_disk
                if self.should_write_to_disk:
                    self.dump_frames()
                else:
                    self.create_video()
            elif self.window.event.key in [ti.GUI.SPACE, "p"]:
                self.is_paused = not self.is_paused
            elif self.window.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                self.window.running = False  # Stop the simulation

    @ti.kernel
    def update_temperature_p(self):
        max_temperature = Simulation.MaxTemperature
        min_temperature = Simulation.MinTemperature

        # Get min, max values for normalization:
        if self.should_normalize_temperature:
            for p in self.temperature_colors_p:
                if self.solver.state_p[p] == State.Hidden:
                    continue  # ignore uninitialized particles
                temperature = self.solver.temperature_p[p]
                if temperature > max_temperature:
                    max_temperature = temperature
                elif temperature < min_temperature:
                    min_temperature = temperature

        # Affine combination of the colors based on min, max values and the temperature:
        max_index, min_index = self.heat_map_length - 1, 0
        factor = ti.cast(max_index, ti.f32)
        for p in self.temperature_colors_p:
            if self.solver.state_p[p] == State.Hidden:
                continue  # ignore uninitialized particles
            t = self.solver.temperature_p[p]
            a = (t - min_temperature) / (max_temperature - min_temperature)
            color1 = self.heat_map[ti.max(min_index, ti.floor(factor * a, ti.i8))]
            color2 = self.heat_map[ti.min(max_index, ti.ceil(factor * a, ti.i8))]
            self.temperature_colors_p[p] = ((1 - a) * color1) + (a * color2)

    def draw_temperature_p(self) -> None:
        """
        Draw the temperature for each particle.
        """
        # self.update_temperature_p()
        self.canvas.circles(
            per_vertex_color=self.temperature_colors_p,
            centers=self.solver.position_p,
            radius=self.radius,
        )

    def draw_phase_p(self) -> None:
        """
        Draw the phase for each particle.
        """
        # self.canvas.circles(
        #     per_vertex_color=self.solver.color_p,
        #     centers=self.solver.position_p,
        #     radius=self.radius,
        # )
        self.scene.particles(per_vertex_color=self.solver.color_p, centers=self.solver.position_p, radius=self.radius)

    @ti.kernel
    def update_scratch_field(self, scalar_field: ti.template()):  # pyright: ignore
        for i, j in ti.ndrange(self.solver.n_grid, self.solver.n_grid):
            self.scratch_field[i, j] = scalar_field[i, j]

    def show_contour(self, scalar_field) -> None:
        """
        Show the contour of a given scalar field.
        """
        # Because the fields of the solver are offset to move the boundary beyond the visible
        # portion of the window, we need to remove the invisible parts, otherwise the contour
        # will be offset.
        self.update_scratch_field(scalar_field)
        self.canvas.contour(self.scratch_field, cmap_name="magma", normalize=True)

    def render(self) -> None:
        """Render the simulation."""
        # Draw chosen foreground/background, NOTE: foreground must be drawn last.
        for option in self.background_options + self.foreground_options:
            if option.is_active:
                option.draw()

        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
        self.scene.point_light(pos=(0.5, 1.0, 0.5), color=(0.8, 0.8, 0.8))
        self.scene.point_light(pos=(0.5, 1.0, 1.5), color=(0.8, 0.8, 0.8))
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.8, 0.8, 0.8))
        self.canvas.scene(self.scene)  # NOTE: 3D

        if self.should_write_to_disk and not self.is_paused and not self.is_showing_settings:
            self.video_manager.write_frame(self.window.get_image_buffer_as_numpy())
        
        #n_pts = self.solver.n_particles.to_numpy().item()
        #if n_pts > 0:
        #    p_pos = self.solver.position_p.to_numpy()
        #    writer = ti.tools.PLYWriter(n_pts)
        #    writer.add_vertex_pos(p_pos[:, 0], p_pos[:, 1], p_pos[:, 2])
        #    writer.export_frame_ascii(self.frame_counter, "export.ply")
        #    self.frame_counter += 1

        self.window.show()

    def run(self) -> None:
        """Run the simulation."""
        while self.window.running:
            self.handle_events()
            self.show_settings()
            if not self.is_paused:
                self.substep()
            self.render()
