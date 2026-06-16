from src.samplers import PoissonDiskSampler
from src.configurations import Configuration
from src.simulation import BaseSimulation
from src.constants import ColorHEX, Water
from src.solvers import CollocatedSolver

import taichi as ti


@ti.data_oriented
class GUI_Simulation(BaseSimulation):
    frame_counter = 0

    def __init__(
        self,
        configurations: list[Configuration],
        sampler: PoissonDiskSampler,
        solver: CollocatedSolver,
        radius: float,
        prefix: str,
        name: str,
        res: int,
        initial_configuration: int = 0,
    ) -> None:
        """Constructs a  GUI renderer, this advances the MLS-MPM solver and renders the updated particle positions.
        ---
        Parameters:
            name: string displayed at the top of the window
            res: tuple holding window width and height
            solver: the MLS-MPM solver
            configuration: the one configuration for the solver
        """
        super().__init__(
            initial_configuration=initial_configuration,
            configurations=configurations,
            prefix=prefix,
            sampler=sampler,
            radius=radius,
            solver=solver,
            name=name,
        )

        # GUI.
        self.gui = ti.GUI(name, res=res, background_color=ColorHEX.Background)

    def render(self) -> None:
        """Render the simulation."""
        indices = [0 if p == Water.Phase else 1 for p in self.solver.phase_p.to_numpy()]
        position = self.solver.position_p.to_numpy()
        # position
        palette = [ColorHEX.Water, ColorHEX.Ice]
        radius = self.radius * 1000
        self.gui.circles(position, radius, palette=palette, palette_indices=indices)  # pyright: ignore

        writer = ti.tools.PLYWriter(num_vertices=len(position))
        writer.add_vertex_pos(position[:, 0], position[:, 1], position[:, 2])
        writer.export_frame_ascii(self.frame_counter, "export.ply")
        print('Exported PLY successfully!')
        self.frame_counter += 1

        self.gui.show()

    def run(self) -> None:
        """Run the simulation."""
        while self.gui.running:
            if self.gui.get_event(ti.GUI.PRESS):
                if self.gui.event.key == "r":  # pyright: ignore
                    self.reset()
                elif self.gui.event.key == ti.GUI.SPACE:  # pyright: ignore
                    self.is_paused = not self.is_paused
                elif self.gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:  # pyright: ignore
                    break
            if not self.is_paused:
                self.substep()
            self.render()
