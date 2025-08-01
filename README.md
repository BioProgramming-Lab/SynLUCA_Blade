# SynLUCA_Blade

This project is a simulation program composed of four main Python files.

---

### **Core Components**

* `Container.py`: Defines the container's mesh. Run this file to preview the mesh based on the specified resolution and `shape.txt` file.

* `Diffusion.py`: Calculates the diffusion properties between meshes.

* `Reactions.py`: Defines the biochemical reactions. Run this file to simulate the MinDE skeleton model. It calls `Container.py`, `Diffusion.py`, and `Animation.py`. The simulation results are automatically saved to `MinDE_solution.pkl` and animated as `concentration_evolution.mp4`.

* `Animation.py`: Configures the animation. Run this file to modify the animation style without re-running the full simulation. It can also be used to verify the conservation of matter, which serves as a qualification check for the simulation.