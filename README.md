# 2D Physics Simulator

A production-ready 2D physics engine with visualization capabilities built in Python.

## Features

### Core Physics Engine
- Support for point masses and rigid bodies (circles)
- Newtonian mechanics with force integration
- Multiple numerical integrators: Euler, Semi-implicit Euler, and RK4
- Configurable time stepping with stability checks

### Forces and Interactions  
- Gravitational force (configurable direction and magnitude)
- Linear drag/air resistance
- Spring forces with damping (Hooke's law)
- Constant external forces
- Impulse application

### Collision System
- Circle-circle, point-circle, and point-plane collision detection
- Configurable restitution (bounciness) and friction
- Position correction for penetration resolution
- Baumgarte stabilization for numerical stability

### Constraints
- Distance constraints between objects
- Fixed joints (to world or between objects)
- Iterative constraint solver with convergence checking

### Spatial Acceleration
- Uniform spatial grid partitioning
- QuadTree implementation
- Automatic fallback to brute-force for small scenes
- Configurable acceleration structures

### Visualization
- Real-time matplotlib animation
- Velocity vector display
- Object trails/paths
- Debug overlays (spatial grid visualization)
- Configurable rendering options
- Frame export for offline analysis

### Configuration & I/O
- JSON and YAML configuration file support
- Simulation state save/load
- Time-series data export (CSV/JSON)
- Comprehensive parameter validation

### Debugging & Monitoring
- Energy conservation tracking
- Collision statistics
- Penetration depth monitoring
- Configurable logging with multiple verbosity levels
- Runtime performance metrics

## Installation

Requires Python 3.7+ with the following packages:
```bash
pip install numpy matplotlib
# Optional: for YAML config support
pip install pyyaml
```

## Usage

### Basic Usage

```python
from physics2d import PhysicsEngine, Circle, Visualizer

# Create engine
engine = PhysicsEngine(dt=0.016, bounds=[-10, -10, 10, 10])

# Add objects
ball1 = Circle(position=[0, 5], radius=0.5, velocity=[2, 0], mass=1.0)
ball2 = Circle(position=[3, 0], radius=0.8, velocity=[-1, 1], mass=2.0)
engine.add_object(ball1)
engine.add_object(ball2)

# Add ground plane
engine.add_plane(point=[0, -5], normal=[0, 1])

# Create visualizer and run
viz = Visualizer(engine, xlim=(-10, 10), ylim=(-6, 8))
viz.animate()
```

### Configuration Files

Create simulation scenarios using JSON/YAML:

```json
{
  "engine": {
    "integrator": "semi_implicit_euler",
    "dt": 0.016,
    "bounds": [-10, -5, 10, 5],
    "use_spatial_acceleration": true
  },
  "collision": {
    "restitution": 0.8,
    "friction": 0.3
  },
  "objects": [
    {
      "type": "Circle",
      "id": "ball1",
      "position": [0, 3],
      "velocity": [2, 0],
      "radius": 0.5,
      "mass": 1.0
    }
  ],
  "forces": {
    "gravity": {"vector": [0, -9.81]},
    "drag": {"coefficient": 0.1},
    "springs": [
      {
        "obj1_id": "ball1",
        "obj2_id": "ball2", 
        "spring_constant": 50.0,
        "rest_length": 2.0,
        "damping": 1.0
      }
    ]
  },
  "planes": [
    {"point": [0, -4], "normal": [0, 1]}
  ]
}
```

Load and run:
```python
from physics2d.io import ConfigLoader

config = ConfigLoader.load_config("simulation.json")
engine = ConfigLoader.create_engine_from_config(config)
```

### Advanced Features

```python
# Switch integrators at runtime
engine.set_integrator('rk4')

# Apply impulses
engine.apply_impulse('ball1', [10, 5])

# Export simulation data
from physics2d.io import StateExporter
exporter = StateExporter(engine)

# Record states during simulation
for i in range(1000):
    engine.step()
    if i % 10 == 0:  # Record every 10th step
        exporter.record_state()

# Export time-series data
exporter.export_time_series("simulation_data.csv", format="csv")
```

### Constraints

```python
from physics2d.constraints import DistanceConstraint, ConstraintSolver

# Create constraint system
solver = ConstraintSolver()

# Add distance constraint
constraint = DistanceConstraint("obj1", "obj2", target_distance=3.0)
solver.add_constraint(constraint)

# Solve constraints each step (integrate into engine if needed)
solver.solve_constraints(engine.objects)
```

## Architecture

The simulator is organized into clear modules:

- `primitives.py` - Physics object classes (PointMass, RigidBody, Circle)
- `integrators.py` - Numerical integration schemes
- `forces.py` - Force generator classes  
- `collisions.py` - Collision detection and response
- `constraints.py` - Constraint system for joints/connections
- `spatial.py` - Spatial partitioning for performance
- `engine.py` - Main physics engine coordination
- `visualization.py` - Matplotlib-based rendering
- `io.py` - Configuration loading and data export

## Performance Considerations

- **Spatial Acceleration**: Automatically enabled for >50 objects
- **Integration Method**: Semi-implicit Euler recommended for most scenarios, RK4 for high accuracy needs
- **Time Step**: Default 0.016s (60 FPS), reduce for stiff systems
- **Collision Optimization**: Spatial grid cell size auto-calculated, can be tuned manually

## Numerical Stability

The engine includes several stability features:

- Position correction for collision penetration
- Velocity clamping for extreme speeds
- Time step validation and warnings
- Energy conservation monitoring
- NaN detection and error reporting

## Validation and Error Handling

- Parameter validation with informative error messages
- Stability warnings for large time steps or stiff springs
- Graceful degradation when spatial acceleration fails
- Comprehensive logging system with configurable verbosity

## Extending the System

The modular design allows easy extension:

- Add new force generators by subclassing `ForceGenerator`
- Implement custom integrators via the `Integrator` interface
- Create new object types by extending `PointMass` or `RigidBody`
- Add constraint types by subclassing `Constraint`

## License

This physics simulator is provided as a complete, production-ready implementation for educational and research purposes.


