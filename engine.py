"""
Main physics engine coordinating simulation components.
"""

from typing import List, Dict, Optional, Callable, Any, Union
import numpy as np
import logging
from .primitives import PointMass, RigidBody, Circle
from .integrators import Integrator, EulerIntegrator, SemiImplicitEulerIntegrator, RK4Integrator
from .forces import ForceGenerator, GravityForce, LinearDragForce
from .collisions import CollisionDetector, CollisionResolver, CollisionInfo
from .spatial import SpatialGrid, QuadTree

logger = logging.getLogger(__name__)


class PhysicsEngine:
    """Main physics simulation engine."""
    
    def __init__(self, integrator: str = 'semi_implicit_euler',
                 dt: float = 0.016, bounds: Optional[List[float]] = None,
                 use_spatial_acceleration: bool = True,
                 spatial_type: str = 'grid'):
        """
        Initialize physics engine.
        
        Args:
            integrator: Integration method ('euler', 'semi_implicit_euler', 'rk4')
            dt: Fixed time step
            bounds: Simulation bounds [min_x, min_y, max_x, max_y]
            use_spatial_acceleration: Whether to use spatial partitioning
            spatial_type: Type of spatial acceleration ('grid' or 'quadtree')
            
        Raises:
            ValueError: If dt is non-positive or integrator is invalid
        """
        if dt <= 0:
            raise ValueError("Time step must be positive")
            
        self.dt = float(dt)
        self.time = 0.0
        self.paused = False
        self.objects: List[PointMass] = []
        self.force_generators: List[ForceGenerator] = []
        self.planes: List[Dict] = []
        
        # Set up integrator
        integrator_map = {
            'euler': EulerIntegrator(),
            'semi_implicit_euler': SemiImplicitEulerIntegrator(),
            'rk4': RK4Integrator()
        }
        
        if integrator not in integrator_map:
            raise ValueError(f"Unknown integrator: {integrator}")
        self.integrator = integrator_map[integrator]
        
        # Collision system
        self.collision_detector = CollisionDetector()
        self.collision_resolver = CollisionResolver()
        
        # Spatial acceleration
        self.use_spatial_acceleration = use_spatial_acceleration
        self.spatial_structure = None
        if use_spatial_acceleration and bounds:
            if spatial_type == 'grid':
                cell_size = min(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 20
                self.spatial_structure = SpatialGrid(tuple(bounds), cell_size)
            elif spatial_type == 'quadtree':
                self.spatial_structure = QuadTree(tuple(bounds))
            else:
                logger.warning(f"Unknown spatial type: {spatial_type}, falling back to brute force")
                
        # Statistics and debugging
        self.collision_count = 0
        self.energy_history = []
        self.max_penetration = 0.0
        
        # Default forces
        self.add_force_generator(GravityForce())
        self.add_force_generator(LinearDragForce(0.01))
        
        # Validation parameters
        self.max_safe_dt = 0.1
        self.max_velocity = 1000.0
        
        logger.info(f"Physics engine initialized with {integrator} integrator, dt={dt}")
        
    def add_object(self, obj: PointMass) -> None:
        """Add a physics object to the simulation."""
        self.objects.append(obj)
        logger.debug(f"Added object {obj.id} to simulation")
        
    def remove_object(self, obj_id: str) -> bool:
        """Remove object by ID. Returns True if found and removed."""
        for i, obj in enumerate(self.objects):
            if obj.id == obj_id:
                del self.objects[i]
                logger.debug(f"Removed object {obj_id}")
                return True
        return False
        
    def add_force_generator(self, force_gen: ForceGenerator) -> None:
        """Add a force generator to the simulation."""
        self.force_generators.append(force_gen)
        
    def add_plane(self, point: Union[List[float], np.ndarray], 
                  normal: Union[List[float], np.ndarray]) -> None:
        """Add a static plane for collision detection."""
        plane = {
            'point': np.array(point, dtype=np.float64),
            'normal': np.array(normal, dtype=np.float64)
        }
        self.planes.append(plane)
        logger.debug("Added collision plane")
        
    def set_integrator(self, integrator_name: str) -> None:
        """Change the numerical integrator at runtime."""
        integrator_map = {
            'euler': EulerIntegrator(),
            'semi_implicit_euler': SemiImplicitEulerIntegrator(),
            'rk4': RK4Integrator()
        }
        
        if integrator_name in integrator_map:
            self.integrator = integrator_map[integrator_name]
            logger.info(f"Switched to {integrator_name} integrator")
        else:
            raise ValueError(f"Unknown integrator: {integrator_name}")
            
    def set_collision_parameters(self, restitution: float = None, 
                               friction: float = None) -> None:
        """Update collision response parameters."""
        if restitution is not None:
            if not (0 <= restitution <= 1):
                raise ValueError("Restitution must be between 0 and 1")
            self.collision_resolver.restitution = restitution
            
        if friction is not None:
            if friction < 0:
                raise ValueError("Friction must be non-negative")
            self.collision_resolver.friction = friction
            
    def step(self, dt: Optional[float] = None) -> None:
        """Advance simulation by one time step."""
        if self.paused:
            return
            
        step_dt = dt if dt is not None else self.dt
        
        # Validate time step
        if step_dt > self.max_safe_dt:
            logger.warning(f"Time step {step_dt} exceeds maximum safe value {self.max_safe_dt}")
            
        # Update spatial acceleration structure
        if self.use_spatial_acceleration and self.spatial_structure:
            self.spatial_structure.clear()
            for obj in self.objects:
                self.spatial_structure.insert(obj)
                
        # Apply forces through integrator
        def apply_all_forces(objects: List[PointMass]) -> None:
            for force_gen in self.force_generators:
                force_gen.apply_force(objects)
                
        # Integration step
        self.integrator.integrate(self.objects, step_dt, apply_all_forces)
        
        # Collision detection and response
        collisions = self.collision_detector.detect_collisions(self.objects, self.planes)
        if collisions:
            self.collision_count += len(collisions)
            self.max_penetration = max(self.max_penetration, 
                                     max(c.penetration for c in collisions))
            self.collision_resolver.resolve_collisions(collisions)
            
        # Validate object states
        self._validate_simulation_state()
        
        # Update time
        self.time += step_dt
        
        # Record energy for debugging
        if len(self.energy_history) < 1000:  # Limit history size
            total_energy = sum(obj.get_kinetic_energy() for obj in self.objects)
            if hasattr(self.objects[0], 'get_rotational_energy'):
                total_energy += sum(obj.get_rotational_energy() 
                                  for obj in self.objects 
                                  if isinstance(obj, RigidBody))
            self.energy_history.append(total_energy)
            
    def step_n(self, n_steps: int) -> None:
        """Advance simulation by n steps."""
        for _ in range(n_steps):
            self.step()
            
    def run_for_time(self, duration: float) -> None:
        """Run simulation for specified duration."""
        end_time = self.time + duration
        while self.time < end_time and not self.paused:
            remaining = end_time - self.time
            step_dt = min(self.dt, remaining)
            self.step(step_dt)
            
    def pause(self) -> None:
        """Pause the simulation."""
        self.paused = True
        logger.debug("Simulation paused")
        
    def resume(self) -> None:
        """Resume the simulation."""
        self.paused = False
        logger.debug("Simulation resumed")
        
    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.time = 0.0
        self.collision_count = 0
        self.max_penetration = 0.0
        self.energy_history.clear()
        
        # Reset object states - this would need initial state storage
        # For now, just clear forces
        for obj in self.objects:
            obj.clear_forces()
            if isinstance(obj, RigidBody):
                obj.clear_torques()
                
        logger.info("Simulation reset")
        
    def get_state(self) -> Dict[str, Any]:
        """Get complete simulation state."""
        return {
            'time': self.time,
            'objects': [self._serialize_object(obj) for obj in self.objects],
            'paused': self.paused,
            'collision_count': self.collision_count,
            'max_penetration': self.max_penetration
        }
        
    def _serialize_object(self, obj: PointMass) -> Dict[str, Any]:
        """Serialize object state."""
        state = {
            'id': obj.id,
            'type': obj.__class__.__name__,
            'position': obj.position.tolist(),
            'velocity': obj.velocity.tolist(),
            'mass': obj.mass
        }
        
        if isinstance(obj, RigidBody):
            state.update({
                'orientation': obj.orientation,
                'angular_velocity': obj.angular_velocity,
                'moment_inertia': obj.moment_inertia
            })
            
        if isinstance(obj, Circle):
            state['radius'] = obj.radius
            
        return state
        
    def _validate_simulation_state(self) -> None:
        """Validate simulation state and warn about potential issues."""
        for obj in self.objects:
            # Check for NaN values
            if np.any(np.isnan(obj.position)) or np.any(np.isnan(obj.velocity)):
                logger.error(f"NaN detected in object {obj.id}")
                
            # Check for extreme velocities
            speed = np.linalg.norm(obj.velocity)
            if speed > self.max_velocity:
                logger.warning(f"Object {obj.id} has extreme velocity: {speed}")
                
    def apply_impulse(self, obj_id: str, impulse: Union[List[float], np.ndarray]) -> None:
        """Apply instantaneous impulse to object."""
        obj = next((o for o in self.objects if o.id == obj_id), None)
        if obj:
            impulse_array = np.array(impulse, dtype=np.float64)
            obj.velocity += impulse_array / obj.mass
            logger.debug(f"Applied impulse {impulse} to object {obj_id}")
        else:
            logger.warning(f"Object {obj_id} not found for impulse application")
            
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debugging information."""
        info = {
            'time': self.time,
            'dt': self.dt,
            'object_count': len(self.objects),
            'collision_count': self.collision_count,
            'max_penetration': self.max_penetration,
            'total_energy': sum(obj.get_kinetic_energy() for obj in self.objects),
            'paused': self.paused
        }
        
        if self.spatial_structure:
            info['spatial_debug'] = self.spatial_structure.get_debug_info()
            
        return info

