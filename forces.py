"""
Force generators for the physics simulation.
"""

from typing import List, Tuple, Union
import numpy as np
from .primitives import PointMass, RigidBody


class ForceGenerator:
    """Base class for force generators."""
    
    def apply_force(self, objects: List[PointMass]) -> None:
        """Apply forces to objects."""
        raise NotImplementedError


class GravityForce(ForceGenerator):
    """Constant gravitational force."""
    
    def __init__(self, gravity: Union[List[float], np.ndarray] = None):
        """
        Initialize gravity force.
        
        Args:
            gravity: Gravity vector [gx, gy], defaults to [0, -9.81]
        """
        self.gravity = np.array(gravity if gravity is not None else [0.0, -9.81], dtype=np.float64)
        
    def apply_force(self, objects: List[PointMass]) -> None:
        """Apply gravitational force to all objects."""
        for obj in objects:
            obj.add_force(obj.mass * self.gravity)


class LinearDragForce(ForceGenerator):
    """Linear drag force: F = -k * v"""
    
    def __init__(self, drag_coefficient: float = 0.1):
        """
        Initialize linear drag force.
        
        Args:
            drag_coefficient: Drag coefficient (k)
        """
        self.drag_coefficient = float(drag_coefficient)
        
    def apply_force(self, objects: List[PointMass]) -> None:
        """Apply linear drag force to all objects."""
        for obj in objects:
            drag_force = -self.drag_coefficient * obj.velocity
            obj.add_force(drag_force)


class SpringForce(ForceGenerator):
    """Spring force between two objects: F = -k * (|r| - rest_length) * r_hat"""
    
    def __init__(self, obj1_id: str, obj2_id: str, spring_constant: float, 
                 rest_length: float = 0.0, damping: float = 0.0):
        """
        Initialize spring force between two objects.
        
        Args:
            obj1_id: ID of first object
            obj2_id: ID of second object  
            spring_constant: Spring constant (k)
            rest_length: Rest length of spring
            damping: Damping coefficient for velocity-dependent damping
        """
        self.obj1_id = obj1_id
        self.obj2_id = obj2_id
        self.spring_constant = float(spring_constant)
        self.rest_length = float(rest_length)
        self.damping = float(damping)
        
    def apply_force(self, objects: List[PointMass]) -> None:
        """Apply spring force between connected objects."""
        obj1 = next((obj for obj in objects if obj.id == self.obj1_id), None)
        obj2 = next((obj for obj in objects if obj.id == self.obj2_id), None)
        
        if obj1 is None or obj2 is None:
            return
            
        # Vector from obj1 to obj2
        r_vec = obj2.position - obj1.position
        distance = np.linalg.norm(r_vec)
        
        if distance < 1e-10:  # Avoid division by zero
            return
            
        r_hat = r_vec / distance
        
        # Spring force magnitude
        force_magnitude = self.spring_constant * (distance - self.rest_length)
        
        # Damping force (proportional to relative velocity along spring direction)
        if self.damping > 0:
            relative_velocity = obj2.velocity - obj1.velocity
            relative_vel_along_spring = np.dot(relative_velocity, r_hat)
            damping_force_magnitude = self.damping * relative_vel_along_spring
            force_magnitude += damping_force_magnitude
            
        spring_force = force_magnitude * r_hat
        
        # Apply equal and opposite forces
        obj1.add_force(spring_force)
        obj2.add_force(-spring_force)


class ConstantForce(ForceGenerator):
    """Constant external force on specific objects."""
    
    def __init__(self, force: Union[List[float], np.ndarray], object_ids: List[str] = None):
        """
        Initialize constant force.
        
        Args:
            force: Constant force vector [fx, fy]
            object_ids: List of object IDs to apply force to, None for all objects
        """
        self.force = np.array(force, dtype=np.float64)
        self.object_ids = set(object_ids) if object_ids else None
        
    def apply_force(self, objects: List[PointMass]) -> None:
        """Apply constant force to specified objects."""
        for obj in objects:
            if self.object_ids is None or obj.id in self.object_ids:
                obj.add_force(self.force)


