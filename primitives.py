"""
Physics objects and primitives for the 2D physics simulator.
"""

from typing import List, Tuple, Optional, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PointMass:
    """A point mass object with position, velocity, and mass."""
    
    def __init__(self, position: Union[List[float], np.ndarray], 
                 velocity: Union[List[float], np.ndarray] = None,
                 mass: float = 1.0, object_id: Optional[str] = None):
        """
        Initialize a point mass.
        
        Args:
            position: Initial position [x, y]
            velocity: Initial velocity [vx, vy], defaults to [0, 0]
            mass: Mass of the object, must be positive
            object_id: Optional identifier for the object
            
        Raises:
            ValueError: If mass is non-positive
        """
        if mass <= 0:
            raise ValueError("Mass must be positive")
            
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity if velocity is not None else [0.0, 0.0], dtype=np.float64)
        self.mass = float(mass)
        self.force = np.zeros(2, dtype=np.float64)
        self.id = object_id or f"pm_{id(self)}"
        
        # For visualization trails
        self.trail_positions = []
        self.max_trail_length = 50
        
    def add_force(self, force: Union[List[float], np.ndarray]) -> None:
        """Add a force to the accumulated force for this object."""
        self.force += np.array(force, dtype=np.float64)
        
    def clear_forces(self) -> None:
        """Clear accumulated forces."""
        self.force.fill(0.0)
        
    def update_trail(self) -> None:
        """Update position trail for visualization."""
        self.trail_positions.append(self.position.copy())
        if len(self.trail_positions) > self.max_trail_length:
            self.trail_positions.pop(0)
            
    def get_kinetic_energy(self) -> float:
        """Calculate kinetic energy: 0.5 * m * v^2"""
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)


class RigidBody(PointMass):
    """A rigid body with orientation and angular velocity."""
    
    def __init__(self, position: Union[List[float], np.ndarray],
                 velocity: Union[List[float], np.ndarray] = None,
                 mass: float = 1.0,
                 orientation: float = 0.0,
                 angular_velocity: float = 0.0,
                 moment_inertia: float = 1.0,
                 object_id: Optional[str] = None):
        """
        Initialize a rigid body.
        
        Args:
            position: Initial position [x, y]
            velocity: Initial velocity [vx, vy]
            mass: Mass of the object
            orientation: Initial orientation in radians
            angular_velocity: Initial angular velocity in rad/s
            moment_inertia: Moment of inertia for rotation
            object_id: Optional identifier
        """
        super().__init__(position, velocity, mass, object_id)
        self.orientation = float(orientation)
        self.angular_velocity = float(angular_velocity)
        self.moment_inertia = float(moment_inertia)
        self.torque = 0.0
        self.id = object_id or f"rb_{id(self)}"
        
    def add_torque(self, torque: float) -> None:
        """Add torque to the accumulated torque."""
        self.torque += torque
        
    def clear_torques(self) -> None:
        """Clear accumulated torques."""
        self.torque = 0.0
        
    def get_rotational_energy(self) -> float:
        """Calculate rotational kinetic energy: 0.5 * I * ω^2"""
        return 0.5 * self.moment_inertia * self.angular_velocity**2
        
    def get_total_energy(self) -> float:
        """Get total kinetic energy (translational + rotational)."""
        return self.get_kinetic_energy() + self.get_rotational_energy()


class Circle(RigidBody):
    """A circular rigid body for collision detection."""
    
    def __init__(self, position: Union[List[float], np.ndarray],
                 radius: float,
                 velocity: Union[List[float], np.ndarray] = None,
                 mass: float = 1.0,
                 orientation: float = 0.0,
                 angular_velocity: float = 0.0,
                 object_id: Optional[str] = None):
        """
        Initialize a circular rigid body.
        
        Args:
            position: Initial position [x, y]
            radius: Radius of the circle
            velocity: Initial velocity [vx, vy]
            mass: Mass of the object
            orientation: Initial orientation in radians
            angular_velocity: Initial angular velocity in rad/s
            object_id: Optional identifier
            
        Raises:
            ValueError: If radius is non-positive
        """
        if radius <= 0:
            raise ValueError("Radius must be positive")
            
        # For a solid disk: I = 0.5 * m * r^2
        moment_inertia = 0.5 * mass * radius**2
        
        super().__init__(position, velocity, mass, orientation, 
                        angular_velocity, moment_inertia, object_id)
        self.radius = float(radius)
        self.id = object_id or f"circle_{id(self)}"
