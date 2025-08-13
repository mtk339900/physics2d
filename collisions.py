"""
Collision detection and response system.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import logging
from .primitives import PointMass, Circle

logger = logging.getLogger(__name__)


class CollisionInfo:
    """Information about a collision."""
    
    def __init__(self, obj1: PointMass, obj2: Optional[PointMass], 
                 contact_point: np.ndarray, normal: np.ndarray, 
                 penetration: float, collision_type: str):
        """
        Initialize collision information.
        
        Args:
            obj1: First object in collision
            obj2: Second object in collision (None for static collisions)
            contact_point: Point of contact
            normal: Collision normal (pointing from obj1 to obj2)
            penetration: Penetration depth
            collision_type: Type of collision ('circle-circle', 'point-plane', etc.)
        """
        self.obj1 = obj1
        self.obj2 = obj2
        self.contact_point = contact_point
        self.normal = normal
        self.penetration = penetration
        self.collision_type = collision_type


class CollisionDetector:
    """Collision detection system."""
    
    def __init__(self):
        """Initialize collision detector."""
        self.collision_pairs = []
        
    def detect_collisions(self, objects: List[PointMass], 
                         planes: List[Dict] = None) -> List[CollisionInfo]:
        """
        Detect all collisions in the system.
        
        Args:
            objects: List of physics objects
            planes: List of plane definitions [{'point': [x,y], 'normal': [nx,ny]}]
            
        Returns:
            List of collision information
        """
        collisions = []
        
        # Object-object collisions
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                collision = self._check_object_collision(objects[i], objects[j])
                if collision:
                    collisions.append(collision)
                    
        # Object-plane collisions
        if planes:
            for obj in objects:
                for plane in planes:
                    collision = self._check_plane_collision(obj, plane)
                    if collision:
                        collisions.append(collision)
                        
        return collisions
        
    def _check_object_collision(self, obj1: PointMass, obj2: PointMass) -> Optional[CollisionInfo]:
        """Check collision between two objects."""
        if isinstance(obj1, Circle) and isinstance(obj2, Circle):
            return self._check_circle_circle_collision(obj1, obj2)
        else:
            # Point-circle or point-point collisions
            return self._check_point_collision(obj1, obj2)
            
    def _check_circle_circle_collision(self, circle1: Circle, circle2: Circle) -> Optional[CollisionInfo]:
        """Check collision between two circles."""
        distance_vec = circle2.position - circle1.position
        distance = np.linalg.norm(distance_vec)
        min_distance = circle1.radius + circle2.radius
        
        if distance < min_distance and distance > 1e-10:
            normal = distance_vec / distance
            penetration = min_distance - distance
            contact_point = circle1.position + normal * circle1.radius
            
            return CollisionInfo(circle1, circle2, contact_point, normal, 
                               penetration, 'circle-circle')
        return None
        
    def _check_point_collision(self, obj1: PointMass, obj2: PointMass) -> Optional[CollisionInfo]:
        """Check collision between point masses or point-circle."""
        if isinstance(obj2, Circle):
            obj1, obj2 = obj2, obj1  # Ensure circle is first
            
        if isinstance(obj1, Circle):
            # Point-circle collision
            distance_vec = obj2.position - obj1.position
            distance = np.linalg.norm(distance_vec)
            
            if distance < obj1.radius and distance > 1e-10:
                normal = distance_vec / distance
                penetration = obj1.radius - distance
                contact_point = obj1.position + normal * obj1.radius
                
                return CollisionInfo(obj1, obj2, contact_point, normal,
                                   penetration, 'point-circle')
        else:
            # Point-point collision (very close points)
            distance = np.linalg.norm(obj2.position - obj1.position)
            if distance < 0.01:  # Small threshold for point collision
                if distance > 1e-10:
                    normal = (obj2.position - obj1.position) / distance
                else:
                    normal = np.array([1.0, 0.0])  # Arbitrary direction
                
                return CollisionInfo(obj1, obj2, obj1.position, normal,
                                   0.01 - distance, 'point-point')
        return None
        
    def _check_plane_collision(self, obj: PointMass, plane: Dict) -> Optional[CollisionInfo]:
        """Check collision between object and plane."""
        plane_point = np.array(plane['point'])
        plane_normal = np.array(plane['normal'])
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        
        if isinstance(obj, Circle):
            # Circle-plane collision
            to_center = obj.position - plane_point
            distance_to_plane = np.dot(to_center, plane_normal)
            
            if distance_to_plane < obj.radius:
                penetration = obj.radius - distance_to_plane
                contact_point = obj.position - plane_normal * distance_to_plane
                
                return CollisionInfo(obj, None, contact_point, -plane_normal,
                                   penetration, 'circle-plane')
        else:
            # Point-plane collision
            to_point = obj.position - plane_point
            distance_to_plane = np.dot(to_point, plane_normal)
            
            if distance_to_plane < 0:
                penetration = -distance_to_plane
                contact_point = obj.position - plane_normal * distance_to_plane
                
                return CollisionInfo(obj, None, contact_point, -plane_normal,
                                   penetration, 'point-plane')
        return None


class CollisionResolver:
    """Collision response and resolution system."""
    
    def __init__(self, restitution: float = 0.8, friction: float = 0.3,
                 position_correction: bool = True, correction_factor: float = 0.8):
        """
        Initialize collision resolver.
        
        Args:
            restitution: Coefficient of restitution (bounciness)
            friction: Friction coefficient
            position_correction: Whether to apply position correction
            correction_factor: Position correction factor (Baumgarte stabilization)
        """
        if not (0 <= restitution <= 1):
            raise ValueError("Restitution must be between 0 and 1")
        if friction < 0:
            raise ValueError("Friction must be non-negative")
            
        self.restitution = float(restitution)
        self.friction = float(friction)
        self.position_correction = position_correction
        self.correction_factor = float(correction_factor)
        
    def resolve_collisions(self, collisions: List[CollisionInfo]) -> None:
        """Resolve all collisions with impulse-based response."""
        for collision in collisions:
            self._resolve_collision(collision)
            
    def _resolve_collision(self, collision: CollisionInfo) -> None:
        """Resolve a single collision."""
        obj1 = collision.obj1
        obj2 = collision.obj2
        normal = collision.normal
        
        # Calculate relative velocity at contact point
        if obj2 is not None:
            relative_velocity = obj2.velocity - obj1.velocity
        else:
            relative_velocity = -obj1.velocity
            
        # Relative velocity in collision normal direction
        vel_along_normal = np.dot(relative_velocity, normal)
        
        # Objects separating, no collision response needed
        if vel_along_normal > 0:
            return
            
        # Calculate impulse scalar
        impulse_magnitude = -(1 + self.restitution) * vel_along_normal
        
        if obj2 is not None:
            # Two-body collision
            inv_mass_sum = 1/obj1.mass + 1/obj2.mass
            impulse_magnitude /= inv_mass_sum
            
            impulse = impulse_magnitude * normal
            
            # Apply impulses
            obj1.velocity -= impulse / obj1.mass
            obj2.velocity += impulse / obj2.mass
            
        else:
            # Single body collision with static object
            impulse = impulse_magnitude * normal
            obj1.velocity -= impulse / obj1.mass
            
        # Apply friction
        self._apply_friction(collision, relative_velocity, normal)
        
        # Position correction to reduce penetration
        if self.position_correction and collision.penetration > 1e-6:
            self._apply_position_correction(collision)
            
    def _apply_friction(self, collision: CollisionInfo, relative_velocity: np.ndarray, 
                       normal: np.ndarray) -> None:
        """Apply friction forces."""
        if self.friction == 0:
            return
            
        # Calculate tangential component of relative velocity
        tangent = relative_velocity - np.dot(relative_velocity, normal) * normal
        tangent_magnitude = np.linalg.norm(tangent)
        
        if tangent_magnitude < 1e-6:
            return
            
        tangent = tangent / tangent_magnitude
        
        # Calculate friction impulse
        obj1 = collision.obj1
        obj2 = collision.obj2
        
        friction_impulse_magnitude = self.friction * tangent_magnitude
        
        if obj2 is not None:
            inv_mass_sum = 1/obj1.mass + 1/obj2.mass
            friction_impulse_magnitude /= inv_mass_sum
            
            friction_impulse = friction_impulse_magnitude * tangent
            
            obj1.velocity += friction_impulse / obj1.mass
            obj2.velocity -= friction_impulse / obj2.mass
        else:
            friction_impulse = friction_impulse_magnitude * tangent
            obj1.velocity += friction_impulse / obj1.mass
            
    def _apply_position_correction(self, collision: CollisionInfo) -> None:
        """Apply position correction to reduce penetration."""
        correction = self.correction_factor * collision.penetration * collision.normal
        
        obj1 = collision.obj1
        obj2 = collision.obj2
        
        if obj2 is not None:
            inv_mass_sum = 1/obj1.mass + 1/obj2.mass
            obj1.position -= correction * (1/obj1.mass) / inv_mass_sum
            obj2.position += correction * (1/obj2.mass) / inv_mass_sum
        else:
            obj1.position -= correction


