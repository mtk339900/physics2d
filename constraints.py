"""
Constraint system for connecting objects with joints and springs.
"""

from typing import List, Dict, Optional, Any
import numpy as np
import logging
from .primitives import PointMass, RigidBody

logger = logging.getLogger(__name__)


class Constraint:
    """Base class for constraints between objects."""
    
    def __init__(self, obj1_id: str, obj2_id: Optional[str] = None):
        """
        Initialize constraint.
        
        Args:
            obj1_id: ID of first object
            obj2_id: ID of second object (None for world constraints)
        """
        self.obj1_id = obj1_id
        self.obj2_id = obj2_id
        self.enabled = True
        
    def apply_constraint(self, objects: List[PointMass]) -> None:
        """Apply constraint forces/corrections."""
        if not self.enabled:
            return
        raise NotImplementedError
        
    def get_constraint_info(self) -> Dict[str, Any]:
        """Get constraint information for debugging."""
        return {
            'type': self.__class__.__name__,
            'obj1_id': self.obj1_id,
            'obj2_id': self.obj2_id,
            'enabled': self.enabled
        }


class DistanceConstraint(Constraint):
    """Distance constraint maintaining fixed distance between objects."""
    
    def __init__(self, obj1_id: str, obj2_id: str, target_distance: float, 
                 stiffness: float = 1.0):
        """
        Initialize distance constraint.
        
        Args:
            obj1_id: ID of first object
            obj2_id: ID of second object
            target_distance: Target distance to maintain
            stiffness: Constraint stiffness (0-1)
        """
        super().__init__(obj1_id, obj2_id)
        self.target_distance = float(target_distance)
        self.stiffness = float(stiffness)
        
    def apply_constraint(self, objects: List[PointMass]) -> None:
        """Apply distance constraint via position correction."""
        obj1 = next((obj for obj in objects if obj.id == self.obj1_id), None)
        obj2 = next((obj for obj in objects if obj.id == self.obj2_id), None)
        
        if obj1 is None or obj2 is None:
            return
            
        # Calculate current distance
        delta = obj2.position - obj1.position
        current_distance = np.linalg.norm(delta)
        
        if current_distance < 1e-10:
            return
            
        # Calculate correction
        error = current_distance - self.target_distance
        correction_magnitude = error * self.stiffness * 0.5
        correction = (delta / current_distance) * correction_magnitude
        
        # Apply position corrections based on mass ratio
        total_inv_mass = 1/obj1.mass + 1/obj2.mass
        obj1_correction = correction * (1/obj1.mass) / total_inv_mass
        obj2_correction = -correction * (1/obj2.mass) / total_inv_mass
        
        obj1.position += obj1_correction
        obj2.position += obj2_correction


class FixedJoint(Constraint):
    """Fixed joint constraint maintaining relative positions."""
    
    def __init__(self, obj1_id: str, obj2_id: Optional[str], 
                 local_anchor1: List[float], local_anchor2: Optional[List[float]] = None,
                 stiffness: float = 1.0):
        """
        Initialize fixed joint.
        
        Args:
            obj1_id: ID of first object
            obj2_id: ID of second object (None for world joint)
            local_anchor1: Anchor point in obj1's local space
            local_anchor2: Anchor point in obj2's local space (or world space if obj2 is None)
            stiffness: Joint stiffness
        """
        super().__init__(obj1_id, obj2_id)
        self.local_anchor1 = np.array(local_anchor1, dtype=np.float64)
        self.local_anchor2 = np.array(local_anchor2 if local_anchor2 else [0, 0], dtype=np.float64)
        self.stiffness = float(stiffness)
        
    def apply_constraint(self, objects: List[PointMass]) -> None:
        """Apply fixed joint constraint."""
        obj1 = next((obj for obj in objects if obj.id == self.obj1_id), None)
        if obj1 is None:
            return
            
        # Calculate world anchor positions
        if isinstance(obj1, RigidBody):
            cos_theta = np.cos(obj1.orientation)
            sin_theta = np.sin(obj1.orientation)
            rotation_matrix = np.array([[cos_theta, -sin_theta],
                                       [sin_theta, cos_theta]])
            world_anchor1 = obj1.position + rotation_matrix @ self.local_anchor1
        else:
            world_anchor1 = obj1.position + self.local_anchor1
            
        if self.obj2_id is not None:
            # Joint between two objects
            obj2 = next((obj for obj in objects if obj.id == self.obj2_id), None)
            if obj2 is None:
                return
                
            if isinstance(obj2, RigidBody):
                cos_theta = np.cos(obj2.orientation)
                sin_theta = np.sin(obj2.orientation)
                rotation_matrix = np.array([[cos_theta, -sin_theta],
                                           [sin_theta, cos_theta]])
                world_anchor2 = obj2.position + rotation_matrix @ self.local_anchor2
            else:
                world_anchor2 = obj2.position + self.local_anchor2
        else:
            # Joint to world
            world_anchor2 = self.local_anchor2
            
        # Calculate position error
        error = world_anchor1 - world_anchor2
        correction = -error * self.stiffness
        
        if self.obj2_id is not None:
            obj2 = next((obj for obj in objects if obj.id == self.obj2_id), None)
            total_inv_mass = 1/obj1.mass + 1/obj2.mass
            
            obj1_correction = correction * (1/obj1.mass) / total_inv_mass
            obj2_correction = -correction * (1/obj2.mass) / total_inv_mass
            
            obj1.position += obj1_correction
            obj2.position += obj2_correction
        else:
            # Only obj1 moves for world joint
            obj1.position += correction / obj1.mass


class ConstraintSolver:
    """Solver for constraint systems."""
    
    def __init__(self, max_iterations: int = 10, tolerance: float = 1e-6):
        """
        Initialize constraint solver.
        
        Args:
            max_iterations: Maximum solver iterations per step
            tolerance: Convergence tolerance
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.constraints: List[Constraint] = []
        
    def add_constraint(self, constraint: Constraint) -> None:
        """Add constraint to solver."""
        self.constraints.append(constraint)
        logger.debug(f"Added constraint: {constraint.get_constraint_info()}")
        
    def remove_constraint(self, constraint: Constraint) -> bool:
        """Remove constraint from solver."""
        try:
            self.constraints.remove(constraint)
            logger.debug(f"Removed constraint: {constraint.get_constraint_info()}")
            return True
        except ValueError:
            return False
            
    def solve_constraints(self, objects: List[PointMass]) -> int:
        """
        Solve all constraints iteratively.
        
        Args:
            objects: List of physics objects
            
        Returns:
            Number of iterations performed
        """
        for iteration in range(self.max_iterations):
            max_error = 0.0
            
            for constraint in self.constraints:
                if constraint.enabled:
                    # Store positions before constraint
                    old_positions = [obj.position.copy() for obj in objects]
                    
                    # Apply constraint
                    constraint.apply_constraint(objects)
                    
                    # Calculate position change magnitude
                    for i, obj in enumerate(objects):
                        error = np.linalg.norm(obj.position - old_positions[i])
                        max_error = max(max_error, error)
                        
            # Check convergence
            if max_error < self.tolerance:
                return iteration + 1
                
        logger.debug(f"Constraint solver reached max iterations ({self.max_iterations})")
        return self.max_iterations
        
    def get_solver_info(self) -> Dict[str, Any]:
        """Get solver information for debugging."""
        return {
            'constraint_count': len(self.constraints),
            'enabled_constraints': sum(1 for c in self.constraints if c.enabled),
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance
        }
