"""
Numerical integration schemes for physics simulation.
"""

from typing import List, Callable
import numpy as np
from .primitives import PointMass, RigidBody


class Integrator:
    """Base class for numerical integrators."""
    
    def integrate(self, objects: List[PointMass], dt: float,
                 force_function: Callable[[List[PointMass]], None]) -> None:
        """
        Integrate the system forward by time step dt.
        
        Args:
            objects: List of physics objects to integrate
            dt: Time step
            force_function: Function that computes forces for all objects
        """
        raise NotImplementedError


class EulerIntegrator(Integrator):
    """Explicit Euler integrator (first-order, not recommended for long simulations)."""
    
    def integrate(self, objects: List[PointMass], dt: float,
                 force_function: Callable[[List[PointMass]], None]) -> None:
        """Explicit Euler integration step."""
        # Clear forces and compute new forces
        for obj in objects:
            obj.clear_forces()
            if isinstance(obj, RigidBody):
                obj.clear_torques()
                
        force_function(objects)
        
        # Update positions and velocities
        for obj in objects:
            # Translational motion
            acceleration = obj.force / obj.mass
            obj.position += obj.velocity * dt
            obj.velocity += acceleration * dt
            
            # Rotational motion for rigid bodies
            if isinstance(obj, RigidBody):
                angular_acceleration = obj.torque / obj.moment_inertia
                obj.orientation += obj.angular_velocity * dt
                obj.angular_velocity += angular_acceleration * dt
                
            obj.update_trail()


class SemiImplicitEulerIntegrator(Integrator):
    """Semi-implicit (symplectic) Euler integrator - more stable for oscillatory systems."""
    
    def integrate(self, objects: List[PointMass], dt: float,
                 force_function: Callable[[List[PointMass]], None]) -> None:
        """Semi-implicit Euler integration step."""
        # Clear forces and compute new forces
        for obj in objects:
            obj.clear_forces()
            if isinstance(obj, RigidBody):
                obj.clear_torques()
                
        force_function(objects)
        
        # Update velocities first, then positions (symplectic)
        for obj in objects:
            # Translational motion
            acceleration = obj.force / obj.mass
            obj.velocity += acceleration * dt
            obj.position += obj.velocity * dt
            
            # Rotational motion for rigid bodies
            if isinstance(obj, RigidBody):
                angular_acceleration = obj.torque / obj.moment_inertia
                obj.angular_velocity += angular_acceleration * dt
                obj.orientation += obj.angular_velocity * dt
                
            obj.update_trail()


class RK4Integrator(Integrator):
    """Fourth-order Runge-Kutta integrator - high accuracy."""
    
    def integrate(self, objects: List[PointMass], dt: float,
                 force_function: Callable[[List[PointMass]], None]) -> None:
        """Fourth-order Runge-Kutta integration step."""
        # Store initial state
        initial_states = []
        for obj in objects:
            state = {
                'position': obj.position.copy(),
                'velocity': obj.velocity.copy(),
            }
            if isinstance(obj, RigidBody):
                state.update({
                    'orientation': obj.orientation,
                    'angular_velocity': obj.angular_velocity
                })
            initial_states.append(state)
        
        # RK4 coefficients storage
        k_pos = [np.zeros_like(obj.position) for obj in objects]
        k_vel = [np.zeros_like(obj.velocity) for obj in objects]
        k_ori = [0.0 for obj in objects if isinstance(obj, RigidBody)]
        k_ang_vel = [0.0 for obj in objects if isinstance(obj, RigidBody)]
        
        # RK4 stages
        for stage in range(4):
            # Clear forces and compute
            for obj in objects:
                obj.clear_forces()
                if isinstance(obj, RigidBody):
                    obj.clear_torques()
                    
            force_function(objects)
            
            # Compute derivatives
            for i, obj in enumerate(objects):
                if stage == 0:
                    k_pos[i] = obj.velocity.copy()
                    k_vel[i] = obj.force / obj.mass
                    
                # Apply intermediate updates for next stage
                if stage < 3:
                    h = dt * (0.5 if stage < 2 else 1.0)
                    obj.position = initial_states[i]['position'] + h * k_pos[i]
                    obj.velocity = initial_states[i]['velocity'] + h * k_vel[i]
                    
                    if isinstance(obj, RigidBody):
                        if stage == 0:
                            k_ori[i] = obj.angular_velocity
                            k_ang_vel[i] = obj.torque / obj.moment_inertia
                        obj.orientation = initial_states[i]['orientation'] + h * k_ori[i]
                        obj.angular_velocity = initial_states[i]['angular_velocity'] + h * k_ang_vel[i]
        
        # Final update (RK4 combination)
        for i, obj in enumerate(objects):
            obj.position = initial_states[i]['position'] + dt * k_pos[i]
            obj.velocity = initial_states[i]['velocity'] + dt * k_vel[i]
            
            if isinstance(obj, RigidBody):
                obj.orientation = initial_states[i]['orientation'] + dt * k_ori[i]
                obj.angular_velocity = initial_states[i]['angular_velocity'] + dt * k_ang_vel[i]
                
            obj.update_trail()
