"""
2D Physics Simulator Package

A production-ready 2D physics engine with visualization capabilities.
"""

__version__ = "1.0.0"

from .engine import PhysicsEngine
from .primitives import PointMass, RigidBody, Circle
from .visualization import Visualizer
from .io import ConfigLoader, StateExporter

__all__ = [
    'PhysicsEngine',
    'PointMass', 
    'RigidBody',
    'Circle',
    'Visualizer',
    'ConfigLoader',
    'StateExporter'
]
