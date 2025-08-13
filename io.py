"""
Configuration and state I/O for the physics simulator.
"""

from typing import Dict, List, Any, Optional, Union
import json
import logging
import numpy as np
from .primitives import PointMass, RigidBody, Circle
from .forces import GravityForce, LinearDragForce, SpringForce, ConstantForce
from .engine import PhysicsEngine

logger = logging.getLogger(__name__)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    logger.warning("PyYAML not available, YAML configuration disabled")


class ConfigLoader:
    """Load simulation configuration from JSON/YAML files."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file (.json or .yaml)
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If file format is unsupported
        """
        import os
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        ext = os.path.splitext(config_path)[1].lower()
        
        with open(config_path, 'r') as f:
            if ext == '.json':
                config = json.load(f)
            elif ext in ['.yaml', '.yml'] and HAS_YAML:
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {ext}")
                
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    @staticmethod
    def create_engine_from_config(config: Dict[str, Any]) -> PhysicsEngine:
        """
        Create physics engine from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured PhysicsEngine instance
        """
        # Engine parameters
        engine_config = config.get('engine', {})
        
        engine = PhysicsEngine(
            integrator=engine_config.get('integrator', 'semi_implicit_euler'),
            dt=engine_config.get('dt', 0.016),
            bounds=engine_config.get('bounds'),
            use_spatial_acceleration=engine_config.get('use_spatial_acceleration', True),
            spatial_type=engine_config.get('spatial_type', 'grid')
        )
        
        # Set collision parameters
        collision_config = config.get('collision', {})
        engine.set_collision_parameters(
            restitution=collision_config.get('restitution', 0.8),
            friction=collision_config.get('friction', 0.3)
        )
        
        # Add objects
        for obj_config in config.get('objects', []):
            obj = ConfigLoader._create_object_from_config(obj_config)
            engine.add_object(obj)
            
        # Add forces
        ConfigLoader._setup_forces_from_config(engine, config.get('forces', {}))
        
        # Add planes
        for plane_config in config.get('planes', []):
            engine.add_plane(plane_config['point'], plane_config['normal'])
            
        logger.info("Created physics engine from configuration")
        return engine
        
    @staticmethod
    def _create_object_from_config(obj_config: Dict[str, Any]) -> PointMass:
        """Create physics object from configuration."""
        obj_type = obj_config.get('type', 'PointMass')
        
        common_params = {
            'position': obj_config['position'],
            'velocity': obj_config.get('velocity', [0, 0]),
            'mass': obj_config.get('mass', 1.0),
            'object_id': obj_config.get('id')
        }
        
        if obj_type == 'PointMass':
            return PointMass(**common_params)
        elif obj_type == 'RigidBody':
            return RigidBody(
                **common_params,
                orientation=obj_config.get('orientation', 0.0),
                angular_velocity=obj_config.get('angular_velocity', 0.0),
                moment_inertia=obj_config.get('moment_inertia', 1.0)
            )
        elif obj_type == 'Circle':
            return Circle(
                **common_params,
                radius=obj_config['radius'],
                orientation=obj_config.get('orientation', 0.0),
                angular_velocity=obj_config.get('angular_velocity', 0.0)
            )
        else:
            raise ValueError(f"Unknown object type: {obj_type}")
            
    @staticmethod
    def _setup_forces_from_config(engine: PhysicsEngine, forces_config: Dict[str, Any]) -> None:
        """Setup force generators from configuration."""
        # Clear default forces if specified
        if forces_config.get('clear_defaults', False):
            engine.force_generators.clear()
            
        # Gravity
        if 'gravity' in forces_config:
            gravity_config = forces_config['gravity']
            if isinstance(gravity_config, list):
                engine.add_force_generator(GravityForce(gravity_config))
            elif isinstance(gravity_config, dict):
                engine.add_force_generator(GravityForce(gravity_config.get('vector', [0, -9.81])))
                
        # Linear drag
        if 'drag' in forces_config:
            drag_config = forces_config['drag']
            if isinstance(drag_config, (int, float)):
                engine.add_force_generator(LinearDragForce(drag_config))
            elif isinstance(drag_config, dict):
                engine.add_force_generator(LinearDragForce(drag_config.get('coefficient', 0.1)))
                
        # Springs
        for spring_config in forces_config.get('springs', []):
            engine.add_force_generator(SpringForce(
                obj1_id=spring_config['obj1_id'],
                obj2_id=spring_config['obj2_id'],
                spring_constant=spring_config['spring_constant'],
                rest_length=spring_config.get('rest_length', 0.0),
                damping=spring_config.get('damping', 0.0)
            ))
            
        # Constant forces
        for const_force_config in forces_config.get('constant_forces', []):
            engine.add_force_generator(ConstantForce(
                force=const_force_config['force'],
                object_ids=const_force_config.get('object_ids')
            ))


class StateExporter:
    """Export simulation state and time-series data."""
    
    def __init__(self, engine: PhysicsEngine):
        """
        Initialize state exporter.
        
        Args:
            engine: Physics engine to export data from
        """
        self.engine = engine
        self.time_series_data = []
        
    def record_state(self) -> None:
        """Record current simulation state for time-series export."""
        state = {
            'time': self.engine.time,
            'objects': []
        }
        
        for obj in self.engine.objects:
            obj_data = {
                'id': obj.id,
                'position': obj.position.tolist(),
                'velocity': obj.velocity.tolist(),
                'kinetic_energy': obj.get_kinetic_energy()
            }
            
            if isinstance(obj, RigidBody):
                obj_data.update({
                    'orientation': obj.orientation,
                    'angular_velocity': obj.angular_velocity,
                    'rotational_energy': obj.get_rotational_energy()
                })
                
            state['objects'].append(obj_data)
            
        self.time_series_data.append(state)
        
    def export_current_state(self, output_path: str) -> None:
        """
        Export current simulation state to file.
        
        Args:
            output_path: Path for output file (.json)
        """
        state = self.engine.get_state()
        
        with open(output_path, 'w') as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Exported current state to {output_path}")
        
    def export_time_series(self, output_path: str, format: str = 'json') -> None:
        """
        Export time-series data to file.
        
        Args:
            output_path: Path for output file
            format: Export format ('json' or 'csv')
        """
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(self.time_series_data, f, indent=2)
        elif format == 'csv':
            self._export_csv(output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        logger.info(f"Exported time-series data to {output_path}")
        
    def _export_csv(self, output_path: str) -> None:
        """Export time-series data as CSV."""
        import csv
        
        if not self.time_series_data:
            logger.warning("No time-series data to export")
            return
            
        # Flatten data for CSV format
        rows = []
        for state in self.time_series_data:
            for obj_data in state['objects']:
                row = {
                    'time': state['time'],
                    'object_id': obj_data['id'],
                    'pos_x': obj_data['position'][0],
                    'pos_y': obj_data['position'][1],
                    'vel_x': obj_data['velocity'][0],
                    'vel_y': obj_data['velocity'][1],
                    'kinetic_energy': obj_data['kinetic_energy']
                }
                
                if 'orientation' in obj_data:
                    row.update({
                        'orientation': obj_data['orientation'],
                        'angular_velocity': obj_data['angular_velocity'],
                        'rotational_energy': obj_data['rotational_energy']
                    })
                    
                rows.append(row)
                
        # Write CSV
        with open(output_path, 'w', newline='') as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
                
    def clear_time_series(self) -> None:
        """Clear recorded time-series data."""
        self.time_series_data.clear()
        logger.debug("Cleared time-series data")

