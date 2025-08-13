"""
Matplotlib-based visualization for the physics simulation.
"""

from typing import List, Dict, Optional, Callable, Any, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle as MPLCircle
import logging
from .primitives import PointMass, RigidBody, Circle
from .engine import PhysicsEngine

logger = logging.getLogger(__name__)


class Visualizer:
    """Real-time visualization using matplotlib."""
    
    def __init__(self, engine: PhysicsEngine, figsize: Tuple[int, int] = (10, 8),
                 xlim: Tuple[float, float] = (-10, 10),
                 ylim: Tuple[float, float] = (-10, 10)):
        """
        Initialize visualizer.
        
        Args:
            engine: Physics engine to visualize
            figsize: Figure size (width, height)
            xlim: X-axis limits
            ylim: Y-axis limits
        """
        self.engine = engine
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Visualization options
        self.show_velocities = True
        self.show_trails = True
        self.show_forces = False
        self.show_debug = False
        self.trail_length = 50
        self.velocity_scale = 1.0
        
        # Visual elements
        self.patches = {}
        self.velocity_arrows = {}
        self.trails = {}
        self.debug_patches = {}
        
        # Animation state
        self.animation = None
        self.frame_rate = 60
        self.real_time = True
        
        # Statistics display
        self.stats_text = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes,
                                      verticalalignment='top', fontfamily='monospace')
                                      
        logger.info("Visualizer initialized")
        
    def set_limits(self, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> None:
        """Set axis limits."""
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        
    def set_visualization_options(self, show_velocities: bool = None,
                                 show_trails: bool = None,
                                 show_forces: bool = None,
                                 show_debug: bool = None,
                                 trail_length: int = None,
                                 velocity_scale: float = None) -> None:
        """Configure visualization options."""
        if show_velocities is not None:
            self.show_velocities = show_velocities
        if show_trails is not None:
            self.show_trails = show_trails
        if show_forces is not None:
            self.show_forces = show_forces
        if show_debug is not None:
            self.show_debug = show_debug
        if trail_length is not None:
            self.trail_length = trail_length
        if velocity_scale is not None:
            self.velocity_scale = velocity_scale
            
    def update_frame(self, frame_num: int = 0) -> List:
        """Update visualization frame."""
        # Step physics if in real-time mode
        if self.real_time and not self.engine.paused:
            self.engine.step()
            
        # Clear previous dynamic elements
        for arrow in self.velocity_arrows.values():
            arrow.remove()
        self.velocity_arrows.clear()
        
        # Update object patches
        artists = []
        for obj in self.engine.objects:
            # Update or create patch for object
            if obj.id not in self.patches:
                self.patches[obj.id] = self._create_patch(obj)
                self.ax.add_patch(self.patches[obj.id])
                
            patch = self.patches[obj.id]
            self._update_patch(patch, obj)
            artists.append(patch)
            
            # Draw velocity vectors
            if self.show_velocities:
                arrow = self._draw_velocity_vector(obj)
                if arrow:
                    artists.append(arrow)
                    
            # Draw trails
            if self.show_trails and len(obj.trail_positions) > 1:
                if obj.id not in self.trails:
                    self.trails[obj.id], = self.ax.plot([], [], 'b-', alpha=0.5, linewidth=1)
                    
                trail_line = self.trails[obj.id]
                positions = np.array(obj.trail_positions[-self.trail_length:])
                trail_line.set_data(positions[:, 0], positions[:, 1])
                artists.append(trail_line)
                
        # Remove patches for deleted objects
        current_ids = {obj.id for obj in self.engine.objects}
        for obj_id in list(self.patches.keys()):
            if obj_id not in current_ids:
                self.patches[obj_id].remove()
                del self.patches[obj_id]
                if obj_id in self.trails:
                    self.trails[obj_id].remove()
                    del self.trails[obj_id]
                    
        # Draw debug information
        if self.show_debug:
            self._draw_debug_info()
            
        # Update statistics
        self._update_stats()
        artists.append(self.stats_text)
        
        return artists
        
    def _create_patch(self, obj: PointMass) -> Any:
        """Create matplotlib patch for object."""
        if isinstance(obj, Circle):
            patch = MPLCircle((0, 0), obj.radius, 
                            facecolor='lightblue', 
                            edgecolor='blue',
                            alpha=0.7)
        else:
            # Point mass as small circle
            patch = MPLCircle((0, 0), 0.1,
                            facecolor='red',
                            edgecolor='darkred',
                            alpha=0.8)
        return patch
        
    def _update_patch(self, patch: Any, obj: PointMass) -> None:
        """Update patch position and appearance."""
        patch.center = (obj.position[0], obj.position[1])
        
        # Color based on speed for visual feedback
        speed = np.linalg.norm(obj.velocity)
        if isinstance(obj, Circle):
            # Interpolate color based on speed
            color_intensity = min(1.0, speed / 10.0)
            patch.set_facecolor((color_intensity, 0.5, 1.0 - color_intensity))
            
    def _draw_velocity_vector(self, obj: PointMass) -> Optional[Any]:
        """Draw velocity vector as arrow."""
        if np.linalg.norm(obj.velocity) < 0.1:
            return None
            
        arrow = self.ax.arrow(obj.position[0], obj.position[1],
                             obj.velocity[0] * self.velocity_scale,
                             obj.velocity[1] * self.velocity_scale,
                             head_width=0.2, head_length=0.3,
                             fc='green', ec='green', alpha=0.7)
        self.velocity_arrows[obj.id] = arrow
        return arrow
        
    def _draw_debug_info(self) -> None:
        """Draw debug overlays."""
        # Draw spatial grid if available
        if (hasattr(self.engine, 'spatial_structure') and 
            self.engine.spatial_structure and 
            hasattr(self.engine.spatial_structure, 'grid')):
            
            grid = self.engine.spatial_structure
            for (row, col), objects in grid.grid.items():
                if objects:
                    x = grid.min_x + col * grid.cell_size
                    y = grid.min_y + row * grid.cell_size
                    rect = plt.Rectangle((x, y), grid.cell_size, grid.cell_size,
                                       fill=False, edgecolor='gray', alpha=0.5)
                    self.ax.add_patch(rect)
                    
    def _update_stats(self) -> None:
        """Update statistics display."""
        debug_info = self.engine.get_debug_info()
        total_energy = debug_info['total_energy']
        
        stats_text = (f"Time: {self.engine.time:.2f}s\n"
                     f"Objects: {len(self.engine.objects)}\n"
                     f"Collisions: {self.engine.collision_count}\n"
                     f"Energy: {total_energy:.2f}\n"
                     f"Max Penetration: {self.engine.max_penetration:.4f}")
                     
        if self.engine.paused:
            stats_text += "\nPAUSED"
            
        self.stats_text.set_text(stats_text)
        
    def animate(self, interval: float = None, save_path: str = None) -> None:
        """Start real-time animation."""
        if interval is None:
            interval = 1000 / self.frame_rate
            
        self.animation = animation.FuncAnimation(
            self.fig, self.update_frame, interval=interval,
            blit=False, repeat=True
        )
        
        if save_path:
            logger.info(f"Saving animation to {save_path}")
            writer = animation.PillowWriter(fps=self.frame_rate)
            self.animation.save(save_path, writer=writer)
        else:
            plt.show()
            
    def render_frame(self, save_path: str = None) -> None:
        """Render a single frame."""
        self.update_frame()
        if save_path:
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
            
    def export_frames(self, duration: float, output_dir: str, 
                     fps: int = 30) -> None:
        """Export sequence of frames as images."""
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        frame_dt = 1.0 / fps
        total_frames = int(duration * fps)
        
        for frame in range(total_frames):
            # Step simulation to match desired frame rate
            target_time = frame * frame_dt
            while self.engine.time < target_time:
                self.engine.step()
                
            self.update_frame()
            frame_path = os.path.join(output_dir, f"frame_{frame:06d}.png")
            self.fig.savefig(frame_path, dpi=150, bbox_inches='tight')
            
            if frame % 30 == 0:
                logger.info(f"Exported frame {frame}/{total_frames}")
                
    def close(self) -> None:
        """Close visualization."""
        if self.animation:
            self.animation.event_source.stop()
        plt.close(self.fig)
