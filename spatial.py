"""
Spatial partitioning for collision detection acceleration.
"""

from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from .primitives import PointMass, Circle


class SpatialGrid:
    """Uniform spatial grid for collision detection acceleration."""
    
    def __init__(self, bounds: Tuple[float, float, float, float], cell_size: float):
        """
        Initialize spatial grid.
        
        Args:
            bounds: Grid bounds (min_x, min_y, max_x, max_y)
            cell_size: Size of each grid cell
        """
        self.min_x, self.min_y, self.max_x, self.max_y = bounds
        self.cell_size = float(cell_size)
        
        self.cols = int((self.max_x - self.min_x) / cell_size) + 1
        self.rows = int((self.max_y - self.min_y) / cell_size) + 1
        
        self.grid: Dict[Tuple[int, int], List[PointMass]] = {}
        
    def clear(self) -> None:
        """Clear the grid."""
        self.grid.clear()
        
    def insert(self, obj: PointMass) -> None:
        """Insert object into appropriate grid cells."""
        cells = self._get_cells_for_object(obj)
        for cell in cells:
            if cell not in self.grid:
                self.grid[cell] = []
            self.grid[cell].append(obj)
            
    def _get_cells_for_object(self, obj: PointMass) -> List[Tuple[int, int]]:
        """Get all grid cells that the object occupies."""
        if isinstance(obj, Circle):
            # Circle occupies multiple cells
            min_x = obj.position[0] - obj.radius
            max_x = obj.position[0] + obj.radius
            min_y = obj.position[1] - obj.radius
            max_y = obj.position[1] + obj.radius
        else:
            # Point mass occupies single cell
            min_x = max_x = obj.position[0]
            min_y = max_y = obj.position[1]
            
        min_col = max(0, int((min_x - self.min_x) / self.cell_size))
        max_col = min(self.cols - 1, int((max_x - self.min_x) / self.cell_size))
        min_row = max(0, int((min_y - self.min_y) / self.cell_size))
        max_row = min(self.rows - 1, int((max_y - self.min_y) / self.cell_size))
        
        cells = []
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                cells.append((row, col))
        return cells
        
    def get_potential_collisions(self) -> List[Tuple[PointMass, PointMass]]:
        """Get potential collision pairs using spatial partitioning."""
        pairs = set()
        
        for cell_objects in self.grid.values():
            # Check all pairs within each cell
            for i in range(len(cell_objects)):
                for j in range(i + 1, len(cell_objects)):
                    obj1, obj2 = cell_objects[i], cell_objects[j]
                    # Ensure consistent ordering
                    if id(obj1) > id(obj2):
                        obj1, obj2 = obj2, obj1
                    pairs.add((obj1, obj2))
                    
        return list(pairs)
        
    def get_debug_info(self) -> Dict:
        """Get debug information about the spatial grid."""
        occupied_cells = len([cell for cell in self.grid.values() if cell])
        total_objects = sum(len(cell) for cell in self.grid.values())
        
        return {
            'total_cells': self.rows * self.cols,
            'occupied_cells': occupied_cells,
            'total_object_entries': total_objects,
            'cell_size': self.cell_size,
            'bounds': (self.min_x, self.min_y, self.max_x, self.max_y)
        }


class QuadTree:
    """QuadTree implementation for spatial partitioning."""
    
    def __init__(self, bounds: Tuple[float, float, float, float], max_objects: int = 10, max_depth: int = 5):
        """
        Initialize QuadTree.
        
        Args:
            bounds: Bounding box (min_x, min_y, max_x, max_y)
            max_objects: Maximum objects per node before subdivision
            max_depth: Maximum tree depth
        """
        self.bounds = bounds
        self.max_objects = max_objects
        self.max_depth = max_depth
        self.objects: List[PointMass] = []
        self.children: List[Optional['QuadTree']] = [None, None, None, None]
        self.depth = 0
        
    def clear(self) -> None:
        """Clear the QuadTree."""
        self.objects.clear()
        self.children = [None, None, None, None]
        
    def insert(self, obj: PointMass) -> None:
        """Insert object into QuadTree."""
        if not self._contains_object(obj):
            return
            
        if len(self.objects) < self.max_objects or self.depth >= self.max_depth:
            self.objects.append(obj)
        else:
            if self.children[0] is None:
                self._subdivide()
                
            for child in self.children:
                if child is not None:
                    child.insert(obj)
                    
    def _contains_object(self, obj: PointMass) -> bool:
        """Check if object is within this node's bounds."""
        min_x, min_y, max_x, max_y = self.bounds
        
        if isinstance(obj, Circle):
            return (obj.position[0] - obj.radius <= max_x and 
                   obj.position[0] + obj.radius >= min_x and
                   obj.position[1] - obj.radius <= max_y and 
                   obj.position[1] + obj.radius >= min_y)
        else:
            return (min_x <= obj.position[0] <= max_x and 
                   min_y <= obj.position[1] <= max_y)
                   
    def _subdivide(self) -> None:
        """Subdivide this node into four children."""
        min_x, min_y, max_x, max_y = self.bounds
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2
        
        # Create four children: NE, NW, SW, SE
        child_bounds = [
            (mid_x, mid_y, max_x, max_y),  # NE
            (min_x, mid_y, mid_x, max_y),  # NW  
            (min_x, min_y, mid_x, mid_y),  # SW
            (mid_x, min_y, max_x, mid_y)   # SE
        ]
        
        for i, bounds in enumerate(child_bounds):
            self.children[i] = QuadTree(bounds, self.max_objects, self.max_depth)
            self.children[i].depth = self.depth + 1
            
        # Redistribute objects to children
        for obj in self.objects:
            for child in self.children:
                if child is not None:
                    child.insert(obj)
        self.objects.clear()
        
    def get_potential_collisions(self) -> List[Tuple[PointMass, PointMass]]:
        """Get potential collision pairs from QuadTree."""
        pairs = set()
        self._collect_pairs(pairs)
        return list(pairs)
        
    def _collect_pairs(self, pairs: Set[Tuple[PointMass, PointMass]]) -> None:
        """Recursively collect collision pairs."""
        # Pairs within this node
        for i in range(len(self.objects)):
            for j in range(i + 1, len(self.objects)):
                obj1, obj2 = self.objects[i], self.objects[j]
                if id(obj1) > id(obj2):
                    obj1, obj2 = obj2, obj1
                pairs.add((obj1, obj2))
                
        # Pairs from children
        for child in self.children:
            if child is not None:
                child._collect_pairs(pairs)


