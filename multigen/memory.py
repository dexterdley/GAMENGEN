import math
import numpy as np
import torch

class Map:
    """
    Represents the static level map as described in Section 3.1 of the MultiGen paper.
    Maintains a set of 2D vertices and line segments representing walls and layout.
    """
    def __init__(self, vertices=None, edges=None):
        # vertices: list of (x, y) tuples
        # edges: list of (index_v1, index_v2) tuples
        self.vertices = vertices if vertices is not None else []
        self.edges = edges if edges is not None else []
    
    @classmethod
    def create_simple_box(cls, size=10.0):
        """Creates a simple square room of the given size."""
        v = [
            (0.0, 0.0),
            (size, 0.0),
            (size, size),
            (0.0, size)
        ]
        e = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0)
        ]
        return cls(vertices=v, edges=e)
    
    def get_lines(self):
        """Returns the line segments as pairs of coordinates: [((x1,y1), (x2,y2)), ...]"""
        return [
            (self.vertices[e[0]], self.vertices[e[1]])
            for e in self.edges
        ]

class PlayerState:
    """
    Maintains the player pose (x, y, yaw) as described in Section 3.1.
    """
    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw  # Angle in radians
    
    def get_pose(self):
        return (self.x, self.y, self.yaw)
    
    def update(self, dx, dy, dyaw):
        """Update the pose, wrapping the yaw angle to [-pi, pi]."""
        self.x += dx
        self.y += dy
        self.yaw = (self.yaw + dyaw + math.pi) % (2 * math.pi) - math.pi

class MemoryModule:
    """
    The external memory system (Section 3.1 & 3.2).
    Maintains Map, PlayerState, and contexts, and provides geometric readout via ray-tracing.
    """
    def __init__(self, game_map: Map, initial_pose: PlayerState, fov_deg=90.0, num_rays=64, max_depth=20.0):
        self.map = game_map
        self.player = initial_pose
        
        # Ray-tracing parameters
        self.fov = math.radians(fov_deg)
        self.num_rays = num_rays
        self.max_depth = max_depth
        
        # We will keep visual context external to this class for simplicity 
        # (usually handled by the Engine loop), but this class generates the geometric conditional rt.

    def _ray_intersect_segment(self, ray_origin, ray_dir, pt1, pt2):
        """
        Calculates intersection between a ray and a line segment.
        Returns distance to intersection, or float('inf') if no intersection.
        """
        # Ray: origin + t * dir  (t >= 0)
        # Segment: pt1 + u * (pt2 - pt1) (0 <= u <= 1)
        v1 = ray_origin[0] - pt1[0], ray_origin[1] - pt1[1]
        v2 = pt2[0] - pt1[0], pt2[1] - pt1[1]
        v3 = -ray_dir[1], ray_dir[0]
        
        dot = v2[0] * v3[0] + v2[1] * v3[1]
        if abs(dot) < 1e-6:
            return float('inf') # Parallel
        
        t1 = (v2[0] * v1[1] - v2[1] * v1[0]) / dot
        t2 = (ray_dir[0] * v1[1] - ray_dir[1] * v1[0]) / dot
        
        if t1 >= 0.0 and 0.0 <= t2 <= 1.0:
            return t1
        return float('inf')

    def get_geometric_readout(self, spatial_size=(64, 64)):
        """
        Ray-traces a 1D depth vector and converts it to disparity (inverse depth).
        Returns a spatial tensor mapped to the UNet input resolution.
        """
        x, y, yaw = self.player.get_pose()
        origin = (x, y)
        
        # Calculate ray angles spanning the FOV, centered at current yaw
        start_angle = yaw - self.fov / 2
        end_angle = yaw + self.fov / 2
        angles = np.linspace(start_angle, end_angle, self.num_rays)
        
        lines = self.map.get_lines()
        depths = np.full(self.num_rays, self.max_depth)
        
        for i, angle in enumerate(angles):
            ray_dir = (math.cos(angle), math.sin(angle))
            min_dist = self.max_depth
            
            for pt1, pt2 in lines:
                dist = self._ray_intersect_segment(origin, ray_dir, pt1, pt2)
                if dist < min_dist:
                    min_dist = dist
                    
            depths[i] = min_dist
            
        # Convert to disparity representation (Section 3.2): disparity = 1 / depth
        # Add epsilon to prevent division by zero
        disparity = 1.0 / (depths + 1e-3)
        
        # Normalize disparity to roughly [0, 1] based on max expected disparity
        max_disp = 1.0 / 1e-3
        min_disp = 1.0 / (self.max_depth + 1e-3)
        disparity_norm = (disparity - min_disp) / (max_disp - min_disp + 1e-6)
        
        # The paper says: "mapping the 1D disparity to a spatial tensor at the UNet input resolution"
        # We broadcast the 1D vector (width) along the height dimension.
        # This provides a (1, H, W) tensor where columns represent ray-traced disparity.
        
        # Resize 1D vector to the target width (W)
        if self.num_rays != spatial_size[1]:
            # Simple linear interpolation to match target spatial width
            x_old = np.linspace(0, 1, self.num_rays)
            x_new = np.linspace(0, 1, spatial_size[1])
            disparity_w = np.interp(x_new, x_old, disparity_norm)
        else:
            disparity_w = disparity_norm
            
        # Broadcast to shape (1, H, W)
        disparity_tensor = torch.tensor(disparity_w, dtype=torch.float32)
        disparity_spatial = disparity_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 1, spatial_size[0], 1)
        # shape: (1, 1, H, W)
        
        return disparity_spatial

if __name__ == "__main__":
    # Quick simple test
    test_map = Map.create_simple_box(10.0)
    player = PlayerState(x=5.0, y=5.0, yaw=0.0) # looking +x
    memory = MemoryModule(test_map, player, fov_deg=90, num_rays=64)
    
    readout = memory.get_geometric_readout(spatial_size=(64, 64))
    print(f"Readout shape: {readout.shape}")
    print(f"Readout center val: {readout[0, 0, 32, 32].item():.4f}") # Looking straight at wall 5 units away
