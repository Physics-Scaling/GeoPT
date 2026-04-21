import os
import argparse
import numpy as np
import trimesh
import time
import fcpw
import polyscope as ps

# Default sampling bounds for general ShapeNet categories
GENERAL_BOUNDS = ((-5.0, 5.0), (-2.0, 4.0), (-5.0, 5.0))


def compute_normals_improved(mesh, sampled_points):
    """
    Improved normal computation: recompute mesh normals.
    """
    # Ensure mesh has normals
    _ = mesh.face_normals
    _ = mesh.vertex_normals

    point_cloud = trimesh.PointCloud(vertices=mesh.vertices)
    distances, indices = point_cloud.kdtree.query(sampled_points, k=1)
    nearest_vertices_indices = indices.flatten()

    normals_at_sampled_points = mesh.vertex_normals[nearest_vertices_indices]

    # Check and handle zero normals
    zero_mask = np.linalg.norm(normals_at_sampled_points, axis=1) < 1e-6
    if np.any(zero_mask):
        print(f"Warning: found {np.sum(zero_mask)} zero normals")
        # Use face normals as fallback
        normals_at_sampled_points = handle_zero_normals(mesh, sampled_points, normals_at_sampled_points, zero_mask)

    return normals_at_sampled_points


def handle_zero_normals(mesh, points, current_normals, zero_mask):
    """
    Handle zero normal vectors.
    """
    # Method 1: use nearest face normals
    from trimesh.proximity import closest_point

    # Find nearest face for each point
    closest, distance, triangle_id = closest_point(mesh, points[zero_mask])

    # Get face normals
    face_normals = mesh.face_normals
    replacement_normals = face_normals[triangle_id]

    # Replace zero normals
    result_normals = current_normals.copy()
    result_normals[zero_mask] = replacement_normals

    # Check again for remaining zero normals
    still_zero = np.linalg.norm(result_normals, axis=1) < 1e-6
    if np.any(still_zero):
        print(f"Still {np.sum(still_zero)} zero normals, using random directions")
        # For remaining zeros, use random normalized directions
        random_dirs = np.random.randn(np.sum(still_zero), 3)
        random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)
        result_normals[still_zero] = random_dirs

    return result_normals


class FCPWScene:
    """
    Wrapper around FCPW scene for efficient geometric queries.
    Builds the acceleration structure once and provides methods for:
    - Ray intersection
    - Closest point queries
    - Inside/outside testing
    """

    def __init__(self, mesh: trimesh.Trimesh, build_vectorized: bool = True):
        """
        Initialize FCPW scene from a trimesh mesh.

        Args:
            mesh: trimesh.Trimesh object
            build_vectorized: Whether to build vectorized BVH for faster batch queries
        """
        self.mesh = mesh

        # Get vertices and faces as contiguous float32/int32 arrays
        positions = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
        indices = np.ascontiguousarray(mesh.faces, dtype=np.int32)

        # Create and build FCPW scene
        self.scene = fcpw.scene_3D()
        self.scene.set_object_count(1)
        self.scene.set_object_vertices(positions, 0)
        self.scene.set_object_triangles(indices, 0)

        # Build BVH acceleration structure
        aggregate_type = fcpw.aggregate_type.bvh_surface_area
        self.scene.build(aggregate_type, build_vectorized,
                         print_stats=False, reduce_memory_footprint=False)

    def find_closest_points(self, query_points: np.ndarray,
                            return_normals: bool = False) -> tuple:
        """
        Find closest points on mesh surface to query points.

        Args:
            query_points: (N, 3) array of query points
            return_normals: Whether to return surface normals at closest points

        Returns:
            distances: (N,) array of distances to closest points
            closest_points: (N, 3) array of closest points on surface
            normals: (N, 3) array of normals (only if return_normals=True)
        """
        query_points = np.ascontiguousarray(query_points, dtype=np.float32)
        n_points = len(query_points)

        # Use infinite search radius
        squared_max_radii = np.full(n_points, np.inf, dtype=np.float32)

        # Perform closest point queries
        interactions = fcpw.interaction_3D_list()
        self.scene.find_closest_points(query_points, squared_max_radii, interactions,
                                       record_normal=return_normals)
        inside_2 = time.time()
        # Extract results
        closest_points = np.array([i.p for i in interactions], dtype=np.float32)
        distances = np.array([i.d for i in interactions], dtype=np.float32)
        if return_normals:
            normals = np.array([i.n for i in interactions], dtype=np.float32)
            return distances, closest_points, normals

        return distances, closest_points

    def intersect_rays(self, ray_origins: np.ndarray, ray_directions: np.ndarray,
                       return_all_hits: bool = False) -> tuple:
        """
        Intersect rays with the mesh.

        Args:
            ray_origins: (N, 3) array of ray origins
            ray_directions: (N, 3) array of ray directions (will be normalized)
            return_all_hits: If True, count all intersections; if False, only closest

        Returns:
            hit_mask: (N,) boolean array indicating which rays hit
            hit_distances: (N,) array of hit distances (inf for misses)
            hit_points: (N, 3) array of hit points (origin for misses)
            hit_counts: (N,) array of hit counts (only if return_all_hits=True)
        """
        ray_origins = np.ascontiguousarray(ray_origins, dtype=np.float32)
        ray_directions = np.ascontiguousarray(ray_directions, dtype=np.float32)
        n_rays = len(ray_origins)

        # Normalize directions
        norms = np.linalg.norm(ray_directions, axis=1, keepdims=True)
        ray_directions = ray_directions / (norms + 1e-10)

        # Set very large distance bounds
        ray_distance_bounds = np.full(n_rays, 1e10, dtype=np.float32)

        # Perform intersection queries
        interactions = fcpw.interaction_3D_list()
        self.scene.intersect(ray_origins, ray_directions, ray_distance_bounds,
                             interactions, check_for_occlusion=False)

        # Extract results
        hit_distances = np.full(n_rays, np.inf, dtype=np.float32)
        hit_points = ray_origins.copy()
        hit_mask = np.zeros(n_rays, dtype=bool)

        for i, interaction in enumerate(interactions):
            if interaction.d < np.inf and interaction.d >= 0:
                hit_mask[i] = True
                hit_distances[i] = interaction.d
                hit_points[i] = interaction.p

        if return_all_hits:
            # For all hits, we need to count intersections using multiple ray passes
            # This is a simplified version - FCPW doesn't directly support counting all hits
            # in batch mode, so we keep the hit_mask as the main return
            return hit_mask, hit_distances, hit_points, hit_mask.astype(int)

        return hit_mask, hit_distances, hit_points

    def contains(self, points: np.ndarray) -> np.ndarray:
        """
        Check if points are inside the mesh.

        Uses FCPW's contains method with BVH acceleration.
        Note: Works best with watertight meshes.

        Args:
            points: (N, 3) array of query points

        Returns:
            inside: (N,) boolean array, True if point is inside mesh
        """
        points = np.ascontiguousarray(points, dtype=np.float32)
        result = np.zeros(len(points), dtype=np.int32)
        self.scene.contains(points, result)
        return result.astype(bool)


def transform_mesh(mesh: trimesh.Trimesh, target_length: float = 5.0):
    """
    Rotate and scale mesh to align with downstream simulation.

    Args:
        mesh: Input trimesh mesh
        target_length: Target length along z-axis

    Returns:
        Transformed mesh (modified in-place)
    """
    V = mesh.vertices.copy()
    new_V = V

    # Rotate: swap axes
    new_V[:, 0] = -V[:, 0]

    bound_max = np.max(new_V, axis=0)
    bound_min = np.min(new_V, axis=0)
    obj_length = bound_max - bound_min
    # Shift y to ground
    new_V[:, 1] -= bound_min[1]

    # Scale to target length
    length = bound_max[0] - bound_min[0]
    scale = target_length / length
    if obj_length[0] * scale > 5.5 or obj_length[1] * scale > 3 or obj_length[2] * scale > 6:
        scale = scale * 0.5
    new_V *= scale

    # Center x
    x_avg = np.mean(new_V[:, 0])
    new_V[:, 0] -= x_avg

    # Center y
    y_avg = np.mean(new_V[:, 2])
    new_V[:, 2] -= y_avg

    mesh.vertices = new_V
    print(f"Mesh bounds: max={np.max(new_V, axis=0)}, min={np.min(new_V, axis=0)}")
    return mesh, bound_min[1], x_avg, y_avg, scale


def transform_pointcloud(surf_points, surf_normal, z_min, x_avg, y_avg, scale):
    new_surf_points = surf_points
    new_surf_normal = surf_normal
    new_surf_points[:, 0] = -surf_points[:, 0]
    new_surf_normal[:, 0] = -surf_normal[:, 0]

    # shift z
    new_surf_points[:, 1] -= z_min
    # scale
    new_surf_points = new_surf_points * scale
    # shift x
    new_surf_points[:, 0] -= x_avg
    # shift y
    new_surf_points[:, 2] -= y_avg
    print(np.max(new_surf_points, axis=0))
    print(np.min(new_surf_points, axis=0))
    return new_surf_points, new_surf_normal


def get_sdf(fcpw_scene: FCPWScene, target_points: np.ndarray) -> tuple:
    """
    Compute signed distance field (unsigned distance) and direction to mesh surface.

    Uses FCPW find_closest_points to compute distance to the actual mesh surface
    (not just to sampled surface points).

    Args:
        fcpw_scene: FCPWScene wrapper with built acceleration structure
        target_points: (N, 3) array of query points

    Returns:
        distances: (N,) array of distances to mesh surface
        directions: (N, 3) array of normalized directions to closest points
    """
    distances, closest_points = fcpw_scene.find_closest_points(target_points)

    # Compute directions from target to closest point on surface
    diff = closest_points - target_points
    norms = distances.reshape(-1, 1) + 1e-8
    directions = diff / norms

    return distances, directions


def sample_volume_outside_mesh(
        fcpw_scene: FCPWScene,
        N: int = 32768,
        bounds: tuple = ((-5, 5.0), (-2, 4), (-5, 5)),
        batch_size: int = 65536,
        max_iter: int = 50
) -> np.ndarray:
    """
    Sample points outside mesh using FCPW contains test.

    Uses FCPW's BVH-accelerated contains() method for fast inside/outside testing.

    Args:
        fcpw_scene: FCPWScene with built acceleration structure
        N: Number of outside points to sample
        bounds: Sampling bounds ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        batch_size: Number of points to sample per iteration
        max_iter: Maximum iterations to prevent infinite loop

    Returns:
        outside_points: (N, 3) array of points outside the mesh
    """
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    z_min, z_max = bounds[2]

    outside_points = []
    total_count = 0

    for iter_count in range(max_iter):
        if total_count >= N:
            break

        print(f"iter_count: {iter_count + 1}")

        # Sample random points in bounds
        points = np.random.uniform(
            low=[x_min, y_min, z_min],
            high=[x_max, y_max, z_max],
            size=(batch_size, 3)
        ).astype(np.float32)

        # Use FCPW contains (BVH accelerated)
        inside_mask = fcpw_scene.contains(points)

        # Select outside points (not inside)
        outside_mask = ~inside_mask
        new_points = points[outside_mask]

        if len(new_points) > 0:
            outside_points.append(new_points)
            total_count += len(new_points)

        print(f"total outside points: {total_count}")

    # Concatenate and truncate
    outside_points = np.concatenate(outside_points, axis=0)[:N]

    if len(outside_points) < N:
        print(f"Warning: After {max_iter} iterations, only sampled {len(outside_points)} points")

    return outside_points


def multi_step_constrained_walk_with_surface(
        fcpw_scene: FCPWScene,
        volume_points: np.ndarray,
        surf_points: np.ndarray,
        steps: int = 3,
        min_step: float = 0.0,
        max_step: float = 2.0,
        sigma: float = 0.1,
        k_neighbors: int = 32,
        init_directions: np.ndarray = None,
        init_step_lengths: np.ndarray = None
) -> dict:
    """
    Perform multi-step constrained random walk for volume points.

    Volume points walk in random directions with collision detection against mesh.
    Surface points remain fixed. At each step, compute Gaussian-weighted supervision
    towards surface points.

    Key optimizations:
    - Use FCPW for ray-mesh intersection (collision detection)
    - Use FCPW find_closest_points for surface projection (instead of KDTree)

    Args:
        fcpw_scene: FCPWScene wrapper with built acceleration structure
        volume_points: (N, 3) array of volume points
        surf_points: (M, 3) array of surface points
        steps: Number of walk steps
        min_step: Minimum step length
        max_step: Maximum step length
        sigma: Gaussian weight parameter for supervision
        k_neighbors: Number of neighbors for Gaussian weighting

    Returns:
        dict with keys:
            supervise: (N+M, 3*steps) supervision vectors
            condition: (N+M, 4) walk conditions [direction, step_length]
            vis_data: list of `steps` dicts with:
                - actual_start: (N, 3) actual start positions (volume only)
                - intended_end: (N, 3) intended end positions without collision
                - actual_end: (N, 3) actual end positions after collision
                - collision_mask: (N,) bool, True if collision occurred
                Note: The last entry (step=steps-1) shows final positions with no movement.
    """
    N = volume_points.shape[0]
    M = surf_points.shape[0]

    # Combine all points
    all_points = np.vstack([volume_points, surf_points]).astype(np.float32)
    positions = all_points.copy()

    supervise_list = []
    vis_data_list = []  # Visualization data per step transition

    # Volume mask (surface points don't move)
    vol_mask = np.ones(N + M, dtype=bool)
    vol_mask[-M:] = False

    if init_directions is not None:
        directions = init_directions.copy()
    else:
        phi = np.random.uniform(0, 2 * np.pi, size=(N + M, 1))
        cos_theta = np.random.uniform(-1, 1, size=(N + M, 1))
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        directions = np.concatenate([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta
        ], axis=1).astype(np.float32)

    if init_step_lengths is not None:
        step_lengths = init_step_lengths.copy()
    else:
        step_lengths = np.random.uniform(min_step, max_step, size=(N + M,)).astype(np.float32)
        step_lengths[-M:] = 0

    # Build KDTree for surf_points once (for Gaussian weighting)
    # from scipy.spatial import cKDTree
    # surf_tree = cKDTree(surf_points)

    for step_idx in range(steps):
        # Compute Gaussian-weighted supervision (direction to surface)
        _, weighted_mean = fcpw_scene.find_closest_points(positions, return_normals=False)
        supervise_list.append(positions - weighted_mean)

        if step_idx == steps - 1:
            # Last step: no movement
            break

        # Current volume positions (start of this step)
        vol_positions = positions[vol_mask].copy()
        vol_dirs = directions[vol_mask]
        vol_step_lengths = step_lengths[vol_mask]

        # Intended end positions (without collision)
        intended_end = vol_positions + vol_dirs * vol_step_lengths[:, None]
        # Collision detection using FCPW ray intersection
        hit_mask, hit_distances, _ = fcpw_scene.intersect_rays(vol_positions, vol_dirs)

        # Determine which points collide before reaching intended step
        collision_mask = hit_mask & (hit_distances < vol_step_lengths)

        # Compute actual steps (clamped by collision)
        actual_steps = np.where(collision_mask, hit_distances * 0.99, vol_step_lengths)

        # Actual end positions (after collision clamping)
        actual_end = vol_positions + vol_dirs * actual_steps[:, None]

        # Store visualization data for this step transition
        vis_data_list.append({
            'actual_start': vol_positions.copy(),
            'intended_end': intended_end.copy(),
            'actual_end': actual_end.copy(),
            'collision_mask': collision_mask.copy()
        })

        # Update positions
        positions[vol_mask] = actual_end

        # Ensure surface points stay fixed
        positions[-M:] = surf_points

    # Add final state visualization (no movement, just show final positions)
    final_vol_positions = positions[vol_mask].copy()
    vis_data_list.append({
        'actual_start': final_vol_positions.copy(),
        'intended_end': final_vol_positions.copy(),  # No movement
        'actual_end': final_vol_positions.copy(),  # No movement
        'collision_mask': np.zeros(N, dtype=bool)  # No collisions
    })

    # Concatenate supervision from all steps
    supervise = np.concatenate(supervise_list, axis=1)
    condition = np.concatenate([directions, step_lengths[:, None]], axis=1)

    return {
        'supervise': supervise,
        'condition': condition,
        'directions': directions,
        'step_lengths': step_lengths,
        'vis_data': vis_data_list
    }


def visualize_walk_results(
        mesh: trimesh.Trimesh,
        walk_results_list: list,
        steps: int = 3,
        show_mesh: bool = True,
        point_radius: float = 0.005,
        subsample: int = None,
        bounds: tuple = None
):
    """
    Interactive visualization of random walk: start, intended end, actual end.

    For each step shows:
    - Start positions (yellow)
    - Intended end positions (transparent blue)
    - Actual end positions (solid green)
    - Edges from start to actual: gray for no collision, red for collision

    Args:
        mesh: trimesh.Trimesh object
        walk_results_list: List of dicts from multi_step_constrained_walk_with_surface
        steps: Number of walk steps
        show_mesh: Whether to show mesh
        point_radius: Point display radius
        subsample: If set, subsample to this many points for performance
        bounds: If provided, draw the sampling bounds box with XYZ axis indicators
    """
    n_walks = len(walk_results_list)
    n_steps = steps  # Number of steps (0 to steps-1)

    # Get N from first walk result
    first_vis_data = walk_results_list[0]['vis_data']
    if len(first_vis_data) == 0:
        print("No visualization data (steps=1?)")
        return
    N = first_vis_data[0]['actual_start'].shape[0]

    # Subsample if requested
    if subsample is not None and subsample < N:
        vol_idx = np.random.choice(N, subsample, replace=False)
    else:
        vol_idx = np.arange(N)
    N_vis = len(vol_idx)

    # Extract subsampled data for all walks and steps
    walk_data = []
    for result in walk_results_list:
        vis_data = result['vis_data']
        transitions = []
        for step_data in vis_data:
            transitions.append({
                'start': step_data['actual_start'][vol_idx],
                'intended': step_data['intended_end'][vol_idx],
                'actual': step_data['actual_end'][vol_idx],
                'collision': step_data['collision_mask'][vol_idx]
            })
        walk_data.append(transitions)

    # Initialize polyscope
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    if show_mesh:
        ps_mesh = ps.register_surface_mesh("mesh", mesh.vertices, mesh.faces, transparency=0.3)
        ps_mesh.set_color((0.8, 0.8, 0.8))

    # Draw sampling bounds box and XYZ axis indicators
    if bounds is not None:
        bmin = np.array([bounds[0][0], bounds[1][0], bounds[2][0]], dtype=np.float32)
        bmax = np.array([bounds[0][1], bounds[1][1], bounds[2][1]], dtype=np.float32)
        corners = np.array([
            [bmin[0], bmin[1], bmin[2]], [bmax[0], bmin[1], bmin[2]],
            [bmin[0], bmax[1], bmin[2]], [bmax[0], bmax[1], bmin[2]],
            [bmin[0], bmin[1], bmax[2]], [bmax[0], bmin[1], bmax[2]],
            [bmin[0], bmax[1], bmax[2]], [bmax[0], bmax[1], bmax[2]],
        ], dtype=np.float32)
        box_edges = np.array([
            [0, 1], [2, 3], [4, 5], [6, 7],
            [0, 2], [1, 3], [4, 6], [5, 7],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ], dtype=np.int32)
        ps_bbox = ps.register_curve_network("sampling_bounds", corners, box_edges)
        ps_bbox.set_radius(point_radius * 0.3)
        ps_bbox.set_color((0.9, 0.9, 0.2))
        extent = bmax - bmin
        for axis_name, axis_idx, color in [
            ("bounds_X", 0, (1.0, 0.2, 0.2)),
            ("bounds_Y", 1, (0.2, 1.0, 0.2)),
            ("bounds_Z", 2, (0.2, 0.2, 1.0)),
        ]:
            axis_end = bmin.copy()
            axis_end[axis_idx] += extent[axis_idx]
            ps_ax = ps.register_curve_network(
                axis_name, np.array([bmin, axis_end]), np.array([[0, 1]], dtype=np.int32))
            ps_ax.set_radius(point_radius * 0.8)
            ps_ax.set_color(color)

    # Helper to update visualization
    def update_vis(data):
        collision = data['collision']

        # Start positions (yellow)
        ps_start = ps.register_point_cloud("start", data['start'])
        ps_start.set_radius(point_radius)
        ps_start.set_color((1.0, 0.8, 0.0))

        # Intended end positions (transparent blue)
        ps_intended = ps.register_point_cloud("intended_end", data['intended'])
        ps_intended.set_radius(point_radius)
        ps_intended.set_color((0.3, 0.3, 1.0))
        ps_intended.set_transparency(0.3)

        # Actual end positions (solid green)
        ps_actual = ps.register_point_cloud("actual_end", data['actual'])
        ps_actual.set_radius(point_radius)
        ps_actual.set_color((0.2, 0.8, 0.2))

        # Edges: start -> intended (gray, for all points)
        n_pts = len(data['start'])
        nodes_intended = np.vstack([data['start'], data['intended']])
        edges_intended = np.column_stack([np.arange(n_pts), np.arange(n_pts) + n_pts])
        ps_edges_intended = ps.register_curve_network("edges_to_intended", nodes_intended, edges_intended)
        ps_edges_intended.set_radius(point_radius * 0.3)
        ps_edges_intended.set_color((0.6, 0.6, 0.6))

        # Edges: start -> actual for collision points only (red, thicker)
        coll_idx = np.where(collision)[0]
        if len(coll_idx) > 0:
            nodes = np.vstack([data['start'][coll_idx], data['actual'][coll_idx]])
            edges = np.column_stack([np.arange(len(coll_idx)),
                                     np.arange(len(coll_idx)) + len(coll_idx)])
            ps_edges_c = ps.register_curve_network("edges_collision", nodes, edges)
            ps_edges_c.set_radius(point_radius * 0.5)
            ps_edges_c.set_color((1.0, 0.0, 0.0))

    # Register initial state
    update_vis(walk_data[0][0])

    # State for callback
    state = {'walk': 0, 'step': 0}

    def callback():
        updated = False

        changed_walk, new_walk = ps.imgui.SliderInt("Walk", state['walk'], 0, n_walks - 1)
        if changed_walk:
            state['walk'] = new_walk
            updated = True

        changed_step, new_step = ps.imgui.SliderInt("Step", state['step'], 0, n_steps - 1)
        if changed_step:
            state['step'] = new_step
            updated = True

        if updated:
            # Remove old structures
            ps.remove_point_cloud("start")
            ps.remove_point_cloud("intended_end")
            ps.remove_point_cloud("actual_end")
            ps.remove_curve_network("edges_to_intended")
            try:
                ps.remove_curve_network("edges_collision")
            except:
                pass

            # Update
            data = walk_data[state['walk']][state['step']]
            update_vis(data)

        # Stats
        data = walk_data[state['walk']][state['step']]
        n_coll = data['collision'].sum()

        ps.imgui.Separator()
        ps.imgui.TextUnformatted(f"Walk {state['walk'] + 1}/{n_walks}")
        ps.imgui.TextUnformatted(f"Step {state['step']}/{n_steps - 1}")
        ps.imgui.TextUnformatted(f"Points: {N_vis}")
        ps.imgui.TextUnformatted(f"Collisions: {n_coll} ({100 * n_coll / N_vis:.1f}%)")
        ps.imgui.Separator()
        ps.imgui.TextUnformatted("Yellow: start")
        ps.imgui.TextUnformatted("Blue (transparent): intended end")
        ps.imgui.TextUnformatted("Green: actual end")
        ps.imgui.TextUnformatted("Gray edges: start -> intended")
        ps.imgui.TextUnformatted("Red edges: start -> actual (collision)")

    ps.set_user_callback(callback)
    ps.show()


def process_single_mesh(
        mesh_path: str,
        save_dir: str,
        name: str,
        n_volume_points: int = 32768,
        n_random_walks: int = 10,
        visualize: bool = False,
        visualize_n_points: int = 1000
):
    """
    Process a single mesh: sample volume points, compute SDF, perform random walks.

    Args:
        mesh_path: Path to mesh file (.obj)
        point_cloud_path: Path to point cloud file (.npy)
        save_dir: Directory to save results
        name: Name prefix for saved files
        n_volume_points: Number of volume points to sample
        n_random_walks: Number of random walk iterations
        visualize: If True, visualize random walk results with interactive slider
        visualize_n_points: Number of points to visualize (subsampled for performance)
    """
    total_start = time.time()

    # 1. Load mesh
    start = time.time()
    mesh = trimesh.load(mesh_path, force='mesh')
    # 2. transform point cloud
    surf_points = mesh.sample(4096)
    surf_normal = compute_normals_improved(mesh, surf_points)
    surf_points = np.array(surf_points)
    surf_normal = np.array(surf_normal)
    surf_sdf = np.zeros([surf_points.shape[0], 1])
    # transform mesh
    mesh, z_min, x_avg, y_avg, scale = transform_mesh(mesh)
    surf_points, surf_normal = transform_pointcloud(surf_points, surf_normal, z_min, x_avg, y_avg, scale)
    print(f"Load & transform mesh: {time.time() - start:.2f}s")

    # 3. Build FCPW acceleration structure
    start = time.time()
    fcpw_scene = FCPWScene(mesh, build_vectorized=True)
    print(f"Build FCPW scene: {time.time() - start:.2f}s")

    # 4. Sample volume points outside mesh (uses FCPW contains with BVH)
    start = time.time()
    volume_points = sample_volume_outside_mesh(fcpw_scene, N=n_volume_points)
    print(f"Sample outside points: {time.time() - start:.2f}s")

    # 5. Compute SDF for volume points (using FCPW to mesh surface)
    start = time.time()
    volume_sdf, volume_normal = get_sdf(fcpw_scene, volume_points)
    print(f"Compute SDF: {time.time() - start:.2f}s")

    # 6. Prepare output arrays
    init_ext = np.c_[volume_points, volume_sdf, volume_normal]
    init_surf = np.c_[surf_points, surf_sdf, surf_normal]
    x = np.concatenate((init_ext, init_surf), axis=0)

    # 7. Save inputs
    os.makedirs(os.path.join(save_dir, f'{name}'), exist_ok=True)
    np.save(os.path.join(save_dir, f'{name}/x.npy'), x.astype(np.float16))

    # 8. Perform random walks
    # For training stability, we adopt the first 10 sampled dynamics as base data.
    BASE_WALKS = 10
    PERTURB_SIGMA = 0.05
    start = time.time()
    walk_results_list = []
    base_directions = []
    base_step_lengths = []
    for j in range(n_random_walks):
        if j < BASE_WALKS:
            result = multi_step_constrained_walk_with_surface(
                fcpw_scene,
                volume_points[:, :3],
                surf_points[:, :3],
                steps=3
            )
            base_directions.append(result['directions'])
            base_step_lengths.append(result['step_lengths'])
        else:
            base_idx = j % BASE_WALKS
            perturbed_dirs = base_directions[base_idx] + \
                np.random.randn(*base_directions[base_idx].shape).astype(np.float32) * PERTURB_SIGMA
            norms = np.linalg.norm(perturbed_dirs, axis=1, keepdims=True)
            perturbed_dirs = perturbed_dirs / (norms + 1e-10)
            result = multi_step_constrained_walk_with_surface(
                fcpw_scene,
                volume_points[:, :3],
                surf_points[:, :3],
                steps=3,
                init_directions=perturbed_dirs,
                init_step_lengths=base_step_lengths[base_idx]
            )
        walk_results_list.append(result)
        np.save(os.path.join(save_dir, f'{name}/supervise_{j}.npy'), result['supervise'].astype(np.float16))
        np.save(os.path.join(save_dir, f'{name}/condition_{j}.npy'), result['condition'].astype(np.float16))
    print(f"Random walks ({n_random_walks}x): {time.time() - start:.2f}s")

    last_result = walk_results_list[-1]
    print(f"Output shapes: x={x.shape}, "
          f"supervise={last_result['supervise'].shape}, condition={last_result['condition'].shape}")
    print(f"Total time: {time.time() - total_start:.2f}s")

    # 9. Optional visualization
    if visualize:
        visualize_walk_results(
            mesh=mesh,
            walk_results_list=walk_results_list,
            steps=3,
            subsample=visualize_n_points,
            bounds=GENERAL_BOUNDS
        )


DEFAULT_CATEGORIES = [
    "02801938", "02828884", "02876657", "02933112",
    "02954340", "03001627", "03207941", "03325088", "03513137",
    "03636649", "03710193", "03790512", "03938244", "04004475",
    "04099429", "04330267", "04460130", "04554684",
    "02747177", "02808440", "02843684", "02880940", "02942699",
    "03046257", "03211117", "03337140", "03593526",
    "03642806", "03759954", "03797390", "03948459", "04074963",
    "04225987", "04379243", "04468005",
    "02773838", "02818832", "02871439", "02924116", "02946921",
    "02992529", "03085013", "03261776", "03467517", "03624134",
    "03691459", "03761084", "03928116", "03991062", "04090263",
    "04256520", "04401088",
]


def main():
    parser = argparse.ArgumentParser(description="ShapeNet general mesh preprocessing (all categories)")
    parser.add_argument("--mesh_root", default="../../",
                        help="Root directory containing ShapeNet category folders")
    parser.add_argument("--save_root", default="../../pretrain_data",
                        help="Root directory to save processed data")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Category IDs to process (default: all general categories)")
    parser.add_argument("--n_volume_points", type=int, default=32768)
    parser.add_argument("--n_random_walks", type=int, default=100)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize_n_points", type=int, default=1000)
    args = parser.parse_args()

    categories = args.categories if args.categories is not None else DEFAULT_CATEGORIES

    for category in categories:
        mesh_root = os.path.join(args.mesh_root, category)
        save_root = os.path.join(args.save_root, category)
        os.makedirs(save_root, exist_ok=True)

        dirs = [d for d in os.listdir(mesh_root)
                if os.path.isdir(os.path.join(mesh_root, d))]
        print(f"\nCategory {category}: {len(dirs)} meshes")

        for i, d in enumerate(dirs):
            print(f"\n{'=' * 50}")
            print(f"Processing [{i + 1}/{len(dirs)}]: {d}")
            print('=' * 50)

            save_path = os.path.join(save_root, d, 'x.npy')
            if os.path.exists(save_path):
                print(f"Already processed, skipping.")
                continue

            mesh_path = os.path.join(mesh_root, d, "./models/model_normalized.obj")

            try:
                process_single_mesh(
                    mesh_path=mesh_path,
                    save_dir=save_root,
                    name=d,
                    n_volume_points=args.n_volume_points,
                    n_random_walks=args.n_random_walks,
                    visualize=args.visualize,
                    visualize_n_points=args.visualize_n_points
                )
            except Exception as e:
                print(f"Error processing {d}: {e}")
                import traceback
                traceback.print_exc()
                continue


if __name__ == "__main__":
    main()
