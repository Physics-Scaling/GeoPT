import os
import argparse
import numpy as np
import trimesh
import time
import fcpw
import polyscope as ps

# Per-category ShapeNet IDs and sampling bounds
CATEGORY_CONFIG = {
    "ship": {
        "id": "04530566",
        "bounds": ((-4.0, 5.0), (-1.0, 1.0), (-1.5, 1.5)),
    },
    "car": {
        "id": "02958343",
        "bounds": ((-3.0, 4.2), (0.0, 2.5), (-1.5, 1.5)),
    },
    "airplane": {
        "id": "02691156",
        "bounds": ((-3.0, 4.5), (0.0, 2.5), (-2.8, 2.8)),
    },
}


def compute_normals_improved(mesh, sampled_points):
    _ = mesh.face_normals
    _ = mesh.vertex_normals

    point_cloud = trimesh.PointCloud(vertices=mesh.vertices)
    distances, indices = point_cloud.kdtree.query(sampled_points, k=1)
    nearest_vertices_indices = indices.flatten()

    normals_at_sampled_points = mesh.vertex_normals[nearest_vertices_indices]

    zero_mask = np.linalg.norm(normals_at_sampled_points, axis=1) < 1e-6
    if np.any(zero_mask):
        print(f"Warning: found {np.sum(zero_mask)} zero normals")
        normals_at_sampled_points = handle_zero_normals(mesh, sampled_points, normals_at_sampled_points, zero_mask)

    return normals_at_sampled_points


def handle_zero_normals(mesh, points, current_normals, zero_mask):
    from trimesh.proximity import closest_point

    closest, distance, triangle_id = closest_point(mesh, points[zero_mask])

    face_normals = mesh.face_normals
    replacement_normals = face_normals[triangle_id]

    result_normals = current_normals.copy()
    result_normals[zero_mask] = replacement_normals

    still_zero = np.linalg.norm(result_normals, axis=1) < 1e-6
    if np.any(still_zero):
        print(f"Still {np.sum(still_zero)} zero normals, using random directions")
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
        self.mesh = mesh

        positions = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
        indices = np.ascontiguousarray(mesh.faces, dtype=np.int32)

        self.scene = fcpw.scene_3D()
        self.scene.set_object_count(1)
        self.scene.set_object_vertices(positions, 0)
        self.scene.set_object_triangles(indices, 0)

        aggregate_type = fcpw.aggregate_type.bvh_surface_area
        self.scene.build(aggregate_type, build_vectorized,
                         print_stats=False, reduce_memory_footprint=False)

    def find_closest_points(self, query_points: np.ndarray,
                            return_normals: bool = False) -> tuple:
        query_points = np.ascontiguousarray(query_points, dtype=np.float32)
        n_points = len(query_points)

        squared_max_radii = np.full(n_points, np.inf, dtype=np.float32)

        interactions = fcpw.interaction_3D_list()
        self.scene.find_closest_points(query_points, squared_max_radii, interactions,
                                       record_normal=return_normals)

        closest_points = np.array([i.p for i in interactions], dtype=np.float32)
        distances = np.array([i.d for i in interactions], dtype=np.float32)
        if return_normals:
            normals = np.array([i.n for i in interactions], dtype=np.float32)
            return distances, closest_points, normals

        return distances, closest_points

    def intersect_rays(self, ray_origins: np.ndarray, ray_directions: np.ndarray,
                       return_all_hits: bool = False) -> tuple:
        ray_origins = np.ascontiguousarray(ray_origins, dtype=np.float32)
        ray_directions = np.ascontiguousarray(ray_directions, dtype=np.float32)
        n_rays = len(ray_origins)

        norms = np.linalg.norm(ray_directions, axis=1, keepdims=True)
        ray_directions = ray_directions / (norms + 1e-10)

        ray_distance_bounds = np.full(n_rays, 1e10, dtype=np.float32)

        interactions = fcpw.interaction_3D_list()
        self.scene.intersect(ray_origins, ray_directions, ray_distance_bounds,
                             interactions, check_for_occlusion=False)

        hit_distances = np.full(n_rays, np.inf, dtype=np.float32)
        hit_points = ray_origins.copy()
        hit_mask = np.zeros(n_rays, dtype=bool)

        for i, interaction in enumerate(interactions):
            if interaction.d < np.inf and interaction.d >= 0:
                hit_mask[i] = True
                hit_distances[i] = interaction.d
                hit_points[i] = interaction.p

        if return_all_hits:
            return hit_mask, hit_distances, hit_points, hit_mask.astype(int)

        return hit_mask, hit_distances, hit_points

    def contains(self, points: np.ndarray) -> np.ndarray:
        points = np.ascontiguousarray(points, dtype=np.float32)
        result = np.zeros(len(points), dtype=np.int32)
        self.scene.contains(points, result)
        return result.astype(bool)


def transform_mesh(mesh: trimesh.Trimesh, target_length: float = 5.0):
    """
    Scale and orient mesh: swap X↔Z axes, shift Y to ground, scale to target
    length along X, then center X and Z.
    """
    V = mesh.vertices.copy()

    new_V = np.zeros(V.shape)
    new_V[:, 0] = V[:, 2]
    new_V[:, 1] = V[:, 1]
    new_V[:, 2] = V[:, 0]

    bound_max = np.max(new_V, axis=0)
    bound_min = np.min(new_V, axis=0)

    new_V[:, 1] -= bound_min[1]

    length = bound_max[0] - bound_min[0]
    scale = target_length / length
    new_V *= scale

    x_avg = np.mean(new_V[:, 0])
    new_V[:, 0] -= x_avg

    y_avg = np.mean(new_V[:, 2])
    new_V[:, 2] -= y_avg

    mesh.vertices = new_V
    print(f"Mesh bounds: max={np.max(new_V, axis=0)}, min={np.min(new_V, axis=0)}")
    return mesh, bound_min[1], x_avg, y_avg, scale


def transform_pointcloud(surf_points, surf_normal, z_min, x_avg, y_avg, scale):
    """Apply the same axis-swap transform to a point cloud (mirrors transform_mesh)."""
    new_surf_points = np.zeros_like(surf_points)
    new_surf_normal = np.zeros_like(surf_normal)
    new_surf_points[:, 0] = surf_points[:, 2]
    new_surf_points[:, 1] = surf_points[:, 1]
    new_surf_points[:, 2] = surf_points[:, 0]
    new_surf_normal[:, 0] = surf_normal[:, 2]
    new_surf_normal[:, 1] = surf_normal[:, 1]
    new_surf_normal[:, 2] = surf_normal[:, 0]

    new_surf_points[:, 1] -= z_min
    new_surf_points = new_surf_points * scale
    new_surf_points[:, 0] -= x_avg
    new_surf_points[:, 2] -= y_avg

    print(np.max(new_surf_points, axis=0))
    print(np.min(new_surf_points, axis=0))
    return new_surf_points, new_surf_normal


def get_sdf(fcpw_scene: FCPWScene, target_points: np.ndarray) -> tuple:
    distances, closest_points = fcpw_scene.find_closest_points(target_points)

    diff = closest_points - target_points
    norms = distances.reshape(-1, 1) + 1e-8
    directions = diff / norms

    return distances, directions


def sample_volume_outside_mesh(
        fcpw_scene: FCPWScene,
        N: int = 32768,
        bounds: tuple = None,
        batch_size: int = 65536,
        max_iter: int = 50
) -> np.ndarray:
    """Sample N points outside the mesh within bounds using FCPW contains()."""
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    z_min, z_max = bounds[2]

    outside_points = []
    total_count = 0

    for iter_count in range(max_iter):
        if total_count >= N:
            break

        print(f"iter_count: {iter_count + 1}")

        points = np.random.uniform(
            low=[x_min, y_min, z_min],
            high=[x_max, y_max, z_max],
            size=(batch_size, 3)
        ).astype(np.float32)

        inside_mask = fcpw_scene.contains(points)
        new_points = points[~inside_mask]

        if len(new_points) > 0:
            outside_points.append(new_points)
            total_count += len(new_points)

        print(f"total outside points: {total_count}")

    outside_points = np.concatenate(outside_points, axis=0)[:N]

    if len(outside_points) < N:
        print(f"Warning: after {max_iter} iterations, only sampled {len(outside_points)} points")

    return outside_points


def multi_step_constrained_walk_with_surface(
        fcpw_scene: FCPWScene,
        volume_points: np.ndarray,
        surf_points: np.ndarray,
        steps: int = 3,
        min_step: float = 0.0,
        max_step: float = 2.0,
        init_directions: np.ndarray = None,
        init_step_lengths: np.ndarray = None
) -> dict:
    """
    Perform multi-step constrained random walk for volume points.

    Volume points walk in random directions with FCPW collision detection.
    Surface points remain fixed. At each step, FCPW find_closest_points provides
    the supervision target (closest point on mesh surface).

    Args:
        fcpw_scene: FCPWScene wrapper
        volume_points: (N, 3) array of volume points
        surf_points: (M, 3) array of surface points
        steps: Number of walk steps
        min_step: Minimum step length
        max_step: Maximum step length
        init_directions: (N+M, 3) pre-defined unit directions; if None, sampled randomly
        init_step_lengths: (N+M,) pre-defined step lengths; if None, sampled randomly

    Returns:
        dict with keys:
            supervise: (N+M, 3*steps) supervision vectors
            condition: (N+M, 4) walk conditions [direction, step_length]
            directions: (N+M, 3) directions used (for perturbation reuse)
            step_lengths: (N+M,) step lengths used (for perturbation reuse)
            vis_data: list of `steps` dicts with per-step position data
    """
    N = volume_points.shape[0]
    M = surf_points.shape[0]

    all_points = np.vstack([volume_points, surf_points]).astype(np.float32)
    positions = all_points.copy()

    supervise_list = []
    vis_data_list = []

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

    for step_idx in range(steps):
        _, closest = fcpw_scene.find_closest_points(positions)
        supervise_list.append(positions - closest)

        if step_idx == steps - 1:
            break

        vol_positions = positions[vol_mask].copy()
        vol_dirs = directions[vol_mask]
        vol_step_lengths = step_lengths[vol_mask]

        intended_end = vol_positions + vol_dirs * vol_step_lengths[:, None]

        hit_mask, hit_distances, _ = fcpw_scene.intersect_rays(vol_positions, vol_dirs)

        collision_mask = hit_mask & (hit_distances < vol_step_lengths)
        actual_steps = np.where(collision_mask, hit_distances * 0.99, vol_step_lengths)
        actual_end = vol_positions + vol_dirs * actual_steps[:, None]

        vis_data_list.append({
            'actual_start': vol_positions.copy(),
            'intended_end': intended_end.copy(),
            'actual_end': actual_end.copy(),
            'collision_mask': collision_mask.copy()
        })

        positions[vol_mask] = actual_end
        positions[-M:] = surf_points

    final_vol_positions = positions[vol_mask].copy()
    vis_data_list.append({
        'actual_start': final_vol_positions.copy(),
        'intended_end': final_vol_positions.copy(),
        'actual_end': final_vol_positions.copy(),
        'collision_mask': np.zeros(N, dtype=bool)
    })

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
    Interactive polyscope visualization of random walk results.

    Sliders for Walk and Step index. Shows start (yellow), intended end
    (transparent blue), actual end (green), collision edges (red).
    If bounds is provided, draws the sampling region box with RGB XYZ axes.
    """
    n_walks = len(walk_results_list)
    n_steps = steps

    first_vis_data = walk_results_list[0]['vis_data']
    if len(first_vis_data) == 0:
        print("No visualization data (steps=1?)")
        return
    N = first_vis_data[0]['actual_start'].shape[0]

    if subsample is not None and subsample < N:
        vol_idx = np.random.choice(N, subsample, replace=False)
    else:
        vol_idx = np.arange(N)
    N_vis = len(vol_idx)

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

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    if show_mesh:
        ps_mesh = ps.register_surface_mesh("mesh", mesh.vertices, mesh.faces, transparency=0.3)
        ps_mesh.set_color((0.8, 0.8, 0.8))

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

    def update_vis(data):
        collision = data['collision']

        ps_start = ps.register_point_cloud("start", data['start'])
        ps_start.set_radius(point_radius)
        ps_start.set_color((1.0, 0.8, 0.0))

        ps_intended = ps.register_point_cloud("intended_end", data['intended'])
        ps_intended.set_radius(point_radius)
        ps_intended.set_color((0.3, 0.3, 1.0))
        ps_intended.set_transparency(0.3)

        ps_actual = ps.register_point_cloud("actual_end", data['actual'])
        ps_actual.set_radius(point_radius)
        ps_actual.set_color((0.2, 0.8, 0.2))

        n_pts = len(data['start'])
        nodes_intended = np.vstack([data['start'], data['intended']])
        edges_intended = np.column_stack([np.arange(n_pts), np.arange(n_pts) + n_pts])
        ps_edges_intended = ps.register_curve_network("edges_to_intended", nodes_intended, edges_intended)
        ps_edges_intended.set_radius(point_radius * 0.3)
        ps_edges_intended.set_color((0.6, 0.6, 0.6))

        coll_idx = np.where(collision)[0]
        if len(coll_idx) > 0:
            nodes = np.vstack([data['start'][coll_idx], data['actual'][coll_idx]])
            edges = np.column_stack([np.arange(len(coll_idx)),
                                     np.arange(len(coll_idx)) + len(coll_idx)])
            ps_edges_c = ps.register_curve_network("edges_collision", nodes, edges)
            ps_edges_c.set_radius(point_radius * 0.5)
            ps_edges_c.set_color((1.0, 0.0, 0.0))

    update_vis(walk_data[0][0])

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
            ps.remove_point_cloud("start")
            ps.remove_point_cloud("intended_end")
            ps.remove_point_cloud("actual_end")
            ps.remove_curve_network("edges_to_intended")
            try:
                ps.remove_curve_network("edges_collision")
            except:
                pass
            update_vis(walk_data[state['walk']][state['step']])

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
        bounds: tuple,
        n_volume_points: int = 32768,
        n_random_walks: int = 10,
        visualize: bool = False,
        visualize_n_points: int = 1000
):
    """
    Process a single mesh: transform, build FCPW, sample outside points,
    compute SDF, and perform perturbation-based random walks.
    """
    total_start = time.time()

    # 1. Load mesh
    start = time.time()
    mesh = trimesh.load(mesh_path, force='mesh')

    # 2. Sample surface points before transform
    surf_points = mesh.sample(4096)
    surf_normal = compute_normals_improved(mesh, surf_points)
    surf_points = np.array(surf_points)
    surf_normal = np.array(surf_normal)
    surf_sdf = np.zeros([surf_points.shape[0], 1])

    # 3. Apply axis-swap transform
    mesh, z_min, x_avg, y_avg, scale = transform_mesh(mesh)
    surf_points, surf_normal = transform_pointcloud(surf_points, surf_normal, z_min, x_avg, y_avg, scale)
    print(f"Load & transform mesh: {time.time() - start:.2f}s")

    # 4. Build FCPW BVH
    start = time.time()
    fcpw_scene = FCPWScene(mesh, build_vectorized=True)
    print(f"Build FCPW scene: {time.time() - start:.2f}s")

    # 5. Sample volume points outside mesh
    start = time.time()
    volume_points = sample_volume_outside_mesh(fcpw_scene, N=n_volume_points, bounds=bounds)
    print(f"Sample outside points: {time.time() - start:.2f}s")

    # 6. Compute SDF
    start = time.time()
    volume_sdf, volume_normal = get_sdf(fcpw_scene, volume_points)
    print(f"Compute SDF: {time.time() - start:.2f}s")

    # 7. Prepare and save input arrays
    init_ext = np.c_[volume_points, volume_sdf, volume_normal]
    init_surf = np.c_[surf_points, surf_sdf, surf_normal]
    x = np.concatenate((init_ext, init_surf), axis=0)

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
                fcpw_scene, volume_points[:, :3], surf_points[:, :3], steps=3
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
                fcpw_scene, volume_points[:, :3], surf_points[:, :3], steps=3,
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
            bounds=bounds
        )


def main():
    parser = argparse.ArgumentParser(description="ShapeNet mesh preprocessing for ship/car/airplane")
    parser.add_argument("--category", required=True, choices=list(CATEGORY_CONFIG.keys()),
                        help="Object category to process")
    parser.add_argument("--mesh_root", default="../../",
                        help="Root directory containing ShapeNet category folders")
    parser.add_argument("--save_root", default="../../pretrain_data",
                        help="Root directory to save processed data")
    parser.add_argument("--n_volume_points", type=int, default=32768)
    parser.add_argument("--n_random_walks", type=int, default=100)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize_n_points", type=int, default=100)
    args = parser.parse_args()

    cfg = CATEGORY_CONFIG[args.category]
    category_id = cfg["id"]
    bounds = cfg["bounds"]

    mesh_root = os.path.join(args.mesh_root, category_id)
    save_root = os.path.join(args.save_root, category_id)
    os.makedirs(save_root, exist_ok=True)

    dirs = [d for d in os.listdir(mesh_root)
            if os.path.isdir(os.path.join(mesh_root, d))]
    print(f"Category: {args.category} ({category_id})")
    print(f"Found {len(dirs)} meshes to process")

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
                bounds=bounds,
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