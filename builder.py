from datetime import datetime, timezone
from collections import defaultdict
from stl_parser import *
import numpy as np

def validate_geometry(vertices, edges, triangles):
    # First filter out degenerate triangles
    valid_triangles = []
    for i, (v1, v2, v3, normal) in enumerate(triangles):
        if v1 == v2 or v2 == v3 or v3 == v1:
            print(f"Removing degenerate triangle {i}: vertices ({v1}, {v2}, {v3})")
            print(f"v1 {vertices[v1]} v2 {vertices[v2]} v3 {vertices[v3]}")
            continue
        valid_triangles.append((v1, v2, v3, normal))
    
    print(f"Removed {len(triangles) - len(valid_triangles)} degenerate triangles")
    
    # Now generate edges from valid triangles only
    edge_set = set()
    for v1, v2, v3, _ in valid_triangles:
        # Ensure vertex indices are valid
        norm_v1 = v1 % len(vertices)
        norm_v2 = v2 % len(vertices)
        norm_v3 = v3 % len(vertices)
        
        # Add edges with normalized indices
        edge_set.add(tuple(sorted([norm_v1, norm_v2])))
        edge_set.add(tuple(sorted([norm_v2, norm_v3])))
        edge_set.add(tuple(sorted([norm_v3, norm_v1])))
    
    # Convert edge set back to list
    normalized_edges = list(edge_set)
    
    return normalized_edges, valid_triangles

def get_perpendicular_vectors(normal):
    normal = np.array(normal)
    # First perpendicular vector using cross product with (0,0,1) or (1,0,0)
    if abs(normal[2]) < 0.9:
        perp1 = np.cross(normal, [0, 0, 1])
    else:
        perp1 = np.cross(normal, [1, 0, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    
    # Second perpendicular vector
    perp2 = np.cross(normal, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)
    
    return tuple(perp1), tuple(perp2)
def merge_coplanar_triangles(vertices, triangles, normal_tolerance=0.001):
    def are_normals_equal(n1, n2):
        return np.allclose(n1, n2, rtol=normal_tolerance)
    
    # Create adjacency graph
    adj_graph = defaultdict(list)
    for i, (v1, v2, v3, normal1) in enumerate(triangles):
        for j, (v4, v5, v6, normal2) in enumerate(triangles[i+1:], i+1):
            # Check if triangles share two vertices
            verts1 = {v1, v2, v3}
            verts2 = {v4, v5, v6}
            if len(verts1.intersection(verts2)) == 2:
                # Check if normals are parallel
                if are_normals_equal(normal1, normal2):
                    adj_graph[i].append(j)
                    adj_graph[j].append(i)
    
    # Find connected components (merged triangles)
    merged_groups = []
    visited = set()
    
    for start_tri in range(len(triangles)):
        if start_tri in visited:
            continue
            
        # BFS to find all connected coplanar triangles
        group = []
        queue = [start_tri]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
                
            visited.add(current)
            group.append(current)
            
            for neighbor in adj_graph[current]:
                if neighbor not in visited:
                    queue.append(neighbor)
        
        if group:
            merged_groups.append(group)
    
    return merged_groups

def calculate_cylinder_properties(vertices, face_indices, triangles):
    # Get face centers and normals
    face_centers = []
    face_normals = []
    for idx in face_indices:
        v1, v2, v3, normal = triangles[idx]
        center = np.mean([vertices[v1], vertices[v2], vertices[v3]], axis=0)
        face_centers.append(center)
        face_normals.append(np.array(normal))
    
    face_centers = np.array(face_centers)
    face_normals = np.array(face_normals)
    
    # Calculate axis from cross products
    axis_vector = np.zeros(3)
    for i in range(len(face_normals)):
        cross = np.cross(face_normals[i], face_normals[(i+1)%len(face_normals)])
        if np.linalg.norm(cross) > 1e-10:
            cross = cross / np.linalg.norm(cross)
            if np.dot(cross, axis_vector) < 0:
                cross = -cross
            axis_vector += cross
    axis_vector = axis_vector / np.linalg.norm(axis_vector)
    
    # Project centers and normals onto plane perpendicular to axis
    proj_centers = face_centers - np.outer(np.dot(face_centers - face_centers[0], axis_vector), axis_vector)
    proj_normals = face_normals - np.outer(np.dot(face_normals, axis_vector), axis_vector)
    proj_normals = np.array([n/np.linalg.norm(n) if np.linalg.norm(n) > 1e-10 else n for n in proj_normals])
    
    # Find center by intersecting normal lines
    center_point = np.zeros(3)
    count = 0
    for i in range(len(proj_centers)):
        for j in range(i+1, len(proj_centers)):
            n1, n2 = proj_normals[i], proj_normals[j]
            p1, p2 = proj_centers[i], proj_centers[j]
            
            # Solve for intersection of lines p1 + t1*n1 = p2 + t2*n2
            A = np.column_stack([n1, -n2])
            if np.linalg.matrix_rank(A) == 2:
                b = p2 - p1
                t1, t2 = np.linalg.solve(A.T @ A, A.T @ b)
                intersection = p1 + t1 * n1
                center_point += intersection
                count += 1
    
    if count > 0:
        center_point = center_point / count
    else:
        center_point = np.mean(proj_centers, axis=0)
    
    # Calculate diameter using distance from center to projected points
    radii = np.linalg.norm(proj_centers - center_point, axis=1)
    diameter = 2 * np.mean(radii)
    
    return diameter, tuple(center_point), tuple(axis_vector)

def detect_cylindrical_faces(vertices, triangles, angle_tolerance=0.001, area_tolerance=0.05, normal_tolerance=0.001, max_angle=np.pi/12):
    # First merge coplanar triangles
    merged_groups = merge_coplanar_triangles(vertices, triangles, normal_tolerance)
    
    def calculate_triangle_area(v1_idx, v2_idx, v3_idx):
        v1 = np.array(vertices[v1_idx])
        v2 = np.array(vertices[v2_idx])
        v3 = np.array(vertices[v3_idx])
        return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

    # Calculate areas and normals for merged groups
    merged_faces = []
    for group in merged_groups:
        total_area = sum(calculate_triangle_area(
            triangles[idx][0], 
            triangles[idx][1], 
            triangles[idx][2]
        ) for idx in group)
        normal = triangles[group[0]][3]
        merged_faces.append((group, total_area, normal))

    # Group merged faces by area with tolerance
    area_groups = defaultdict(list)
    for i, (group, area, normal) in enumerate(merged_faces):
        rounded_area = round(area / area_tolerance) * area_tolerance
        area_groups[rounded_area].append((i, group, normal))

    # Handle split faces
    small_groups = {area: faces for area, faces in area_groups.items() 
                   if len(faces) < 3}
    
    for small_area, small_faces in small_groups.items():
        double_area = small_area * 2
        if double_area in area_groups and len(area_groups[double_area]) >= 3:
            # Check if faces are adjacent and have consistent normals
            for small_idx, small_group, small_normal in small_faces:
                for large_idx, large_group, large_normal in area_groups[double_area]:
                    if are_groups_adjacent(small_group, large_group, triangles):
                        angle = calculate_angle_between_faces(small_normal, large_normal)
                        if angle < max_angle:
                            # Move to larger group and adjust area
                            area_groups[double_area].append((small_idx, small_group, small_normal))
                            area_groups[small_area].remove((small_idx, small_group, small_normal))

    # Rest of the function remains the same...
    # (Process area groups to find cylindrical components)
    cylindrical_faces = []
    processed = set()

    for area, group_list in area_groups.items():
        if len(group_list) < 3:
            continue
        
        # Build adjacency graph for merged faces
        adj_graph = defaultdict(list)
        for i, (idx1, group1, normal1) in enumerate(group_list):
            if idx1 in processed:
                continue
            
            for j, (idx2, group2, normal2) in enumerate(group_list[i+1:], i+1):
                if idx2 in processed:
                    continue
                
                if are_groups_adjacent(group1, group2, triangles):
                    angle = calculate_angle_between_faces(normal1, normal2)
                    if angle > max_angle:
                        continue
                    adj_graph[idx1].append((idx2, angle))
                    adj_graph[idx2].append((idx1, angle))
        
        # Continue with existing component finding logic...
        visited = set()
        for start_idx, start_group, _ in group_list:
            if start_idx in visited or start_idx in processed:
                continue
            
            component = []
            queue = [(start_idx, None)]
            
            while queue:
                current_idx, expected_angle = queue.pop(0)
                if current_idx in visited:
                    continue
                
                visited.add(current_idx)
                component.append(current_idx)
                
                for neighbor_idx, angle in adj_graph[current_idx]:
                    if neighbor_idx not in visited:
                        if expected_angle is None or abs(angle - expected_angle) < angle_tolerance:
                            queue.append((neighbor_idx, angle))
            if len(component) >= 3:
                # Get all triangle indices in the component
                all_triangles = []
                for comp_idx in component:
                    all_triangles.extend(merged_faces[comp_idx][0])
                
                # Calculate total angle sweep
                normals = [merged_faces[idx][2] for idx in component]
                total_angle = sum(calculate_angle_between_faces(normals[i], normals[i+1])
                                for i in range(len(normals)-1))
                total_angle += calculate_angle_between_faces(normals[-1], normals[0])
                
            if len(component) >= 3:
                # Process component as before...
                all_triangles = []
                for comp_idx in component:
                    group_info = next(g for i, g, _ in group_list if i == comp_idx)
                    all_triangles.extend(group_info)
                diameter, center, axis = calculate_cylinder_properties(vertices, all_triangles, triangles)

                if abs(total_angle - 2 * np.pi) < angle_tolerance:
                    cylindrical_faces.append(("complete_cylinder", all_triangles, diameter, center, axis))
                    print("complete cylinder")
                    print(total_angle)
                else:
                    cylindrical_faces.append(("partial_cylinder", all_triangles, diameter, center, axis))
                    processed.update(component)
    return cylindrical_faces

def are_groups_adjacent(group1, group2, triangles):
    for idx1 in group1:
        verts1 = set([triangles[idx1][0], triangles[idx1][1], triangles[idx1][2]])
        for idx2 in group2:
            verts2 = set([triangles[idx2][0], triangles[idx2][1], triangles[idx2][2]])
            if len(verts1.intersection(verts2)) >= 2:
                return True
    return False

def calculate_angle_between_faces(normal1, normal2):
    n1 = np.array(normal1)
    n2 = np.array(normal2)
    cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
    return np.arccos(cos_angle)


def find_cylinder_boundaries(vertices, triangles, face_indices, axis, center):
    boundary_edges = set()
    cylinder_triangles = set(face_indices)

    # Find boundary edges
    for idx in face_indices:
       v1, v2, v3, _ = triangles[idx]
       for edge in [(v1,v2), (v2,v3), (v3,v1)]:
           edge = tuple(sorted(edge))
           
           # Find triangles sharing this edge
           connecting_faces = []
           for i, (tv1, tv2, tv3, _) in enumerate(triangles):
               if edge[0] in (tv1, tv2, tv3) and edge[1] in (tv1, tv2, tv3):
                   connecting_faces.append(i)
           
           if any(face_idx not in cylinder_triangles for face_idx in connecting_faces):
               boundary_edges.add(edge)

    # Convert to cylinder space
    axis = np.array(axis)
    center = np.array(center)
    ref_dir = get_perpendicular_vectors(axis)[0]

    # Calculate angles
    boundary_points = []
    for v1, v2 in boundary_edges:
       for vertex in [vertices[v1], vertices[v2]]:
           point_rel = np.array(vertex) - center
           point_proj = point_rel - np.dot(point_rel, axis) * axis
           
           angle = np.arctan2(
               np.dot(point_proj, np.cross(axis, ref_dir)),
               np.dot(point_proj, ref_dir)
           )
           # Normalize to [0, 2π]
           if angle < 0:
               angle += 2 * np.pi
           boundary_points.append((angle, vertex))

    # Sort and find gap
    boundary_points.sort(key=lambda x: x[0])
    max_gap = 0
    gap_start_idx = 0

    for i in range(len(boundary_points)):
       angle1 = boundary_points[i][0]
       angle2 = boundary_points[(i+1) % len(boundary_points)][0]
       if angle2 < angle1:
           angle2 += 2 * np.pi
       gap = angle2 - angle1
       if gap > max_gap:
           max_gap = gap
           gap_start_idx = i

    # Points on either side of largest gap are our boundaries
    start_angle = boundary_points[(gap_start_idx + 1) % len(boundary_points)][0]
    end_angle = boundary_points[gap_start_idx][0]
    start_point = boundary_points[(gap_start_idx + 1) % len(boundary_points)][1]
    end_point = boundary_points[gap_start_idx][1]

    return start_angle, end_angle, start_point, end_point

def generate_partial_cylinder(vertices, triangles, face_indices, diameter, center, axis):
    start_angle, end_angle, start_point, end_point = find_cylinder_boundaries(
        vertices, triangles, face_indices, axis, center
    )
    
    axis = np.array(axis)
    center = np.array(center)
    radius = diameter / 2
    
    points = np.array([vertices[i] for tri in face_indices for i in triangles[tri][:3]])
    heights = np.dot(points - center, axis)
    min_height = np.min(heights)
    max_height = np.max(heights)
    height = max_height - min_height
    
    ref_dir = np.array(get_perpendicular_vectors(axis)[0])  # Convert to numpy array
    cross_dir = np.cross(axis, ref_dir)
    
    bottom_center = center + min_height * axis
    top_center = center + max_height * axis
    
    start_bottom = bottom_center + radius * (
        ref_dir * np.cos(start_angle) + 
        cross_dir * np.sin(start_angle)
    )
    end_bottom = bottom_center + radius * (
        ref_dir * np.cos(end_angle) + 
        cross_dir * np.sin(end_angle)
    )
    
    start_top = start_bottom + height * axis
    end_top = end_bottom + height * axis
    
    return start_bottom, end_bottom, start_top, end_top

def generate_step_entities(vertices, edges, triangles, start_id=100):
    entities = []
    current_id = start_id
    all_face_ids = []
    
    geometry = {
        'cartesian_points': {},  # vertex_idx -> id
        'vertex_points': {},     # vertex_idx -> id
        'directions': {},        # (dx,dy,dz) -> id
        'vectors': {},          # edge_idx -> id
        'lines': {},            # edge_idx -> id
        'edge_curves': {},      # edge_idx -> id
        'face_mappings': {
            'planar_to_cylindrical': {},  # planar_face_id -> cylindrical_face_id
            'cylindrical_components': {}   # cylindrical_face_id -> [replaced_planar_faces]
        },
        'edge_mappings': {
            'planar': {
                'axis_start_end': {},      # cyl_id -> {edge_id: oriented_edge_id}
                'axis_intermediate': {},    # cyl_id -> {edge_id: oriented_edge_id}
                'circumferential': {},      # cyl_id -> {edge_id: oriented_edge_id}
            },
            'cylindrical': {
                'vertical_edges': {},       # cyl_id -> {start: oriented_edge_id, end: oriented_edge_id}
                'curved_edges': {},         # cyl_id -> {top: oriented_edge_id, bottom: oriented_edge_id}
            },
            'replacements': {
                'vertical': {},            # planar_edge_id -> cylindrical_edge_id
                'curved': {}               # list of planar_edge_ids -> cylindrical_edge_id
            }
        }
    }
    # Generate base points and vertices
    for i, (x, y, z) in enumerate(vertices):
        entities.append(f"#{current_id}=CARTESIAN_POINT('',({x:.6f},{y:.6f},{z:.6f}));")
        geometry['cartesian_points'][i] = current_id
        current_id += 1
        
        entities.append(f"#{current_id}=VERTEX_POINT('',#{geometry['cartesian_points'][i]});")
        geometry['vertex_points'][i] = current_id
        current_id += 1
    
    # Generate edge geometry
    for i, (v1, v2) in enumerate(edges):
        p1 = np.array(vertices[v1])
        p2 = np.array(vertices[v2])
        direction = p2 - p1
        direction = direction / np.linalg.norm(direction)
        direction_tuple = tuple(direction)
        
        if direction_tuple not in geometry['directions']:
            entities.append(f"#{current_id}=DIRECTION('',({direction[0]:.6f},{direction[1]:.6f},{direction[2]:.6f}));")
            geometry['directions'][direction_tuple] = current_id
            current_id += 1
        
        entities.append(f"#{current_id}=VECTOR('',#{geometry['directions'][direction_tuple]},1.0);")
        geometry['vectors'][i] = current_id
        current_id += 1
        
        entities.append(f"#{current_id}=LINE('',#{geometry['cartesian_points'][v1]},#{geometry['vectors'][i]});")
        geometry['lines'][i] = current_id
        current_id += 1
        
        entities.append(f"#{current_id}=EDGE_CURVE('',#{geometry['vertex_points'][v1]},#{geometry['vertex_points'][v2]},#{geometry['lines'][i]},.T.);")
        geometry['edge_curves'][i] = current_id
        current_id += 1

        # First detect cylindrical faces and collect their triangles
    cylinder_faces = detect_cylindrical_faces(vertices, triangles)
    excluded_triangles = set()
    cylinder_triangle_map = {}  # triangle_idx -> (cyl_idx, type)
    
    for cyl_idx, (type_name, face_indices, diameter, center, axis) in enumerate(cylinder_faces):
        # Calculate axis direction and reference vectors for later use
        axis_np = np.array(axis)
        for tri_idx in face_indices:
            excluded_triangles.add(tri_idx)
            cylinder_triangle_map[tri_idx] = (cyl_idx, type_name)
            
        # Pre-analyze edges for this cylinder
        # Collect all edges from triangles in this cylinder
        cylinder_edges = set()
        for tri_idx in face_indices:
            v1, v2, v3, _ = triangles[tri_idx]
            for edge in [(v1, v2), (v2, v3), (v3, v1)]:
                cylinder_edges.add(tuple(sorted(edge)))
                
        # Classify edges based on their orientation relative to cylinder axis
        parallel_edges = []
        perpendicular_edges = []
        
        for edge in cylinder_edges:
            v1, v2 = edge
            edge_vector = np.array(vertices[v2]) - np.array(vertices[v1])
            edge_vector = edge_vector / np.linalg.norm(edge_vector)
            
            # Check alignment with axis
            alignment = abs(np.dot(edge_vector, axis_np))
            if alignment > 0.95:  # Nearly parallel
                parallel_edges.append(edge)
            else:  # Perpendicular or angled
                perpendicular_edges.append(edge)
        
        # Store in geometry structure for later use
        geometry['edge_mappings']['planar']['axis_start_end'][cyl_idx] = {}
        geometry['edge_mappings']['planar']['axis_intermediate'][cyl_idx] = {}
        geometry['edge_mappings']['planar']['circumferential'][cyl_idx] = {}
        
        # Further analyze parallel edges to identify start/end vs intermediate
        if type_name == "partial_cylinder":
            # Calculate angle around axis for each edge
            edge_angles = []
            for edge in parallel_edges:
                v1, v2 = edge
                p1 = np.array(vertices[v1]) - np.array(center)
                p1 = p1 - np.dot(p1, axis_np) * axis_np
                ref_dir = np.array(get_perpendicular_vectors(axis)[0])
                angle = np.arctan2(np.dot(p1, np.cross(axis_np, ref_dir)), np.dot(p1, ref_dir))
                if angle < 0:
                    angle += 2 * np.pi
                edge_angles.append((angle, edge))
            
            # Sort by angle and identify extremes
            edge_angles.sort()
            start_angle = edge_angles[0][0]
            end_angle = edge_angles[-1][0]
            
            # Classify parallel edges
            for angle, edge in edge_angles:
                if abs(angle - start_angle) < 0.01 or abs(angle - end_angle) < 0.01:
                    geometry['edge_mappings']['planar']['axis_start_end'][cyl_idx][edge] = None
                else:
                    geometry['edge_mappings']['planar']['axis_intermediate'][cyl_idx][edge] = None
        
        # Store perpendicular edges
        for edge in perpendicular_edges:
            geometry['edge_mappings']['planar']['circumferential'][cyl_idx][edge] = None
    
    merged_groups = merge_coplanar_triangles(vertices, triangles, normal_tolerance=0.001)
    planar_face_groups = {}  # Maps triangle_idx -> planar_face_id
    
    for group_idx, group_triangles in enumerate(merged_groups):
        # Skip if all triangles are part of cylinder
        if all(idx in excluded_triangles for idx in group_triangles):
            # Map these triangles to their cylinder
            cyl_idx = cylinder_triangle_map[group_triangles[0]]
            if cyl_idx not in geometry['face_mappings']['cylindrical_components']:
                geometry['face_mappings']['cylindrical_components'][cyl_idx] = []
            geometry['face_mappings']['cylindrical_components'][cyl_idx].extend(group_triangles)
            continue
        
        v1, v2, v3, normal = triangles[group_triangles[0]]
        outer_loop, inner_loops = get_face_boundaries(vertices, triangles, group_triangles, edges)
    
        # Add after outer_loop, inner_loops = get_face_boundaries(...)
        normal_tuple = tuple(normal)
        if normal_tuple not in geometry['directions']:
            entities.append(f"#{current_id}=DIRECTION('',({normal[0]:.6f},{normal[1]:.6f},{normal[2]:.6f}));")
            geometry['directions'][normal_tuple] = current_id
            current_id += 1

        perp1, _ = get_perpendicular_vectors(normal)
        perp_tuple = tuple(perp1)
        if perp_tuple not in geometry['directions']:
            entities.append(f"#{current_id}=DIRECTION('',({perp1[0]:.6f},{perp1[1]:.6f},{perp1[2]:.6f}));")
            geometry['directions'][perp_tuple] = current_id
            current_id += 1
        # Create placements and plane
        entities.append(f"#{current_id}=AXIS2_PLACEMENT_3D('',#{geometry['cartesian_points'][v1]},#{geometry['directions'][normal_tuple]},#{geometry['directions'][perp_tuple]});")
        axis_placement_id = current_id
        current_id += 1

        entities.append(f"#{current_id}=PLANE('',#{axis_placement_id});")
        plane_id = current_id
        current_id += 1

        # Initialize for edge processing
        edge_lookup = {tuple(sorted([v1, v2])): i for i, (v1, v2) in enumerate(edges)}
        outer_oriented_edges = []
    for group_triangles in merged_groups:
        if all(idx in excluded_triangles for idx in group_triangles):
            continue
        
        # When creating ORIENTED_EDGEs for planar faces, store their IDs:
        for edge in outer_loop:
            sorted_edge = tuple(sorted(edge))
            edge_idx = edge_lookup[sorted_edge]
            orientation = '.T.' if edge == sorted_edge else '.F.'
            
            entities.append(f"#{current_id}=ORIENTED_EDGE('',*,*,#{geometry['edge_curves'][edge_idx]},{orientation});")
            
            # Store in appropriate mapping if this edge is part of a cylinder
            for cyl_idx in geometry['edge_mappings']['planar']['axis_start_end']:
                if sorted_edge in geometry['edge_mappings']['planar']['axis_start_end'][cyl_idx]:
                    geometry['edge_mappings']['planar']['axis_start_end'][cyl_idx][sorted_edge] = current_id
                elif sorted_edge in geometry['edge_mappings']['planar']['axis_intermediate'][cyl_idx]:
                    geometry['edge_mappings']['planar']['axis_intermediate'][cyl_idx][sorted_edge] = current_id
                elif sorted_edge in geometry['edge_mappings']['planar']['circumferential'][cyl_idx]:
                    geometry['edge_mappings']['planar']['circumferential'][cyl_idx][sorted_edge] = current_id
            
            outer_oriented_edges.append(current_id)
            current_id += 1
            
    # Process cylindrical faces
    for cyl_idx, (type_name, face_indices, diameter, center, axis) in enumerate(cylinder_faces):
        replaced_faces = geometry['face_mappings']['cylindrical_components'].get(cyl_idx, [])

        center_np = np.array(center)
        axis_np = np.array(axis)
        ref_dir = np.array(get_perpendicular_vectors(axis)[0])
        radius = diameter/2
        
        # Get height bounds
        points = np.array([vertices[i] for tri in face_indices for i in triangles[tri][:3]])
        heights = np.dot(points - center_np, axis_np)
        min_height = np.min(heights)
        max_height = np.max(heights)
        height = max_height - min_height
        
        bottom_center = center_np + min_height * axis_np
        top_center = center_np + max_height * axis_np
        
        entities.append(f"#{current_id}=DIRECTION('',({axis[0]:.6f},{axis[1]:.6f},{axis[2]:.6f}));")
        direction_ids = {'axis': current_id}
        current_id += 1

        entities.append(f"#{current_id}=DIRECTION('',({ref_dir[0]:.6f},{ref_dir[1]:.6f},{ref_dir[2]:.6f}));")
        direction_ids['ref'] = current_id
        current_id += 1

        # Create center points for placements
        entities.append(f"#{current_id}=CARTESIAN_POINT('',({bottom_center[0]:.6f},{bottom_center[1]:.6f},{bottom_center[2]:.6f}));")
        bottom_center_id = current_id
        current_id += 1

        entities.append(f"#{current_id}=CARTESIAN_POINT('',({top_center[0]:.6f},{top_center[1]:.6f},{top_center[2]:.6f}));")
        top_center_id = current_id
        current_id += 1

        # Create placements
        placement_ids = {}
        for name, center_id in [('bottom_circle', bottom_center_id), ('top_circle', top_center_id)]:
            entities.append(f"#{current_id}=AXIS2_PLACEMENT_3D('',#{center_id},#{direction_ids['axis']},#{direction_ids['ref']});")
            placement_ids[name] = current_id
            current_id += 1
        
        if type_name == "complete_cylinder":
            # Create points
            bottom_point = bottom_center + radius * ref_dir
            top_point = top_center + radius * ref_dir
            
            point_ids = {}
            for name, point in [
                ('surface_origin', bottom_center),
                ('bottom_center', bottom_center),
                ('top_center', top_center),
                ('bottom_point', bottom_point),
                ('top_point', top_point)
            ]:
                entities.append(f"#{current_id}=CARTESIAN_POINT('',({point[0]:.6f},{point[1]:.6f},{point[2]:.6f}));")
                point_ids[name] = current_id
                current_id += 1
            
            # Vector for vertical line
            entities.append(f"#{current_id}=VECTOR('',#{direction_ids['axis']},{height:.6f});")
            line_vector_id = current_id
            current_id += 1
            
            # Create vertex points
            entities.append(f"#{current_id}=VERTEX_POINT('',#{point_ids['bottom_point']});")
            bottom_point_id = current_id
            current_id += 1
            
            entities.append(f"#{current_id}=VERTEX_POINT('',#{point_ids['top_point']});")
            top_point_id = current_id
            current_id += 1
            
            # Line and circles
            entities.append(f"#{current_id}=LINE('',#{bottom_point_id},#{line_vector_id});")
            line_id = current_id
            current_id += 1

            # Line and circles
            entities.append(f"#{current_id}=LINE('',#{bottom_point_id},#{line_vector_id});")
            line_id = current_id
            current_id += 1

            placement_ids = {}
            for name, point_id in [
                ('cylinder', 'surface_origin'),
                ('bottom_circle', 'bottom_center'),
                ('top_circle', 'top_center')
            ]:
                entities.append(f"#{current_id}=AXIS2_PLACEMENT_3D('',#{point_ids[point_id]},#{direction_ids['axis']},#{direction_ids['ref']});")
                placement_ids[name] = current_id
                current_id += 1


            entities.append(f"#{current_id}=CIRCLE('',#{placement_ids['bottom_circle']},{radius:.6f});")
            bottom_circle_id = current_id
            current_id += 1

            entities.append(f"#{current_id}=CIRCLE('',#{placement_ids['top_circle']},{radius:.6f});")
            top_circle_id = current_id
            current_id += 1

            # Edge curves
            entities.append(f"#{current_id}=EDGE_CURVE('',#{bottom_point_id},#{top_point_id},#{line_id},.T.);")
            line_edge_id = current_id
            current_id += 1

            entities.append(f"#{current_id}=EDGE_CURVE('',#{bottom_point_id},#{bottom_point_id},#{bottom_circle_id},.T.);")
            bottom_edge_id = current_id
            current_id += 1

            entities.append(f"#{current_id}=EDGE_CURVE('',#{top_point_id},#{top_point_id},#{top_circle_id},.T.);")
            top_edge_id = current_id
            current_id += 1
            
            # Create oriented edges with mapping
            geometry['edge_mappings']['cylindrical']['vertical_edges'][cyl_idx] = {}
            geometry['edge_mappings']['cylindrical']['curved_edges'][cyl_idx] = {}
            
            oriented_edge_ids = []
            # Map oriented edges as they're created
            for edge_type, edge_id, orientation in [
                ('top', top_edge_id, '.T.'),
                ('vertical_start', line_edge_id, '.F.'),
                ('bottom', bottom_edge_id, '.F.'),
                ('vertical_end', line_edge_id, '.T.')
            ]:
                entities.append(f"#{current_id}=ORIENTED_EDGE('',*,*,#{edge_id},{orientation});")
                
                if edge_type in ('vertical_start', 'vertical_end'):
                    geometry['edge_mappings']['cylindrical']['vertical_edges'][cyl_idx][edge_type] = current_id
                else:
                    geometry['edge_mappings']['cylindrical']['curved_edges'][cyl_idx][edge_type] = current_id
                
                oriented_edge_ids.append(current_id)
                current_id += 1

            entities.append(f"#{current_id}=EDGE_LOOP('',({','.join(f'#{e}' for e in oriented_edge_ids)}));")
            edge_loop_id = current_id
            current_id += 1
            
            # Create face bound
            entities.append(f"#{current_id}=FACE_BOUND('',#{edge_loop_id},.T.);")
            face_bound_id = current_id
            current_id += 1
            
            # Create cylindrical surface
            entities.append(f"#{current_id}=CYLINDRICAL_SURFACE('',#{placement_ids['cylinder']},{radius:.6f});")
            surface_id = current_id
            current_id += 1
            
            # Create advanced face
            entities.append(f"#{current_id}=ADVANCED_FACE('',(#{face_bound_id}),#{surface_id},.T.);")
            all_face_ids.append(current_id)
            current_id += 1
        else:  # partial cylinder
            start_bottom, end_bottom, start_top, end_top = generate_partial_cylinder(
                vertices, triangles, face_indices, diameter, center, axis)
            
            # Create points
            point_ids = {}
            for name, point in [
                ('surface_origin', bottom_center),
                ('bottom_center', bottom_center),
                ('top_center', top_center),
                ('start_bottom', start_bottom),
                ('end_bottom', end_bottom),
                ('start_top', start_top),
                ('end_top', end_top)
            ]:
                entities.append(f"#{current_id}=CARTESIAN_POINT('',({point[0]:.6f},{point[1]:.6f},{point[2]:.6f}));")
                point_ids[name] = current_id
                current_id += 1
            
            # Create directions
            direction_ids = {}
            for name, dir_vector in [('axis', axis), ('ref', ref_dir)]:
                entities.append(f"#{current_id}=DIRECTION('',({dir_vector[0]:.6f},{dir_vector[1]:.6f},{dir_vector[2]:.6f}));")
                direction_ids[name] = current_id
                current_id += 1
            
            # Create vertex points
            vertex_ids = {}
            for name in ['start_bottom', 'end_bottom', 'start_top', 'end_top']:
                entities.append(f"#{current_id}=VERTEX_POINT('',#{point_ids[name]});")
                vertex_ids[name] = current_id
                current_id += 1
            
            # Create placements
            placement_ids = {}
            for name in ['bottom', 'top']:
                center_id = point_ids[f'{name}_center']
                entities.append(f"#{current_id}=AXIS2_PLACEMENT_3D('',#{center_id},#{direction_ids['axis']},#{direction_ids['ref']});")
                placement_ids[name] = current_id
                current_id += 1
            
            # Create circles
            entities.append(f"#{current_id}=CIRCLE('',#{placement_ids['bottom']},{radius:.6f});")
            bottom_circle_id = current_id
            current_id += 1
            
            entities.append(f"#{current_id}=CIRCLE('',#{placement_ids['top']},{radius:.6f});")
            top_circle_id = current_id
            current_id += 1
            
            # Create vectors and lines for vertical edges
            entities.append(f"#{current_id}=VECTOR('',#{direction_ids['axis']},{height:.6f});")
            line_vector_id = current_id
            current_id += 1
            
            line_ids = {}
            for name in ['start', 'end']:
                entities.append(f"#{current_id}=LINE('',#{point_ids[f'{name}_bottom']},#{line_vector_id});")
                line_ids[name] = current_id
                current_id += 1
            # Create oriented edges with mapping
            geometry['edge_mappings']['cylindrical']['vertical_edges'][cyl_idx] = {}
            geometry['edge_mappings']['cylindrical']['curved_edges'][cyl_idx] = {}
            
            # Map oriented edges as they're created
            # Create edge curves
            edge_curve_ids = {
                'bottom_arc': (vertex_ids['start_bottom'], vertex_ids['end_bottom'], bottom_circle_id),
                'top_arc': (vertex_ids['start_top'], vertex_ids['end_top'], top_circle_id),
                'start_line': (vertex_ids['start_bottom'], vertex_ids['start_top'], line_ids['start']),
                'end_line': (vertex_ids['end_bottom'], vertex_ids['end_top'], line_ids['end'])
            }
            
            for name, (start_vertex, end_vertex, curve) in edge_curve_ids.items():
                entities.append(f"#{current_id}=EDGE_CURVE('',#{start_vertex},#{end_vertex},#{curve},.T.);")
                edge_curve_ids[name] = current_id
                current_id += 1
                entities.append(f"#{current_id}=ORIENTED_EDGE('',*,*,#{edge_id},{orientation});")
                
                if edge_type.startswith('vertical_'):
                    geometry['edge_mappings']['cylindrical']['vertical_edges'][cyl_idx][edge_type] = current_id
                else:
                    geometry['edge_mappings']['cylindrical']['curved_edges'][cyl_idx][edge_type] = current_id
                
                oriented_edge_ids.append(current_id)
                current_id += 1
                
    # Create final mapping between planar and cylindrical edges
    for cyl_idx in geometry['edge_mappings']['planar']['axis_start_end']:
        # Map vertical edges (start/end)
        for planar_edge, planar_oriented_id in geometry['edge_mappings']['planar']['axis_start_end'][cyl_idx].items():
            if planar_oriented_id is not None:
                cylindrical_oriented_id = geometry['edge_mappings']['cylindrical']['vertical_edges'][cyl_idx].get('vertical_start')
                if cylindrical_oriented_id:
                    geometry['edge_mappings']['replacements']['vertical'][planar_oriented_id] = cylindrical_oriented_id
        
        # Map curved edges (replacing circumferential)
        circumferential_edges = []
        for planar_edge, planar_oriented_id in geometry['edge_mappings']['planar']['circumferential'][cyl_idx].items():
            if planar_oriented_id is not None:
                circumferential_edges.append(planar_oriented_id)
        
        if circumferential_edges:
            top_curve_id = geometry['edge_mappings']['cylindrical']['curved_edges'][cyl_idx].get('top')
            bottom_curve_id = geometry['edge_mappings']['cylindrical']['curved_edges'][cyl_idx].get('bottom')
            if top_curve_id and bottom_curve_id:
                geometry['edge_mappings']['replacements']['curved'][tuple(circumferential_edges)] = [top_curve_id, bottom_curve_id]

                    # Create edge loop
            entities.append(f"#{current_id}=EDGE_LOOP('',({','.join(f'#{e}' for e in oriented_edge_ids)}));")
            edge_loop_id = current_id
            current_id += 1
            
            # Create face bound
            entities.append(f"#{current_id}=FACE_BOUND('',#{edge_loop_id},.T.);")
            face_bound_id = current_id
            current_id += 1
            
            # Create cylindrical surface
            entities.append(f"#{current_id}=CYLINDRICAL_SURFACE('',#{placement_ids['top']},{radius:.6f});")
            surface_id = current_id
            current_id += 1
            
            # Create advanced face
            entities.append(f"#{current_id}=ADVANCED_FACE('',(#{face_bound_id}),#{surface_id},.T.);")
            all_face_ids.append(current_id)
            current_id += 1

    # Create closed shell
    face_list = ",".join(f"#{id}" for id in all_face_ids)
    entities.append(f"#{current_id}=CLOSED_SHELL('',({face_list}));")
    closed_shell_id = current_id
    current_id += 1
    
    # Debug output for face and edge relationships
    print("\n=== Face Mapping Analysis ===")
    for planar_id, cylindrical_id in geometry['face_mappings']['planar_to_cylindrical'].items():
        print(f"Planar Face #{planar_id} was replaced by Cylindrical Face #{cylindrical_id}")
        
    for cyl_id, planar_faces in geometry['face_mappings']['cylindrical_components'].items():
        print(f"\nCylindrical Face Component #{cyl_id} replaced planar triangles:", planar_faces)
    
    print("\n=== Edge Mapping Analysis ===")
    for cyl_idx in geometry['edge_mappings']['planar']['axis_start_end']:
        print(f"\nCylinder #{cyl_idx} Edge Replacements:")
        
        print("\nVertical Edges:")
        print("Start/End edges (original -> replacement):")
        print(geometry['edge_mappings']['planar']['axis_start_end'][cyl_idx].items())
        for edge, oriented_id in geometry['edge_mappings']['planar']['axis_start_end'][cyl_idx].items():
            if oriented_id is not None:
                replaced_by = geometry['edge_mappings']['replacements']['vertical'].get(oriented_id, "Not replaced")
                print(f"  Edge {edge} (#{oriented_id}) -> #{replaced_by}")
        
        print("\nIntermediate vertical edges that were removed:")
        print(geometry['edge_mappings']['planar']['axis_intermediate'][cyl_idx].items())
        for edge, oriented_id in geometry['edge_mappings']['planar']['axis_intermediate'][cyl_idx].items():
            if oriented_id is not None:
                print(f"  Edge {edge} (#{oriented_id})")
        
        print("\nCircumferential edges replaced by curves:")
        print(geometry['edge_mappings']['planar']['circumferential'][cyl_idx].items())
        for edge, oriented_id in geometry['edge_mappings']['planar']['circumferential'][cyl_idx].items():
            if oriented_id is not None:
                print(f"  Edge {edge} (#{oriented_id})")
        
        print("\nNew cylindrical edges:")
        if cyl_idx in geometry['edge_mappings']['cylindrical']['vertical_edges']:
            print("  Vertical edges:", geometry['edge_mappings']['cylindrical']['vertical_edges'][cyl_idx])
        if cyl_idx in geometry['edge_mappings']['cylindrical']['curved_edges']:
            print("  Curved edges:", geometry['edge_mappings']['cylindrical']['curved_edges'][cyl_idx])
    
    print("\n=== Edge Replacement Summary ===")
    print("\nVertical edge replacements (old -> new):")
    for old_id, new_id in geometry['edge_mappings']['replacements']['vertical'].items():
        print(f"  #{old_id} -> #{new_id}")
    
    print("\nCurved edge replacements (multiple old -> new curved edges):")
    for old_edges, new_curves in geometry['edge_mappings']['replacements']['curved'].items():
        print(f"  Original edges {[f'#{e}' for e in old_edges]} replaced by curves {[f'#{c}' for c in new_curves]}")

    return "\n".join(entities), current_id, geometry, closed_shell_id
"""def generate_step_entities(vertices, edges, triangles, start_id=100):
    entities = []
    current_id = start_id
    all_face_ids = []
    
    # Basic geometry data structures
    geometry = {
        'cartesian_points': {},
        'vertex_points': {},
        'directions': {},
        'vectors': {},
        'lines': {},
        'edge_curves': {},
        'face_mappings': {
            'planar_to_cylindrical': {},  # planar_face_id -> cylindrical_face_id
            'cylindrical_components': {}   # cylindrical_face_id -> [replaced_planar_faces]
        }
    }
    
    # Generate base points and vertices
    for i, (x, y, z) in enumerate(vertices):
        entities.append(f"#{current_id}=CARTESIAN_POINT('',({x:.6f},{y:.6f},{z:.6f}));")
        geometry['cartesian_points'][i] = current_id
        current_id += 1
        
        entities.append(f"#{current_id}=VERTEX_POINT('',#{geometry['cartesian_points'][i]});")
        geometry['vertex_points'][i] = current_id
        current_id += 1
    
    # Generate edge geometry
    for i, (v1, v2) in enumerate(edges):
        p1 = np.array(vertices[v1])
        p2 = np.array(vertices[v2])
        direction = p2 - p1
        direction = direction / np.linalg.norm(direction)
        direction_tuple = tuple(direction)
        
        if direction_tuple not in geometry['directions']:
            entities.append(f"#{current_id}=DIRECTION('',({direction[0]:.6f},{direction[1]:.6f},{direction[2]:.6f}));")
            geometry['directions'][direction_tuple] = current_id
            current_id += 1
        
        entities.append(f"#{current_id}=VECTOR('',#{geometry['directions'][direction_tuple]},1.0);")
        geometry['vectors'][i] = current_id
        current_id += 1
        
        entities.append(f"#{current_id}=LINE('',#{geometry['cartesian_points'][v1]},#{geometry['vectors'][i]});")
        geometry['lines'][i] = current_id
        current_id += 1
        
        entities.append(f"#{current_id}=EDGE_CURVE('',#{geometry['vertex_points'][v1]},#{geometry['vertex_points'][v2]},#{geometry['lines'][i]},.T.);")
        geometry['edge_curves'][i] = current_id
        current_id += 1
    
    # Collect all geometry before creating faces
    cylinder_faces = detect_cylindrical_faces(vertices, triangles)
    excluded_triangles = set()
    cylinder_triangle_map = {}  # Maps triangle_idx -> cylinder_idx

    for cyl_idx, (_, face_indices, _, _, _) in enumerate(cylinder_faces):
        for tri_idx in face_indices:
            excluded_triangles.add(tri_idx)
            cylinder_triangle_map[tri_idx] = cyl_idx
    
    # Process planar faces
    merged_groups = merge_coplanar_triangles(vertices, triangles, normal_tolerance=0.001)
    planar_face_groups = {}  # Maps triangle_idx -> planar_face_id
    
    for group_idx, group_triangles in enumerate(merged_groups):
        # Skip if all triangles are part of cylinder
        if all(idx in excluded_triangles for idx in group_triangles):
            # Map these triangles to their cylinder
            cyl_idx = cylinder_triangle_map[group_triangles[0]]
            if cyl_idx not in geometry['face_mappings']['cylindrical_components']:
                geometry['face_mappings']['cylindrical_components'][cyl_idx] = []
            geometry['face_mappings']['cylindrical_components'][cyl_idx].extend(group_triangles)
            continue
        
        v1, v2, v3, normal = triangles[group_triangles[0]]
        outer_loop, inner_loops = get_face_boundaries(vertices, triangles, group_triangles, edges)
        
        # Create normal direction
        normal_tuple = tuple(normal)
        if normal_tuple not in geometry['directions']:
            entities.append(f"#{current_id}=DIRECTION('',({normal[0]:.6f},{normal[1]:.6f},{normal[2]:.6f}));")
            geometry['directions'][normal_tuple] = current_id
            current_id += 1
        
        # Create reference direction
        perp1, _ = get_perpendicular_vectors(normal)
        perp_tuple = tuple(perp1)
        if perp_tuple not in geometry['directions']:
            entities.append(f"#{current_id}=DIRECTION('',({perp1[0]:.6f},{perp1[1]:.6f},{perp1[2]:.6f}));")
            geometry['directions'][perp_tuple] = current_id
            current_id += 1
        
        # Create axis placement
        entities.append(f"#{current_id}=AXIS2_PLACEMENT_3D('',#{geometry['cartesian_points'][v1]},#{geometry['directions'][normal_tuple]},#{geometry['directions'][perp_tuple]});")
        axis_placement_id = current_id
        current_id += 1
        
        # Create plane
        entities.append(f"#{current_id}=PLANE('',#{axis_placement_id});")
        plane_id = current_id
        current_id += 1
        
        # Process outer loop
        edge_lookup = {tuple(sorted([v1, v2])): i for i, (v1, v2) in enumerate(edges)}
        outer_oriented_edges = []
        
        for edge in outer_loop:
            sorted_edge = tuple(sorted(edge))
            edge_idx = edge_lookup[sorted_edge]
            orientation = '.T.' if edge == sorted_edge else '.F.'
            
            entities.append(f"#{current_id}=ORIENTED_EDGE('',*,*,#{geometry['edge_curves'][edge_idx]},{orientation});")
            outer_oriented_edges.append(current_id)
            current_id += 1
        
        # Create outer edge loop and bound
        entities.append(f"#{current_id}=EDGE_LOOP('',({','.join(f'#{e}' for e in outer_oriented_edges)}));")
        outer_loop_id = current_id
        current_id += 1
        
        entities.append(f"#{current_id}=FACE_BOUND('',#{outer_loop_id},.T.);")
        outer_bound_id = current_id
        current_id += 1
        
        # Process inner loops
        inner_bound_ids = []
        for inner_loop in inner_loops:
            inner_oriented_edges = []
            for edge in inner_loop:
                sorted_edge = tuple(sorted(edge))
                edge_idx = edge_lookup[sorted_edge]
                orientation = '.T.' if edge == sorted_edge else '.F.'
                
                entities.append(f"#{current_id}=ORIENTED_EDGE('',*,*,#{geometry['edge_curves'][edge_idx]},{orientation});")
                inner_oriented_edges.append(current_id)
                current_id += 1
            
            entities.append(f"#{current_id}=EDGE_LOOP('',({','.join(f'#{e}' for e in inner_oriented_edges)}));")
            inner_loop_id = current_id
            current_id += 1
            
            entities.append(f"#{current_id}=FACE_BOUND('',#{inner_loop_id},.F.);")
            inner_bound_ids.append(current_id)
            current_id += 1
        
        # Create advanced face
        bound_list = f"#{outer_bound_id}"
        if inner_bound_ids:
            bound_list += "," + ",".join(f"#{id}" for id in inner_bound_ids)
        entities.append(f"#{current_id}=ADVANCED_FACE('',({bound_list}),#{plane_id},.T.);")
        all_face_ids.append(current_id)
        current_id += 1
        
        for tri_idx in group_triangles:
            planar_face_groups[tri_idx] = current_id
        all_face_ids.append(current_id)
        current_id += 1
        
    # Process cylindrical faces
    for cyl_idx, (type_name, face_indices, diameter, center, axis) in enumerate(cylinder_faces):
        replaced_faces = geometry['face_mappings']['cylindrical_components'].get(cyl_idx, [])

        center_np = np.array(center)
        axis_np = np.array(axis)
        ref_dir = np.array(get_perpendicular_vectors(axis)[0])
        radius = diameter/2
        
        # Get height bounds
        points = np.array([vertices[i] for tri in face_indices for i in triangles[tri][:3]])
        heights = np.dot(points - center_np, axis_np)
        min_height = np.min(heights)
        max_height = np.max(heights)
        height = max_height - min_height
        
        bottom_center = center_np + min_height * axis_np
        top_center = center_np + max_height * axis_np
        
        if type_name == "complete_cylinder":
            # Create points
            bottom_point = bottom_center + radius * ref_dir
            top_point = top_center + radius * ref_dir
            
            point_ids = {}
            for name, point in [
                ('surface_origin', bottom_center),
                ('bottom_center', bottom_center),
                ('top_center', top_center),
                ('bottom_point', bottom_point),
                ('top_point', top_point)
            ]:
                entities.append(f"#{current_id}=CARTESIAN_POINT('',({point[0]:.6f},{point[1]:.6f},{point[2]:.6f}));")
                point_ids[name] = current_id
                current_id += 1
            
            # Create directions
            direction_ids = {}
            for name, dir_vector in [('axis', axis), ('ref', ref_dir)]:
                entities.append(f"#{current_id}=DIRECTION('',({dir_vector[0]:.6f},{dir_vector[1]:.6f},{dir_vector[2]:.6f}));")
                direction_ids[name] = current_id
                current_id += 1
            
            # Create vertex points
            entities.append(f"#{current_id}=VERTEX_POINT('',#{point_ids['bottom_point']});")
            bottom_vertex_id = current_id
            current_id += 1
            
            entities.append(f"#{current_id}=VERTEX_POINT('',#{point_ids['top_point']});")
            top_vertex_id = current_id
            current_id += 1
            
            # Create placements
            placement_ids = {}
            for name, point_id in [
                ('cylinder', 'surface_origin'),
                ('bottom_circle', 'bottom_center'),
                ('top_circle', 'top_center')
            ]:
                entities.append(f"#{current_id}=AXIS2_PLACEMENT_3D('',#{point_ids[point_id]},#{direction_ids['axis']},#{direction_ids['ref']});")
                placement_ids[name] = current_id
                current_id += 1
            
            # Create circles
            entities.append(f"#{current_id}=CIRCLE('',#{placement_ids['bottom_circle']},{radius:.6f});")
            bottom_circle_id = current_id
            current_id += 1
            
            entities.append(f"#{current_id}=CIRCLE('',#{placement_ids['top_circle']},{radius:.6f});")
            top_circle_id = current_id
            current_id += 1
            
            # Create line for vertical edges
            entities.append(f"#{current_id}=VECTOR('',#{direction_ids['axis']},{height:.6f});")
            line_vector_id = current_id
            current_id += 1
            
            entities.append(f"#{current_id}=LINE('',#{point_ids['bottom_point']},#{line_vector_id});")
            line_id = current_id
            current_id += 1
            
            # Create edge curves
            entities.append(f"#{current_id}=EDGE_CURVE('',#{bottom_vertex_id},#{top_vertex_id},#{line_id},.T.);")
            line_edge_id = current_id
            current_id += 1
            
            entities.append(f"#{current_id}=EDGE_CURVE('',#{bottom_vertex_id},#{bottom_vertex_id},#{bottom_circle_id},.T.);")
            bottom_edge_id = current_id
            current_id += 1
            
            entities.append(f"#{current_id}=EDGE_CURVE('',#{top_vertex_id},#{top_vertex_id},#{top_circle_id},.T.);")
            top_edge_id = current_id
            current_id += 1
            
            # Create oriented edges
            oriented_edge_ids = []
            for edge_id, orientation in [
                (top_edge_id, '.T.'),
                (line_edge_id, '.F.'),
                (bottom_edge_id, '.F.'),
                (line_edge_id, '.T.')
            ]:
                entities.append(f"#{current_id}=ORIENTED_EDGE('',*,*,#{edge_id},{orientation});")
                oriented_edge_ids.append(current_id)
                current_id += 1
            
            # Create edge loop
            entities.append(f"#{current_id}=EDGE_LOOP('',({','.join(f'#{e}' for e in oriented_edge_ids)}));")
            edge_loop_id = current_id
            current_id += 1
            
            # Create face bound
            entities.append(f"#{current_id}=FACE_BOUND('',#{edge_loop_id},.T.);")
            face_bound_id = current_id
            current_id += 1
            
            # Create cylindrical surface
            entities.append(f"#{current_id}=CYLINDRICAL_SURFACE('',#{placement_ids['cylinder']},{radius:.6f});")
            surface_id = current_id
            current_id += 1
            
            # Create advanced face
            entities.append(f"#{current_id}=ADVANCED_FACE('',(#{face_bound_id}),#{surface_id},.T.);")
            all_face_ids.append(current_id)
            current_id += 1
            
        else:  # partial cylinder
            start_bottom, end_bottom, start_top, end_top = generate_partial_cylinder(
                vertices, triangles, face_indices, diameter, center, axis)
            
            # Create points
            point_ids = {}
            for name, point in [
                ('surface_origin', bottom_center),
                ('bottom_center', bottom_center),
                ('top_center', top_center),
                ('start_bottom', start_bottom),
                ('end_bottom', end_bottom),
                ('start_top', start_top),
                ('end_top', end_top)
            ]:
                entities.append(f"#{current_id}=CARTESIAN_POINT('',({point[0]:.6f},{point[1]:.6f},{point[2]:.6f}));")
                point_ids[name] = current_id
                current_id += 1
            
            # Create directions
            direction_ids = {}
            for name, dir_vector in [('axis', axis), ('ref', ref_dir)]:
                entities.append(f"#{current_id}=DIRECTION('',({dir_vector[0]:.6f},{dir_vector[1]:.6f},{dir_vector[2]:.6f}));")
                direction_ids[name] = current_id
                current_id += 1
            
            # Create vertex points
            vertex_ids = {}
            for name in ['start_bottom', 'end_bottom', 'start_top', 'end_top']:
                entities.append(f"#{current_id}=VERTEX_POINT('',#{point_ids[name]});")
                vertex_ids[name] = current_id
                current_id += 1
            
            # Create placements
            placement_ids = {}
            for name in ['bottom', 'top']:
                center_id = point_ids[f'{name}_center']
                entities.append(f"#{current_id}=AXIS2_PLACEMENT_3D('',#{center_id},#{direction_ids['axis']},#{direction_ids['ref']});")
                placement_ids[name] = current_id
                current_id += 1
            
            # Create circles
            entities.append(f"#{current_id}=CIRCLE('',#{placement_ids['bottom']},{radius:.6f});")
            bottom_circle_id = current_id
            current_id += 1
            
            entities.append(f"#{current_id}=CIRCLE('',#{placement_ids['top']},{radius:.6f});")
            top_circle_id = current_id
            current_id += 1
            
            # Create vectors and lines for vertical edges
            entities.append(f"#{current_id}=VECTOR('',#{direction_ids['axis']},{height:.6f});")
            line_vector_id = current_id
            current_id += 1
            
            line_ids = {}
            for name in ['start', 'end']:
                entities.append(f"#{current_id}=LINE('',#{point_ids[f'{name}_bottom']},#{line_vector_id});")
                line_ids[name] = current_id
                current_id += 1
            
            # Create edge curves
            edge_curve_ids = {
                'bottom_arc': (vertex_ids['start_bottom'], vertex_ids['end_bottom'], bottom_circle_id),
                'top_arc': (vertex_ids['start_top'], vertex_ids['end_top'], top_circle_id),
                'start_line': (vertex_ids['start_bottom'], vertex_ids['start_top'], line_ids['start']),
                'end_line': (vertex_ids['end_bottom'], vertex_ids['end_top'], line_ids['end'])
            }
            
            for name, (start_vertex, end_vertex, curve) in edge_curve_ids.items():
                entities.append(f"#{current_id}=EDGE_CURVE('',#{start_vertex},#{end_vertex},#{curve},.T.);")
                edge_curve_ids[name] = current_id
                current_id += 1
            
            # Create oriented edges
            oriented_edges = []
            for edge_id, orientation in [
                (edge_curve_ids['bottom_arc'], '.T.'),
                (edge_curve_ids['end_line'], '.T.'),
                (edge_curve_ids['top_arc'], '.F.'),
                (edge_curve_ids['start_line'], '.F.')
            ]:
                entities.append(f"#{current_id}=ORIENTED_EDGE('',*,*,#{edge_id},{orientation});")
                oriented_edges.append(current_id)
                current_id += 1
            
            # Create edge loop
            entities.append(f"#{current_id}=EDGE_LOOP('',({','.join(f'#{e}' for e in oriented_edges)}));")
            edge_loop_id = current_id
            current_id += 1
            
            # Create face bound
            entities.append(f"#{current_id}=FACE_BOUND('',#{edge_loop_id},.T.);")
            face_bound_id = current_id
            current_id += 1
            
            # Create cylindrical surface
            entities.append(f"#{current_id}=CYLINDRICAL_SURFACE('',#{placement_ids['top']},{radius:.6f});")
            surface_id = current_id
            current_id += 1
            
            # Create advanced face
            entities.append(f"#{current_id}=ADVANCED_FACE('',(#{face_bound_id}),#{surface_id},.T.);")
            all_face_ids.append(current_id)
            current_id += 1
    
        cylinder_face_id = current_id
        for tri_idx in face_indices:
            if tri_idx in planar_face_groups:
                geometry['face_mappings']['planar_to_cylindrical'][planar_face_groups[tri_idx]] = cylinder_face_id
        
        all_face_ids.append(cylinder_face_id)
        current_id += 1
        
    # Create closed shell
    face_list = ",".join(f"#{id}" for id in all_face_ids)
    entities.append(f"#{current_id}=CLOSED_SHELL('',({face_list}));")
    closed_shell_id = current_id
    current_id += 1
    
    return "\n".join(entities), current_id, geometry, closed_shell_id"""

def get_face_boundaries(vertices, triangles, group_indices, edges):
    """Get outer and inner loops for a merged face group"""
    # Create edge lookup for faster reference
    edge_lookup = {tuple(sorted([v1, v2])): i for i, (v1, v2) in enumerate(edges)}
    
    # First collect all edges from the face group
    boundary_edges = defaultdict(int)
    edge_vertices = {}  # Keep track of actual vertices for each edge
    
    for tri_idx in group_indices:
        v1, v2, v3, _ = triangles[tri_idx]
        for edge in [(v1, v2), (v2, v3), (v3, v1)]:
            sorted_edge = tuple(sorted(edge))
            if sorted_edge in edge_lookup:  # Only process valid edges
                boundary_edges[sorted_edge] += 1
                edge_vertices[sorted_edge] = edge  # Store original orientation
    
    # Edges appearing once are boundary edges
    boundary = [edge_vertices[edge] for edge, count in boundary_edges.items() 
               if count == 1 and edge in edge_vertices]
    
    if not boundary:
        return [], []  # Return empty lists if no boundary found
        
    # Order edges into continuous loops
    loops = []
    remaining = boundary.copy()
    
    while remaining:
        current_loop = []
        start_edge = remaining.pop(0)
        current_loop.append(start_edge)
        current_vertex = start_edge[1]
        
        while current_vertex != start_edge[0] and remaining:  # Added remaining check
            found_next = False
            for i, edge in enumerate(remaining):
                if edge[0] == current_vertex:
                    current_vertex = edge[1]
                    current_loop.append(edge)
                    remaining.pop(i)
                    found_next = True
                    break
                elif edge[1] == current_vertex:
                    current_vertex = edge[0]
                    current_loop.append((edge[1], edge[0]))
                    remaining.pop(i)
                    found_next = True
                    break
            if not found_next:
                break  # Break if we can't find the next edge
        
        if len(current_loop) > 2:  # Only add loops with at least 3 edges
            loops.append(current_loop)
    
    if not loops:
        return [], []
    
    # Helper function to calculate loop length    
    def calculate_loop_length(loop):
        total_length = 0
        for e in loop:
            if e[0] >= len(vertices) or e[1] >= len(vertices):  # Validate indices
                continue
            v1_coords = np.array(vertices[e[0]])
            v2_coords = np.array(vertices[e[1]])
            total_length += np.linalg.norm(v2_coords - v1_coords)
        return total_length
    
    # Find outer loop (largest perimeter)
    outer_loop = max(loops, key=calculate_loop_length)
    inner_loops = [loop for loop in loops if loop != outer_loop]
    
    return outer_loop, inner_loops

def find_index(arr_list, target_array):
    for idx, arr in enumerate(arr_list):
        if np.array_equal(arr, target_array):
            return idx
    raise ValueError("Array not found in list.")

def write_step_file(vertices, edges, triangles, filename="output.step"):
    # Validate geometry
    edges, triangles = validate_geometry(vertices, edges, triangles)
    
    # Generate main entity content
    entity_text, final_id, mappings, closed_shell_id = generate_step_entities(vertices, edges, triangles)
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    intro = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(
/* description */ ('STEP AP203'),
/* implementation_level */ '2;1');
FILE_NAME(
/* name */ '{0}',
/* time_stamp */ '{1}',
/* author */ ('Marcus'),
/* organization */ ('Marcus'),
/* preprocessor_version */ 'Wrote this manually',
/* originating_system */ 'Wrote this manually',
/* authorisation */ '  ');
FILE_SCHEMA (('CONFIG_CONTROL_DESIGN'));
ENDSEC;
DATA;
#10=(
GEOMETRIC_REPRESENTATION_CONTEXT(3)
GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#11))
GLOBAL_UNIT_ASSIGNED_CONTEXT((#12,#16,#17))
REPRESENTATION_CONTEXT('Part 1','TOP_LEVEL_ASSEMBLY_PART'));
 /* Units Setup */
 #11=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(3.93700787401575E-7),#12,
 'DISTANCE_ACCURACY_VALUE','Maximum Tolerance applied to model');
 #12=(CONVERSION_BASED_UNIT('INCH',#14)
 LENGTH_UNIT()
 NAMED_UNIT(#13)); 
  #13=DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.); 
  #14=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#15);
  #15=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));
  #16=(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.));
  #17=(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT());
 /* Product and Context */ 
 #18=PRODUCT_DEFINITION_SHAPE('','',#19);
  #19=PRODUCT_DEFINITION('','',#21,#20); 
   #20=DESIGN_CONTEXT('',#24,'design');
   #21=PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE('','',#22, .NOT_KNOWN.);
    #22=PRODUCT('Part 1','Part 1','Part 1',(#23));
     #23=MECHANICAL_CONTEXT('',#24,'mechanical'); 
      #24=APPLICATION_CONTEXT('configuration controlled 3D designs of mechanical parts and assemblies');
     #25=PRODUCT_RELATED_PRODUCT_CATEGORY('','',(#22)); 
     #26=PRODUCT_CATEGORY('',''); 
  /* Representation */
#27=SHAPE_DEFINITION_REPRESENTATION(#18,#39);
#28=REPRESENTATION('',(#16),#10); 
#29=REPRESENTATION('',(#17),#10);
#30=PROPERTY_DEFINITION_REPRESENTATION(#15,#13); 
#31=VALUE_REPRESENTATION_ITEM('number of annotations',COUNT_MEASURE(0.)); 
#32=VALUE_REPRESENTATION_ITEM('number of views',COUNT_MEASURE(0.)); 
#33=APPLICATION_PROTOCOL_DEFINITION('international standard','config_control_design',2010,#24);#34=PROPERTY_DEFINITION_REPRESENTATION(#35,#39); /* 34 */
 #35=PROPERTY_DEFINITION('pmi validation property','',#18); 
 #36=PROPERTY_DEFINITION('pmi validation property','',#18);
#37=ADVANCED_BREP_SHAPE_REPRESENTATION('',(#44),#10);
#38=SHAPE_REPRESENTATION_RELATIONSHIP('','',#39,#37);
 /* Origin */
 #39=SHAPE_REPRESENTATION('Part 1',(#40),#10); 
  #40=AXIS2_PLACEMENT_3D('',#41,#42,#43); 
   #41=CARTESIAN_POINT('',(0.,0.,0.)); 
   #42=DIRECTION('',(0.,0.,1.)); 
   #43=DIRECTION('',(1.,0.,0.)); 
#44=MANIFOLD_SOLID_BREP('Part 1',#{2}); 
""".format(filename, timestamp, closed_shell_id)
    

    outro = """ENDSEC;
END-ISO-10303-21;"""

    # Combine all content
    step_content = intro + "\n" + entity_text + "\n" + outro
    
    # Write file
    with open(filename, 'w') as f:
        f.write(step_content)
    return True