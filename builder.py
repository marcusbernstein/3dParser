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
def merge_coplanar_triangles(vertices, triangles, normal_tolerance=0.01):
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

def detect_cylindrical_faces(vertices, triangles, angle_tolerance=0.001, area_tolerance=0.001, normal_tolerance=0.01, max_angle=np.pi/12):
    print("\n=== Starting Cylindrical Face Detection ===")
    def calculate_triangle_area(v1_idx, v2_idx, v3_idx):
        v1 = np.array(vertices[v1_idx])
        v2 = np.array(vertices[v2_idx])
        v3 = np.array(vertices[v3_idx])
        return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    # First merge coplanar triangles
    merged_groups = merge_coplanar_triangles(vertices, triangles, normal_tolerance)
    print(f"Found {len(merged_groups)} merged groups")
    
    def get_triangle_edges(triangle):
        v1, v2, v3, _ = triangle
        return {tuple(sorted([v1, v2])), tuple(sorted([v2, v3])), tuple(sorted([v3, v1]))}
    
    # Calculate areas and edges for merged groups
    merged_faces = []
    for group in merged_groups:
        total_area = sum(calculate_triangle_area(
            triangles[idx][0], 
            triangles[idx][1], 
            triangles[idx][2]
        ) for idx in group)
        normal = triangles[group[0]][3]
        
        group_edges = set()
        for idx in group:
            edges = get_triangle_edges(triangles[idx])
            group_edges.update(edges)
            
        merged_faces.append((group, total_area, normal, group_edges))

    # Group faces by area with tolerance
    area_groups = defaultdict(list)
    removed_edges = set()
    
    # First pass: identify potential cylinder areas (those with many faces)
    potential_cylinder_areas = set()
    for i, (group, area, normal, edges) in enumerate(merged_faces):
        rounded_area = round(area / area_tolerance) * area_tolerance
        area_groups[rounded_area].append((i, group, normal, edges))
        if len(area_groups[rounded_area]) >= 3:
            potential_cylinder_areas.add(rounded_area)
    
    print("\n=== Analyzing Face Areas ===")
    for area in sorted(area_groups.keys()):
        print(f"Area {area:.6f}: {len(area_groups[area])} faces")
        if area in potential_cylinder_areas:
            print(f"  - Potential cylinder area")
    
    # Second pass: look for split faces that could belong to cylinders
    print("\n=== Looking for Split Faces ===")
    for cylinder_area in potential_cylinder_areas:
        cylinder_faces = area_groups[cylinder_area]
        cylinder_edges = set()
        for _, _, _, edges in cylinder_faces:
            cylinder_edges.update(edges)
        
        # Look for small faces that might be part of this cylinder
        for small_area, small_faces in area_groups.items():
            if small_area >= cylinder_area or small_area in potential_cylinder_areas:
                continue
                
            for small_idx, small_group, small_normal, small_edges in small_faces[:]:  # Use slice to allow removal
                # Check if this small face shares edges with cylinder faces
                if small_edges & cylinder_edges:
                    print(f"\nFound small face (area={small_area:.6f}) adjacent to cylinder (area={cylinder_area:.6f})")
                    # Check if normals align
                    for _, _, cylinder_normal, _ in cylinder_faces:
                        angle = calculate_angle_between_faces(small_normal, cylinder_normal)
                        if angle < max_angle:
                            print(f"  - Normals align (angle={angle:.6f})")
                            # Add small face to cylinder group
                            area_groups[cylinder_area].append((small_idx, small_group, small_normal, small_edges))
                            # Remove from small group
                            small_faces.remove((small_idx, small_group, small_normal, small_edges))
                            # Add edges to removed set
                            removed_edges.update(small_edges)
                            print(f"  - Added {len(small_edges)} edges to removed set")
                            break
    
    # Now process cylinders with updated groups
    cylindrical_faces = []
    processed = set()
    
    for area, group_list in area_groups.items():
        if len(group_list) < 3:
            continue
                                    
        print(f"\nProcessing area {area:.6f} with {len(group_list)} faces")
        
        # Build adjacency graph
        adj_graph = defaultdict(list)
        for i, (idx1, group1, normal1, edges1) in enumerate(group_list):
            if idx1 in processed:
                continue
            
            for j, (idx2, group2, normal2, edges2) in enumerate(group_list[i+1:], i+1):
                if idx2 in processed:
                    continue
                
                if are_groups_adjacent(group1, group2, triangles):
                    angle = calculate_angle_between_faces(normal1, normal2)
                    if angle > max_angle:
                        continue
                    adj_graph[idx1].append((idx2, angle))
                    adj_graph[idx2].append((idx1, angle))
        
        # Find connected components
        visited = set()
        for start_idx, start_group, _, start_edges in group_list:
            if start_idx in visited or start_idx in processed:
                continue
            
            component = []
            component_edges = set()
            queue = [(start_idx, None)]
            current_edges = []
            
            while queue:
                current_idx, expected_angle = queue.pop(0)
                if current_idx in visited:
                    continue
                
                visited.add(current_idx)
                component.append(current_idx)
                current_edges = next(e for i, _, _, e in group_list if i == current_idx)
                component_edges.update(current_edges)
                
                for neighbor_idx, angle in adj_graph[current_idx]:
                    if neighbor_idx not in visited:
                        if expected_angle is None or abs(angle - expected_angle) < angle_tolerance:
                            queue.append((neighbor_idx, angle))
            
            if len(component) >= 3:
                print(f"\nFound component with {len(component)} faces")
                all_triangles = []
                for comp_idx in component:
                    group_info = next(g for i, g, _, _ in group_list if i == comp_idx)
                    all_triangles.extend(group_info)
                
                normals = [norm for idx, _, norm, _ in group_list if idx in component]
                total_angle = sum(calculate_angle_between_faces(normals[i], normals[i+1])
                                for i in range(len(normals)-1))
                total_angle += calculate_angle_between_faces(normals[-1], normals[0])
                
                print(f"Total angle sweep: {total_angle:.6f}")
                
                diameter, center, axis = calculate_cylinder_properties(vertices, all_triangles, triangles)
                
                if abs(total_angle - 2 * np.pi) < angle_tolerance:
                    print("Complete cylinder detected!")
                    print(f"Adding {len(component_edges)} edges to removed set")
                    cylindrical_faces.append(("complete_cylinder", all_triangles, diameter, center, axis, component_edges))
                    removed_edges.update(component_edges)
                    print(f"Current removed_edges size: {len(removed_edges)}")
                else:
                    print("Partial cylinder detected")
                    boundary_faces = [component[0], component[-1]]
                    boundary_edges = set()
                    
                    for face_idx in boundary_faces:
                        face_edges = next(e for i, _, _, e in group_list if i == face_idx)
                        for edge in face_edges:
                            v1, v2 = edge
                            p1 = np.array(vertices[v1])
                            p2 = np.array(vertices[v2])
                            edge_dir = p2 - p1
                            if abs(abs(np.dot(edge_dir, axis)) - np.linalg.norm(edge_dir)) < 0.001:
                                boundary_edges.add(edge)
                    
                    interior_edges = component_edges - boundary_edges
                    print(f"Adding {len(interior_edges)} interior edges to removed set")
                    cylindrical_faces.append(("partial_cylinder", all_triangles, diameter, center, axis, component_edges))
                    removed_edges.update(interior_edges)
                    print(f"Current removed_edges size: {len(removed_edges)}")
                
                processed.update(component)
    
    print("\n=== Final Statistics ===")
    print(f"Total cylindrical faces found: {len(cylindrical_faces)}")
    print(f"Total edges marked for removal: {len(removed_edges)}")
    
    return cylindrical_faces, removed_edges

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
    
    # Basic geometry data structures
    geometry = {
        'cartesian_points': {},  # vertex_idx -> id
        'vertex_points': {},     # vertex_idx -> id
        'directions': {},        # (dx,dy,dz) -> id
        'vectors': {},          # edge_idx -> id
        'lines': {},            # edge_idx -> id
        'edge_curves': {},      # edge_idx -> id
    }
    oriented_edges_map = {}  # oriented_edge_id -> (v1, v2)
    curved_edge_map = {}  # (v1, v2) -> (edge_id, orientation)
    curved_edge_vertices = {}  # curved_edge_id -> set of vertices it encompasses


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
    cylinder_faces, removed_edges = detect_cylindrical_faces(vertices, triangles)
    excluded_triangles = set()
    for _, face_indices, _, _, _, _ in cylinder_faces:
        excluded_triangles.update(face_indices)
    print("found cylinderss qty: ")
    print(len(cylinder_faces))
    # Process cylindrical faces
    for type_name, face_indices, diameter, center, axis, component_edges in cylinder_faces:
        print("component edges")
        print(component_edges)
        center_np = np.array(center)
        axis_np = np.array(axis)
        ref_dir = np.array(get_perpendicular_vectors(axis)[0])
        radius = diameter/2
        
        ref_dir = np.array(get_perpendicular_vectors(axis)[0])
        # Fixes corrupt partial cylinders Flip ref_dir if points in wrong direction
        first_vertex = np.array(vertices[triangles[face_indices[0]][0]]) - center_np
        first_proj = first_vertex - np.dot(first_vertex, axis_np) * axis_np
        if np.dot(first_proj, ref_dir) >= 0:
            print("FLIPPED!")
            print(ref_dir)
            ref_dir = -ref_dir
            print(ref_dir)
        
        # Get height bounds
        points = np.array([vertices[i] for tri in face_indices for i in triangles[tri][:3]])
        heights = np.dot(points - center_np, axis_np)
        min_height = np.min(heights)
        max_height = np.max(heights)
        height = max_height - min_height
        
        bottom_center = center_np + min_height * axis_np
        top_center = center_np + max_height * axis_np
        
        if type_name == "complete_cylinder":
            print("complete cylinder ID")
            radius = diameter/2
            bottom_point = bottom_center + radius * np.array(ref_dir)
            top_point = top_center + radius * np.array(ref_dir)

            # 1. Create all CARTESIAN_POINTs
            # Center point for cylindrical surface (like #214 in example)
            entities.append(f"#{current_id}=CARTESIAN_POINT('',({bottom_center[0]:.6f},{bottom_center[1]:.6f},{bottom_center[2]:.6f}));")
            surface_origin_id = current_id
            bottom_center_id = current_id
            current_id += 1

            entities.append(f"#{current_id}=CARTESIAN_POINT('',({top_center[0]:.6f},{top_center[1]:.6f},{top_center[2]:.6f}));")
            top_center_id = current_id
            current_id += 1

            # Points for vertices
            entities.append(f"#{current_id}=CARTESIAN_POINT('',({bottom_point[0]:.6f},{bottom_point[1]:.6f},{bottom_point[2]:.6f}));")
            bottom_point_id = current_id
            current_id += 1

            entities.append(f"#{current_id}=CARTESIAN_POINT('',({top_point[0]:.6f},{top_point[1]:.6f},{top_point[2]:.6f}));")
            top_point_id = current_id
            current_id += 1

            # 2. Create DIRECTIONs (following example #176, #177)
            entities.append(f"#{current_id}=DIRECTION('',({axis[0]:.6f},{axis[1]:.6f},{axis[2]:.6f}));")
            axis_direction_id = current_id
            current_id += 1

            entities.append(f"#{current_id}=DIRECTION('',({ref_dir[0]:.6f},{ref_dir[1]:.6f},{ref_dir[2]:.6f}));")
            ref_direction_id = current_id
            current_id += 1

            # 3. Create VERTEX_POINTs (like #69, #71, #76, #77 in example)
            entities.append(f"#{current_id}=VERTEX_POINT('',#{bottom_point_id});")
            bottom_vertex_id = current_id
            current_id += 1

            entities.append(f"#{current_id}=VERTEX_POINT('',#{top_point_id});")
            top_vertex_id = current_id
            current_id += 1

            # Create placements for circles (separate for top and bottom)
            entities.append(f"#{current_id}=AXIS2_PLACEMENT_3D('',#{bottom_center_id},#{axis_direction_id},#{ref_direction_id});")
            cylinder_placement_id = current_id
            bottom_circle_placement_id = current_id
            current_id += 1

            entities.append(f"#{current_id}=AXIS2_PLACEMENT_3D('',#{top_center_id},#{axis_direction_id},#{ref_direction_id});")
            top_circle_placement_id = current_id
            current_id += 1

            # 5. Create CIRCLEs (like #20, #21 in example)
            entities.append(f"#{current_id}=CIRCLE('',#{bottom_circle_placement_id},{radius:.6f});")
            bottom_circle_id = current_id
            current_id += 1

            entities.append(f"#{current_id}=CIRCLE('',#{top_circle_placement_id},{radius:.6f});")
            top_circle_id = current_id
            current_id += 1

            # 6. Create LINE between points (like #81, #89 in example)
            # Create vector for the line direction
            entities.append(f"#{current_id}=VECTOR('',#{axis_direction_id},{height:.6f});")
            line_vector_id = current_id
            current_id += 1

            entities.append(f"#{current_id}=LINE('',#{bottom_point_id},#{line_vector_id});")
            line_id = current_id
            current_id += 1

            # 7. Create EDGE_CURVEs (like #56, #64, #66, #67 in example)
            entities.append(f"#{current_id}=EDGE_CURVE('',#{bottom_vertex_id},#{top_vertex_id},#{line_id},.T.);")
            line_edge_id = current_id
            current_id += 1

            entities.append(f"#{current_id}=EDGE_CURVE('',#{bottom_vertex_id},#{bottom_vertex_id},#{bottom_circle_id},.T.);")
            bottom_edge_id = current_id
            current_id += 1

            entities.append(f"#{current_id}=EDGE_CURVE('',#{top_vertex_id},#{top_vertex_id},#{top_circle_id},.T.);")
            top_edge_id = current_id
            current_id += 1

            # Create ORIENTED_EDGEs with improved vertex tracking
            entities.append(f"#{current_id}=ORIENTED_EDGE('',*,*,#{top_edge_id},.T.);")
            top_circle_oriented_id = current_id
            # Track vertices for top circle
            top_circle_vertices = set()
            for v1, v2 in component_edges:
                v1_height = np.dot(np.array(vertices[v1]) - bottom_center, axis_np)
                v2_height = np.dot(np.array(vertices[v2]) - bottom_center, axis_np)
                if abs(v1_height - height) < abs(v1_height):
                    top_circle_vertices.add(v1)
                if abs(v2_height - height) < abs(v2_height):
                    top_circle_vertices.add(v2)
            curved_edge_vertices[top_circle_oriented_id] = top_circle_vertices
            current_id += 1

            entities.append(f"#{current_id}=ORIENTED_EDGE('',*,*,#{line_edge_id},.F.);")
            line_oriented_id_1 = current_id
            current_id += 1

            entities.append(f"#{current_id}=ORIENTED_EDGE('',*,*,#{bottom_edge_id},.F.);")
            bottom_circle_oriented_id = current_id
            # Track vertices for bottom circle
            bottom_circle_vertices = set()
            for v1, v2 in component_edges:
                v1_height = np.dot(np.array(vertices[v1]) - bottom_center, axis_np)
                v2_height = np.dot(np.array(vertices[v2]) - bottom_center, axis_np)
                if abs(v1_height) < abs(v1_height - height):
                    bottom_circle_vertices.add(v1)
                if abs(v2_height) < abs(v2_height - height):
                    bottom_circle_vertices.add(v2)
            curved_edge_vertices[bottom_circle_oriented_id] = bottom_circle_vertices
            current_id += 1

            entities.append(f"#{current_id}=ORIENTED_EDGE('',*,*,#{line_edge_id},.T.);")
            line_oriented_id_2 = current_id
            current_id += 1

            excluded_triangles.update(face_indices)

            # Create EDGE_LOOP
            edge_loop_str = f"#{top_circle_oriented_id},#{line_oriented_id_1},#{bottom_circle_oriented_id},#{line_oriented_id_2}"
            entities.append(f"#{current_id}=EDGE_LOOP('',({edge_loop_str})); /* 1 2 3 4 edge loop */")
            edge_loop_id = current_id
            current_id += 1

            # 10. Create FACE_BOUND (like #115 in example)
            entities.append(f"#{current_id}=FACE_BOUND('',#{edge_loop_id},.T.);")
            face_bound_id = current_id
            current_id += 1

            # 11. Create CYLINDRICAL_SURFACE (like #22 in example)
            entities.append(f"#{current_id}=CYLINDRICAL_SURFACE('',#{cylinder_placement_id},{radius:.6f});")
            surface_id = current_id
            current_id += 1

            # 12. Create ADVANCED_FACE (like #128 in example)
            entities.append(f"#{current_id}=ADVANCED_FACE('',(#{face_bound_id}),#{surface_id},.T.);")
            all_face_ids.append(current_id)
            current_id += 1
            
            # Create top circular face
            entities.append(f"#{current_id}=ORIENTED_EDGE('',*,*,#{top_edge_id},.T.);")
            top_circle_oriented_id = current_id
            current_id += 1
            
            entities.append(f"#{current_id}=EDGE_LOOP('',(#{top_circle_oriented_id})); /* top circle */")
            top_loop_id = current_id
            current_id += 1
            
            entities.append(f"#{current_id}=FACE_BOUND('',#{top_loop_id},.T.);")
            top_bound_id = current_id
            current_id += 1
            
            # Create plane for top face
            entities.append(f"#{current_id}=PLANE('',#{top_circle_placement_id});")
            top_plane_id = current_id
            current_id += 1
            
            '''entities.append(f"#{current_id}=ADVANCED_FACE('',(#{top_bound_id}),#{top_plane_id},.F.);")
            all_face_ids.append(current_id)
            current_id += 1'''
            
            # Create bottom circular face
            entities.append(f"#{current_id}=ORIENTED_EDGE('',*,*,#{bottom_edge_id},.T.);")
            bottom_circle_oriented_id = current_id
            current_id += 1

            '''bottom_vertices = {i for tri in face_indices for i in triangles[tri][:3] 
                              if abs(np.dot(np.array(vertices[i]) - bottom_center, axis_np)) < 0.001}

            curved_edge_vertices[oriented_edge3_id] = bottom_vertices'''
            
            entities.append(f"#{current_id}=EDGE_LOOP('',(#{bottom_circle_oriented_id})); /* bottom circle */")
            bottom_loop_id = current_id
            current_id += 1
            
            entities.append(f"#{current_id}=FACE_BOUND('',#{bottom_loop_id},.T.);")
            bottom_bound_id = current_id
            current_id += 1
            
            # Create plane for bottom face
            entities.append(f"#{current_id}=PLANE('',#{bottom_circle_placement_id});")
            bottom_plane_id = current_id
            current_id += 1
            
            '''entities.append(f"#{current_id}=ADVANCED_FACE('',(#{bottom_bound_id}),#{bottom_plane_id},.T.);")
            all_face_ids.append(current_id)
            current_id += 1'''
            
        elif type_name == "partial_cylinder":  # partial cylinder
            print("partial cylinder")
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
            curve_mappings = {
                'start_line': (vertex_ids['start_top'], vertex_ids['start_bottom'], line_ids['start']),
                'bottom_arc': (vertex_ids['start_bottom'], vertex_ids['end_bottom'], bottom_circle_id),
                'end_line': (vertex_ids['end_bottom'], vertex_ids['end_top'], line_ids['end']),
                'top_arc': (vertex_ids['start_top'], vertex_ids['end_top'], top_circle_id)
            }

            # Dictionary to store IDs
            edge_curve_ids = {}

            # Create edge curves and store both IDs and original mappings
            for name, (start_vertex, end_vertex, curve) in curve_mappings.items():
                # First create the EDGE_CURVE
                entities.append(f"#{current_id}=EDGE_CURVE('',#{start_vertex},#{end_vertex},#{curve},.T.); /*{name}*/")
                edge_curve_id = current_id
                current_id += 1
                
                # Then create the ORIENTED_EDGE and store its ID for arcs
                entities.append(f"#{current_id}=ORIENTED_EDGE('',*,*,#{edge_curve_id},.T.);")
                oriented_edge_id = current_id
                
                if 'arc' in name:
                    arc_vertices = set()
                    for v1, v2 in component_edges:
                        v1_height = np.dot(np.array(vertices[v1]) - bottom_center, axis_np)
                        v2_height = np.dot(np.array(vertices[v2]) - bottom_center, axis_np)
                        
                        if name == 'bottom_arc':
                            # Only add vertices that are closer to bottom than top
                            if abs(v1_height - min_height) < abs(v1_height - max_height):
                                arc_vertices.add(v1)
                            if abs(v2_height - min_height) < abs(v2_height - max_height):
                                arc_vertices.add(v2)
                        elif name == 'top_arc':
                            # Only add vertices that are closer to top than bottom
                            if abs(v1_height - max_height) < abs(v1_height - min_height):
                                arc_vertices.add(v1)
                            if abs(v2_height - max_height) < abs(v2_height - min_height):
                                arc_vertices.add(v2)
                                
                    curved_edge_vertices[oriented_edge_id] = arc_vertices  # Store with ORIENTED_EDGE ID

                edge_curve_ids[name] = oriented_edge_id  # Store the ORIENTED_EDGE ID instead
                current_id += 1            
            bottom_vertices = set()
            top_vertices = set()
            for v1, v2 in component_edges:
                # Determine if vertices are on bottom or top based on height
                v1_height = np.dot(np.array(vertices[v1]) - bottom_center, axis_np)
                v2_height = np.dot(np.array(vertices[v2]) - bottom_center, axis_np)
                
                if abs(v1_height - min_height) < abs(v1_height - max_height):
                    bottom_vertices.add(v1)
                else:
                    top_vertices.add(v1)
                    
                if abs(v2_height - min_height) < abs(v2_height - max_height):
                    bottom_vertices.add(v2)
                else:
                    top_vertices.add(v2)

            excluded_triangles.update(face_indices)

            # Create edge loop using the existing ORIENTED_EDGE IDs
            oriented_edges = [
                edge_curve_ids['start_line'],   # Already an ORIENTED_EDGE ID
                edge_curve_ids['bottom_arc'],  # Already an ORIENTED_EDGE ID
                edge_curve_ids['end_line'],    # Already an ORIENTED_EDGE ID
                edge_curve_ids['top_arc']     # Already an ORIENTED_EDGE ID
            ]

            # Create edge loop directly with existing oriented edges
            entities.append(f"#{current_id}=EDGE_LOOP('',({','.join(f'#{e}' for e in oriented_edges)})) /* oriented edge loop */;")
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

    # Process planar faces
    merged_groups = merge_coplanar_triangles(vertices, triangles)
    planar_faces = []
    
    for group_triangles in merged_groups:
        if all(idx in excluded_triangles for idx in group_triangles):
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
            oriented_edges_map[current_id] = edge if orientation == '.T.' else edge[::-1]
            outer_oriented_edges.append(current_id)
            current_id += 1
        
        edge_loop_str = "(" + ",".join([f"#{e}" for e in outer_oriented_edges]) + ")"
        print("\nProposed edge loop:", edge_loop_str)
        edges_to_replace = analyze_edge_loop_simple(edge_loop_str, removed_edges, edge_lookup, geometry, oriented_edges_map)
        
        # Replace straight edges with curved ones in outer loop
        if edges_to_replace:
            seen_curved_edges = set()  # Track which curved edges we've already used
            
            # Check if we need to replace the entire loop
            if len(edges_to_replace) == len(outer_oriented_edges):
                # Find a suitable curved edge that contains all vertices
                vertex_set = set()
                for edge_id in outer_oriented_edges:
                    if edge_id in oriented_edges_map:
                        vertex_pair = oriented_edges_map[edge_id]
                        vertex_set.update(vertex_pair)
                
                for curved_id, vertices in curved_edge_vertices.items():
                    if vertex_set.issubset(vertices):
                        # Found a curved edge that encompasses all vertices
                        outer_oriented_edges = [curved_id]
                        break
            else:
                # Regular edge-by-edge replacement
                modified_outer_edges = []
                for edge_id in outer_oriented_edges:
                    if edge_id in edges_to_replace:
                        vertex_pair = oriented_edges_map[edge_id]
                        for curved_id, vertices in curved_edge_vertices.items():
                            if (vertex_pair[0] in vertices and 
                                vertex_pair[1] in vertices and 
                                curved_id not in seen_curved_edges):
                                modified_outer_edges.append(curved_id)
                                seen_curved_edges.add(curved_id)
                                break
                    else:
                        modified_outer_edges.append(edge_id)
                
                outer_oriented_edges = modified_outer_edges
        print("\n final edge loop : ", outer_oriented_edges)
        # Create outer edge loop and bound
        entities.append(f"#{current_id}=EDGE_LOOP('',({','.join(f'#{e}' for e in outer_oriented_edges)})); /* outer edge */ ")
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
                oriented_edges_map[current_id] = edge if orientation == '.T.' else edge[::-1]

                inner_oriented_edges.append(current_id)
                
                current_id += 1
            
            # Look for edges to replace with curves in inner loop
            edge_loop_str = "(" + ",".join([f"#{e}" for e in inner_oriented_edges]) + ")"
            print(f"\nProposed inner loop: {edge_loop_str}")
            edges_to_replace = analyze_edge_loop_simple(edge_loop_str, removed_edges, edge_lookup, geometry, oriented_edges_map)
            
            # Replace straight edges with curved ones in inner loop
            if edges_to_replace:
                seen_curved_edges = set()
                modified_inner_edges = []
                
                for edge_id in inner_oriented_edges:
                    if edge_id in edges_to_replace:
                        vertex_pair = oriented_edges_map[edge_id]
                        for curved_id, vertices in curved_edge_vertices.items():
                            if vertex_pair[0] in vertices and vertex_pair[1] in vertices:
                                if curved_id not in seen_curved_edges:
                                    modified_inner_edges.append(curved_id)
                                    seen_curved_edges.add(curved_id)
                                break
                    else:
                        modified_inner_edges.append(edge_id)
                
                inner_oriented_edges = modified_inner_edges
            
            entities.append(f"#{current_id}=EDGE_LOOP('',({','.join(f'#{e}' for e in inner_oriented_edges)})) /* inner edge loop */;")
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
    
    # Create closed shell
    face_list = ",".join(f"#{id}" for id in all_face_ids)
    entities.append(f"#{current_id}=CLOSED_SHELL('',({face_list}));")
    closed_shell_id = current_id
    current_id += 1
    
    return "\n".join(entities), current_id, geometry, closed_shell_id

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
                edge_vertices[tuple(reversed(sorted_edge))] = tuple(reversed(edge))
    
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
            verts = sorted(vertices)
            v1_coords = np.array(verts[e[0]])
            v2_coords = np.array(verts[e[1]])
            total_length += np.linalg.norm(v2_coords - v1_coords)
        return total_length
    
    # Find outer loop (largest perimeter)
    outer_loop = max(loops, key=calculate_loop_length)
    inner_loops = [loop for loop in loops if loop != outer_loop]
    
    return outer_loop, inner_loops

def analyze_edge_loop_simple(edge_loop_str, edges_to_remove, edge_lookup, geometry, oriented_edges_map):
    """Analyze which edges in an edge loop should be replaced with curves."""
    print("\nAnalyzing edge loop:", edge_loop_str)
    
    # Parse edge loop string to get IDs
    edge_ids = [int(x.strip('(#)')) for x in edge_loop_str.strip("'{}").split(',')]
    print("Edge IDs in loop:", edge_ids)
    
    # If there's only one edge and it's a complete circle (same start/end point),
    # this is already a complete cylinder edge - don't modify it
    if len(edge_ids) == 1:
        return []
        
    # Look for complete cylinder edges in the loop
    # These will have equal start and end points
    '''    for edge_id in edge_ids:
        if edge_id in oriented_edges_map:
            vertex_pair = oriented_edges_map[edge_id]
            if vertex_pair[0] == vertex_pair[1]:  # Same start and end point
                # This is a complete cylinder - it should be the only edge in the loop
                print(f"Found complete cylinder edge {edge_id}, replacing entire loop")
                # Return all other edges for removal
                return [eid for eid in edge_ids if eid != edge_id]'''
    
    # If no complete cylinder found, proceed with normal edge replacement
    edges_to_replace = []
    for edge_id in edge_ids:
        if edge_id in oriented_edges_map:
            vertex_pair = oriented_edges_map[edge_id]
            if vertex_pair in edges_to_remove or vertex_pair[::-1] in edges_to_remove:
                edges_to_replace.append(edge_id)
    
    print("\nEdges to replace:", edges_to_replace)
    return edges_to_replace

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