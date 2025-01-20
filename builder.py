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

    norm = np.linalg.norm(axis_vector)
    if norm < 1e-10:  # If no valid cross products found
        # Find most different normals and use their cross product
        max_angle = 0
        best_cross = None
        for i in range(len(face_normals)):
            for j in range(i+1, len(face_normals)):
                angle = np.arccos(np.clip(np.dot(face_normals[i], face_normals[j]), -1.0, 1.0))
                if angle > max_angle:
                    cross = np.cross(face_normals[i], face_normals[j])
                    if np.linalg.norm(cross) > 1e-10:
                        max_angle = angle
                        best_cross = cross
        
        if best_cross is not None:
            axis_vector = best_cross / np.linalg.norm(best_cross)
        else:
            # Last resort: use the normal of first face to generate perpendicular vector
            axis_vector = get_perpendicular_vectors(face_normals[0])[0]
    else:
        axis_vector = axis_vector / norm
    
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

def merge_coplanar_triangles(vertices, triangles, normal_tolerance):
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
    
    # Find connected components
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

def detect_cylindrical_faces(vertices, triangles, angle_tolerance=0.01, area_tolerance=0.01, normal_tolerance=0.001, max_angle=np.pi/12):
    print(f"\nStarting cylindrical face detection with parameters:")
    print(f"angle_tolerance: {angle_tolerance:.4f} radians ({np.degrees(angle_tolerance):.2f}°)")
    print(f"area_tolerance: {area_tolerance:.4f}")
    print(f"normal_tolerance: {normal_tolerance:.4f}")
    print(f"max_angle: {max_angle:.4f} radians ({np.degrees(max_angle):.2f}°)")
    
    merged_groups = merge_coplanar_triangles(vertices, triangles, normal_tolerance)
    print(f"\nFound {len(merged_groups)} initial coplanar groups")
    
    def calculate_triangle_area(v1_idx, v2_idx, v3_idx):
        v1 = np.array(vertices[v1_idx])
        v2 = np.array(vertices[v2_idx])
        v3 = np.array(vertices[v3_idx])
        return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    
    def calculate_angle_between_faces(normal1, normal2):
        n1 = np.array(normal1)
        n2 = np.array(normal2)
        cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
        return np.arccos(cos_angle)
    
    def calculate_triangle_area(v1_idx, v2_idx, v3_idx):
        v1 = np.array(vertices[v1_idx])
        v2 = np.array(vertices[v2_idx])
        v3 = np.array(vertices[v3_idx])
        return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    
    def calculate_angle_between_faces(normal1, normal2):
        n1 = np.array(normal1)
        n2 = np.array(normal2)
        cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
        return np.arccos(cos_angle)
    
    def are_groups_adjacent(group1_indices, group2_indices):
        for idx1 in group1_indices:
            verts1 = set([triangles[idx1][0], triangles[idx1][1], triangles[idx1][2]])
            for idx2 in group2_indices:
                verts2 = set([triangles[idx2][0], triangles[idx2][1], triangles[idx2][2]])
                if len(verts1.intersection(verts2)) >= 2:
                    return True
        return False
    
    # Calculate areas and normals for merged groups
    merged_faces = []
    for i, group in enumerate(merged_groups):
        total_area = sum(calculate_triangle_area(
            triangles[idx][0], 
            triangles[idx][1], 
            triangles[idx][2]
        ) for idx in group)
        normal = triangles[group[0]][3]
        merged_faces.append((group, total_area, normal))
        print(f"\nGroup {i}: {len(group)} triangles, area: {total_area:.4f}")
    
    # Group by area pattern
    area_groups = defaultdict(list)
    for i, (group, area, normal) in enumerate(merged_faces):
        rounded_area = round(area / area_tolerance) * area_tolerance
        area_groups[rounded_area].append((i, group, normal))
        print(f"Face {i} with area {area:.4f} grouped into bucket {rounded_area:.4f}")
    
    print(f"\nFound {len(area_groups)} distinct area groups")
    for area, group_list in area_groups.items():
        print(f"Area {area:.4f}: {len(group_list)} faces")
    
    cylindrical_faces = []
    processed = set()
    
    for area, group_list in area_groups.items():
        if len(group_list) < 3:
            print(f"\nSkipping area group {area:.4f} - insufficient faces ({len(group_list)} < 3)")
            continue
            
        print(f"\nProcessing area group {area:.4f} with {len(group_list)} faces")
        
        # Build adjacency graph
        adj_graph = defaultdict(list)
        angle_patterns = defaultdict(int)
        
        for i, (idx1, group1, normal1) in enumerate(group_list):
            if idx1 in processed:
                continue
                
            for j, (idx2, group2, normal2) in enumerate(group_list[i+1:], i+1):
                if idx2 in processed:
                    continue
                    
                if are_groups_adjacent(group1, group2):
                    angle = calculate_angle_between_faces(normal1, normal2)
                    print(f"Faces {idx1}-{idx2}: angle {np.degrees(angle):.2f}°")
                    
                    if angle > max_angle:
                        print(f"Skipping faces {idx1}-{idx2}: angle {np.degrees(angle):.2f}° > max {np.degrees(max_angle):.2f}°")
                        continue
                        
                    rounded_angle = round(angle / angle_tolerance) * angle_tolerance
                    angle_patterns[rounded_angle] += 1
                    adj_graph[idx1].append((idx2, angle))
                    adj_graph[idx2].append((idx1, angle))
        
        if not angle_patterns:
            print("No valid angle patterns found in this area group")
            continue
            
        dominant_angle = max(angle_patterns.items(), key=lambda x: x[1])[0]
        print(f"Dominant angle pattern: {np.degrees(dominant_angle):.2f}° (frequency: {angle_patterns[dominant_angle]})")
        
        # Find connected components
        visited = set()
        for start_idx, start_group, _ in group_list:
            if start_idx in visited or start_idx in processed:
                continue
                
            print(f"\nStarting new component from face {start_idx}")
            component = []
            angles = []
            queue = [(start_idx, None)]
            
            while queue:
                current_idx, prev_angle = queue.pop(0)
                if current_idx in visited:
                    continue
                
                visited.add(current_idx)
                component.append(current_idx)
                
                if prev_angle is not None:
                    angles.append(prev_angle)
                    print(f"Added angle {np.degrees(prev_angle):.2f}° to component")
                
                for neighbor_idx, angle in adj_graph[current_idx]:
                    if neighbor_idx not in visited:
                        rounded_angle = round(angle / angle_tolerance) * angle_tolerance
                        if abs(rounded_angle - dominant_angle) < angle_tolerance:
                            queue.append((neighbor_idx, angle))
            
            if len(component) >= 3:
                print(f"\nFound component with {len(component)} faces")
                all_triangles = []
                for comp_idx in component:
                    all_triangles.extend(merged_faces[comp_idx][0])
                
                total_angle = sum(angles)
                print(f"Total angle sweep: {np.degrees(total_angle):.2f}°")
                
                if len(angles) >= 2:
                    angle_diffs = np.diff(sorted(angles))
                    if all(abs(angle_diffs) < .5):
                        angle_ratio = max(angle_diffs) / min(angle_diffs)
                        print(f"Angle spacing ratio: {angle_ratio:.4f}")
                        consistent_spacing = True
                    else:
                        consistent_spacing = True
                        print("Zero angle differences found - treating as consistent")
                else:
                    consistent_spacing = True
                    print("Too few angles to check spacing consistency")
                
                if consistent_spacing:
                    print("Angles are consistently spaced")
                    if abs(total_angle - 2 * np.pi) < angle_tolerance * len(angles):
                        print("Complete cylinder detected (2π angle sweep)")
                        # Verify physical connectivity
                        connected = True
                        for i in range(len(component)):
                            if not are_groups_adjacent(merged_faces[component[i]][0], 
                                                     merged_faces[component[(i+1)%len(component)]][0]):
                                connected = False
                                print(f"Connectivity break detected between faces {component[i]} and {component[(i+1)%len(component)]}")
                                break
                        
                        if connected:
                            print("Adding complete cylinder")
                            cylindrical_faces.append(("complete_cylinder", all_triangles))
                        else:
                            print("Adding as partial cylinder due to connectivity break")
                            cylindrical_faces.append(("partial_cylinder", all_triangles))
                    else:
                        print(f"Adding partial cylinder (angle sweep != 2π)")
                        cylindrical_faces.append(("partial_cylinder", all_triangles))
                else:
                    print("Rejecting component due to inconsistent angle spacing")
                
                processed.update(component)
    
    print(f"\nFinal detection results: {len(cylindrical_faces)} cylindrical features")
    result = []
    for i, (type_name, face_indices) in enumerate(cylindrical_faces):
        diameter, center, axis = calculate_cylinder_properties(vertices, face_indices, triangles)
        print(f"\nCylinder {i} ({type_name}):")
        print(f"  Diameter: {diameter:.4f}")
        print(f"  Center: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
        print(f"  Axis: ({axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f})")
        result.append((type_name, face_indices, diameter, center, axis))
    
    return result

def detect_cylindrical_faces(vertices, triangles, angle_tolerance=0.1, area_tolerance=0.1, normal_tolerance=0.01, max_angle=np.pi/12):
    print(f"\nStarting cylindrical face detection with parameters:")
    print(f"angle_tolerance: {angle_tolerance:.4f} radians ({np.degrees(angle_tolerance):.2f}°)")
    print(f"area_tolerance: {area_tolerance:.4f}")
    print(f"normal_tolerance: {normal_tolerance:.4f}")
    print(f"max_angle: {max_angle:.4f} radians ({np.degrees(max_angle):.2f}°)")
    
    merged_groups = merge_coplanar_triangles(vertices, triangles, normal_tolerance)
    print(f"\nFound {len(merged_groups)} initial coplanar groups")
    
    def calculate_triangle_area(v1_idx, v2_idx, v3_idx):
        v1 = np.array(vertices[v1_idx])
        v2 = np.array(vertices[v2_idx])
        v3 = np.array(vertices[v3_idx])
        return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    
    def calculate_angle_between_faces(normal1, normal2):
        n1 = np.array(normal1)
        n2 = np.array(normal2)
        cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
        return np.arccos(cos_angle)
    
    def calculate_triangle_area(v1_idx, v2_idx, v3_idx):
        v1 = np.array(vertices[v1_idx])
        v2 = np.array(vertices[v2_idx])
        v3 = np.array(vertices[v3_idx])
        return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    
    def calculate_angle_between_faces(normal1, normal2):
        n1 = np.array(normal1)
        n2 = np.array(normal2)
        cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
        return np.arccos(cos_angle)
    
    def are_groups_adjacent(group1_indices, group2_indices):
        for idx1 in group1_indices:
            verts1 = set([triangles[idx1][0], triangles[idx1][1], triangles[idx1][2]])
            for idx2 in group2_indices:
                verts2 = set([triangles[idx2][0], triangles[idx2][1], triangles[idx2][2]])
                if len(verts1.intersection(verts2)) >= 2:
                    return True
        return False
    
    # Calculate areas and normals for merged groups
    merged_faces = []
    for i, group in enumerate(merged_groups):
        total_area = sum(calculate_triangle_area(
            triangles[idx][0], 
            triangles[idx][1], 
            triangles[idx][2]
        ) for idx in group)
        normal = triangles[group[0]][3]
        merged_faces.append((group, total_area, normal))
        print(f"\nGroup {i}: {len(group)} triangles, area: {total_area:.4f}")
    
    # Group by area pattern
    area_groups = defaultdict(list)
    for i, (group, area, normal) in enumerate(merged_faces):
        rounded_area = round(area / area_tolerance) * area_tolerance
        area_groups[rounded_area].append((i, group, normal))
        print(f"Face {i} with area {area:.4f} grouped into bucket {rounded_area:.4f}")
    
    print(f"\nFound {len(area_groups)} distinct area groups")
    for area, group_list in area_groups.items():
        print(f"Area {area:.4f}: {len(group_list)} faces")
    
    cylindrical_faces = []
    processed = set()
    
    for area, group_list in area_groups.items():
        if len(group_list) < 3:
            print(f"\nSkipping area group {area:.4f} - insufficient faces ({len(group_list)} < 3)")
            continue
            
        print(f"\nProcessing area group {area:.4f} with {len(group_list)} faces")
        
        # Build adjacency graph
        adj_graph = defaultdict(list)
        angle_patterns = defaultdict(int)
        
        for i, (idx1, group1, normal1) in enumerate(group_list):
            if idx1 in processed:
                continue
                
            for j, (idx2, group2, normal2) in enumerate(group_list[i+1:], i+1):
                if idx2 in processed:
                    continue
                    
                if are_groups_adjacent(group1, group2):
                    angle = calculate_angle_between_faces(normal1, normal2)
                    print(f"Faces {idx1}-{idx2}: angle {np.degrees(angle):.2f}°")
                    
                    if angle > max_angle:
                        print(f"Skipping faces {idx1}-{idx2}: angle {np.degrees(angle):.2f}° > max {np.degrees(max_angle):.2f}°")
                        continue
                        
                    rounded_angle = round(angle / angle_tolerance) * angle_tolerance
                    angle_patterns[rounded_angle] += 1
                    adj_graph[idx1].append((idx2, angle))
                    adj_graph[idx2].append((idx1, angle))
        
        if not angle_patterns:
            print("No valid angle patterns found in this area group")
            continue
            
        dominant_angle = max(angle_patterns.items(), key=lambda x: x[1])[0]
        print(f"Dominant angle pattern: {np.degrees(dominant_angle):.2f}° (frequency: {angle_patterns[dominant_angle]})")
        
        # Find connected components
        visited = set()
        for start_idx, start_group, _ in group_list:
            if start_idx in visited or start_idx in processed:
                continue
                
            print(f"\nStarting new component from face {start_idx}")
            component = []
            angles = []
            queue = [(start_idx, None)]
            
            while queue:
                current_idx, prev_angle = queue.pop(0)
                if current_idx in visited:
                    continue
                
                visited.add(current_idx)
                component.append(current_idx)
                
                if prev_angle is not None:
                    angles.append(prev_angle)
                    print(f"Added angle {np.degrees(prev_angle):.2f}° to component")
                
                for neighbor_idx, angle in adj_graph[current_idx]:
                    if neighbor_idx not in visited:
                        rounded_angle = round(angle / angle_tolerance) * angle_tolerance
                        if abs(rounded_angle - dominant_angle) < angle_tolerance:
                            queue.append((neighbor_idx, angle))
            
            if len(component) >= 3:
                print(f"\nFound component with {len(component)} faces")
                all_triangles = []
                for comp_idx in component:
                    all_triangles.extend(merged_faces[comp_idx][0])
                
                total_angle = sum(angles)
                print(f"Total angle sweep: {np.degrees(total_angle):.2f}°")
                
                if len(angles) >= 2:
                    angle_diffs = np.diff(sorted(angles))
                    if all(abs(angle_diffs) < .5):
                        angle_ratio = max(angle_diffs) / min(angle_diffs)
                        print(f"Angle spacing ratio: {angle_ratio:.4f}")
                        consistent_spacing = True
                    else:
                        consistent_spacing = True
                        print("Zero angle differences found - treating as consistent")
                else:
                    consistent_spacing = True
                    print("Too few angles to check spacing consistency")
                
                if consistent_spacing:
                    print("Angles are consistently spaced")
                    if abs(total_angle - 2 * np.pi) < angle_tolerance * len(angles):
                        print("Complete cylinder detected (2π angle sweep)")
                        # Verify physical connectivity
                        connected = True
                        for i in range(len(component)):
                            if not are_groups_adjacent(merged_faces[component[i]][0], 
                                                     merged_faces[component[(i+1)%len(component)]][0]):
                                connected = False
                                print(f"Connectivity break detected between faces {component[i]} and {component[(i+1)%len(component)]}")
                                break
                        
                        if connected:
                            print("Adding complete cylinder")
                            cylindrical_faces.append(("complete_cylinder", all_triangles))
                        else:
                            print("Adding as partial cylinder due to connectivity break")
                            cylindrical_faces.append(("partial_cylinder", all_triangles))
                    else:
                        print(f"Adding partial cylinder (angle sweep != 2π)")
                        cylindrical_faces.append(("partial_cylinder", all_triangles))
                else:
                    print("Rejecting component due to inconsistent angle spacing")
                
                processed.update(component)
    
    print(f"\nFinal detection results: {len(cylindrical_faces)} cylindrical features")
    result = []
    for i, (type_name, face_indices) in enumerate(cylindrical_faces):
        diameter, center, axis = calculate_cylinder_properties(vertices, face_indices, triangles)
        print(f"\nCylinder {i} ({type_name}):")
        print(f"  Diameter: {diameter:.4f}")
        print(f"  Center: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
        print(f"  Axis: ({axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f})")
        result.append((type_name, face_indices, diameter, center, axis))
    
    return result

def generate_step_entities(vertices, edges, triangles, start_id=100):
    entities = []
    current_id = start_id
        
    # Tracking dictionaries for entity references
    mappings = {
        'cartesian_points': {},  # vertex_idx -> id
        'vertex_points': {},     # vertex_idx -> id
        'directions': {},        # (dx,dy,dz) -> id
        'vectors': {},          # edge_idx -> id
        'lines': {},            # edge_idx -> id
        'edge_curves': {},      # edge_idx -> id
        'oriented_edges': {},   # (edge_idx, direction) -> id
        'edge_loops': {},       # triangle_idx -> id
        'face_bounds': {},      # triangle_idx -> id
        'axis_placements': {},  # triangle_idx -> id
        'planes': {},           # triangle_idx -> id
        'faces': {}            # triangle_idx -> id
    }
    
    cylindrical_mappings = {
        'cartesian_points': {},  # (x,y,z) -> id
        'directions': {},        # (dx,dy,dz) -> id
        'axis_placements': {},   # (center,axis,ref) -> id
        'circles': {},          # (placement,diameter) -> id
        'edge_curves': {},      # (circle/line,start,end) -> id
        'oriented_edges': {},   # (curve,orientation) -> id
        'edge_loops': {},       # (edges_tuple) -> id
        'face_bounds': {},      # (loop) -> id
        'cylindrical_surfaces': {}, # (placement,diameter) -> id
        'faces': {}            # surface_idx -> id
    }
    
    enable_curved_surfaces = True
    # Detect cylindrical faces if enabled
    cylinder_faces = []
    excluded_triangles = set()
    if enable_curved_surfaces:
        cylinder_faces = detect_cylindrical_faces(vertices, triangles)
        # Collect all triangle indices that are part of cylindrical surfaces
        for _, face_indices, _, _, _ in cylinder_faces:
            excluded_triangles.update(face_indices)
    
    # Generate planar geometry entities (excluding cylindrical triangles)
    for i, (x, y, z) in enumerate(vertices):
        entities.append(f"#{current_id}=CARTESIAN_POINT('',({x:.6f},{y:.6f},{z:.6f})); /* vertex {i} */")
        mappings['cartesian_points'][i] = current_id
        current_id += 1
    
    # 2. Vertex Points
    for i in range(len(vertices)):
        entities.append(f"#{current_id}=VERTEX_POINT('',#{mappings['cartesian_points'][i]}); /* vertex point {i} */")
        mappings['vertex_points'][i] = current_id
        current_id += 1
    
    # 3. Edge Directions and Vectors
    for i, (v1, v2) in enumerate(edges):  # Make sure this loops through ALL edges
        # Calculate direction vector
        p1 = np.array(vertices[v1])
        p2 = np.array(vertices[v2])
        direction = p2 - p1
        direction = direction / np.linalg.norm(direction)
        direction_tuple = tuple(direction)
    
        # Create or reuse direction
        if direction_tuple not in mappings['directions']:
            entities.append(f"#{current_id}=DIRECTION('',({direction[0]:.6f},{direction[1]:.6f},{direction[2]:.6f})); /* edge {i} direction */")
            mappings['directions'][direction_tuple] = current_id
            current_id += 1
    
        # Create vector
        entities.append(f"#{current_id}=VECTOR('',#{mappings['directions'][direction_tuple]},1.0); /* edge {i} vector */")
        mappings['vectors'][i] = current_id
        current_id += 1

    # 4. Lines (CARTESIAN_POINT, VECTOR)
    for i, (v1, v2) in enumerate(edges):  # This should process ALL edges
        entities.append(f"#{current_id}=LINE('',#{mappings['cartesian_points'][v1]},#{mappings['vectors'][i]}); /* edge {i} line */")
        mappings['lines'][i] = current_id
        current_id += 1

    # 5. Edge Curves (VERTEX, VERTEX, LINE)
    for i, (v1, v2) in enumerate(edges):  # This should process ALL edges
        entities.append(f"#{current_id}=EDGE_CURVE('',#{mappings['vertex_points'][v1]},#{mappings['vertex_points'][v2]},#{mappings['lines'][i]},.T.); /* edge {i} curve */")
        mappings['edge_curves'][i] = current_id
        current_id += 1
    print(f"Created edge curves: {len(mappings['edge_curves'])}")
    print(f"Total edges: {len(edges)}")
    assert len(mappings['edge_curves']) == len(edges), "Not all edges have corresponding curves!"
    
    # 6. Oriented Edges (EDGE_CURVE)
    edge_lookup = {}
    for i, (v1, v2) in enumerate(edges):
        edge_lookup[(v1, v2)] = i
        edge_lookup[(v2, v1)] = i  # Add reverse direction since edge direction doesn't matter

    for i, (v1, v2, v3, normal) in enumerate(triangles):
        if i not in excluded_triangles:  # Only create if not part of cylinder
            # Get edge indices directly from lookup
            edge1 = edge_lookup[(v1, v2)] if (v1, v2) in edge_lookup else edge_lookup[(v2, v1)]
            edge2 = edge_lookup[(v2, v3)] if (v2, v3) in edge_lookup else edge_lookup[(v3, v2)]
            edge3 = edge_lookup[(v3, v1)] if (v3, v1) in edge_lookup else edge_lookup[(v1, v3)]
    
        # Create oriented edges
        for edge_idx in (edge1, edge2, edge3):
            key = (i, edge_idx)
            entities.append(f"#{current_id}=ORIENTED_EDGE('',*,*,#{mappings['edge_curves'][edge_idx]},.T.); /* triangle {i} oriented edge */")
            mappings['oriented_edges'][key] = current_id
            current_id += 1
    # 7. Edge Loops (ORIENTED_EDGES)
    for i, (v1, v2, v3, normal) in enumerate(triangles):
        if i not in excluded_triangles:  # Only create if not part of cylinder
            edge1 = edge_lookup.get((v1, v2)) or edge_lookup.get((v2, v1))
            edge2 = edge_lookup.get((v2, v3)) or edge_lookup.get((v3, v2))
            edge3 = edge_lookup.get((v3, v1)) or edge_lookup.get((v1, v3))
        
            entities.append(f"#{current_id}=EDGE_LOOP('',(#{mappings['oriented_edges'][(i,edge1)]},#{mappings['oriented_edges'][(i,edge2)]},#{mappings['oriented_edges'][(i,edge3)]})); /* triangle {i} loop */")
            mappings['edge_loops'][i] = current_id
            current_id += 1
    
    # 8. Face Bounds (EDGE_LOOPS)
    for i in range(len(triangles)):
        if i not in excluded_triangles:  # Only create if not part of cylinder
            entities.append(f"#{current_id}=FACE_BOUND('',#{mappings['edge_loops'][i]},.T.); /* triangle {i} bound */")
            mappings['face_bounds'][i] = current_id
            current_id += 1
    
    # 9. Triangle Normal Directions and Axis Placements
    for i, (v1, v2, v3, normal) in enumerate(triangles):
        if i not in excluded_triangles:  # Only create if not part of cylinder

            # Get perpendicular vectors
            perp1, perp2 = get_perpendicular_vectors(normal)
        
            # Add normal direction
            entities.append(f"#{current_id}=DIRECTION('',({normal[0]:.6f},{normal[1]:.6f},{normal[2]:.6f})); /* triangle {i} normal */")
            normal_id = current_id
            current_id += 1
        
            # Add perpendicular direction 
            entities.append(f"#{current_id}=DIRECTION('',({perp1[0]:.6f},{perp1[1]:.6f},{perp1[2]:.6f})); /* triangle {i} reference direction */")
            perp_id = current_id
            current_id += 1
        
            # Create axis placement (DIRECTION, DIRECTION)
            entities.append(f"#{current_id}=AXIS2_PLACEMENT_3D('',#{mappings['cartesian_points'][v1]},#{normal_id},#{perp_id}); /* triangle {i} axis */")
            mappings['axis_placements'][i] = current_id
            current_id += 1
    
    # 10. Planes (AXIS2_PLACEMENT_3D)
    for i in range(len(triangles)):
        if i not in excluded_triangles:  # Only create if not part of cylinder

            entities.append(f"#{current_id}=PLANE('',#{mappings['axis_placements'][i]}); /* triangle {i} plane */")
            mappings['planes'][i] = current_id
            current_id += 1
    
    # 11. Advanced Faces (FACE_BOUNDS, PLANES)
    for i in range(len(triangles)):
        if i not in excluded_triangles:  # Only create if not part of cylinder
            entities.append(f"#{current_id}=ADVANCED_FACE('',(#{mappings['face_bounds'][i]}),#{mappings['planes'][i]},.T.); /* triangle {i} face */")
            mappings['faces'][i] = current_id
            current_id += 1
    
    # 12. Closed Shell (ADVANCED_FACES)
    '''face_list = ",".join([f"#{mappings['faces'][i]}" for i in range(len(triangles))])
    entities.append(f"#{current_id}=CLOSED_SHELL('',({face_list})); /* complete shell */")
    closed_shell_id = current_id
    current_id += 1 '''
    
    # END RECTILINEAR
    
    if enable_curved_surfaces:
        # Process each detected cylindrical face
        for cyl_idx, (type_name, face_indices, diameter, center, axis) in enumerate(cylinder_faces):
            center_np = np.array(center)
            axis_np = np.array(axis)
            
            # Get all vertices involved in this cylindrical face
            face_vertices = set()
            for idx in face_indices:
                v1, v2, v3, _ = triangles[idx]
                face_vertices.update([v1, v2, v3])
            
            # Calculate actual height from vertices
            points = np.array([vertices[i] for i in face_vertices])
            heights = np.dot(points - center_np, axis_np)
            min_height = np.min(heights)
            max_height = np.max(heights)
            height = max_height - min_height
            
            # Calculate top and bottom centers accurately
            bottom_center = center_np + min_height * axis_np
            top_center = center_np + max_height * axis_np
            
            # Create reference directions (following example structure)
            # Primary axis is the cylinder axis (like #176 in example)
            # Reference direction is perpendicular (like #177 in example)
            #ref_dir = get_perpendicular_vectors(axis)[0]
            ref_dir = np.array(get_perpendicular_vectors(axis)[0])

            
            if type_name == "complete_cylinder":
                print("complete cylinder ID")

                # Create points (following example structure)
                # Bottom reference point
                radius = diameter/2
                bottom_point = bottom_center + radius * np.array(ref_dir)
                # Top reference point
                top_point = top_center + radius * np.array(ref_dir)
            
                # 1. Create all CARTESIAN_POINTs
                # Center point for cylindrical surface (like #214 in example)
                entities.append(f"#{current_id}=CARTESIAN_POINT('',({bottom_center[0]:.6f},{bottom_center[1]:.6f},{bottom_center[2]:.6f}));")
                surface_origin_id = current_id
                current_id += 1
            
                # Points for the circles' centers
                entities.append(f"#{current_id}=CARTESIAN_POINT('',({bottom_center[0]:.6f},{bottom_center[1]:.6f},{bottom_center[2]:.6f}));")
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
            
                # 4. Create AXIS2_PLACEMENT_3D for the cylindrical surface (like #148 in example)
                entities.append(f"#{current_id}=AXIS2_PLACEMENT_3D('',#{surface_origin_id},#{axis_direction_id},#{ref_direction_id});")
                cylinder_placement_id = current_id
                current_id += 1
            
                # Create placements for circles (separate for top and bottom)
                entities.append(f"#{current_id}=AXIS2_PLACEMENT_3D('',#{bottom_center_id},#{axis_direction_id},#{ref_direction_id});")
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
            
                # 8. Create ORIENTED_EDGEs (like #39, #40, #41, #42 in example)
                entities.append(f"#{current_id}=ORIENTED_EDGE('',*,*,#{top_edge_id},.T.);")
                oriented_edge1_id = current_id
                current_id += 1
            
                entities.append(f"#{current_id}=ORIENTED_EDGE('',*,*,#{line_edge_id},.F.);")
                oriented_edge2_id = current_id
                current_id += 1
            
                entities.append(f"#{current_id}=ORIENTED_EDGE('',*,*,#{bottom_edge_id},.F.);")
                oriented_edge3_id = current_id
                current_id += 1
            
                entities.append(f"#{current_id}=ORIENTED_EDGE('',*,*,#{line_edge_id},.T.);")
                oriented_edge4_id = current_id
                current_id += 1
            
                # 9. Create EDGE_LOOP (like #108 in example)
                edge_loop_str = f"#{oriented_edge1_id},#{oriented_edge2_id},#{oriented_edge3_id},#{oriented_edge4_id}"
                entities.append(f"#{current_id}=EDGE_LOOP('',({edge_loop_str}));")
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
                cylindrical_mappings['faces'][cyl_idx] = current_id
                current_id += 1
                pass
                
            else:  # "partial_cylinder"
                # partial cylinder section
                face_set = set(face_indices)
                edge_count = {}
                
                # Find edges that appear once in cylindrical faces
                for idx in face_indices:
                    v1, v2, v3, _ = triangles[idx]
                    for edge in [(v1, v2), (v2, v3), (v3, v1)]:
                        edge = tuple(sorted(edge))
                        edge_count[edge] = edge_count.get(edge, 0) + 1
                
                # Find boundary edges connecting to non-cylindrical faces
                boundary_edges = []
                boundary_vertices = set()
                
                for edge, count in edge_count.items():
                    if count == 1:  # Edge appears once in cylindrical faces
                        v1, v2 = edge
                        # Check if edge connects to non-cylindrical face
                        for i, (tv1, tv2, tv3, _) in enumerate(triangles):
                            if i not in face_set:  # Non-cylindrical face
                                if {v1, v2}.issubset({tv1, tv2, tv3}):
                                    boundary_edges.append(edge)
                                    boundary_vertices.update(edge)
                                    break
                
                # Order vertices along boundary
                ordered_vertices = []
                if boundary_edges:
                    current_edge = boundary_edges[0]
                    ordered_vertices.extend(current_edge)
                    used_edges = {current_edge}
                    
                    while len(used_edges) < len(boundary_edges):
                        found_next = False
                        current_vertex = ordered_vertices[-1]
                        
                        for edge in boundary_edges:
                            if edge not in used_edges:
                                if current_vertex in edge:
                                    next_vertex = edge[1] if edge[0] == current_vertex else edge[0]
                                    ordered_vertices.append(next_vertex)
                                    used_edges.add(edge)
                                    found_next = True
                                    break
                        
                        if not found_next:
                            break
                
                # Project boundary points onto plane perpendicular to axis
                boundary_points = np.array([vertices[idx] for idx in ordered_vertices])
                rel_vectors = boundary_points - bottom_center
                proj_vectors = rel_vectors - np.outer(np.dot(rel_vectors, axis_np), axis_np)
                
                # Normalize projected vectors
                proj_vectors = np.array([v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-10 else v 
                                       for v in proj_vectors])
                
                # Calculate reference vector and angles
                ref_vector = proj_vectors[0]
                angles = np.arctan2(
                    np.array([np.dot(np.cross(ref_vector, v), axis_np) for v in proj_vectors]),
                    np.array([np.dot(ref_vector, v) for v in proj_vectors])
                )
                
                diameter /= 2
                
                boundary_edges = find_boundary_edges(face_indices, triangles, edges)
                print("Length Boundary Edges!")
                print(boundary_edges)
                if len(boundary_edges) < 2:
                    print("Insufficient boundary edges found")
                    continue

                # Calculate start and end angles using existing vertices
                start_angle, end_angle = calculate_cylinder_angles(boundary_edges, vertices, bottom_center, axis)
                
                # Get points from existing vertices
                start_bottom = vertices[boundary_edges[0][0]]
                end_bottom = vertices[boundary_edges[-1][0]]
                start_top = start_bottom + height * axis_np
                end_top = end_bottom + height * axis_np
                ref_dir = ref_vector  # Use the calculated reference vector for surface orientation

                # 1. CARTESIAN_POINTs
                entities.append(f"#{current_id}=CARTESIAN_POINT('',({bottom_center[0]:.6f},{bottom_center[1]:.6f},{bottom_center[2]:.6f}));")
                surface_origin_id = current_id
                current_id += 1
                
                entities.append(f"#{current_id}=CARTESIAN_POINT('',({bottom_center[0]:.6f},{bottom_center[1]:.6f},{bottom_center[2]:.6f}));")
                bottom_center_id = current_id
                current_id += 1

                entities.append(f"#{current_id}=CARTESIAN_POINT('',({top_center[0]:.6f},{top_center[1]:.6f},{top_center[2]:.6f}));")
                top_center_id = current_id
                current_id += 1
                
                entities.append(f"#{current_id}=CARTESIAN_POINT('',({start_bottom[0]:.6f},{start_bottom[1]:.6f},{start_bottom[2]:.6f}));")
                start_bottom_id = current_id
                current_id += 1

                entities.append(f"#{current_id}=CARTESIAN_POINT('',({end_bottom[0]:.6f},{end_bottom[1]:.6f},{end_bottom[2]:.6f}));")
                end_bottom_id = current_id
                current_id += 1

                entities.append(f"#{current_id}=CARTESIAN_POINT('',({start_top[0]:.6f},{start_top[1]:.6f},{start_top[2]:.6f}));")
                start_top_id = current_id
                current_id += 1

                entities.append(f"#{current_id}=CARTESIAN_POINT('',({end_top[0]:.6f},{end_top[1]:.6f},{end_top[2]:.6f}));")
                end_top_id = current_id
                current_id += 1

                # 2. DIRECTIONs
                entities.append(f"#{current_id}=DIRECTION('',({axis[0]:.6f},{axis[1]:.6f},{axis[2]:.6f}));")
                axis_dir_id = current_id
                current_id += 1

                entities.append(f"#{current_id}=DIRECTION('',({ref_dir[0]:.6f},{ref_dir[1]:.6f},{ref_dir[2]:.6f}));")
                ref_dir_id = current_id
                current_id += 1

                # 3. VERTEX_POINTs
                entities.append(f"#{current_id}=VERTEX_POINT('',#{start_bottom_id});")
                start_bottom_vertex_id = current_id
                current_id += 1

                entities.append(f"#{current_id}=VERTEX_POINT('',#{end_bottom_id});")
                end_bottom_vertex_id = current_id
                current_id += 1

                entities.append(f"#{current_id}=VERTEX_POINT('',#{start_top_id});")
                start_top_vertex_id = current_id
                current_id += 1

                entities.append(f"#{current_id}=VERTEX_POINT('',#{end_top_id});")
                end_top_vertex_id = current_id
                current_id += 1

                # Create AXIS2_PLACEMENT_3D for bottom and top circles
                entities.append(f"#{current_id}=AXIS2_PLACEMENT_3D('',#{bottom_center_id},#{axis_dir_id},#{ref_dir_id});")
                bottom_placement_id = current_id
                current_id += 1

                entities.append(f"#{current_id}=AXIS2_PLACEMENT_3D('',#{top_center_id},#{axis_dir_id},#{ref_dir_id});")
                top_placement_id = current_id
                current_id += 1

                # 5. CIRCLEs for top and bottom arcs
                # Create CIRCLE entities for top and bottom
                entities.append(f"#{current_id}=CIRCLE('',#{bottom_placement_id},{diameter:.6f});")
                bottom_circle_id = current_id
                current_id += 1

                entities.append(f"#{current_id}=CIRCLE('',#{top_placement_id},{diameter:.6f});")
                top_circle_id = current_id
                current_id += 1

                # 6. Create VECTORs and LINEs for vertical edges
                entities.append(f"#{current_id}=VECTOR('',#{axis_dir_id},{height:.6f});")
                line_vector_id = current_id
                current_id += 1

                entities.append(f"#{current_id}=LINE('',#{start_bottom_id},#{line_vector_id});")
                line1_id = current_id
                current_id += 1

                entities.append(f"#{current_id}=LINE('',#{end_bottom_id},#{line_vector_id});")
                line2_id = current_id
                current_id += 1

                # 7. Create EDGE_CURVEs
                entities.append(f"#{current_id}=EDGE_CURVE('',#{start_bottom_vertex_id},#{end_bottom_vertex_id},#{bottom_circle_id},.T.);")
                bottom_arc_id = current_id
                current_id += 1
                
                entities.append(f"#{current_id}=EDGE_CURVE('',#{start_top_vertex_id},#{end_top_vertex_id},#{top_circle_id},.T.);")
                top_arc_id = current_id
                current_id += 1

                entities.append(f"#{current_id}=EDGE_CURVE('',#{start_bottom_vertex_id},#{start_top_vertex_id},#{line1_id},.T.);")
                line1_edge_id = current_id
                current_id += 1

                entities.append(f"#{current_id}=EDGE_CURVE('',#{end_bottom_vertex_id},#{end_top_vertex_id},#{line2_id},.T.);")
                line2_edge_id = current_id
                current_id += 1

                # 8. Create ORIENTED_EDGEs
                oriented_edges = []
                for edge_id, orientation in [
                    (bottom_arc_id, '.T.'),
                    (line2_edge_id, '.T.'),
                    (top_arc_id, '.F.'),
                    (line1_edge_id, '.F.')
                ]:
                    entities.append(f"#{current_id}=ORIENTED_EDGE('',*,*,#{edge_id},{orientation});")
                    oriented_edges.append(current_id)
                    current_id += 1

                # 9. Create EDGE_LOOP
                entities.append(f"#{current_id}=EDGE_LOOP('',({','.join(f'#{e}' for e in oriented_edges)}));")
                edge_loop_id = current_id
                current_id += 1

                # 10. Create FACE_BOUND
                entities.append(f"#{current_id}=FACE_BOUND('',#{edge_loop_id},.T.);")
                face_bound_id = current_id
                current_id += 1

                # 11. Create CYLINDRICAL_SURFACE
                entities.append(f"#{current_id}=CYLINDRICAL_SURFACE('',#{top_placement_id},{diameter:.6f});")
                surface_id = current_id
                current_id += 1

                # 12. Create ADVANCED_FACE
                entities.append(f"#{current_id}=ADVANCED_FACE('',(#{face_bound_id}),#{surface_id},.T.);")
                cylindrical_mappings['faces'][cyl_idx] = current_id
                current_id += 1
    
    # Combine all faces for closed shell
    all_face_ids = []
    all_face_ids.extend([f"#{mappings['faces'][i]}" for i in range(len(triangles)) if i not in excluded_triangles])
    if enable_curved_surfaces:
        all_face_ids.extend([f"#{cylindrical_mappings['faces'][i]}" for i in range(len(cylinder_faces))])
    
    # Create closed shell with all faces
    face_list = ",".join(all_face_ids)
    entities.append(f"#{current_id}=CLOSED_SHELL('',({face_list})); /* complete shell */")
    closed_shell_id = current_id
    current_id += 1
    
    return "\n".join(entities), current_id, mappings, closed_shell_id

def find_boundary_edges(face_indices, triangles, edges):
    """Find the two boundary edges of a partial cylinder that are parallel to its axis."""
    edge_count = defaultdict(int)
    
    # Count edge occurrences in cylindrical faces
    for idx in face_indices:
        v1, v2, v3, _ = triangles[idx]
        for edge in [(v1, v2), (v2, v3), (v3, v1)]:
            edge = tuple(sorted(edge))
            edge_count[edge] += 1
    
    # Get edges that appear once (boundary edges)
    return [edge for edge, count in edge_count.items() if count == 1]

def calculate_cylinder_angles(boundary_edges, vertices, center, axis):
    """Calculate start and end angles of partial cylinder from boundary edges."""
    axis_np = np.array(axis)
    center_np = np.array(center)
    
    # Project boundary vertices onto plane perpendicular to axis
    angles = []
    for v1, v2 in boundary_edges:
        point = np.array(vertices[v1]) - center_np
        proj = point - np.dot(point, axis_np) * axis_np
        if np.linalg.norm(proj) > 1e-10:
            proj = proj / np.linalg.norm(proj)
            angle = np.arctan2(proj[1], proj[0])
            angles.append(angle)
    
    return min(angles), max(angles)

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

vertices, triangles, edges = parse('cylinder.stl')