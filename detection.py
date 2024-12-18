import numpy as np
from rendering import *
from collections import defaultdict
from math import isclose

def detect_sketch_planes(vertices: np.ndarray, triangles: list, tolerance: float = 1e-3) -> list:
    # Step 1: Detect flat surfaces aligned with primary axes
    planes = {'x': set(), 'y': set(), 'z': set()}
    for vertex in vertices:
        x, y, z = vertex
        planes['x'].add(round(x, 5))
        planes['y'].add(round(y, 5))
        planes['z'].add(round(z, 5))

    def collect_surface_triangles(axis_planes, axis_index):
        surfaces = {}
        for plane in axis_planes:
            points_in_plane = [i for i, v in enumerate(vertices) if abs(v[axis_index] - plane) < tolerance]
            if len(points_in_plane) > 3:
                # Find edges for this plane
                plane_edges = set()
                for tri in triangles:
                    v1, v2, v3, _ = tri
                    if v1 in points_in_plane and v2 in points_in_plane and v3 in points_in_plane:
                        edges = {(min(v1, v2), max(v1, v2)),
                                (min(v2, v3), max(v2, v3)),
                                (min(v3, v1), max(v3, v1))}
                        plane_edges.update(edges)
                if plane_edges:  # Only add if there are valid edges
                    surfaces[plane] = (set(points_in_plane), plane_edges)
        return surfaces

    # Axis to normal vector mapping
    axis_to_normal = {
        'x': (1, 0, 0),
        'y': (0, 1, 0),
        'z': (0, 0, 1)
    }

    # Store flat surfaces on each axis with vertex coverage information
    flat_surfaces = {
        'x': collect_surface_triangles(planes['x'], axis_index=0),
        'y': collect_surface_triangles(planes['y'], axis_index=1),
        'z': collect_surface_triangles(planes['z'], axis_index=2)
    }

    # Step 2: Convert surfaces to unified format with coverage information
    candidate_planes = []
    for axis, surfaces in flat_surfaces.items():
        for magnitude, (verts, edges) in surfaces.items():
            candidate_planes.append({
                'normal': axis_to_normal[axis],
                'magnitude': magnitude,
                'vertices': verts,
                'edges': edges
            })
    # Step 3: Use greedy set cover to select minimal set of planes
    all_vertices = set(range(len(vertices)))
    selected_planes = []
    covered_vertices = set()
    
    while covered_vertices != all_vertices and candidate_planes:
        # Find plane that covers most uncovered vertices
        best_plane = max(candidate_planes, 
                        key=lambda p: len(p['vertices'] - covered_vertices))
        
        # If best plane adds no new coverage, check for off-angle planes
        if not (best_plane['vertices'] - covered_vertices):
            break
            
        # Add selected plane
        plane_index = len(selected_planes)
        selected_planes.append((
            plane_index,
            best_plane['normal'],
            best_plane['magnitude'],
            best_plane['edges']
        ))
        covered_vertices.update(best_plane['vertices'])
        
        # Remove selected plane from candidates
        candidate_planes.remove(best_plane)

    # Step 4: Handle remaining vertices with off-angle planes if needed
    if covered_vertices != all_vertices:
        leftover_vertices = all_vertices - covered_vertices
        normal_groups = {}
        
        for tri_idx, (v1, v2, v3, normal) in enumerate(triangles):
            if {v1, v2, v3} & leftover_vertices:
                normal_key = tuple(round(n, 5) for n in normal)
                if normal_key not in normal_groups:
                    normal_groups[normal_key] = (set(), set())
                normal_groups[normal_key][0].update({v1, v2, v3})
                edges = {(min(v1, v2), max(v1, v2)),
                        (min(v2, v3), max(v2, v3)),
                        (min(v3, v1), max(v3, v1))}
                normal_groups[normal_key][1].update(edges)

        # Add off-angle planes that provide the most additional coverage
        for normal, (verts, edges) in sorted(normal_groups.items(), 
                                           key=lambda x: len(x[1][0] - covered_vertices),
                                           reverse=True):
            if not (verts - covered_vertices):
                continue
                
            # Calculate magnitude as distance from origin to plane
            magnitude = sum(n * vertices[list(verts)[0]][i] for i, n in enumerate(normal))
            selected_planes.append((
                len(selected_planes),
                normal,
                magnitude,
                edges
            ))
            covered_vertices.update(verts)
            
            if covered_vertices == all_vertices:
                break

    return selected_planes

def detect_extrudes(sketch_planes: list, vertices: np.ndarray, tolerance: float = 1e-3, angle_tolerance: float = 0.005) -> dict:
    # Sort planes by index for consistent processing
    sorted_planes = sorted(sketch_planes, key=lambda x: x[0])  # x[0] is now the index
    plane_vertex_groups = []
    
    for plane in sorted_planes:
        index, normal_vector, magnitude, edges = plane
        # Extract vertices from edges
        plane_vertex_indices = set()
        for edge in edges:
            plane_vertex_indices.update(edge)
            
        # Group vertices based on their projected position
        vertex_groups = defaultdict(list)
        for idx in plane_vertex_indices:
            v = vertices[idx]  # Get actual vertex coordinates
            
            # Project vertices based on normal vector
            if np.allclose(normal_vector, (1, 0, 0)):  # X-axis
                key = (round(v[1], 5), round(v[2], 5))
            elif np.allclose(normal_vector, (0, 1, 0)):  # Y-axis
                key = (round(v[0], 5), round(v[2], 5))
            elif np.allclose(normal_vector, (0, 0, 1)):  # Z-axis
                key = (round(v[0], 5), round(v[1], 5))
            else:  # Off-angle plane
                # Project point onto plane's normal direction
                key = (round(np.dot(normal_vector, v), 5))
                
            vertex_groups[key].append(idx)
            
        plane_vertex_groups.append((normal_vector, magnitude, vertex_groups))

    # Detect parallel extrudes
    extrudes = []
    for i, (normal1, magnitude1, vertices1) in enumerate(plane_vertex_groups):
        for j in range(i + 1, len(plane_vertex_groups)):
            normal2, magnitude2, vertices2 = plane_vertex_groups[j]
            
            # Check if planes are parallel
            cos_angle = np.dot(normal1, normal2) / (np.linalg.norm(normal1) * np.linalg.norm(normal2))
            if not isclose(abs(cos_angle), 1, abs_tol=angle_tolerance):
                continue
                
            # Calculate distance between planes
            distance = round((magnitude2 - magnitude1), 5)
            
            # Find matching vertex pairs
            matching_pairs = []
            common_keys = set(vertices1.keys()) & set(vertices2.keys())
            for key in common_keys:
                for idx1 in vertices1[key]:
                    for idx2 in vertices2[key]:
                        matching_pairs.append((idx1, idx2))
                        
            # If enough matching pairs found, record the extrude
            if len(matching_pairs) > 3:
                direction = 1 if normal2 == normal1 else -1
                
                # Determine if this is an axis-aligned or off-angle extrude
                if np.allclose(normal1, (1, 0, 0)):
                    axis = 'x'
                elif np.allclose(normal1, (0, 1, 0)):
                    axis = 'y'
                elif np.allclose(normal1, (0, 0, 1)):
                    axis = 'z'
                else:
                    axis = 'normal'
                extrude_info = {
                    'start_plane': (normal1, magnitude1),
                    'end_plane': (normal2, magnitude2),
                    'distance': distance,
                    'direction': direction,
                    'matching_pairs': matching_pairs
                }
                extrudes.append(extrude_info)
    def group_extrudes_by_sketch(extrudes):
        """Group extrudes by their start planes to identify features created from the same sketch"""
        sketch_groups = {}
        sketch_index = 0
        
        # First pass: group extrudes by their start planes
        for i, extrude in enumerate(extrudes):
            start_plane = (extrude['start_plane'][0], round(extrude['start_plane'][1], 1))
            
            # If this start plane hasn't been seen, assign it a new sketch index
            if start_plane not in sketch_groups:
                sketch_groups[start_plane] = {
                    'sketch_index': sketch_index,
                    'extrudes': []
                }
                sketch_index += 1
                
            sketch_groups[start_plane]['extrudes'].append(i)
        
        return sketch_groups
    return {
        'extrudes': extrudes,
        'sketch_groups': group_extrudes_by_sketch(extrudes)
    }

def build_sketches(sketch_planes, vertices, triangles):
    sketches = []
    used_edges = set()
    triangle_normals = defaultdict(set)
    
    def normalize_vector(vector):
        normalized = np.array(vector)
        flipped = False
        if np.linalg.norm(normalized) != 0:
            normalized = normalized / np.linalg.norm(normalized)
            if normalized[0] < 0 or (normalized[0] == 0 and normalized[1] < 0) or \
               (normalized[0] == 0 and normalized[1] == 0 and normalized[2] < 0):
                normalized = -normalized
                flipped = True
        return tuple(round(float(c), 3) for c in normalized), flipped

    def create_2d_local_sketch(edges, vertices, transformation_matrix):
        local_edges = []
        for edge in edges:
            v1, v2 = edge
            vertex1, vertex2 = vertices[v1], vertices[v2]
            local_v1 = np.dot(transformation_matrix, np.array([*vertex1, 1]))[:2]
            local_v2 = np.dot(transformation_matrix, np.array([*vertex2, 1]))[:2]
            local_edges.append((tuple(local_v1), tuple(local_v2)))
        return local_edges

    def build_transformation_matrix(normalized_normal):
        z_axis = np.array(normalized_normal)
        x_axis = np.cross(z_axis, [0, 1, 0])
        if np.allclose(x_axis, 0):
            x_axis = np.cross(z_axis, [1, 0, 0])
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        transformation_matrix = np.array([
            [*x_axis, 0],
            [*y_axis, 0],
            [*z_axis, 0],
            [0, 0, 0, 1]
        ])
        return np.linalg.inv(transformation_matrix)

    # Store triangle normals for edge filtering
    for v1, v2, v3, normal in triangles:
        normalized_normal, _ = normalize_vector(normal)
        edges = [(v1, v2), (v2, v3), (v3, v1)]
        for edge in edges:
            edge_key = tuple(sorted(edge))
            triangle_normals[edge_key].add(normalized_normal)

    # Create unsorted sketches list
    unsorted_sketches = []
    for plane in sketch_planes:
        if len(plane[3]) < 3:
            continue
            
        index, normal, magnitude, plane_edges = plane
        normalized_normal, flipped = normalize_vector(normal)
        adjusted_magnitude = magnitude
        if flipped:
            adjusted_magnitude *= -1

        transformation_matrix = build_transformation_matrix(normalized_normal)

        # Collect unique edges associated with the sketch plane
        unique_edges = set()
        for edge in plane_edges:
            edge_key = tuple(sorted(edge))
            if len(triangle_normals[edge_key]) > 1:
                unique_edges.add(edge)

        formatted_edges = []
        for edge in unique_edges:
            if edge not in used_edges:
                v1, v2 = edge
                vertex1, vertex2 = vertices[v1], vertices[v2]
                formatted_edges.append(((round(vertex1[0], 3), round(vertex1[1], 3)),
                                      (round(vertex2[0], 3), round(vertex2[1], 3))))
                used_edges.add(edge)

        sketch_vertices = [
            np.dot(transformation_matrix, np.array([*v, 1]))[:2]
            for v in vertices if np.isclose(np.dot(v, normalized_normal), adjusted_magnitude, atol=1e-5)
        ]

        if formatted_edges:
            sketch = {
                'normal': normalized_normal,
                'magnitude': adjusted_magnitude,
                'edges': create_2d_local_sketch(unique_edges, vertices, transformation_matrix),
                'vertices': [(round(v[0], 3), round(v[1], 3)) for v in sketch_vertices],
                'transformation_matrix': transformation_matrix,
                'global_edges': formatted_edges,
                'abs_magnitude': abs(adjusted_magnitude)
            }
            unsorted_sketches.append(sketch)

    # Sort sketches by absolute magnitude
    sorted_sketches = sorted(unsorted_sketches, key=lambda x: x['abs_magnitude'])
    
    # Assign indices after sorting
    sketches = []
    for i, sketch in enumerate(sorted_sketches):
        sketch['index'] = i
        sketches.append(sketch)

    return sketches

def build_extrudes(raw_extrude_data, sketches, vertices, tolerance=1e-3):
    filtered_extrudes = []
    sketch_groups = {}
    
    print(f"Number of raw extrudes: {len(raw_extrude_data['extrudes'])}")
    print(f"Number of sketches: {len(sketches)}")
    
    # Create a mapping of sketch planes to their indices with consistent rounding
    sketch_planes = {}
    for sketch in sketches:
        normal = tuple(round(n, 1) for n in sketch['normal'])
        magnitude = round(sketch['magnitude'], 1)
        key = (normal, magnitude)
        sketch_planes[key] = sketch['index']
        print(f"Stored sketch key: {key} -> index {sketch['index']}")
    
    print("\nProcessing extrudes:")
    potential_extrudes = {}
    
    # Sort sketches by magnitude for proper matching
    sorted_sketches = sorted(sketch_planes.items(), key=lambda x: abs(x[0][1]))
    
    # Track which sketches are used as end planes
    end_plane_sketches = set()
    
    for i, extrude in enumerate(raw_extrude_data['extrudes']):
        # Get the original normal directions and magnitudes
        print("reid")
        print(extrude['direction'])
        normal1 = tuple(round(n, 1) for n in extrude['start_plane'][0])
        magnitude1 = round(extrude['start_plane'][1], 1)
        normal2 = tuple(round(n * extrude['direction'], 1) for n in extrude['end_plane'][0])
        magnitude2 = round(extrude['end_plane'][1], 1) * extrude['direction']
        print(f"\nProcessing extrude {i}:")
        print(f"Plane 1: {normal1}, {magnitude1}")
        print(f"Plane 2: {normal2}, {magnitude2},")
        
        # Find matching sketch indices for both planes
        sketch_index1 = None
        sketch_index2 = None
        
        for (sketch_normal, sketch_magnitude), sketch_index in sorted_sketches:
            # Check both planes
            if (np.allclose(normal1, sketch_normal, atol=0.1) and 
                abs(abs(magnitude1) - abs(sketch_magnitude)) < tolerance):
                sketch_index1 = sketch_index
            
            if (np.allclose(normal2, sketch_normal, atol=0.1) and 
                abs(abs(magnitude2) - abs(sketch_magnitude)) < tolerance):
                sketch_index2 = sketch_index
        
        if sketch_index1 is not None and sketch_index2 is not None:
            # Determine which sketch should be the start based on magnitude
            sketch1_mag = sketches[sketch_index1]['magnitude']
            sketch2_mag = sketches[sketch_index2]['magnitude']
            
            # If sketch2 is already used as an end plane, make sketch1 the end plane
            if sketch_index2 in end_plane_sketches:
                start_sketch_index = sketch_index2
                end_sketch_index = sketch_index1
                start_magnitude = sketch2_mag
                end_magnitude = sketch1_mag
            # If sketch1 is already used as an end plane, make sketch2 the end plane
            elif sketch_index1 in end_plane_sketches:
                start_sketch_index = sketch_index1
                end_sketch_index = sketch_index2
                start_magnitude = sketch1_mag
                end_magnitude = sketch2_mag
            # Otherwise, use magnitude to determine order
            else:
                if sketch1_mag < sketch2_mag:
                    start_sketch_index = sketch_index1
                    end_sketch_index = sketch_index2
                    start_magnitude = sketch1_mag
                    end_magnitude = sketch2_mag
                else:
                    start_sketch_index = sketch_index2
                    end_sketch_index = sketch_index1
                    start_magnitude = sketch2_mag
                    end_magnitude = sketch1_mag
            
            # Track this sketch as being used as an end plane
            end_plane_sketches.add(end_sketch_index)
            
            if start_sketch_index not in potential_extrudes:
                potential_extrudes[start_sketch_index] = []
            
            extrude_info = extrude.copy()
            extrude_info['start_sketch_index'] = start_sketch_index
            extrude_info['end_sketch_index'] = end_sketch_index
            extrude_info['start_magnitude'] = start_magnitude
            extrude_info['end_magnitude'] = end_magnitude
            potential_extrudes[start_sketch_index].append((i, extrude_info))
            print(f"Matched: Start sketch {start_sketch_index} -> End sketch {end_sketch_index}")
    
    # Process matched extrudes
    for start_sketch_index in sorted(potential_extrudes.keys()):
        extrudes_for_sketch = potential_extrudes[start_sketch_index]
        
        for i, extrude in extrudes_for_sketch:
            # Get vertices involved in this extrude
            extrude_vertices = set()
            for v1, v2 in extrude['matching_pairs']:
                extrude_vertices.add(v1)
                extrude_vertices.add(v2)
            
            if len(extrude_vertices) > 3:
                filtered_extrudes.append(extrude)
                
                # Update sketch groups
                start_normal = tuple(round(n, 1) for n in sketches[extrude['start_sketch_index']]['normal'])
                start_magnitude = round(sketches[extrude['start_sketch_index']]['magnitude'], 1)
                plane_key = (start_normal, start_magnitude)
                
                if plane_key not in sketch_groups:
                    sketch_groups[plane_key] = {
                        'sketch_index': start_sketch_index,
                        'extrudes': []
                    }
                sketch_groups[plane_key]['extrudes'].append(len(filtered_extrudes) - 1)
    
    return {
        'extrudes': filtered_extrudes,
        'sketch_groups': sketch_groups
    }

def classify_feature_type(sketch, extrude, previous_features):
    """
    Advanced feature classification considering geometric context
    
    Args:
        sketch: Current sketch
        extrude: Current extrude
        previous_features: List of previously processed features
    """
    normal = np.array(sketch['normal'])
    direction = extrude.get('direction', 1)
    magnitude = abs(sketch['magnitude'])
    
    # Base plane detection (horizontal plane)
    if np.allclose(normal, (0, 0, 1), atol=0.1) and direction > 0:
        return 'Base Extrude'
    
    # Emboss vs Cutout detection
    def is_inner_geometry(current_edges):
        """Determine if a sketch represents an inner geometry"""
        return len(current_edges) < 4  # Simple heuristic for inner geometries
    
    current_edges = sketch.get('edges', [])
    
    if direction < 0:
        # Potential cutout
        if is_inner_geometry(current_edges):
            return 'Inner Cutout'
        return 'Outer Cutout'
    
    if direction > 0:
        # Potential emboss
        if is_inner_geometry(current_edges):
            return 'Inner Emboss'
        return 'Side Extension'
    
    return 'Undefined Feature'

def build_feature_tree(sketches, extrude_data):
    feature_tree = []
    used_extrude_indices = set()

    # Step 1: Group sketches into parent-child relationships
    def group_sketches_by_hierarchy(sketches):
        hierarchy = defaultdict(list)
        for i, parent_sketch in enumerate(sketches):
            for j, child_sketch in enumerate(sketches):
                if i != j and is_contained(child_sketch['edges'], parent_sketch['edges']):
                    hierarchy[i].append(j)
        return hierarchy

    def is_contained(inner_edges, outer_edges):
        """Determine if all edges of one sketch are within another."""
        return all(any(e1 == e2 for e2 in outer_edges) for e1 in inner_edges)

    def classify_feature_type(sketch, extrudes, parent_edges):
        """Classify feature type based on geometry relationships."""
        direction = extrudes[0]['direction'] if extrudes else 1
        sketch_edges = sketch.get('edges', [])
        is_inner = len(sketch_edges) < len(parent_edges) and is_contained(sketch_edges, parent_edges)

        if direction < 0:  # Cutout
            return 'Inner Cutout' if is_inner else 'Outer Cutout'
        elif direction > 0:  # Emboss
            return 'Inner Emboss' if is_inner else 'Outer Emboss'
        else:
            return 'Base Feature'

    sketch_hierarchy = group_sketches_by_hierarchy(sketches)

    # Step 2: Build features
    for parent_index, parent_sketch in enumerate(sketches):
        if parent_index in used_extrude_indices:
            continue

        # Identify extrudes starting from this sketch
        parent_extrudes = [
            extrude for extrude_index, extrude in enumerate(extrude_data['extrudes'])
            if extrude['start_sketch_index'] == parent_index and extrude_index not in used_extrude_indices
        ]

        # Classify features for this sketch
        if not parent_extrudes:
            continue

        parent_edges = parent_sketch.get('edges', [])
        feature_type = classify_feature_type(parent_sketch, parent_extrudes, parent_edges)

        # Add parent feature
        feature_tree.append({
            'sketch': parent_sketch,
            'type': feature_type,
            'extrudes': parent_extrudes,
            'children': [
                {
                    'sketch': sketches[child_index],
                    'type': classify_feature_type(sketches[child_index], [], parent_edges),
                    'extrudes': [],
                }
                for child_index in sketch_hierarchy.get(parent_index, [])
            ],
        })

        # Mark extrudes as used by their indices
        for extrude_index, extrude in enumerate(extrude_data['extrudes']):
            if extrude in parent_extrudes:
                used_extrude_indices.add(extrude_index)

    return feature_tree
