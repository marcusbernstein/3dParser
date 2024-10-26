import numpy as np
from rendering import *
from collections import defaultdict

def detect_flat_surfaces(vertices: np.ndarray, triangles: list, tolerance: float = 1e-3) -> dict:
    # Sets to store unique planes along each axis
    planes_x = set()
    planes_y = set()
    planes_z = set()

    # Check all vertices and group them by rounded X, Y, or Z magnitudes
    for vertex in vertices:
        x, y, z = vertex
        # Round the values to avoid floating-point imprecision when grouping
        planes_x.add(round(x, 5))
        planes_y.add(round(y, 5))
        planes_z.add(round(z, 5))

    # Helper function to filter planes based on the number of points in each plane
    def filter_planes(axis_planes, axis_index):
        flat_planes = []
        for plane in axis_planes:
            # Collect points that are within the tolerance of this plane
            points_in_plane = [vertex for vertex in vertices if abs(vertex[axis_index] - plane) < tolerance]
            
            # Only consider planes with more than 3 points
            if len(points_in_plane) > 3:
                flat_planes.append(plane)
        return flat_planes

    # Detect flat surfaces on X, Y, Z axes
    flat_surfaces = {
        'x': filter_planes(planes_x, axis_index=0),
        'y': filter_planes(planes_y, axis_index=1),
        'z': filter_planes(planes_z, axis_index=2)
    }
    return flat_surfaces

def detect_sketch_planes(flat_surfaces: dict, vertices: np.ndarray, tolerance=1e-3) -> list:
    # First, group parallel surfaces together by their axis and position
    parallel_groups = {}
    for axis, planes in flat_surfaces.items():
        for plane in planes:
            # Round plane position to handle floating point comparison
            plane_key = (axis, round(plane/tolerance)*tolerance)
            if plane_key not in parallel_groups:
                parallel_groups[plane_key] = []
            
            # Find vertices on this plane
            if axis == 'x':
                plane_vertices = [i for i, v in enumerate(vertices) if abs(v[0] - plane) < tolerance]
            elif axis == 'y':
                plane_vertices = [i for i, v in enumerate(vertices) if abs(v[1] - plane) < tolerance]
            else:  # z
                plane_vertices = [i for i, v in enumerate(vertices) if abs(v[2] - plane) < tolerance]
                
            if len(plane_vertices) > 3:
                parallel_groups[plane_key].append((plane, set(plane_vertices)))

    # Convert parallel groups to candidate sketch planes
    candidate_planes = []
    for (axis, pos), surfaces in parallel_groups.items():
        if surfaces:  # If we found any valid surfaces on this plane
            # Combine all vertices from parallel surfaces
            all_vertices = set().union(*(verts for _, verts in surfaces))
            if len(all_vertices) > 3:
                candidate_planes.append((axis, pos, list(all_vertices)))
    
    # Sort planes first by distance from origin, then by number of vertices for planes with same magnitude
    def sort_key(plane):
        axis, pos, vertices = plane
        # Primary sort: absolute distance from origin (smaller is better)
        distance_from_origin = abs(pos)
        # Secondary sort: number of vertices (more is better, use negative for reverse sort)
        vertex_count = -len(vertices)  # Negative because we want higher vertex counts first
        return (distance_from_origin, vertex_count)
    
    # Sort using the new criteria
    candidate_planes.sort(key=sort_key)
    
    # Greedily select sketch planes
    covered_vertices = set()
    sketch_planes = []
    
    for plane in candidate_planes:
        axis, pos, plane_vertices = plane
        # Check how many new vertices this plane would cover
        new_vertices = set(plane_vertices) - covered_vertices
        if len(new_vertices) > 3:  # Only select if it covers enough new vertices
            sketch_planes.append(plane)
            covered_vertices.update(new_vertices)
            
            # Optional: If all vertices are covered, we can stop
            if len(covered_vertices) == len(vertices):
                break
    
    return sketch_planes

def detect_sketch_plane_edges(sketch_planes: list, edges: list, vertices: np.ndarray, tolerance=1e-3) -> list:
    sketch_plane_edges = []

    for plane in sketch_planes:
        axis, plane_magnitude, plane_vertices = plane
        plane_edges = []

        # Iterate over all edges and check if both vertices of the edge lie on the sketch plane
        for edge in edges:
            v1, v2 = edge
            vertex1 = vertices[v1]
            vertex2 = vertices[v2]

            # Check if both vertices of the edge lie on the same sketch plane within the tolerance
            if axis == 'x':
                if abs(vertex1[0] - plane_magnitude) < tolerance and abs(vertex2[0] - plane_magnitude) < tolerance:
                    plane_edges.append(edge)
            elif axis == 'y':
                if abs(vertex1[1] - plane_magnitude) < tolerance and abs(vertex2[1] - plane_magnitude) < tolerance:
                    plane_edges.append(edge)
            elif axis == 'z':
                if abs(vertex1[2] - plane_magnitude) < tolerance and abs(vertex2[2] - plane_magnitude) < tolerance:
                    plane_edges.append(edge)

        # Store the edges that belong to this sketch plane
        sketch_plane_edges.append((axis, plane_magnitude, plane_edges))

    return sketch_plane_edges

def detect_extrudes(sketch_planes: list, vertices: np.ndarray, tolerance: float = 1e-5) -> dict:
    # Sort sketch planes by axis, then by magnitude
    sorted_planes = sorted(sketch_planes, key=lambda x: (x[0], x[1]))
    
    extrudes = []
    # Group vertices by their rounded x,y coordinates for each plane
    plane_vertex_groups = []
    for axis, magnitude, plane_vertex_indices in sorted_planes:
        vertex_groups = defaultdict(list)
        for idx in plane_vertex_indices:
            v = vertices[idx]
            # Round the coordinates to ensure consistent matching, avoid floating-point issues
            key = (round(v[0], 5), round(v[1], 5))  # Using 5 decimals for more precision
            vertex_groups[key].append(idx)
        plane_vertex_groups.append((axis, magnitude, vertex_groups))

    for i, (axis1, magnitude1, vertices1) in enumerate(plane_vertex_groups):
        for j in range(i+1, len(plane_vertex_groups)):
            axis2, magnitude2, vertices2 = plane_vertex_groups[j]
            magnitude1 = round(magnitude1, 3)
            magnitude2 = round(magnitude2, 3)
            if axis1 != axis2:
                continue  # Only consider planes on the same axis
            # Find matching vertex pairs based on rounded x, y coordinates
            matching_pairs = []
            for key in set(vertices1.keys()) & set(vertices2.keys()):
                for idx1 in vertices1[key]:
                    for idx2 in vertices2[key]:
                        matching_pairs.append((idx1, idx2))

            if len(matching_pairs) >= 3:
                # Potential extrude found
                direction = 1 if magnitude2 > magnitude1 else -1
                # Round the distance to 5 decimal places to ensure precision
                distance = round(abs(magnitude2 - magnitude1), 5)

                extrude_info = {
                    'start_plane': (axis1, magnitude1),
                    'end_plane': (axis2, magnitude2),
                    'direction': direction,
                    'distance': distance,  # Store the rounded distance
                    'matching_vertices': matching_pairs,
                    'vertex_count': len(matching_pairs)
                }
                extrudes.append(extrude_info)

    def group_extrudes_by_sketch(extrudes):
        """Group extrudes by their start planes to identify features created from the same sketch"""
        sketch_groups = {}
        sketch_index = 0
        
        # First pass: group extrudes by their start planes
        for i, extrude in enumerate(extrudes):
            start_plane = (extrude['start_plane'][0], round(extrude['start_plane'][1], 3))
            
            # If this start plane hasn't been seen, assign it a new sketch index
            if start_plane not in sketch_groups:
                sketch_groups[start_plane] = {
                    'sketch_index': sketch_index,
                    'extrudes': []
                }
                sketch_index += 1
                
            sketch_groups[start_plane]['extrudes'].append(i)
        
        return sketch_groups

    # Add sketch grouping information to the return value
    return {
        'extrudes': extrudes,
        'sketch_groups': group_extrudes_by_sketch(extrudes)
    }

def build_sketches(sketch_planes, sketch_plane_edges, extrude_data, vertices):
    sketches = []
    used_edges = set()  # Keep track of edges we've already assigned to earlier sketches
    
    # Create a mapping of plane locations to their sketch indices
    start_planes = {(plane[0], round(plane[1], 3)): info['sketch_index'] 
                   for plane, info in extrude_data['sketch_groups'].items()}
    
    # Sort sketch planes by sketch index to ensure we process them in order
    sorted_planes = []
    for plane in sketch_planes:
        axis, plane_magnitude, plane_vertices = plane
        plane_key = (axis, round(plane_magnitude, 3))
        if plane_key in start_planes:
            sorted_planes.append((start_planes[plane_key], plane))
    
    sorted_planes.sort(key=lambda x: x[0])  # Sort by sketch index
    
    for sketch_index, plane in sorted_planes:
        axis, plane_magnitude, plane_vertices = plane
        plane_key = (axis, round(plane_magnitude, 3))
        sketch_edges = set()  # Edges for this specific sketch
        
        # First, collect all candidate edges that are directly on this plane
        candidate_edges = set()
        for edge_plane in sketch_plane_edges:
            edge_axis, edge_magnitude, edges = edge_plane
            if (edge_axis == axis and 
                abs(round(edge_magnitude, 3) - round(plane_magnitude, 3)) < 1e-3):
                for edge in edges:
                    candidate_edges.add(tuple(sorted(edge)))  # Sort vertices to ensure consistent ordering
        
        # Find all extrudes that start from this plane
        for extrude in extrude_data['extrudes']:
            start_plane = extrude['start_plane']
            end_plane = extrude['end_plane']
            
            # If this extrude starts from our current plane
            if (start_plane[0] == axis and 
                abs(round(start_plane[1], 3) - round(plane_magnitude, 3)) < 1e-3):
                
                # Add edges from both start and end planes
                for edge_plane in sketch_plane_edges:
                    edge_axis, edge_magnitude, edges = edge_plane
                    
                    # Check both start and end planes
                    if ((edge_axis == start_plane[0] and 
                         abs(round(edge_magnitude, 3) - round(start_plane[1], 3)) < 1e-3) or
                        (edge_axis == end_plane[0] and 
                         abs(round(edge_magnitude, 3) - round(end_plane[1], 3)) < 1e-3)):
                        for edge in edges:
                            candidate_edges.add(tuple(sorted(edge)))
        
        # Filter out edges that have already been used in earlier sketches
        new_edges = candidate_edges - used_edges
        
        # If we found any new edges, create a sketch
        if new_edges:
            # Convert vertex indices to 2D coordinates based on the plane's axis
            edges_2d = []
            for edge in new_edges:
                v1, v2 = edge
                vertex1 = vertices[v1]
                vertex2 = vertices[v2]
                
                # Project vertices to 2D based on the plane's axis
                if axis == 'x':
                    point1 = (vertex1[1], vertex1[2])
                    point2 = (vertex2[1], vertex2[2])
                elif axis == 'y':
                    point1 = (vertex1[0], vertex1[2])
                    point2 = (vertex2[0], vertex2[2])
                else:  # z
                    point1 = (vertex1[0], vertex1[1])
                    point2 = (vertex2[0], vertex2[1])
                
                edges_2d.append((point1, point2))
            
            if edges_2d:  # Only create a sketch if we have edges to show
                sketches.append({
                    'index': sketch_index,
                    'axis': axis,
                    'magnitude': plane_magnitude,
                    'edges': edges_2d
                })
                
                # Add these edges to our used set
                used_edges.update(new_edges)
    
    return sorted(sketches, key=lambda x: x['index'])

def build_extrudes(raw_extrude_data, sketches, vertices, tolerance=1e-3):
    # Initialize storage for filtered extrudes
    filtered_extrudes = []
    sketch_groups = {}
    
    # Create a mapping of sketch planes to their indices
    sketch_planes = {(sketch['axis'], round(sketch['magnitude'], 3)): sketch['index'] 
                    for sketch in sketches}
    
    # Track vertices that have been accounted for
    covered_vertices = set()
    
    # First pass: Group extrudes by their start planes and filter overlapping ones
    potential_extrudes = {}
    for i, extrude in enumerate(raw_extrude_data['extrudes']):
        start_plane = (extrude['start_plane'][0], round(extrude['start_plane'][1], 3))
        if start_plane in sketch_planes:
            sketch_index = sketch_planes[start_plane]
            if sketch_index not in potential_extrudes:
                potential_extrudes[sketch_index] = []
            potential_extrudes[sketch_index].append((i, extrude))
    
    # Second pass: Select the most significant extrudes for each sketch
    for sketch_index in sorted(potential_extrudes.keys()):
        extrudes_for_sketch = potential_extrudes[sketch_index]
        
        # Sort extrudes by number of matching vertices (most significant first)
        extrudes_for_sketch.sort(key=lambda x: len(x[1]['matching_vertices']), reverse=True)
        
        valid_extrudes = []
        sketch_vertices = set()
        
        for i, extrude in extrudes_for_sketch:
            # Get vertices involved in this extrude
            extrude_vertices = set()
            for v1, v2 in extrude['matching_vertices']:
                extrude_vertices.add(v1)
                extrude_vertices.add(v2)
            
            # Check if this extrude contributes enough new vertices
            new_vertices = extrude_vertices - sketch_vertices
            if len(new_vertices) >= 3:  # Require at least 3 new vertices for a meaningful feature
                # Add to valid extrudes
                valid_extrudes.append(i)
                filtered_extrudes.append(extrude)
                sketch_vertices.update(extrude_vertices)
                covered_vertices.update(extrude_vertices)
                
                # Update sketch groups
                if sketch_index not in sketch_groups:
                    sketch_groups[sketch_index] = []
                sketch_groups[sketch_index].append(len(filtered_extrudes) - 1)
    
    return {
        'extrudes': filtered_extrudes,
        'sketch_groups': {
            (sketches[idx]['axis'], round(sketches[idx]['magnitude'], 3)): {
                'sketch_index': idx,
                'extrudes': extrudes
            }
            for idx, extrudes in sketch_groups.items()
        }
    }