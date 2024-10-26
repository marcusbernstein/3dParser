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
    
    # Sort by number of vertices covered
    candidate_planes.sort(key=lambda p: len(p[2]), reverse=True)
    
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
    debug_info = []

    debug_info.append(f"Total number of sketch planes: {len(sorted_planes)}")
    for i, (axis, magnitude, plane_vertices) in enumerate(sorted_planes):
        debug_info.append(f"Plane {i}: axis={axis}, magnitude={magnitude}, vertices={len(plane_vertices)}")

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

            debug_info.append(f"\nComparing planes:")
            debug_info.append(f"  Plane 1: axis={axis1}, magnitude={magnitude1}, vertex groups={len(vertices1)}")
            debug_info.append(f"  Plane 2: axis={axis2}, magnitude={magnitude2}, vertex groups={len(vertices2)}")

            # Find matching vertex pairs based on rounded x, y coordinates
            matching_pairs = []
            for key in set(vertices1.keys()) & set(vertices2.keys()):
                for idx1 in vertices1[key]:
                    for idx2 in vertices2[key]:
                        matching_pairs.append((idx1, idx2))

            debug_info.append(f"  Matching vertex pairs: {len(matching_pairs)}")

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

                debug_info.append(f"Extrude detected:")
                debug_info.append(f"  Start plane: {axis1}-axis, magnitude {magnitude1}")
                debug_info.append(f"  End plane: {axis2}-axis, magnitude {magnitude2}")
                debug_info.append(f"  Direction: {direction}")
                debug_info.append(f"  Distance: {distance}")
                debug_info.append(f"  Matching vertices: {len(matching_pairs)}")
            else:
                debug_info.append("  No extrude detected between these planes.")

    return {
        'extrudes': extrudes,
        'debug_info': debug_info
    }