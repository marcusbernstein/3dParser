import numpy as np
from rendering import *
from collections import defaultdict

def detect_flat_surfaces(vertices: np.ndarray, triangles: list, tolerance: float = 1e-5) -> dict:
    planes_x = set()
    planes_y = set()
    planes_z = set()

    # Check all vertices and group them by rounded X, Y, or Z magnitudes
    for vertex in vertices:
        x, y, z = vertex
        planes_x.add(x)  # Group similar X magnitudes
        planes_y.add(y)  # Group similar Y magnitudes
        planes_z.add(z)  # Group similar Z magnitudes

    # Remove planes where only a few points lie (like single-point planes)
    def filter_planes(axis_planes, axis_index):
        flat_planes = []
        for plane in axis_planes:
            points_in_plane = [vertex for vertex in vertices if (abs(vertex[axis_index]) - plane) < tolerance]
            if len(points_in_plane) > 3:  # Only consider planes with more than 3 points
                flat_planes.append(plane)
        return flat_planes

    # Detect flat surfaces on X, Y, Z axes
    flat_surfaces = {
        'x': filter_planes(planes_x, axis_index=0),
        'y': filter_planes(planes_y, axis_index=1),
        'z': filter_planes(planes_z, axis_index=2)
    }
    return flat_surfaces

def detect_sketch_planes(flat_surfaces: dict, vertices: np.ndarray, tolerance=1e-5) -> list:
    candidate_planes = []

    for axis, planes in flat_surfaces.items():
        for plane in planes:
            # Find the vertices that lie on this plane, applying rounding and tolerance
            if axis == 'x':
                plane_vertices = [i for i, v in enumerate(vertices) if abs(v[0] - plane) < tolerance]
            elif axis == 'y':
                plane_vertices = [i for i, v in enumerate(vertices) if abs(v[1] - plane) < tolerance]
            elif axis == 'z':
                plane_vertices = [i for i, v in enumerate(vertices) if abs(v[2] - plane) < tolerance]

            if len(plane_vertices) > 3:  # Only consider flat surfaces with more than 3 vertices
                candidate_planes.append((axis, plane, plane_vertices))

    # Sort candidate planes by the number of vertices they cover, in descending order
    candidate_planes.sort(key=lambda p: len(p[2]), reverse=True)

    # Set to track the vertices that have been covered
    covered_vertices = set()
    sketch_planes = []

    # Greedily select planes until all relevant vertices are covered
    for plane in candidate_planes:
        axis, magnitude, plane_vertices = plane

        # Find the vertices that are not yet covered
        uncovered_vertices = [v for v in plane_vertices if v not in covered_vertices]
        if uncovered_vertices:  # Only select the plane if it covers new vertices
            sketch_planes.append(plane)
            covered_vertices.update(uncovered_vertices)

        # Stop if we've covered all relevant vertices
        if len(covered_vertices) == len(vertices):
            break

    print("sketch_planes")
    print(len(sketch_planes))
    print(sketch_planes)
    return sketch_planes

def detect_extrudes(sketch_planes: list, vertices: np.ndarray, tolerance: float = 1e-5) -> dict:
    # Sort sketch planes by axis, then by magnitude
    sorted_planes = sorted(sketch_planes, key=lambda x: (x[0], x[1]))
    
    extrudes = []
    debug_info = []

    debug_info.append(f"Total number of sketch planes: {len(sorted_planes)}")
    for i, (axis, magnitude, plane_vertices) in enumerate(sorted_planes):
        debug_info.append(f"Plane {i}: axis={axis}, magnitude={magnitude}, vertices={len(plane_vertices)}")

    # Group vertices by their x,y coordinates for each plane
    plane_vertex_groups = []
    for axis, magnitude, plane_vertex_indices in sorted_planes:
        vertex_groups = defaultdict(list)
        for idx in plane_vertex_indices:
            v = vertices[idx]
            key = (round(v[0]/tolerance), round(v[1]/tolerance))  # Round to handle floating point imprecision
            vertex_groups[key].append(idx)
        plane_vertex_groups.append((axis, magnitude, vertex_groups))

    for i, (axis1, magnitude1, vertices1) in enumerate(plane_vertex_groups):
        for j in range(i+1, len(plane_vertex_groups)):
            axis2, magnitude2, vertices2 = plane_vertex_groups[j]
            
            if axis1 != axis2:
                continue  # Only consider planes on the same axis

            debug_info.append(f"\nComparing planes:")
            debug_info.append(f"  Plane 1: axis={axis1}, magnitude={magnitude1}, vertex groups={len(vertices1)}")
            debug_info.append(f"  Plane 2: axis={axis2}, magnitude={magnitude2}, vertex groups={len(vertices2)}")

            # Find matching vertex pairs
            matching_pairs = []
            for key in set(vertices1.keys()) & set(vertices2.keys()):
                for idx1 in vertices1[key]:
                    for idx2 in vertices2[key]:
                        matching_pairs.append((idx1, idx2))

            debug_info.append(f"  Matching vertex pairs: {len(matching_pairs)}")

            if len(matching_pairs) >= 3:
                # Potential extrude found
                direction = 1 if magnitude2 > magnitude1 else -1
                distance = abs(magnitude2 - magnitude1)

                extrude_info = {
                    'start_plane': (axis1, magnitude1),
                    'end_plane': (axis2, magnitude2),
                    'direction': direction,
                    'distance': distance,
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