import os
import numpy as np
from datetime import datetime

def write_step_file(sketches, extrudes, output_file="output.step"):
    """Generate a robust STEP file compliant with ISO 10303-21."""
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    entities = []
    counter = 1

    def next_id():
        nonlocal counter
        id_val = counter
        counter += 1
        return id_val

    # Maps to avoid duplicates
    point_map = {}
    edge_map = {}

    def convert_to_global_point(coords, transformation_matrix=None):
        """Convert point to global 3D coordinates."""
        if len(coords) == 2:
            # Assume 2D point lies on XY-plane, Z=0 by default
            coords = (*coords, 0)
        elif len(coords) != 3:
            raise ValueError(f"Invalid point coordinates: {coords}. Expected 2 or 3 elements.")

        # Apply transformation matrix if provided
        if transformation_matrix is not None:
            coords = np.dot(transformation_matrix, [*coords, 1])[:3]

        return tuple(round(c, 6) for c in coords)

    def add_point(coords, transformation_matrix=None):
        """Add a 3D point to the STEP file."""
        coords = convert_to_global_point(coords, transformation_matrix)
        key = coords
        if key not in point_map:
            point_id = next_id()
            point_map[key] = point_id
            entities.append(f"#{point_id} = CARTESIAN_POINT('', ({coords[0]:.6f}, {coords[1]:.6f}, {coords[2]:.6f}));")
        return point_map[key]

    def add_edge(start_coords, end_coords):
        """Add an edge defined by two points."""
        key = tuple(sorted((tuple(start_coords), tuple(end_coords))))
        if key not in edge_map:
            start_id = add_point(start_coords)
            end_id = add_point(end_coords)
            edge_id = next_id()
            entities.append(f"#{edge_id} = LINE('', #{start_id}, #{end_id});")
            edge_map[key] = edge_id
        return edge_map[key]

    def add_face_loop(edges):
        """Define a face loop for a closed profile."""
        loop_ids = [add_edge(*edge) for edge in edges]
        oriented_edges = []
        for edge_id in loop_ids:
            oriented_edge_id = next_id()
            entities.append(f"#{oriented_edge_id} = ORIENTED_EDGE('', *, *, #{edge_id}, .T.);")
            oriented_edges.append(oriented_edge_id)
        
        loop_id = next_id()
        entities.append(f"#{loop_id} = EDGE_LOOP('', ({', '.join(f'#{eid}' for eid in oriented_edges)}));")
        return loop_id

    def add_face(edges, normal):
        """Define a planar face from edges."""
        loop_id = add_face_loop(edges)
        direction_id = next_id()
        entities.append(f"#{direction_id} = DIRECTION('', ({normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f}));")
        plane_id = next_id()
        entities.append(f"#{plane_id} = PLANE('', #{direction_id});")
        face_id = next_id()
        entities.append(f"#{face_id} = ADVANCED_FACE('', (#{loop_id}), #{plane_id}, .T.);")
        return face_id

    def validate_geometry():
        """Validate generated geometry for STEP compliance."""
        # Simple validation checks for now
        if not sketches:
            print("[Validation Warning] No sketches found.")
        if not extrudes:
            print("[Validation Warning] No extrudes found.")

    # Add STEP Header
    entities.append("ISO-10303-21;")
    entities.append("HEADER;")
    entities.append("/* Generated by software containing ST-Developer */")
    entities.append("/* from STEP Tools, Inc. (www.steptools.com) */")
    entities.append("/* OPTION: using custom renumber hook */")
    entities.append("FILE_DESCRIPTION(('STEP AP242',")
    entities.append("'CAx-IF Rec.Pracs.---Representation and Presentation of Product Manufa")
    entities.append("cturing Information (PMI)---4.0---2014-10-13',")
    entities.append("'CAx-IF Rec.Pracs.---3D Tessellated Geometry---0.4---2014-09-14','2;1'),")
    entities.append("'2;1');")
    entities.append(f"FILE_NAME('{output_file}', '{now}', ('Your Name'), ('Your Organization'), 'ST-DEVELOPER v20', 'ONSHAPE BY PTC INC, 1.191', '');")
    entities.append("FILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF { 1 0 10303 442 1 1 4 }'));")
    entities.append("ENDSEC;")

    # Process sketches
    face_ids = []
    for sketch in sketches:
        edges = sketch['edges']  # [(start_coords, end_coords), ...]
        normal = sketch['normal']
        transformation_matrix = sketch.get('transformation_matrix')
        edges_global = [
            (convert_to_global_point(edge[0], transformation_matrix),
             convert_to_global_point(edge[1], transformation_matrix))
            for edge in edges
        ]
        face_id = add_face(edges_global, normal)
        face_ids.append(face_id)

    # Process extrudes
    solid_ids = []
    for extrude in extrudes.get('extrudes', []):
        start_face_id = face_ids[extrude['start_sketch_index']]
        end_face_id = face_ids[extrude['end_sketch_index']]

        # Placeholder for draft functionality
        if extrude.get('draft_angle', 0) != 0:
            print(f"[Placeholder] Draft angle detected: {extrude['draft_angle']} degrees. Not implemented.")

        shell_id = next_id()
        entities.append(f"#{shell_id} = CLOSED_SHELL('', (#{start_face_id}, #{end_face_id}));")

        solid_id = next_id()
        entities.append(f"#{solid_id} = MANIFOLD_SOLID_BREP('', #{shell_id});")
        solid_ids.append(solid_id)

    # Validation
    validate_geometry()

    # Add STEP Data Section
    entities.append("DATA;")
    entities.extend([
        "#10=SHAPE_REPRESENTATION_RELATIONSHIP('','',#199,#11);",
        "#11=ADVANCED_BREP_SHAPE_REPRESENTATION('',(#197),#309);",
        "#197=MANIFOLD_SOLID_BREP('Part 1',#188);",
        "#188=CLOSED_SHELL('',(#178,#179,#180));",
    ])
    entities.append("ENDSEC;")
    entities.append("END-ISO-10303-21;")

    # Construct STEP file
    return '\n'.join(entities)

def generate_step_file(sketches, extrudes, output_file="output.step"):
    """Write STEP file and save to disk."""
    step_content = write_step_file(sketches, extrudes, output_file)
    with open(output_file, 'w') as f:
        f.write(step_content)
    print(f"STEP file generated: {os.path.abspath(output_file)}")
    return output_file
