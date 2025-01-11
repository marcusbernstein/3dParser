from datetime import datetime
from stl_parser import *
import numpy as np


def validate_geometry(vertices, edges, triangles):
    """
    Validates geometric consistency and returns normalized indices
    """
    # First filter out degenerate triangles
    valid_triangles = []
    for i, (v1, v2, v3, normal) in enumerate(triangles):
        if v1 == v2 or v2 == v3 or v3 == v1:
            print(f"Removing degenerate triangle {i}: vertices ({v1}, {v2}, {v3})")
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
    """
    Returns two perpendicular vectors to the given normal vector
    """
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

def generate_step_entities(vertices, edges, triangles, start_id=100):
    """
    Generates complete STEP entities with proper sequencing and references
    """
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
    
    # 1. Cartesian Points
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

    # 4. Lines
    for i, (v1, v2) in enumerate(edges):  # This should process ALL edges
        entities.append(f"#{current_id}=LINE('',#{mappings['cartesian_points'][v1]},#{mappings['vectors'][i]}); /* edge {i} line */")
        mappings['lines'][i] = current_id
        current_id += 1

    # 5. Edge Curves
    for i, (v1, v2) in enumerate(edges):  # This should process ALL edges
        entities.append(f"#{current_id}=EDGE_CURVE('',#{mappings['vertex_points'][v1]},#{mappings['vertex_points'][v2]},#{mappings['lines'][i]},.T.); /* edge {i} curve */")
        mappings['edge_curves'][i] = current_id
        current_id += 1
    print(f"Created edge curves: {len(mappings['edge_curves'])}")
    print(f"Total edges: {len(edges)}")
    assert len(mappings['edge_curves']) == len(edges), "Not all edges have corresponding curves!"
    # 6. Oriented Edges
    # Modify the edge lookup creation to be more robust
    edge_lookup = {}
    for i, (v1, v2) in enumerate(edges):
        edge_lookup[(v1, v2)] = i
        edge_lookup[(v2, v1)] = i  # Add reverse direction
        
    # Verify all triangle edges exist
    missing_edges = []
    for i, (v1, v2, v3, normal) in enumerate(triangles):
        triangle_edges = [(v1, v2), (v2, v3), (v3, v1)]
        for edge in triangle_edges:
            if edge not in edge_lookup and edge[::-1] not in edge_lookup:
                missing_edges.append(edge)
                
    if missing_edges:
        # Add missing edges to both edges list and lookup
        for edge in missing_edges:
            edge_idx = len(edges)
            edges.append(edge)
            edge_lookup[edge] = edge_idx
            edge_lookup[edge[::-1]] = edge_idx
            
    print(f"Number of edges: {len(edges)}")
    print(f"Edge indices used in triangles:")
    edge_indices_used = set()
    for i, (v1, v2, v3, normal) in enumerate(triangles):
        edge1 = edge_lookup.get((v1, v2)) or edge_lookup.get((v2, v1))
        edge2 = edge_lookup.get((v2, v3)) or edge_lookup.get((v3, v2))
        edge3 = edge_lookup.get((v3, v1)) or edge_lookup.get((v1, v3))
        edge_indices_used.update([edge1, edge2, edge3])
    print(f"Min edge index: {min(edge_indices_used)}")
    print(f"Max edge index: {max(edge_indices_used)}")
    
    for i, (v1, v2, v3, normal) in enumerate(triangles):
        edge1 = edge_lookup.get((v1, v2)) or edge_lookup.get((v2, v1))
        edge2 = edge_lookup.get((v2, v3)) or edge_lookup.get((v3, v2))
        edge3 = edge_lookup.get((v3, v1)) or edge_lookup.get((v1, v3))
        
        if None in (edge1, edge2, edge3):
            raise ValueError(f"Triangle {i} ({v1}, {v2}, {v3}) contains undefined edges")
            
        # Create oriented edges
        for edge_idx in (edge1, edge2, edge3):
            key = (i, edge_idx)
            entities.append(f"#{current_id}=ORIENTED_EDGE('',*,*,#{mappings['edge_curves'][edge_idx]},.T.); /* triangle {i} oriented edge */")
            mappings['oriented_edges'][key] = current_id
            current_id += 1
    # 7. Edge Loops
    for i, (v1, v2, v3, normal) in enumerate(triangles):
        edge1 = edge_lookup.get((v1, v2)) or edge_lookup.get((v2, v1))
        edge2 = edge_lookup.get((v2, v3)) or edge_lookup.get((v3, v2))
        edge3 = edge_lookup.get((v3, v1)) or edge_lookup.get((v1, v3))
        
        entities.append(f"#{current_id}=EDGE_LOOP('',(#{mappings['oriented_edges'][(i,edge1)]},#{mappings['oriented_edges'][(i,edge2)]},#{mappings['oriented_edges'][(i,edge3)]})); /* triangle {i} loop */")
        mappings['edge_loops'][i] = current_id
        current_id += 1
    
    # 8. Face Bounds
    for i in range(len(triangles)):
        entities.append(f"#{current_id}=FACE_BOUND('',#{mappings['edge_loops'][i]},.T.); /* triangle {i} bound */")
        mappings['face_bounds'][i] = current_id
        current_id += 1
    
    # 9. Triangle Normal Directions and Axis Placements
    for i, (v1, v2, v3, normal) in enumerate(triangles):
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
        
        # Create axis placement
        entities.append(f"#{current_id}=AXIS2_PLACEMENT_3D('',#{mappings['cartesian_points'][v1]},#{normal_id},#{perp_id}); /* triangle {i} axis */")
        mappings['axis_placements'][i] = current_id
        current_id += 1
    
    # 10. Planes
    for i in range(len(triangles)):
        entities.append(f"#{current_id}=PLANE('',#{mappings['axis_placements'][i]}); /* triangle {i} plane */")
        mappings['planes'][i] = current_id
        current_id += 1
    
    # 11. Advanced Faces
    for i in range(len(triangles)):
        entities.append(f"#{current_id}=ADVANCED_FACE('',(#{mappings['face_bounds'][i]}),#{mappings['planes'][i]},.T.); /* triangle {i} face */")
        mappings['faces'][i] = current_id
        current_id += 1
    
    # 12. Closed Shell
    face_list = ",".join([f"#{mappings['faces'][i]}" for i in range(len(triangles))])
    entities.append(f"#100000=CLOSED_SHELL('',({face_list})); /* complete shell */")
    return "\n".join(entities), current_id, mappings




def write_step_file(vertices, edges, triangles, filename="output.step"):
    """
    Main function to write STEP file
    """
    # Validate geometry
    edges, triangles = validate_geometry(vertices, edges, triangles)
    print(f"After validation:")
    print(f"Total edges: {len(edges)}")
    print(f"Edge sample: {edges[:3]}")
    # Generate standard intro content
    intro = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(
/* description */ ('STEP AP203'),
/* implementation_level */ '2;1');
FILE_NAME(
/* name */ 'tester file',
/* time_stamp */ '2024-12-31T10:23:12Z',
/* author */ ('Marcus'),
/* organization */ ('Marcus'),
/* preprocessor_version */ 'manual mode',
/* originating_system */ 'nah',
/* authorisation */ '  ');
FILE_SCHEMA (('CONFIG_CONTROL_DESIGN'));
ENDSEC;
DATA;
#10=(
GEOMETRIC_REPRESENTATION_CONTEXT(3)
GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#11))
GLOBAL_UNIT_ASSIGNED_CONTEXT((#12,#16,#17))
REPRESENTATION_CONTEXT('Part 1','TOP_LEVEL_ASSEMBLY_PART')); /* 10 */
 /* Units Setup */
 #11=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(3.93700787401575E-7),#12,
 'DISTANCE_ACCURACY_VALUE','Maximum Tolerance applied to model'); /* 11 */
 #12=(CONVERSION_BASED_UNIT('INCH',#14)
 LENGTH_UNIT()
 NAMED_UNIT(#13)); /* 12 */
  #13=DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.); /* 13 */
  #14=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#15); /* 14 */
  #15=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.)); /* 15 */
  #16=(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.)); /* 16 */
  #17=(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT()); /* 17 */
 /* Product and Context */ 
 #18=PRODUCT_DEFINITION_SHAPE('','',#19); /* 24 */
  #19=PRODUCT_DEFINITION('','',#21,#20); /* 25 */
   #20=DESIGN_CONTEXT('',#24,'design'); /* 29 */
   #21=PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE('','',#22, .NOT_KNOWN.); /* 26 */
    #22=PRODUCT('Part 1','Part 1','Part 1',(#23)); /* 27 */
     #23=MECHANICAL_CONTEXT('',#24,'mechanical'); /* 200 */
      #24=APPLICATION_CONTEXT('configuration controlled 3D designs of mechanical parts and assemblies');
     #25=PRODUCT_RELATED_PRODUCT_CATEGORY('','',(#22)); /* 31 */
     #26=PRODUCT_CATEGORY('',''); /* 32 */
  /* Representation */
#27=SHAPE_DEFINITION_REPRESENTATION(#18,#39); /* 23 */
#28=REPRESENTATION('',(#16),#10); /* 12 */
#29=REPRESENTATION('',(#17),#10); /* 13 */
#30=PROPERTY_DEFINITION_REPRESENTATION(#15,#13); /* 11 */
#31=VALUE_REPRESENTATION_ITEM('number of annotations',COUNT_MEASURE(0.)); /* 16 */
#32=VALUE_REPRESENTATION_ITEM('number of views',COUNT_MEASURE(0.)); /* 17 */
#33=APPLICATION_PROTOCOL_DEFINITION('international standard','config_control_design',2010,#24);/* 201*/
#34=PROPERTY_DEFINITION_REPRESENTATION(#35,#39); /* 34 */
 #35=PROPERTY_DEFINITION('pmi validation property','',#18); /* 35 */
 #36=PROPERTY_DEFINITION('pmi validation property','',#18); /* 39 */
#37=ADVANCED_BREP_SHAPE_REPRESENTATION('',(#44),#10); /* 42 */
#38=SHAPE_REPRESENTATION_RELATIONSHIP('','',#39,#37); /* 43 */    
 /* Origin */
 #39=SHAPE_REPRESENTATION('Part 1',(#40),#10); /* 18 */
  #40=AXIS2_PLACEMENT_3D('',#41,#42,#43); /* 19 */
   #41=CARTESIAN_POINT('',(0.,0.,0.)); /* 20 */
   #42=DIRECTION('',(0.,0.,1.)); /* 21 */
   #43=DIRECTION('',(1.,0.,0.)); /* 22 */
#44=MANIFOLD_SOLID_BREP('Part 1',#100000); /* 121 */
"""
    
    # Generate main entity content
    entity_text, final_id, mappings = generate_step_entities(vertices, edges, triangles)
    
    outro = """ENDSEC;
END-ISO-10303-21;"""

    # Combine all content
    step_content = intro + "\n" + entity_text + "\n" + outro
    
    # Write file
    with open(filename, 'w') as f:
        f.write(step_content)
    
    # Print debug info
    print(f"Successfully wrote STEP file:")
    print(f"- {len(vertices)} vertices")
    print(f"- {len(edges)} edges")
    print(f"- {len(triangles)} triangles")
    print(f"- {final_id - 100} total entities")
    
    return True

vertices, triangles, edges = parse('Shelf.stl')
print("Sample of first few elements:")
print(f"First 3 vertices: {vertices[:3]}")
print(f"First 3 edges: {edges[:3]}")
print(f"First 3 triangles: {triangles[:3]}")
write_step_file(vertices, edges, triangles, "output.step")

