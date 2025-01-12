from datetime import datetime, timezone
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
        edge1 = edge_lookup.get((v1, v2)) or edge_lookup.get((v2, v1))
        edge2 = edge_lookup.get((v2, v3)) or edge_lookup.get((v3, v2))
        edge3 = edge_lookup.get((v3, v1)) or edge_lookup.get((v1, v3))
        
        entities.append(f"#{current_id}=EDGE_LOOP('',(#{mappings['oriented_edges'][(i,edge1)]},#{mappings['oriented_edges'][(i,edge2)]},#{mappings['oriented_edges'][(i,edge3)]})); /* triangle {i} loop */")
        mappings['edge_loops'][i] = current_id
        current_id += 1
    
    # 8. Face Bounds (EDGE_LOOPS)
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
        
        # Create axis placement (DIRECTION, DIRECTION)
        entities.append(f"#{current_id}=AXIS2_PLACEMENT_3D('',#{mappings['cartesian_points'][v1]},#{normal_id},#{perp_id}); /* triangle {i} axis */")
        mappings['axis_placements'][i] = current_id
        current_id += 1
    
    # 10. Planes (AXIS2_PLACEMENT_3D)
    for i in range(len(triangles)):
        entities.append(f"#{current_id}=PLANE('',#{mappings['axis_placements'][i]}); /* triangle {i} plane */")
        mappings['planes'][i] = current_id
        current_id += 1
    
    # 11. Advanced Faces (FACE_BOUNDS, PLANES)
    for i in range(len(triangles)):
        entities.append(f"#{current_id}=ADVANCED_FACE('',(#{mappings['face_bounds'][i]}),#{mappings['planes'][i]},.T.); /* triangle {i} face */")
        mappings['faces'][i] = current_id
        current_id += 1
    
    # 12. Closed Shell (ADVANCED_FACES)
    face_list = ",".join([f"#{mappings['faces'][i]}" for i in range(len(triangles))])
    entities.append(f"#{current_id}=CLOSED_SHELL('',({face_list})); /* complete shell */")
    closed_shell_id = current_id
    current_id += 1
    
    return "\n".join(entities), current_id, mappings, closed_shell_id

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

vertices, triangles, edges = parse('Shelf.stl')