import math
import numpy as np
from stl_parser import *

from datetime import datetime

vertices, triangles, edges = parse('Unit Test.stl')

def create_file(step_content, filename="Tester File.step"):
    with open(filename, 'w') as f:
        f.write(step_content)

intro = f'''ISO-10303-21;
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
#44=MANIFOLD_SOLID_BREP('Part 1',#200); /* 121 */'''


outro = f'''ENDSEC;
END-ISO-10303-21;
'''
def cartesian_points(vertices, counter=100):
    points_text = []
    for x, y, z in vertices:
        points_text.append(f"#{counter}=CARTESIAN_POINT('',({x:.1f},{y:.1f},{z:.1f}));\n")
        counter += 1
    return points_text, counter

def vertex_points(vertices, counter):
       vertex_text = []
       cartesian_ref = 100
       for x in vertices:
           vertex_text.append(f" #{counter}=VERTEX_POINT('',#{cartesian_ref});\n")
           cartesian_ref += 1
           counter += 1
       return vertex_text, counter
def calculate_line_direction(p1, p2):
    # Convert points to numpy arrays
    p1_arr = np.array(p1)
    p2_arr = np.array(p2)
    
    # Calculate direction vector
    direction = p2_arr - p1_arr
    
    # Normalize the vector
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    
    return tuple(direction)

def directions(edges, vertices, counter):
    direction_text = []
    direction_map = {}
    
    for p1_idx, p2_idx in edges:
        direction = calculate_line_direction(vertices[p1_idx], vertices[p2_idx])
        if direction not in direction_map:
            direction_map[direction] = counter
            x, y, z = direction
            direction_text.append(f"#{counter}=DIRECTION('',({x:.1f},{y:.1f},{z:.1f}));\n")
            counter += 1
            
    return direction_text, counter, direction_map

def vectors(edges, vertices, direction_map, counter):
    vector_text = []
    vector_map = {}
    
    for p1_idx, p2_idx in edges:
        direction = calculate_line_direction(vertices[p1_idx], vertices[p2_idx])
        direction_id = direction_map[direction]
        vector_id = counter
        vector_text.append(f" #{counter}=VECTOR('',#{direction_id},39.3700787401575);\n")
        vector_map[(p1_idx, p2_idx)] = vector_id
        counter += 1
            
    return vector_text, counter, vector_map

def lines(edges, vector_map, counter):
    line_text = []
    
    for i, (p1_idx, p2_idx) in enumerate(edges):
        cartesian_id = 100 + p1_idx  # Starting point reference
        vector_id = vector_map[(p1_idx, p2_idx)]
        line_text.append(f" #{counter}=LINE('',#{cartesian_id},#{vector_id}); /* ({p1_idx}, {p2_idx}) */\n")
        counter += 1
        
    return line_text, counter

def edge_curves(edges, counter):
   edge_curve_text = []
   vertex_start = 100 + len(vertices)  # First vertex_point ID
   line_start = counter - len(edges)   # First line ID
   
   for i, (p1_idx, p2_idx) in enumerate(edges):
       v1_id = vertex_start + p1_idx
       v2_id = vertex_start + p2_idx
       line_id = line_start + i
       edge_curve_text.append(f"#{counter}=EDGE_CURVE('',#{v1_id},#{v2_id},#{line_id},.T.); /* ({p1_idx}, {p2_idx}) */\n")
       counter += 1
       
   return edge_curve_text, counter

def oriented_edges(triangles, edges, counter_start):
    # Map edge curve IDs (starting from edge_curve_start)
    edge_curve_start = counter_start - len(edges)
    edge_to_curve = {tuple(sorted(edge)): edge_curve_start + i for i, edge in enumerate(edges)}
    
    print("Edge to curve mapping:", edge_to_curve)  # Debug
    
    oriented_text = []
    oriented_map = {}
    counter = counter_start
    
    # First create all oriented edges (one per edge curve)
    oriented_edge_lookup = {}
    for curve_id in range(edge_curve_start, edge_curve_start + len(edges)):
        oriented_text.append(f"#{counter}=ORIENTED_EDGE('',*,*,#{curve_id},.T.);\n")
        oriented_edge_lookup[curve_id] = counter
        counter += 1
    
    # Then map triangles to their oriented edges
    for tri_idx, (p1, p2, p3, _) in enumerate(triangles):
        tri_edges = [(p1, p2), (p2, p3), (p3, p1)]
        tri_oriented_edges = []
        
        print(f"Triangle {tri_idx}: {tri_edges}")  # Debug
        
        for edge in tri_edges:
            sorted_edge = tuple(sorted(edge))
            print(f"Looking for edge: {sorted_edge}")  # Debug
            if sorted_edge in edge_to_curve:
                curve_id = edge_to_curve[sorted_edge]
                oriented_edge_id = oriented_edge_lookup[curve_id]
                tri_oriented_edges.append(oriented_edge_id)
            else:
                print(f"Warning: Edge {sorted_edge} not found in edge_to_curve")  # Debug
        
        if len(tri_oriented_edges) != 3:
            print(f"Warning: Triangle {tri_idx} only has {len(tri_oriented_edges)} oriented edges")  # Debug
        
        oriented_map[tri_idx] = tri_oriented_edges
    
    return oriented_text, oriented_map, counter

def edge_loops(triangles, oriented_map, counter):
    loop_text = []
    for tri_idx in range(len(triangles)):
        oriented_edges = oriented_map[tri_idx]
        if len(oriented_edges) == 3:
            e1, e2, e3 = oriented_edges
            loop_text.append(f"#{counter}=EDGE_LOOP('',(#{e1},#{e2},#{e3}));\n")
            counter += 1
        else:
            print(f"Skipping triangle {tri_idx} due to insufficient edges: {oriented_edges}")
    return loop_text, counter

def face_bounds(triangles, counter):
    face_bounds_text = []
    face_ref = counter - len(triangles)
    for x in triangles:
        face_bounds_text.append(f" #{counter}=FACE_BOUND('',#{face_ref});\n")
        face_ref += 1
        counter += 1
    return face_bounds_text, counter

def get_perpendicular_direction(normal):
   # Get perpendicular vector using cross product with (0,0,1) or (1,0,0)
   if abs(normal[2]) < 0.9:
       perp = np.cross(normal, [0,0,1])
   else:
       perp = np.cross(normal, [1,0,0])
   return perp /np.linalg.norm(perp)

def axes(triangles, counter):
   axis_text = []
   for idx, (p1, _, _, normal) in enumerate(triangles):
       cart_point = 100 + p1  # Reference first vertex as placement point
       perp = get_perpendicular_direction(normal)
       
       # Add normal and perpendicular directions
       axis_text.append(f"#{counter}=DIRECTION('',({normal[0]:.1f},{normal[1]:.1f},{normal[2]:.1f}));\n")
       normal_id = counter
       counter += 1
       
       axis_text.append(f"#{counter}=DIRECTION('',({perp[0]:.1f},{perp[1]:.1f},{perp[2]:.1f}));\n")
       perp_id = counter
       counter += 1
       
       # Create placement
       axis_text.append(f"#{counter}=AXIS2_PLACEMENT_3D('',#{cart_point},#{normal_id},#{perp_id});\n")
       counter += 1
       
   return axis_text, counter

def planes(triangles, counter):
    planes_text = []
    planes_ref = counter - (len(triangles)*2)
    for x in triangles:
        planes_text.append(f" #{counter}=PLANE('',#{planes_ref});\n")
        planes_ref += 1
        counter += 1
    return planes_text, counter

def faces(triangles, counter):
    faces_text = []
    faces_ref = counter - (len(triangles)*3)
    for x in triangles:
        faces_text.append(f" #{counter}=ADVANCED_FACE('',#{faces_ref});\n")
        faces_ref += 1
        counter += 1
    return faces_text, counter

def shell(triangles, counter):
    shell_text = []
    #shell_text.append()
    faces_ref = counter - len(triangles)
    for x in triangles:
        shell_text.append(f" #{faces_ref},")
        faces_ref += 1
    return shell_text



points_text, counter = cartesian_points(vertices)
vertex_text, counter = vertex_points(vertices, counter)
direction_text, counter, direction_map = directions(edges, vertices, counter)
vector_text, counter, vector_map = vectors(edges, vertices, direction_map, counter)
line_text, counter = lines(edges, vector_map, counter)
edge_curve_text, counter = edge_curves(edges, counter)
oriented_edges_text, oriented_map, counter = oriented_edges(triangles, edges, counter)
edge_loops_text, counter = edge_loops(triangles, oriented_map, counter)
face_bounds_text, counter = face_bounds(triangles, counter)
axis_text, counter = axes(triangles, counter)
planes_text, counter = planes(triangles, counter)
faces_text, counter = faces(triangles, counter)
shell_text = shell(triangles, counter)

step_content = intro
step_content += (''.join(points_text))
step_content += (''.join(vertex_text))
step_content += (''.join(direction_text))
step_content += (''.join(vector_text))
step_content += (''.join(line_text))
step_content += (''.join(edge_curve_text))
step_content += (''.join(oriented_edges_text))
step_content += (''.join(edge_loops_text))
step_content += (''.join(face_bounds_text))
step_content += (''.join(axis_text))
step_content += (''.join(planes_text))
step_content += (''.join(faces_text))
step_content += (''.join(shell_text))
step_content += (''.join(outro))

create_file(step_content)