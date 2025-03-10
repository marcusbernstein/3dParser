import io
import tempfile
import os
from stl_parser import parse
from step_generator import write_step_file

def get_version():
    """Simple function to verify module is loaded correctly"""
    return "1.0.0"

def convert_file(file_content, file_type):
    """
    Convert STL file to STEP format using the existing parse() and write_step_file() functions.
    
    Args:
        file_content (bytes): The file content
        file_type (str): The file type (e.g., 'stl', 'txt')
        
    Returns:
        bytes: Modified file content
    """
    print(f"Converter module: Processing {file_type} file of size {len(file_content)} bytes")
    
    if file_type.lower() != 'stl':
        print(f"Unsupported file type: {file_type}")
        return file_content
    
    try:
        # Write content to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as temp:
            temp_path = temp.name
            temp.write(file_content)
        
        # Parse the STL file
        print(f"Parsing STL file: {temp_path}")
        vertices, triangles, edges = parse(temp_path)
        print(f"Parsed STL: {len(vertices)} vertices, {len(triangles)} triangles, {len(edges)} edges")
        
        # Create a temporary output file
        output_path = temp_path.replace('.stl', '.step')
        
        # Generate the STEP file
        print(f"Generating STEP file: {output_path}")
        write_step_file(vertices, edges, triangles, output_path)
        
        # Read the generated STEP file
        with open(output_path, 'rb') as f:
            step_content = f.read()
        
        # Clean up temporary files
        try:
            os.remove(temp_path)
            os.remove(output_path)
        except:
            pass
            
        print(f"Generated STEP file of size {len(step_content)} bytes")
        return step_content
        
    except Exception as e:
        print(f"Error converting file: {str(e)}")
        import traceback
        traceback.print_exc()
        return b"CONVERSION ERROR: " + str(e).encode('utf-8')
