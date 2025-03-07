import json
import boto3
import base64
import os
import traceback
import uuid

# Initialize S3 client
s3 = boto3.client('s3')
upload_bucket = 'converter-bucket-uploadds'    # Replace with your upload bucket name
download_bucket = 'converter-bucket-downloads' # Replace with your download bucket name

def lambda_handler(event, context):
    print(f"### LAMBDA INVOKED - Two Bucket Version 1.0 ###")
    print(f"Event: {json.dumps(event)}")
    print(f"Using upload bucket: {upload_bucket}")
    print(f"Using download bucket: {download_bucket}")
    
    # Handle preflight OPTIONS request
    if event.get('requestContext', {}).get('http', {}).get('method') == 'OPTIONS' or event.get('httpMethod') == 'OPTIONS':
        print("OPTIONS request detected, returning CORS headers")
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': 'https://main.dqb4k6muwo5tx.amplifyapp.com',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Methods': 'POST,OPTIONS',
                'Access-Control-Allow-Credentials': 'true'
            },
            'body': ''
        }
    
    # Main processing for POST requests
    try:
        print("Starting main request processing...")
        
        # Check if we have a body
        if 'body' not in event:
            print("ERROR: No body in event")
            return error_response("No request body found")
        
        # Parse request body
        try:
            request_body = json.loads(event['body'])
            print(f"Request body keys: {list(request_body.keys())}")
        except Exception as e:
            print(f"ERROR parsing request body: {str(e)}")
            return error_response(f"Invalid JSON body: {str(e)}")
        
        # Validate required fields
        if 'file' not in request_body:
            print("ERROR: 'file' field missing in request")
            return error_response("Missing 'file' field in request")
        
        if 'filename' not in request_body:
            print("ERROR: 'filename' field missing in request")
            return error_response("Missing 'filename' field in request")
        
        # Get filename
        filename = request_body['filename']
        print(f"Filename: {filename}")
        
        # Generate unique file identifiers
        file_id = str(uuid.uuid4())
        upload_key = f"{file_id}-{filename}"
        download_key = f"{file_id}-{filename}"  # This might be updated later for STL files
        
        # Decode base64 file content
        try:
            print("Decoding file content...")
            file_content = base64.b64decode(request_body['file'])
            print(f"File decoded successfully, size: {len(file_content)} bytes")
        except Exception as e:
            print(f"ERROR decoding file content: {str(e)}")
            return error_response(f"Failed to decode file: {str(e)}")
        
        # Verify buckets exist
        try:
            print(f"Verifying S3 buckets existence...")
            s3.head_bucket(Bucket=upload_bucket)
            s3.head_bucket(Bucket=download_bucket)
            print("S3 buckets verified")
        except Exception as e:
            print(f"ERROR accessing S3 buckets: {str(e)}")
            return error_response(f"S3 bucket error: {str(e)}")
        
        # Upload original file to upload bucket
        try:
            print(f"Uploading original file to upload bucket: {upload_key}")
            upload_response = s3.put_object(
                Bucket=upload_bucket, 
                Key=upload_key, 
                Body=file_content
            )
            print(f"Original file upload response: {upload_response}")
        except Exception as e:
            print(f"ERROR uploading original file: {str(e)}")
            return error_response(f"Failed to upload original file: {str(e)}")
        
        # Process the file based on type
        print("Processing file...")
        processed_content = file_content
        output_filename = filename
        
        # Check if the file is a text file
        if filename.lower().endswith('.txt'):
            try:
                print("Text file detected, adding greeting...")
                # Decode bytes to string
                text_content = file_content.decode('utf-8')
                # Add greeting
                modified_text = "hi marcus\n" + text_content
                # Convert back to bytes
                processed_content = modified_text.encode('utf-8')
                print(f"Text file processed, new size: {len(processed_content)} bytes")
            except Exception as e:
                print(f"ERROR processing text file: {str(e)}")
                return error_response(f"Failed to process text file: {str(e)}")
        # Check if the file is an STL file
        elif filename.lower().endswith('.stl'):
            try:
                print("STL file detected, processing...")
                # Add "hey dude!" at the beginning of the file
                processed_content = b"hey dude!" + file_content
                
                # Change the output filename to .STEP
                base_name = os.path.splitext(filename)[0]
                output_filename = f"{base_name}.STEP"
                print(f"Converted filename from {filename} to {output_filename}")
                download_key = f"{file_id}-{output_filename}"
                
                print(f"STL file processed, new size: {len(processed_content)} bytes")
            except Exception as e:
                print(f"ERROR processing STL file: {str(e)}")
                return error_response(f"Failed to process STL file: {str(e)}")
        else:
            print(f"Not a .txt or .stl file, skipping processing")
        
        # Upload processed file to download bucket
        try:
            print(f"Uploading processed file to download bucket: {download_key}")
            processed_response = s3.put_object(
                Bucket=download_bucket, 
                Key=download_key, 
                Body=processed_content
            )
            print(f"Processed file upload response: {processed_response}")
        except Exception as e:
            print(f"ERROR uploading processed file: {str(e)}")
            return error_response(f"Failed to upload processed file: {str(e)}")
        
        # Generate a pre-signed URL for the processed file
        try:
            print(f"Generating pre-signed URL for: {download_key}")
            url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': download_bucket, 'Key': download_key},
                ExpiresIn=3600  # URL expires in 1 hour
            )
            print(f"Generated URL length: {len(url)} characters")
            print(f"URL starts with: {url[:50]}...")
        except Exception as e:
            print(f"ERROR generating pre-signed URL: {str(e)}")
            return error_response(f"Failed to generate download URL: {str(e)}")
        
        # Create success response
        response = {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': 'https://main.dqb4k6muwo5tx.amplifyapp.com',
                'Access-Control-Allow-Credentials': 'true'
            },
            'body': json.dumps({
                'message': 'File processed successfully',
                'downloadUrl': url,
                'filename': output_filename,
                'originalFilename': filename,
                'fileSize': len(processed_content),
                'uploadBucket': upload_bucket,
                'downloadBucket': download_bucket,
                'processed': filename.lower().endswith('.txt') or filename.lower().endswith('.stl')
            })
        }
        
        print(f"Success response statusCode: {response['statusCode']}")
        print(f"Success response body length: {len(response['body'])}")
        return response
        
    except Exception as e:
        print(f"UNEXPECTED ERROR: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return error_response(f"Unexpected error: {str(e)}")

def error_response(message):
    """Helper function to create error responses with consistent format"""
    response = {
        'statusCode': 500,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': 'https://main.dqb4k6muwo5tx.amplifyapp.com',
            'Access-Control-Allow-Credentials': 'true'
        },
        'body': json.dumps({
            'error': message
        })
    }
    print(f"Returning error response: {message}")
    return response
