document.addEventListener('DOMContentLoaded', function() {
  // Get UI elements
  const fileInput = document.getElementById('file-input');
  const uploadButton = document.getElementById('upload-button');
  const statusDiv = document.getElementById('status');
  const downloadArea = document.getElementById('download-area');
  const downloadLink = document.getElementById('download-link');
  
  // API Gateway endpoint
  const apiEndpoint = 'https://vjfui8m7s6.execute-api.us-east-2.amazonaws.com/default';
  
  console.log("File converter initialized with API endpoint:", apiEndpoint);
  
  // Handle upload button click
  uploadButton.addEventListener('click', function() {
    // Clear previous console logs
    console.clear();
    console.log("Upload button clicked");
    
    // Validate file selection
    if (!fileInput.files.length) {
      statusDiv.innerHTML = "Please select a file first";
      console.log("No file selected");
      return;
    }
    
    const file = fileInput.files[0];
    console.log("File selected:", file.name, "Type:", file.type, "Size:", file.size, "bytes");
    
    // Show upload status
    statusDiv.innerHTML = "Uploading...";
    
    // Read file as base64
    const reader = new FileReader();
    
    // Handle file read errors
    reader.onerror = function() {
      console.error("Error reading file:", reader.error);
      statusDiv.innerHTML = "Error reading file: " + reader.error;
    };
    
    // Process file after it's read
    reader.onload = function() {
      console.log("File read successfully");
      
      // Extract base64 data (remove data URI prefix)
      const base64Data = reader.result.split(',')[1];
      console.log("Base64 data length:", base64Data.length);
      
      // Prepare request
      const requestBody = {
        filename: file.name,
        file: base64Data
      };
      
      console.log("About to send POST request to:", `${apiEndpoint}/convert`);
      console.log("Request includes filename:", file.name);
      
      // Send to API Gateway
      fetch(`${apiEndpoint}/convert`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      })
      .then(response => {
        console.log("Received response:", response.status, response.statusText);
        
        // Check response status
        if (!response.ok) {
          throw new Error(`Server returned error: ${response.status} ${response.statusText}`);
        }
        
        // Try to parse JSON
        return response.json().catch(error => {
          console.error("Error parsing JSON response:", error);
          throw new Error("Invalid response format: " + error.message);
        });
      })
      .then(data => {
        console.log("Parsed response data:", data);
        
        // Validate response data
        if (!data) {
          throw new Error("Empty response received");
        }
        
        if (!data.downloadUrl) {
          console.error("Response missing downloadUrl:", data);
          throw new Error("Missing download URL in server response");
        }
        
        // Update UI with success
        statusDiv.innerHTML = "File converted successfully!";
        downloadArea.style.display = "block";
        
        // Set download link
        console.log("Setting download URL:", data.downloadUrl);
        downloadLink.href = data.downloadUrl;
        downloadLink.download = file.name || "converted-file";
        downloadLink.target = "_blank"; // Open in new tab as fallback
        
        console.log("File processing complete");
      })
      .catch(error => {
        console.error("Request failed:", error);
        statusDiv.innerHTML = "Error: " + error.message;
        
        // Additional error details
        if (error.name === 'SyntaxError') {
          console.error("Response was not valid JSON - likely a server error");
        } else if (error.name === 'TypeError') {
          console.error("Network error - check your connection or CORS settings");
        }
      });
    };
    
    // Start reading the file
    console.log("Starting to read file as Data URL");
    reader.readAsDataURL(file);
  });
  
