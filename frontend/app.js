document.addEventListener('DOMContentLoaded', function() {
  // Get UI elements for file conversion
  const fileInput = document.getElementById('file-input');
  const uploadButton = document.getElementById('upload-button');
  const statusDiv = document.getElementById('status');
  const downloadArea = document.getElementById('download-area');
  const downloadLink = document.getElementById('download-link');
  const fileInputLabel = document.querySelector('.file-input-label');
  
  // Get UI elements for flow diagram
  const diagramBoxes = document.querySelectorAll('.diagram-box');
  const panels = document.querySelectorAll('.panel');
  
  // API Gateway endpoint
  const apiEndpoint = 'https://vjfui8m7s6.execute-api.us-east-2.amazonaws.com/default';
  
  console.log("File converter initialized with API endpoint:", apiEndpoint);
  
  // Update file input label when a file is selected
  fileInput.addEventListener('change', function() {
    if (fileInput.files.length > 0) {
      const fileName = fileInput.files[0].name;
      // Truncate filename if too long
      fileInputLabel.textContent = fileName.length > 25 ? fileName.substring(0, 22) + '...' : fileName;
      fileInputLabel.title = fileName; // Show full name on hover
    } else {
      fileInputLabel.textContent = 'Select STL File';
      fileInputLabel.title = '';
    }
  });
  
  // Handle diagram box clicks
  diagramBoxes.forEach(box => {
    box.addEventListener('click', function() {
      // Remove active class from all boxes
      diagramBoxes.forEach(b => b.classList.remove('active'));
      
      // Add active class to clicked box
      this.classList.add('active');
      
      // Get the panel number
      const panelNumber = this.getAttribute('data-panel');
      
      // Hide all panels
      panels.forEach(panel => panel.classList.remove('active'));
      
      // Show the selected panel
      document.getElementById(`panel-${panelNumber}`).classList.add('active');
      
      // Smooth scroll to panel if needed
      const panelContainer = document.querySelector('.panel-container');
      const containerTop = panelContainer.getBoundingClientRect().top;
      
      if (containerTop < 0 || containerTop > window.innerHeight / 2) {
        panelContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    });
  });
  
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
    
    // Check if file is an STL file
    if (!file.name.toLowerCase().endsWith('.stl')) {
      statusDiv.innerHTML = "Please select an STL file";
      console.log("Not an STL file");
      return;
    }
    
    // Show upload status
    statusDiv.innerHTML = '<span class="status-uploading">Uploading and converting. Please wait...</span>';
    
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
        statusDiv.innerHTML = '<span class="status-success">File converted successfully!</span>';
        downloadArea.style.display = "block";
        
        // Set download link
        console.log("Setting download URL:", data.downloadUrl);
        downloadLink.href = data.downloadUrl;
        
        // Set download filename to original filename with .step extension
        const originalName = file.name.replace(/\.stl$/i, '');
        downloadLink.download = `${originalName}.step`;
        downloadLink.target = "_blank"; // Open in new tab as fallback
        
        console.log("File processing complete");
      })
      .catch(error => {
        console.error("Request failed:", error);
        statusDiv.innerHTML = '<span class="status-error">Error: ' + error.message + '</span>';
        
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
  
  // Add these CSS rules programmatically
  const style = document.createElement('style');
  style.textContent = `
    .status-uploading {
      color: #3498db;
      font-weight: 600;
    }
    .status-success {
      color: #2ecc71;
      font-weight: 600;
    }
    .status-error {
      color: #e74c3c;
      font-weight: 600;
    }
  `;
  document.head.appendChild(style);
});
