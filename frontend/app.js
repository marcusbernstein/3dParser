document.addEventListener('DOMContentLoaded', function() {
  const fileInput = document.getElementById('file-input');
  const uploadButton = document.getElementById('upload-button');
  const statusDiv = document.getElementById('status');
  const downloadArea = document.getElementById('download-area');
  const downloadLink = document.getElementById('download-link');
  
  // Replace with your HTTP API URL
  const apiEndpoint = 'https://vjfui8m7s6.execute-api.us-east-2.amazonaws.com';
  
  uploadButton.addEventListener('click', function() {
    if (!fileInput.files.length) {
      statusDiv.innerHTML = "Please select a file first";
      return;
    }
    
    const file = fileInput.files[0];
    statusDiv.innerHTML = "Uploading...";
    
    // Read file as base64
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = function() {
      // Get base64 data (remove the data:*/*;base64, prefix)
      const base64Data = reader.result.split(',')[1];
      
      // Prepare the request
      fetch(`${apiEndpoint}/convert`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          filename: file.name,
          file: base64Data
        })
      })
      .then(response => response.json())
      .then(data => {
        statusDiv.innerHTML = "File converted successfully!";
        downloadArea.style.display = "block";
        downloadLink.href = data.downloadUrl;
      })
      .catch(error => {
        statusDiv.innerHTML = "Error: " + error;
        console.error('Error:', error);
      });
    };
  });
});
