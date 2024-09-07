document.addEventListener('DOMContentLoaded', function() {
    const imageUpload = document.getElementById('image-upload');
    const analysisResult = document.getElementById('analysis-result');

    imageUpload.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const formData = new FormData();
            formData.append('image', file);

            fetch('/api/image_analysis', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                analysisResult.innerHTML = `<h3>Plant Analysis:</h3><pre>${data.analysis}</pre>`;
            })
            .catch(error => {
                console.error('Error:', error);
                analysisResult.textContent = 'An error occurred during image analysis.';
            });
        }
    });
});
