document.addEventListener('DOMContentLoaded', function() {
    const batchIdInput = document.getElementById('batch-id');
    const predictYieldButton = document.getElementById('predict-yield');
    const yieldResult = document.getElementById('yield-result');

    predictYieldButton.addEventListener('click', function() {
        const batchId = batchIdInput.value.trim();
        if (batchId) {
            fetch(`/api/predict_yield/${batchId}`)
                .then(response => response.json())
                .then(data => {
                    yieldResult.textContent = data.prediction;
                })
                .catch(error => {
                    console.error('Error:', error);
                    yieldResult.textContent = 'An error occurred during yield prediction.';
                });
        } else {
            yieldResult.textContent = 'Please enter a valid Batch ID.';
        }
    });
});
