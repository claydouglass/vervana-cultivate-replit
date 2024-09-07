document.addEventListener('DOMContentLoaded', function() {
    let environmentalChart;

    function updateEnvironmentalData() {
        fetch('/api/environmental_data')
            .then(response => response.json())
            .then(data => {
                if (data.length > 0) {
                    const latestData = data[data.length - 1];
                    document.getElementById('temperature').textContent = latestData.temperature.toFixed(2);
                    document.getElementById('humidity').textContent = latestData.humidity.toFixed(2);
                    document.getElementById('co2-level').textContent = latestData.co2_level.toFixed(2);
                    document.getElementById('last-updated').textContent = new Date(latestData.timestamp).toLocaleString();
                }
                updateChart(data);
            })
            .catch(error => console.error('Error fetching environmental data:', error));
    }

    function updateChart(data) {
        const ctx = document.getElementById('environmental-chart').getContext('2d');
        if (environmentalChart) {
            environmentalChart.destroy();
        }
        environmentalChart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Temperature (°C)',
                    data: data.map(d => ({ x: new Date(d.timestamp), y: d.temperature })),
                    borderColor: 'red',
                    fill: false
                }, {
                    label: 'Humidity (%)',
                    data: data.map(d => ({ x: new Date(d.timestamp), y: d.humidity })),
                    borderColor: 'blue',
                    fill: false
                }, {
                    label: 'CO2 Level (ppm)',
                    data: data.map(d => ({ x: new Date(d.timestamp), y: d.co2_level })),
                    borderColor: 'green',
                    fill: false
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day'
                        }
                    },
                    y: {
                        beginAtZero: false
                    }
                }
            }
        });
    }

    // Update data every 5 minutes
    updateEnvironmentalData();
    setInterval(updateEnvironmentalData, 300000);

    // Add event listener for the Adjust Environment button
    const adjustEnvironmentButton = document.getElementById('adjust-environment');
    const adjustmentRecommendations = document.getElementById('adjustment-recommendations');

    adjustEnvironmentButton.addEventListener('click', function() {
        fetch('/api/adjust_environment')
            .then(response => response.json())
            .then(data => {
                adjustmentRecommendations.innerHTML = `<pre>${data.recommendations}</pre>`;
            })
            .catch(error => {
                console.error('Error fetching adjustment recommendations:', error);
                adjustmentRecommendations.textContent = 'Error fetching recommendations. Please try again.';
            });
    });
});
