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
            .catch(error => {
                console.error('Error fetching environmental data:', error);
                document.getElementById('chart-container').innerHTML = '<p>Error loading environmental data. Please try again later.</p>';
            });
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
                            unit: 'day',
                            displayFormats: {
                                day: 'MMM D'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    },
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: 'Value'
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                    },
                    legend: {
                        position: 'top',
                    },
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

    // Add plant health trends analysis
    const plantHealthTrendsButton = document.getElementById('plant-health-trends');
    const trendsResult = document.getElementById('trends-result');

    plantHealthTrendsButton.addEventListener('click', function() {
        fetch('/api/plant_health_trends')
            .then(response => response.json())
            .then(data => {
                trendsResult.innerHTML = `<pre>${data.trends}</pre>`;
            })
            .catch(error => {
                console.error('Error fetching plant health trends:', error);
                trendsResult.textContent = 'Error fetching plant health trends. Please try again.';
            });
    });
});
