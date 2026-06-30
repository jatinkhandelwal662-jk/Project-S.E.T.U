console.log("TARS Live Pipeline Engine Initialized!");

// 1. Center map around Delhi baseline boundary coordinates
const map = L.map('map').setView([28.6200, 77.2200], 13);

// 2. Load Deep Dark Theme Tiles
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap &copy; CARTO',
    subdomains: 'abcd',
    maxZoom: 20
}).addTo(map);

let geoJsonLayer = null;

// 3. Mathematical color scales for Centrality Score (Vulnerability map)
function getRouteColor(score) {
    return score > 0.8 ? '#ff1744' : // Red: Critical Gatekeeper Intersections
           score > 0.5 ? '#ff9100' : // Orange: Medium Criticality
           score > 0.2 ? '#ffea00' : // Yellow: Secondary Routes
                         #00e5ff';  // Cyan: Low Risk / Peripheral Connected Links
}

function getRouteWeight(score) {
    return 3 + (score * 5); // Thicker line profiles for highly critical arteries
}

// 4. Send Image to Python Server pipeline
async function uploadAndProcessSatelliteImage() {
    const fileInput = document.getElementById('satellite-upload');
    const processBtn = document.getElementById('process-btn');
    const spinner = document.getElementById('loading-spinner');
    const statsContainer = document.getElementById('stats');
    const edgeCountVal = document.getElementById('stat-edges');

    if (!fileInput.files || fileInput.files.length === 0) {
        alert("Please pick an extraction patch file before running processing pipeline.");
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    // Toggle interactive UI waiting indicators
    processBtn.disabled = true;
    spinner.classList.remove('hidden');
    statsContainer.classList.add('hidden');

    try {
        const response = await fetch('http://127.0.0.1:8080/api/process-satellite-mask', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`API returned execution error code: ${response.status}`);
        }

        const geojsonData = await response.json();

        // Remove old topological vectors if reloading a second run
        if (geoJsonLayer) {
            map.removeLayer(geoJsonLayer);
        }

        // Render live extracted nodes & segments
        geoJsonLayer = L.geoJSON(geojsonData, {
            style: function (feature) {
                const score = feature.properties.criticality_score;
                return {
                    color: getRouteColor(score),
                    weight: getRouteWeight(score),
                    opacity: 0.85,
                    lineCap: 'round'
                };
            },
            onEachFeature: function (feature, layer) {
                const score = feature.properties.criticality_score;
                layer.bindPopup(`
                    <strong>Route Segment</strong><br/>
                    Criticality: ${(score * 100).toFixed(1)}%<br/>
                    Status: ${score > 0.6 ? '⚠️ High Vulnerability' : '✅ Stable Network'}
                `);
            }
        }).addTo(map);

        // Pan map automatically to center on the coordinates
        if (geojsonData.features.length > 0) {
            const bounds = geoJsonLayer.getBounds();
            map.fitBounds(bounds);
            
            // Render basic analytics data cards
            edgeCountVal.innerText = geojsonData.features.length;
            statsContainer.classList.remove('hidden');
        } else {
            alert("Model executed successfully but found no continuous road targets in this mask framework.");
        }

    } catch (error) {
        console.error("Pipeline failure:", error);
        alert(`Failed to build network: ${error.message}`);
    } finally {
        processBtn.disabled = false;
        spinner.classList.add('hidden');
    }
}

// Attach event hooks to processing click elements
document.getElementById('process-btn').addEventListener('click', uploadAndProcessSatelliteImage);