let map;
let startMarker;
let destinationMarker;
let routeLayer = null;

let startLocation = null;


// ============================================================
// GET CURRENT LOCATION
// ============================================================

navigator.geolocation.getCurrentPosition(

    function(position) {

        const latitude = position.coords.latitude;
        const longitude = position.coords.longitude;

        startLocation = [
            latitude,
            longitude
        ];

        console.log(
            "CURRENT LOCATION:",
            latitude,
            longitude
        );

        initializeMap();

    },

    function(error) {

        console.error(
            "Location error:",
            error
        );

        alert(
            "Could not get your current location."
        );

        // Fallback
        startLocation = [
            18.4500,
            83.6500
        ];

        initializeMap();
    },

    {
        enableHighAccuracy: true,
        timeout: 15000,
        maximumAge: 0
    }
);


// ============================================================
// INITIALIZE MAP
// ============================================================

function initializeMap() {

    map = L.map("map").setView(
        startLocation,
        14
    );


    // OpenStreetMap
    L.tileLayer(
        "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        {
            maxZoom: 19,
            attribution:
                "&copy; OpenStreetMap contributors"
        }
    ).addTo(map);


    // Starting point
    startMarker = L.marker(
        startLocation
    )
    .addTo(map)
    .bindPopup(
        "<b>START LOCATION</b><br>" +
        "Latitude: " +
        startLocation[0].toFixed(6) +
        "<br>Longitude: " +
        startLocation[1].toFixed(6)
    )
    .openPopup();


    document.getElementById("result").innerHTML =

        "<b>START LOCATION</b><br>" +

        "Latitude: " +
        startLocation[0].toFixed(6) +

        "<br>" +

        "Longitude: " +
        startLocation[1].toFixed(6) +

        "<br><br>" +

        "Enter a destination below.";
}


// ============================================================
// SEARCH DESTINATION
// ============================================================

async function searchDestination() {

    const destination =
        document
        .getElementById("destination")
        .value
        .trim();


    if (!destination) {

        alert(
            "Enter a destination"
        );

        return;
    }


    try {

        console.log(
            "Searching:",
            destination
        );


        const url =
            "https://nominatim.openstreetmap.org/search?" +

            "format=json" +

            "&q=" +

            encodeURIComponent(destination) +

            "&limit=1";


        const response =
            await fetch(url, {

                headers: {
                    "Accept":
                        "application/json"
                }

            });

        if (!response.ok) {
            throw new Error(
                "OpenStreetMap search returned HTTP " + response.status
            );
        }

        const data =
            await response.json();


        if (
            !data ||
            data.length === 0
        ) {

            alert(
                "Destination not found"
            );

            return;
        }


        const latitude =
            parseFloat(
                data[0].lat
            );


        const longitude =
            parseFloat(
                data[0].lon
            );


        console.log(
            "DESTINATION:",
            latitude,
            longitude
        );


        // Remove old marker
        if (destinationMarker) {

            map.removeLayer(
                destinationMarker
            );
        }


        // Add destination marker
        destinationMarker =
            L.marker([
                latitude,
                longitude
            ])
            .addTo(map)
            .bindPopup(
                "<b>DESTINATION</b><br>" +
                data[0].display_name
            )
            .openPopup();


        // Zoom to destination
        map.setView(
            [
                latitude,
                longitude
            ],
            15
        );


        // Draw route
        await createRoute(
            latitude,
            longitude,
            data[0].display_name
        );

    }

    catch (error) {

        console.error(
            error
        );

        document.getElementById("result").textContent =
            "Destination search failed: " + error.message;
    }
}


// ============================================================
// CREATE OSM ROUTE
// ============================================================

async function createRoute(
    destinationLat,
    destinationLon,
    destinationName
) {

    if (!startLocation) {

        alert(
            "Starting location is not available."
        );

        return;
    }


    if (routeLayer) {
        map.removeLayer(routeLayer);
        routeLayer = null;
    }

    const result = document.getElementById("result");
    result.textContent = "Calculating route...";

    try {
        const routeUrl =
            "https://router.project-osrm.org/route/v1/driving/" +
            startLocation[1] + "," + startLocation[0] + ";" +
            destinationLon + "," + destinationLat +
            "?overview=full&geometries=geojson&steps=false";
        const response = await fetch(routeUrl, {
            headers: { "Accept": "application/json" }
        });
        if (!response.ok) {
            throw new Error("Route service returned HTTP " + response.status);
        }

        const data = await response.json();
        if (data.code !== "Ok" || !data.routes || data.routes.length === 0) {
            throw new Error(data.message || "No drivable route was found.");
        }

        const route = data.routes[0];
        const coordinates = route.geometry.coordinates.map(
            point => L.latLng(point[1], point[0])
        );
        routeLayer = L.polyline(coordinates, {
            color: "blue",
            opacity: 0.8,
            weight: 6
        }).addTo(map);
        map.fitBounds(routeLayer.getBounds(), { padding: [20, 20] });

        result.innerHTML =
            "<b>ROUTE CREATED</b><br><br>" +
            "Start:<br>" +
            startLocation[0].toFixed(6) + ", " +
            startLocation[1].toFixed(6) +
            "<br><br>Destination:<br>" +
            destinationLat.toFixed(6) + ", " +
            destinationLon.toFixed(6) +
            "<br><br>Distance: " +
            (route.distance / 1000).toFixed(2) +
            " km<br><br>Sending route to CARLA...";

        await sendRouteToCarla(
            destinationLat,
            destinationLon,
            coordinates,
            destinationName
        );
    } catch (error) {
        console.error("Route creation failed:", error);
        result.textContent = "Route creation failed: " + error.message;
    }
}


// ============================================================
// SEND ROUTE TO PYTHON
// ============================================================


async function sendRouteToCarla(
    destinationLat,
    destinationLon,
    coordinates,
    destinationName
) {
    const routePoints = coordinates.map(point => ({
        lat: point.lat,
        lon: point.lng
    }));

    const data = {
        start_lat: startLocation[0],
        start_lon: startLocation[1],
        destination_lat: destinationLat,
        destination_lon: destinationLon,
        display_name: destinationName || "",
        route: routePoints
    };

    const resultBox = document.getElementById("result");

    try {
        console.log("Sending route to Python:", data);
        console.log("Route point count:", routePoints.length);

        const response = await fetch(
            "http://127.0.0.1:5001/destination",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data)
            }
        );

        const result = await response.json();

        console.log("Python HTTP status:", response.status);
        console.log("Python response:", result);

        if (!response.ok || result.status !== "success") {
            throw new Error(
                result.message ||
                "Python receiver did not accept the route"
            );
        }

        resultBox.innerHTML +=
            "<br><br><b>Route accepted by Python receiver.</b>" +
            "<br>Route points: " + routePoints.length;

    } catch (error) {
        console.error("Route sending failed:", error);

        resultBox.innerHTML +=
            "<br><br><b>Route sending failed:</b> " +
            error.message;
    }
}
