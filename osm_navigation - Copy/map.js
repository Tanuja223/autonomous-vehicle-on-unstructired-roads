let map;
let startMarker;
let destinationMarker;
let routingControl = null;

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
        createRoute(
            latitude,
            longitude
        );

    }

    catch (error) {

        console.error(
            error
        );

        alert(
            "Error connecting to OpenStreetMap."
        );
    }
}


// ============================================================
// CREATE OSM ROUTE
// ============================================================

function createRoute(
    destinationLat,
    destinationLon
) {

    if (!startLocation) {

        alert(
            "Starting location is not available."
        );

        return;
    }


    // Remove previous route
    if (routingControl) {

        map.removeControl(
            routingControl
        );

        routingControl = null;
    }


    routingControl =
        L.Routing.control({

            waypoints: [

                L.latLng(
                    startLocation[0],
                    startLocation[1]
                ),

                L.latLng(
                    destinationLat,
                    destinationLon
                )

            ],

            routeWhileDragging: false,

            addWaypoints: false,

            showAlternatives: true,

            fitSelectedRoutes: true,

            lineOptions: {

                styles: [
                    {
                        color: "blue",
                        opacity: 0.8,
                        weight: 6
                    }
                ]

            },

            createMarker: function(
                i,
                waypoint
            ) {

                if (i === 0) {

                    return L.marker(
                        waypoint.latLng
                    )
                    .bindPopup(
                        "START"
                    );

                }

                return L.marker(
                    waypoint.latLng
                )
                .bindPopup(
                    "DESTINATION"
                );
            }

        })
        .addTo(map);


    // ========================================================
    // ROUTE FOUND
    // ========================================================

    routingControl.on(
        "routesfound",
        function(e) {

            const route =
                e.routes[0];


            const coordinates =
                route.coordinates;


            console.log(
                "ROUTE FOUND"
            );


            console.log(
                "Number of route points:",
                coordinates.length
            );


            document.getElementById(
                "result"
            ).innerHTML =

                "<b>ROUTE CREATED</b><br><br>" +

                "Start:<br>" +

                startLocation[0].toFixed(6) +

                ", " +

                startLocation[1].toFixed(6) +

                "<br><br>" +

                "Destination:<br>" +

                destinationLat.toFixed(6) +

                ", " +

                destinationLon.toFixed(6) +

                "<br><br>" +

                "Distance: " +

                (
                    route.summary.totalDistance /
                    1000
                ).toFixed(2) +

                " km" +

                "<br><br>" +

                "Sending route to CARLA...";


            // Send route to Python
            sendRouteToCarla(
                destinationLat,
                destinationLon,
                coordinates
            );

        }
    );


    routingControl.on(
        "routingerror",
        function(e) {

            console.error(
                "Routing error:",
                e
            );

            alert(
                "Could not create route."
            );

        }
    );
}


// ============================================================
// SEND ROUTE TO PYTHON
// ============================================================

async function sendRouteToCarla(
    destinationLat,
    destinationLon,
    coordinates
) {

    const routePoints =
        coordinates.map(
            point => ({

                lat: point.lat,

                lon: point.lng

            })
        );


    const data = {

        start_lat:
            startLocation[0],

        start_lon:
            startLocation[1],

        destination_lat:
            destinationLat,

        destination_lon:
            destinationLon,

        route:
            routePoints

    };


    console.log(
        "Sending data to CARLA:",
        data
    );


    try {

        const response =
            await fetch(
                "http://127.0.0.1:5001/destination",
                {

                    method: "POST",

                    headers: {

                        "Content-Type":
                            "application/json"

                    },

                    body:
                        JSON.stringify(data)

                }
            );


        if (!response.ok) {

            throw new Error(
                "Server returned " +
                response.status
            );
        }


        const result =
            await response.json();


        console.log(
            "CARLA SERVER:",
            result
        );


        document.getElementById(
            "result"
        ).innerHTML +=

            "<br><br>" +

            "<b>✓ Route sent to CARLA</b>" +

            "<br>" +

            result.message;

    }

    catch (error) {

        console.error(
            "CARLA connection error:",
            error
        );


        alert(
            "Could not connect to CARLA server.\n\n" +
            "Make sure scripts/osm/step2_osm_carla.py is running."
        );

    }
}