async function predictTime() {

    let distance = document.getElementById("distance").value;
    let prepTime = document.getElementById("prep_time").value;
    let weather = document.getElementById("weather").value;
    let traffic = document.getElementById("traffic").value;
    let vehicle = document.getElementById("vehicle").value;

    if (distance === "" || prepTime === "") {
        alert("Please fill all required fields.");
        return;
    }

    try {

        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                distance: parseFloat(distance),
                prep_time: parseFloat(prepTime),
                weather: weather,
                traffic: traffic,
                vehicle: vehicle
            })
        });

        const data = await response.json();

        document.getElementById("prediction").innerHTML =
            "Predicted Delivery Time: " + data.prediction.toFixed(2) + " minutes";

    } catch (error) {
        alert("Error connecting to backend");
    }
}


async function loadMetrics() {

    const response = await fetch("/metrics");
    const data = await response.json();

    const metricValues = document.querySelectorAll(".metric-value");

    metricValues[0].innerText = data.mae;
    metricValues[1].innerText = data.mse;
    metricValues[2].innerText = data.r2;
}

window.onload = loadMetrics;
