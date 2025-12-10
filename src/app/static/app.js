const API_URL = "/predict";

const requestInput = document.getElementById("requestInput");
const responseOutput = document.getElementById("responseOutput");
const statusPrediction = document.getElementById("statusPrediction");
const statusPredictionLabel = document.getElementById("statusPredictionLabel");
const statusProbability = document.getElementById("statusProbability");

const exampleClean = {
  year: 2014,
  day: 150,
  length: 5,
  weight: 0.12,
  count: 3,
  looped: 1,
  neighbors: 15,
  income: 0.003,
};

const exampleHeist = {
  year: 2013,
  day: 45,
  length: 1,
  weight: 0.0002,
  count: 1,
  looped: 0,
  neighbors: 2,
  income: 0.00001,
};

document.getElementById("btnExampleClean").onclick = () => {
  requestInput.value = JSON.stringify(exampleClean, null, 2);
  clearStatus();
};

document.getElementById("btnExampleHeist").onclick = () => {
  requestInput.value = JSON.stringify(exampleHeist, null, 2);
  clearStatus();
};

document.getElementById("btnSend").onclick = async () => {
  let payload;
  try {
    payload = JSON.parse(requestInput.value);
  } catch (e) {
    responseOutput.textContent = "Invalid JSON: " + e.message;
    clearStatus();
    return;
  }

  responseOutput.textContent = "Sending request...";

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const text = await res.text();
    let json;
    try {
      json = JSON.parse(text);
      responseOutput.textContent = JSON.stringify(json, null, 2);
    } catch {
      responseOutput.textContent = text;
      json = null;
    }

    if (!res.ok) {
      clearStatus();
      return;
    }

    if (json && typeof json.prediction === "string") {
      const pred = json.prediction.toLowerCase();
      const prob = json.ransomware_probability;

      statusPrediction.style.display = "inline-flex";
      statusPredictionLabel.textContent =
        pred === "ransomware" ? "Ransomware" : "Clean";

      statusPrediction.classList.remove("clean", "ransomware");
      if (pred === "ransomware") {
        statusPrediction.classList.add("ransomware");
      } else {
        statusPrediction.classList.add("clean");
      }

      if (typeof prob === "number") {
        const pct = (prob * 100).toFixed(2);
        statusProbability.textContent = `ransomware probability: ${pct}%`;
      } else {
        statusProbability.textContent = "";
      }
    } else {
      clearStatus();
    }
  } catch (err) {
    responseOutput.textContent = "Request failed: " + err;
    clearStatus();
  }
};

function clearStatus() {
  statusPrediction.style.display = "none";
  statusPredictionLabel.textContent = "";
  statusProbability.textContent = "";
}

requestInput.value = JSON.stringify(exampleClean, null, 2);
clearStatus();
