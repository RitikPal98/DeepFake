const dropArea = document.getElementById("drop-area");
const fileInput = document.getElementById("file-input");
const gallery = document.getElementById("gallery");
const loader = document.getElementById("loader");
const resultDiv = document.getElementById("result");
const themeToggle = document.getElementById("theme-toggle");
const toggleCheckbox = document.getElementById("toggle-checkbox");
const themeLabel = document.getElementById("theme-label");

// Function to set the theme
function setTheme(isDark) {
  if (isDark) {
    document.body.classList.add("dark");
    toggleCheckbox.checked = true;
    themeLabel.textContent = "Dark Mode";
  } else {
    document.body.classList.remove("dark");
    toggleCheckbox.checked = false;
    themeLabel.textContent = "Light Mode";
  }
}

// Load theme preference from localStorage
const savedTheme = localStorage.getItem("theme") || "light";
setTheme(savedTheme === "dark");

// Event listener for theme toggle
themeToggle.addEventListener("click", () => {
  const isDark = toggleCheckbox.checked;
  setTheme(!isDark);
  localStorage.setItem("theme", isDark ? "light" : "dark");
});

// Prevent default behaviors
["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
  dropArea.addEventListener(eventName, preventDefaults, false);
  document.body.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

// Highlight drop area when item is dragged over it
["dragenter", "dragover"].forEach((eventName) => {
  dropArea.addEventListener(
    eventName,
    () => dropArea.classList.add("highlight"),
    false
  );
});

// Remove highlight when item is dragged away or dropped
["dragleave", "drop"].forEach((eventName) => {
  dropArea.addEventListener(
    eventName,
    () => dropArea.classList.remove("highlight"),
    false
  );
});

// Handle dropped files
dropArea.addEventListener("drop", handleDrop, false);

function handleDrop(e) {
  const dt = e.dataTransfer;
  const files = dt.files;
  handleFiles(files);
}

// Handle selected files
function handleFiles(files) {
  [...files].forEach((file) => {
    previewFile(file);
    uploadFile(file);
  });
}

// Preview the uploaded file
function previewFile(file) {
  const reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onloadend = function () {
    gallery.innerHTML = "";
    if (file.type.startsWith("image/")) {
      const img = document.createElement("img");
      img.src = reader.result;
      img.alt = "Uploaded Image";
      gallery.appendChild(img);
    } else if (file.type.startsWith("video/")) {
      const video = document.createElement("video");
      video.src = reader.result;
      video.controls = true;
      gallery.appendChild(video);
    }
  };
}

// Upload the file to the server
function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  loader.style.display = "block";
  resultDiv.innerHTML = "";

  fetch("/detect", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((result) => {
      loader.style.display = "none";
      displayResult(result);
    })
    .catch((error) => {
      console.error("Error:", error);
      loader.style.display = "none";
      resultDiv.textContent = "An error occurred during processing.";
    });
}

// Display the result from the server
function displayResult(result) {
  console.log("Received result:", result);
  if (result.error) {
    resultDiv.textContent = "Error: " + result.error;
    resultDiv.style.color = "var(--result-error)";
  } else {
    const status = result.is_deepfake ? "DeepFake" : "Real";
    const confidence = (result.confidence * 100).toFixed(2);

    let barChartHtml = "";
    if (result.bar_chart) {
      barChartHtml = `
        <div class="chart">
          <h3>Quantitative Metrics</h3>
          <img src="data:image/png;base64,${result.bar_chart}" alt="Quantitative Metrics Chart">
        </div>
      `;
    } else {
      console.warn("Bar chart data is missing or invalid");
      barChartHtml = "<p>Quantitative metrics chart is not available.</p>";
    }

    resultDiv.innerHTML = `
      <h2>Deepfake Detection Analysis</h2>
      <div class="prediction-result">
        <p><strong>Prediction:</strong> <span class="status">${status}</span></p>
        <p><strong>Confidence:</strong> <span class="confidence">${confidence}%</span></p>
      </div>
      <div class="charts-container">
        <div class="chart">
          <h3>Overall Confidence</h3>
          <img src="data:image/png;base64,${
            result.donut_chart
          }" alt="Overall Confidence Chart">
        </div>
        ${barChartHtml}
      </div>
      <h3>Detailed Report</h3>
      <div class="report-content">${formatReport(result.report)}</div>
    `;
  }
}

function formatReport(report) {
  const paragraphs = report.split("\n\n");

  const formattedParagraphs = paragraphs.map((paragraph) => {
    if (paragraph.startsWith("**") && paragraph.endsWith("**")) {
      return `<h4>${paragraph.replace(/\*\*/g, "")}</h4>`;
    } else if (paragraph.includes("* ")) {
      const listItems = paragraph
        .split("* ")
        .filter((item) => item.trim() !== "");
      return `<ul>${listItems
        .map((item) => `<li>${item.trim()}</li>`)
        .join("")}</ul>`;
    } else {
      return `<p>${paragraph}</p>`;
    }
  });

  return formattedParagraphs.join("");
}
