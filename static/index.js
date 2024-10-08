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
    resultDiv.innerHTML = `
      <h2>Deepfake Detection Analysis</h2>
      <div class="report-meta">
        <div class="meta-item">
          <span class="meta-label">Classification:</span>
          <span class="meta-value ${
            result.is_deepfake ? "deepfake" : "real"
          }">${result.is_deepfake ? "Deepfake" : "Real"}</span>
        </div>
        <div class="meta-item">
          <span class="meta-label">Confidence:</span>
          <span class="meta-value confidence">${(
            result.confidence * 100
          ).toFixed(2)}%</span>
        </div>
      </div>
    `;

    // Display the traced image and heatmaps
    displayTracedImageAndHeatmaps(
      result.traced_image,
      result.face_heatmap,
      result.landmark_heatmap,
      result.attention_heatmap
    );

    // Display charts
    let chartsHtml = '<div class="charts-container">';

    // First row: Overall Confidence and Key Indicators
    chartsHtml += '<div class="charts-row">';
    if (result.donut_chart) {
      chartsHtml += `
        <div class="chart chart-half">
          <h3>Overall Confidence</h3>
          <img src="data:image/png;base64,${result.donut_chart}" alt="Overall Confidence Chart">
        </div>
      `;
    }
    if (result.radar_chart) {
      chartsHtml += `
        <div class="chart chart-half">
          <h3>Key Indicators</h3>
          <img src="data:image/png;base64,${result.radar_chart}" alt="Key Indicators Radar Chart">
        </div>
      `;
    }
    chartsHtml += "</div>"; // Close first row

    chartsHtml += "</div>"; // Close charts-container
    resultDiv.innerHTML += chartsHtml;

    // Display the report content
    resultDiv.innerHTML += `<div class="report-content">${formatReport(
      result
    )}</div>`;
  }
}

function displayTracedImageAndHeatmaps(
  tracedImageData,
  faceHeatmapData,
  landmarkHeatmapData,
  attentionHeatmapData
) {
  if (
    tracedImageData &&
    faceHeatmapData &&
    landmarkHeatmapData &&
    attentionHeatmapData
  ) {
    const imageContainer = document.createElement("div");
    imageContainer.className = "image-container";
    imageContainer.innerHTML = `
      <h3>Visual Analysis</h3>
      <p>These visualizations provide insights into different aspects of the image analysis:</p>
      <div class="image-box">
        <h4>Face Detection</h4>
        <div class="image-wrapper">
          <img src="data:image/jpeg;base64,${tracedImageData}" alt="Traced Face">
          <p class="image-overlay">This image shows the detected face with a bounding box. Accurate face detection is crucial for deepfake analysis.</p>
        </div>
      </div>
      <div class="image-box">
        <h4>Face Heatmap</h4>
        <div class="image-wrapper">
          <img src="data:image/jpeg;base64,${faceHeatmapData}" alt="Face Heatmap">
          <p class="image-overlay">The face heatmap highlights areas of interest within the detected face region. Warmer colors indicate areas of higher importance in the analysis.</p>
        </div>
      </div>
      <div class="image-box">
        <h4>Landmark Heatmap</h4>
        <div class="image-wrapper">
          <img src="data:image/jpeg;base64,${landmarkHeatmapData}" alt="Landmark Heatmap">
          <p class="image-overlay">This heatmap shows detected facial landmarks. Inconsistencies in landmark placement can be indicators of manipulation.</p>
        </div>
      </div>
      <div class="image-box">
        <h4>Attention Heatmap</h4>
        <div class="image-wrapper">
          <img src="data:image/jpeg;base64,${attentionHeatmapData}" alt="Attention Heatmap">
          <p class="image-overlay">The attention heatmap reveals areas where the deepfake detection model is focusing. It can help identify suspicious regions or artifacts.</p>
        </div>
      </div>
    `;
    resultDiv.appendChild(imageContainer);
  }
}

function createChartDiv(title, chartData) {
  const chartDiv = document.createElement("div");
  chartDiv.className = "chart";
  chartDiv.innerHTML = `
    <h3>${title}</h3>
    <img src="data:image/png;base64,${chartData}" alt="${title} Chart">
  `;
  return chartDiv;
}

function formatReport(result) {
  const sections = result.report
    .split(/\n(?=\w+\n[-=]+)/)
    .filter((section) => section.trim());
  let formattedReport = `
    <h3 class="report-main-heading">Deepfake Detection Report</h3>
    <div class="report-meta">
      <div class="meta-item">
        <span class="meta-label">Classification:</span>
        <span class="meta-value ${result.is_deepfake ? "deepfake" : "real"}">${
    result.is_deepfake ? "Deepfake" : "Real"
  }</span>
      </div>
      <div class="meta-item">
        <span class="meta-label">Confidence:</span>
        <span class="meta-value confidence">${(result.confidence * 100).toFixed(
          2
        )}%</span>
      </div>
    </div>
  `;

  sections.forEach((section) => {
    const [title, ...content] = section.split("\n");
    const sectionTitle = title.trim().toLowerCase();

    // Skip the Recommendations and Limitations sections
    if (sectionTitle !== "recommendations" && sectionTitle !== "limitations") {
      const sectionContent = content.join("\n").trim();

      formattedReport += `
        <div class="report-section">
          <h4 class="report-heading">${title.trim()}</h4>
          ${formatSectionContent(sectionContent)}
        </div>
      `;
    }
  });

  return formattedReport;
}

function formatSectionContent(content) {
  const lines = content.split("\n");
  let formattedContent = "";
  let inList = false;

  lines.forEach((line) => {
    line = line.trim();
    if (line.startsWith("**") && line.endsWith("**")) {
      // Handle bold subheadings
      formattedContent += `<h5 class="report-subheading">${line.replace(
        /\*\*/g,
        ""
      )}</h5>`;
    } else if (line.startsWith("-") || line.startsWith("*")) {
      // Handle list items
      if (!inList) {
        formattedContent += "<ul>";
        inList = true;
      }
      formattedContent += `<li>${line.substring(1).trim()}</li>`;
    } else {
      if (inList) {
        formattedContent += "</ul>";
        inList = false;
      }
      // Handle regular paragraphs
      if (line) {
        formattedContent += `<p>${formatInlineStyles(line)}</p>`;
      }
    }
  });

  if (inList) {
    formattedContent += "</ul>";
  }

  return formattedContent;
}

function formatInlineStyles(text) {
  // Handle bold text
  text = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  // Handle italic text
  text = text.replace(/\*(.*?)\*/g, "<em>$1</em>");
  return text;
}
