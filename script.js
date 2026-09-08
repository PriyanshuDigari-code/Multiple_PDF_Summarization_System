const API_URL = "http://127.0.0.1:8000";

const pdfFiles = document.getElementById("pdfFiles");
const fileList = document.getElementById("fileList");
const summarizeBtn = document.getElementById("summarizeBtn");
const status = document.getElementById("status");
const results = document.getElementById("results");

let selectedFiles = [];

pdfFiles.addEventListener("change", function () {
  selectedFiles = Array.from(pdfFiles.files);
  showFiles();
});

function showFiles() {
  fileList.innerHTML = "";
  selectedFiles.forEach((file, index) => {
    const item = document.createElement("div");
    item.className = "file-item";
    item.innerHTML = `${file.name} <button class="remove-btn" onclick="removeFile(${index})">Remove</button>`;
    fileList.appendChild(item);
  });
}

function removeFile(index) {
  selectedFiles.splice(index, 1);
  showFiles();
}

summarizeBtn.addEventListener("click", async function () {
  if (selectedFiles.length === 0) {
    status.textContent = "Please select at least one PDF.";
    return;
  }

  const formData = new FormData();
  selectedFiles.forEach(file => formData.append("files", file));

  summarizeBtn.disabled = true;
  status.textContent = "Processing PDFs... This may take some time.";
  results.innerHTML = "";

  try {
    const response = await fetch(`${API_URL}/summarize`, {
      method: "POST",
      body: formData
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Something went wrong.");

    displayResults(data);
    status.textContent = "Summaries generated successfully.";
  } catch (error) {
    status.textContent = "Error: " + error.message;
  } finally {
    summarizeBtn.disabled = false;
  }
});

function displayResults(data) {
  data.documents.forEach(doc => {
    const card = document.createElement("div");
    card.className = "result-card";
    card.innerHTML = `
      <h2>${doc.filename}</h2>
      <p class="meta">Original words: ${doc.original_words} | Summary words: ${doc.summary_words}</p>
      <h3>Keywords</h3>
      <div class="keywords">${doc.keywords.map(word => `<span>${word}</span>`).join("")}</div>
      <h3>Summary</h3>
      <p class="summary">${doc.summary}</p>
      <button class="download" onclick='downloadText(${JSON.stringify(doc.filename)}, ${JSON.stringify(doc.summary)})'>Download Summary</button>
    `;
    results.appendChild(card);
  });

  const combined = document.createElement("div");
  combined.className = "combined-card";
  combined.innerHTML = `
    <h2>Combined Summary</h2>
    <p class="summary">${data.combined_summary}</p>
    <button onclick='downloadText("combined-summary.txt", ${JSON.stringify(data.combined_summary)})'>Download Combined Summary</button>
  `;
  results.appendChild(combined);
}

function downloadText(filename, text) {
  const blob = new Blob([text], { type: "text/plain" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename.replace(".pdf", "") + "-summary.txt";
  link.click();
  URL.revokeObjectURL(link.href);
}
