const API_URL = "https://multiple-pdf-summarization-system-4qzl.onrender.com";

const pdfFiles = document.getElementById("pdfFiles");
const fileList = document.getElementById("fileList");
const summarizeBtn = document.getElementById("summarizeBtn");
const status = document.getElementById("status");
const results = document.getElementById("results");

let selectedFiles = [];

pdfFiles.addEventListener("change", function () {
  const newFiles = Array.from(pdfFiles.files);
  
  newFiles.forEach(newFile => {
    const isDuplicate = selectedFiles.some(
      existingFile => existingFile.name === newFile.name && existingFile.size === newFile.size
    );
    
    if (!isDuplicate) {
      selectedFiles.push(newFile);
    }
  });

  showFiles();
  
  pdfFiles.value = ""; 
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
  results.innerHTML = "";

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
    `;
    
    const dlBtn = document.createElement("button");
    dlBtn.className = "download";
    dlBtn.textContent = "Download Summary";
    dlBtn.addEventListener("click", () => downloadText(doc.filename, doc.summary));
    
    card.appendChild(dlBtn);
    results.appendChild(card);
  });

  const combined = document.createElement("div");
  combined.className = "combined-card";
  combined.innerHTML = `
    <h2>Combined Summary</h2>
    <p class="summary">${data.combined_summary}</p>
  `;
  
  const dlCombinedBtn = document.createElement("button");
  dlCombinedBtn.textContent = "Download Combined Summary";
  dlCombinedBtn.addEventListener("click", () => downloadText("combined-summary.txt", data.combined_summary));
  
  combined.appendChild(dlCombinedBtn);
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