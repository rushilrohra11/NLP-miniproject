const audioFile = document.getElementById("audioFile");
const fileName = document.getElementById("fileName");
const clinicalText = document.getElementById("clinicalText");
const generateBtn = document.getElementById("generateBtn");
const statusBox = document.getElementById("statusBox");
const errorBox = document.getElementById("errorBox");
const transcriptionOutput = document.getElementById("transcription");
const summaryOutput = document.getElementById("summary");
const soapOutput = document.getElementById("soapNote");

const tabBtns = document.querySelectorAll(".tab-btn");
const tabContents = document.querySelectorAll(".tab-content");

let currentTab = "audio"; // Track which tab is active

// Tab switching
tabBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    const tab = btn.getAttribute("data-tab");
    
    // Update active button
    tabBtns.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    
    // Update active content
    tabContents.forEach((content) => content.classList.remove("active"));
    document.getElementById(`${tab}-tab`).classList.add("active");
    
    currentTab = tab;
    
    // Clear previous input
    audioFile.value = "";
    clinicalText.value = "";
    fileName.textContent = "No file selected";
  });
});

// Audio file selection
audioFile.addEventListener("change", (e) => {
  const file = e.target.files[0];
  fileName.textContent = file ? file.name : "No file selected";
  if (file) {
    setError("");
    setStatus(`Selected file: ${file.name}`);
  }
});

function setOutputs(message) {
  transcriptionOutput.textContent = message;
  summaryOutput.textContent = message;
  soapOutput.innerHTML = `<div class="soap-section"><div class="soap-value">${escapeHtml(message)}</div></div>`;
}

function formatTranscriptionDisplay(transcriptionDialogue, entities) {
  const transcript = (transcriptionDialogue || "").trim() || "No transcription returned.";

  if (!entities || typeof entities !== "object") {
    return transcript;
  }

  const roles = Array.isArray(entities.roles_detected) ? entities.roles_detected : [];
  const doctorMentions = Array.isArray(entities.doctor_mentions) ? entities.doctor_mentions : [];
  const patientMentions = Array.isArray(entities.patient_mentions) ? entities.patient_mentions : [];
  const nurseMentions = Array.isArray(entities.nurse_mentions) ? entities.nurse_mentions : [];
  const accompanierMentions = Array.isArray(entities.accompanier_mentions) ? entities.accompanier_mentions : [];
  const supportStaffMentions = Array.isArray(entities.support_staff_mentions) ? entities.support_staff_mentions : [];
  const turns = entities.turn_count && typeof entities.turn_count === "object" ? entities.turn_count : {};

  const lines = [transcript, "", "Detected Speaker Entities:"];
  if (roles.length > 0) {
    lines.push(`Roles detected: ${roles.join(", ")}`);
  } else {
    lines.push("Roles detected: none");
  }
  lines.push(`Doctor turns: ${turns.doctor || 0}`);
  lines.push(`Patient turns: ${turns.patient || 0}`);
  lines.push(`Nurse turns: ${turns.nurse || 0}`);
  lines.push(`Accompanier turns: ${turns.accompanier || 0}`);
  lines.push(`Support staff turns: ${turns.support_staff || 0}`);
  lines.push(`Other turns: ${turns.other || 0}`);
  if (doctorMentions.length > 0) {
    lines.push(`Doctor names: ${doctorMentions.join(", ")}`);
  }
  if (patientMentions.length > 0) {
    lines.push(`Patient names: ${patientMentions.join(", ")}`);
  }
  if (nurseMentions.length > 0) {
    lines.push(`Nurse names: ${nurseMentions.join(", ")}`);
  }
  if (accompanierMentions.length > 0) {
    lines.push(`Accompanier names: ${accompanierMentions.join(", ")}`);
  }
  if (supportStaffMentions.length > 0) {
    lines.push(`Support staff names: ${supportStaffMentions.join(", ")}`);
  }

  return lines.join("\n");
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatSoapNote(soapNote) {
  if (!soapNote) {
    return "No SOAP note returned.";
  }

  const soap = soapNote.SOAP || soapNote;
  const extractedEntities = soapNote.ExtractedEntities || soapNote.extracted_entities || {};

  const sections = [
    ["Subjective", soap.Subjective, "Patient-reported symptoms, complaints, and history."],
    ["Objective", soap.Objective, "Observed findings, vitals, measurements, and tests."],
    ["Assessment", soap.Assessment, "Likely condition or clinical interpretation."],
    ["Plan", soap.Plan, "Treatment steps, follow-up, and medications."],
  ]
    .map(([label, value, hint]) => `
      <article class="soap-section">
        <div class="soap-label">${label}</div>
        <div class="soap-value">${value ? escapeHtml(value) : "Not available."}</div>
        <div class="soap-hint">${hint}</div>
      </article>
    `)
    .join("");

  const entityGroups = Object.entries(extractedEntities)
    .filter(([, values]) => Array.isArray(values) && values.length > 0)
    .map(
      ([group, values]) => `
        <div>
          <div class="soap-label">${group}</div>
          <div class="entity-grid">
            ${values.map((value) => `<span class="entity-chip">${escapeHtml(value)}</span>`).join("")}
          </div>
        </div>
      `,
    )
    .join("");

  const entities = entityGroups
    ? `<div class="entity-list"><div class="soap-label">Extracted Entities</div>${entityGroups}</div>`
    : "";

  return `${sections}${entities}`;
}

function setError(message) {
  errorBox.hidden = !message;
  errorBox.textContent = message || "";
}

function setStatus(message) {
  statusBox.textContent = message;
}

generateBtn.addEventListener("click", async () => {
  if (currentTab === "audio") {
    const file = audioFile.files[0];

    if (!file) {
      setError("Please choose an audio file first.");
      setStatus("Waiting for an audio file.");
      return;
    }

    setError("");
    generateBtn.disabled = true;
    generateBtn.classList.add("is-loading");
    setStatus("Uploading audio and processing with Gemini AI...");
    setOutputs("Processing audio file...");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://127.0.0.1:8000/api/process", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Request failed with status ${response.status}`);
      }

      const data = await response.json();

      transcriptionOutput.textContent = formatTranscriptionDisplay(
        data.transcription_dialogue || data.transcription,
        data.transcription_entities,
      );
      summaryOutput.textContent = data.summary || "No summary returned.";
      soapOutput.innerHTML = formatSoapNote(data.soap_note);
      setStatus("Processing complete.");
    } catch (error) {
      const message = error instanceof Error ? error.message : "An unknown error occurred.";
      setError(`Processing failed: ${message}`);
      setStatus("Unable to complete processing.");
      setOutputs("No results available.");
    } finally {
      generateBtn.disabled = false;
      generateBtn.classList.remove("is-loading");
    }
  } else {
    // Text tab
    const text = clinicalText.value.trim();

    if (!text) {
      setError("Please enter clinical text first.");
      setStatus("Waiting for clinical text input.");
      return;
    }

    setError("");
    generateBtn.disabled = true;
    generateBtn.classList.add("is-loading");
    setStatus("Processing clinical text...");
    setOutputs("Processing clinical text...");

    try {
      const response = await fetch("http://127.0.0.1:8000/api/process-text", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Request failed with status ${response.status}`);
      }

      const data = await response.json();

      transcriptionOutput.textContent = data.transcription_dialogue || "N/A (text input)";
      summaryOutput.textContent = data.summary || "No summary returned.";
      soapOutput.innerHTML = formatSoapNote(data.soap_note);
      setStatus("Processing complete.");
    } catch (error) {
      const message = error instanceof Error ? error.message : "An unknown error occurred.";
      setError(`Processing failed: ${message}`);
      setStatus("Unable to complete processing.");
      setOutputs("No results available.");
    } finally {
      generateBtn.disabled = false;
      generateBtn.classList.remove("is-loading");
    }
  }
});
