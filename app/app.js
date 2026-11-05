// Agentic RAG — Knowledge Research Engine Client
document.addEventListener("DOMContentLoaded", () => {
  const API_BASE = window.location.origin;

  // Ephemeral Browser Session Management
  let sessionToken = sessionStorage.getItem("rag_session_token");
  let customApiKey = sessionStorage.getItem("rag_user_api_key");

  async function getAuthHeaders() {
    const headers = { "Content-Type": "application/json" };
    if (customApiKey) {
      headers["X-API-Key"] = customApiKey;
      return headers;
    }
    if (!sessionToken) {
      try {
        const res = await fetch(`${API_BASE}/v1/auth/session`, { method: "POST" });
        if (res.ok) {
          const data = await res.json();
          sessionToken = data.session_token;
          sessionStorage.setItem("rag_session_token", sessionToken);
        }
      } catch (err) {
        console.warn("Session token handshake deferred:", err);
      }
    }
    if (sessionToken) {
      headers["X-Session-Token"] = sessionToken;
    }
    return headers;
  }

  // DOM Elements
  const navButtons = document.querySelectorAll(".nav-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const screenTitle = document.getElementById("screen-title");
  const screenDesc = document.getElementById("screen-desc");
  const systemStats = document.getElementById("system-stats");
  const queryInput = document.getElementById("query-input");
  const btnSubmit = document.getElementById("btn-submit-query");
  const answerContainer = document.getElementById("answer-container");
  const statusBadge = document.getElementById("status-badge");
  const loadingSpinner = document.getElementById("loading-spinner");
  const citationsContainer = document.getElementById("citations-container");
  const citationsList = document.getElementById("citations-list");
  const timelineContainer = document.getElementById("timeline-container");
  const evidenceContainer = document.getElementById("evidence-container");
  const evidenceCount = document.getElementById("evidence-count");
  const historyContainer = document.getElementById("history-container");
  const btnClearHistory = document.getElementById("btn-clear-history");
  const btnRunBenchmark = document.getElementById("btn-run-benchmark");

  // Tab Screen Data
  const screenData = {
    workspace: {
      title: "Research Workspace",
      desc: "Execute multi-document grounded queries with autonomous tool planning."
    },
    execution: {
      title: "Agent Execution View",
      desc: "Real-time state transitions, subquery planning, and verification telemetry."
    },
    evidence: {
      title: "Evidence Explorer",
      desc: "Inspect retrieved knowledge passages, relevance scores, and provenance."
    },
    history: {
      title: "Research History",
      desc: "Audit prior research sessions, confidence scores, and citation mappings."
    },
    evaluation: {
      title: "Evaluation Dashboard",
      desc: "Empirical performance comparisons: Standard RAG vs Optimized RAG vs Agentic RAG."
    }
  };

  // Switch tabs
  navButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      navButtons.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      document.getElementById(`tab-${targetTab}`).classList.add("active");

      screenTitle.textContent = screenData[targetTab].title;
      screenDesc.textContent = screenData[targetTab].desc;
    });
  });

  // Query chips
  document.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      queryInput.value = chip.getAttribute("data-q");
    });
  });

  // Check health and update status
  async function fetchHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        systemStats.textContent = `${data.total_documents} docs (${data.total_chunks} chunks indexed)`;
      } else {
        systemStats.textContent = "API Offline";
      }
    } catch {
      systemStats.textContent = "Standalone Mode";
    }
  }
  fetchHealth();

  // Safe DOM Rendering Utilities (Eliminating DOM-Based XSS)
  function renderSafeAnswer(text) {
    answerContainer.replaceChildren();
    const formattedDiv = document.createElement("div");
    formattedDiv.className = "formatted-answer";

    const paragraphs = text.split("\n\n");
    paragraphs.forEach(para => {
      const trimmed = para.trim();
      if (!trimmed) return;
      const p = document.createElement("p");
      p.textContent = trimmed;
      formattedDiv.appendChild(p);
    });

    answerContainer.appendChild(formattedDiv);
  }

  function renderSafeCitations(citations) {
    citationsList.replaceChildren();
    if (!citations || citations.length === 0) {
      citationsContainer.classList.add("hidden");
      return;
    }
    citationsContainer.classList.remove("hidden");
    citations.forEach(c => {
      const li = document.createElement("li");
      li.className = "citation-item";
      li.textContent = `[${c}]`;
      citationsList.appendChild(li);
    });
  }

  function renderSafeTimeline(steps) {
    timelineContainer.replaceChildren();
    if (!steps || steps.length === 0) return;

    steps.forEach(step => {
      const stepDiv = document.createElement("div");
      stepDiv.className = "timeline-step";

      const node = document.createElement("div");
      node.className = "step-node";
      node.textContent = String(step.step_index);

      const body = document.createElement("div");
      body.className = "step-body";

      const header = document.createElement("div");
      header.className = "step-header";

      const stage = document.createElement("span");
      stage.className = "step-stage";
      stage.textContent = step.stage;
      header.appendChild(stage);

      const desc = document.createElement("p");
      desc.className = "step-desc";
      desc.textContent = step.message;

      body.appendChild(header);
      body.appendChild(desc);
      stepDiv.appendChild(node);
      stepDiv.appendChild(body);
      timelineContainer.appendChild(stepDiv);
    });
  }

  function renderSafeEvidence(evidenceList) {
    evidenceContainer.replaceChildren();
    if (!evidenceList) evidenceList = [];
    evidenceCount.textContent = `${evidenceList.length} Chunks Retrieved`;

    evidenceList.forEach(chunk => {
      const card = document.createElement("div");
      card.className = "evidence-card";

      const cardHeader = document.createElement("div");
      cardHeader.className = "evidence-card-header";

      const docIdSpan = document.createElement("span");
      docIdSpan.className = "evidence-doc-id";
      docIdSpan.textContent = chunk.chunk_id;

      const badge = document.createElement("span");
      badge.className = "badge badge-purple";
      badge.textContent = `${chunk.strategy} (Score: ${chunk.score})`;

      cardHeader.appendChild(docIdSpan);
      cardHeader.appendChild(badge);

      const contentP = document.createElement("p");
      contentP.className = "evidence-content";
      contentP.textContent = chunk.content;

      card.appendChild(cardHeader);
      card.appendChild(contentP);
      evidenceContainer.appendChild(card);
    });
  }

  function renderSafeHistory() {
    let history = JSON.parse(localStorage.getItem("rag_history") || "[]");
    historyContainer.replaceChildren();

    if (history.length === 0) {
      const emptyDiv = document.createElement("div");
      emptyDiv.className = "empty-state";
      const p = document.createElement("p");
      p.textContent = "No prior research queries recorded.";
      emptyDiv.appendChild(p);
      historyContainer.appendChild(emptyDiv);
      return;
    }

    history.forEach(h => {
      const item = document.createElement("div");
      item.className = "history-item panel";
      item.style.marginBottom = "12px";
      item.style.padding = "14px";

      const topRow = document.createElement("div");
      topRow.style.display = "flex";
      topRow.style.justifyContent = "space-between";
      topRow.style.alignItems = "center";

      const queryStrong = document.createElement("strong");
      queryStrong.textContent = h.query;

      const badge = document.createElement("span");
      badge.className = `badge ${h.status === "SUCCESS" ? "badge-green" : "badge-yellow"}`;
      badge.textContent = h.status;

      topRow.appendChild(queryStrong);
      topRow.appendChild(badge);

      const bottomRow = document.createElement("div");
      bottomRow.style.fontSize = "0.8rem";
      bottomRow.style.color = "var(--text-muted)";
      bottomRow.style.marginTop = "6px";
      bottomRow.textContent = `Time: ${h.time} | Confidence: ${(h.confidence * 100).toFixed(0)}%`;

      item.appendChild(topRow);
      item.appendChild(bottomRow);
      historyContainer.appendChild(item);
    });
  }

  // Execute Query
  btnSubmit.addEventListener("click", async () => {
    const query = queryInput.value.trim();
    if (!query) return;

    btnSubmit.disabled = true;
    loadingSpinner.classList.remove("hidden");
    answerContainer.classList.add("hidden");
    citationsContainer.classList.add("hidden");
    statusBadge.textContent = "Planning & Executing...";
    statusBadge.className = "badge badge-blue";

    try {
      const headers = await getAuthHeaders();
      const res = await fetch(`${API_BASE}/v1/agent/query`, {
        method: "POST",
        headers: headers,
        body: JSON.stringify({ query: query })
      });

      if (!res.ok) throw new Error(`Execution error (HTTP ${res.status})`);
      const data = await res.json();

      // Render Answer Safely
      answerContainer.classList.remove("hidden");
      renderSafeAnswer(data.answer);

      // Status Badge
      if (data.status === "SUCCESS") {
        statusBadge.textContent = `SUCCESS (Confidence: ${(data.confidence * 100).toFixed(0)}%)`;
        statusBadge.className = "badge badge-green";
      } else {
        statusBadge.textContent = data.status;
        statusBadge.className = "badge badge-yellow";
      }

      // Citations Safely
      renderSafeCitations(data.citations);

      // Fetch trace for execution view
      if (data.request_id) {
        fetchTrace(data.request_id);
      }

      // Save to local history
      saveHistory(data);
    } catch (err) {
      answerContainer.classList.remove("hidden");
      answerContainer.replaceChildren();
      const errP = document.createElement("p");
      errP.className = "text-red";
      errP.textContent = `Error executing query: ${err.message}`;
      answerContainer.appendChild(errP);

      statusBadge.textContent = "ERROR";
      statusBadge.className = "badge badge-red";
    } finally {
      loadingSpinner.classList.add("hidden");
      btnSubmit.disabled = false;
    }
  });

  // Fetch Trace for Execution View
  async function fetchTrace(requestId) {
    try {
      const headers = await getAuthHeaders();
      const res = await fetch(`${API_BASE}/v1/agent/trace/${requestId}`, {
        headers: headers
      });
      if (!res.ok) return;
      const traceData = await res.json();

      const execReqId = document.getElementById("exec-request-id");
      if (execReqId) {
        execReqId.textContent = `Request ID: ${requestId}`;
      }

      renderSafeTimeline(traceData.trace);
      renderSafeEvidence(traceData.evidence);
    } catch (err) {
      console.error("Failed to fetch execution trace", err);
    }
  }

  function saveHistory(item) {
    let history = JSON.parse(localStorage.getItem("rag_history") || "[]");
    history.unshift({
      id: item.request_id,
      query: item.query,
      status: item.status,
      confidence: item.confidence,
      time: new Date().toLocaleTimeString()
    });
    localStorage.setItem("rag_history", JSON.stringify(history.slice(0, 20)));
    renderSafeHistory();
  }

  renderSafeHistory();

  btnClearHistory.addEventListener("click", () => {
    localStorage.removeItem("rag_history");
    renderSafeHistory();
  });

  // Re-run Benchmark
  btnRunBenchmark.addEventListener("click", async () => {
    btnRunBenchmark.disabled = true;
    btnRunBenchmark.textContent = "Running Benchmark...";
    try {
      const headers = await getAuthHeaders();
      const res = await fetch(`${API_BASE}/v1/evaluation`, {
        headers: headers
      });
      if (res.ok) {
        alert("Benchmark completed! Check reports/baseline_comparison.csv and reports/performance_report.md");
      } else {
        alert(`Failed to run benchmark (HTTP ${res.status}).`);
      }
    } catch {
      alert("Failed to run benchmark.");
    } finally {
      btnRunBenchmark.disabled = false;
      btnRunBenchmark.textContent = "Re-Run Benchmark";
    }
  });
});
