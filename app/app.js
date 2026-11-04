// Agentic RAG — Knowledge Research Engine Client
document.addEventListener("DOMContentLoaded", () => {
  const API_BASE = window.location.origin;

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
  const loadingStage = document.getElementById("loading-stage");
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

  // Execute Query
  btnSubmit.addEventListener("click", async () => {
    const query = queryInput.value.trim();
    if (!query) return;

    // UI Loading state
    btnSubmit.disabled = true;
    loadingSpinner.classList.remove("hidden");
    answerContainer.classList.add("hidden");
    citationsContainer.classList.add("hidden");
    statusBadge.textContent = "Planning & Executing...";
    statusBadge.className = "badge badge-blue";

    try {
      const res = await fetch(`${API_BASE}/v1/agent/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query })
      });

      if (!res.ok) throw new Error("Failed to execute research query");
      const data = await res.json();

      // Render Answer
      answerContainer.classList.remove("hidden");
      answerContainer.innerHTML = `<div class="formatted-answer">${data.answer.replace(/\n/g, "<br>")}</div>`;

      // Status Badge
      if (data.status === "SUCCESS") {
        statusBadge.textContent = `SUCCESS (Confidence: ${(data.confidence * 100).toFixed(0)}%)`;
        statusBadge.className = "badge badge-green";
      } else {
        statusBadge.textContent = data.status;
        statusBadge.className = "badge badge-yellow";
      }

      // Citations
      if (data.citations && data.citations.length > 0) {
        citationsContainer.classList.remove("hidden");
        citationsList.innerHTML = data.citations
          .map(c => `<li class="citation-item">[${c}]</li>`)
          .join("");
      }

      // Fetch trace for execution view
      if (data.request_id) {
        fetchTrace(data.request_id);
      }

      // Save to local history
      saveHistory(data);
    } catch (err) {
      answerContainer.classList.remove("hidden");
      answerContainer.innerHTML = `<p class="text-red">Error executing query: ${err.message}</p>`;
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
      const res = await fetch(`${API_BASE}/v1/agent/trace/${requestId}`);
      if (!res.ok) return;
      const traceData = await res.json();

      document.getElementById("exec-request-id").textContent = `Request ID: ${requestId}`;

      // Render timeline
      timelineContainer.innerHTML = traceData.trace.map(step => `
        <div class="timeline-step">
          <div class="step-node">${step.step_index}</div>
          <div class="step-body">
            <div class="step-header">
              <span class="step-stage">${step.stage}</span>
            </div>
            <p class="step-desc">${step.message}</p>
          </div>
        </div>
      `).join("");

      // Render evidence
      evidenceCount.textContent = `${traceData.evidence.length} Chunks Retrieved`;
      evidenceContainer.innerHTML = traceData.evidence.map(chunk => `
        <div class="evidence-card">
          <div class="evidence-card-header">
            <span class="evidence-doc-id">${chunk.chunk_id}</span>
            <span class="badge badge-purple">${chunk.strategy} (Score: ${chunk.score})</span>
          </div>
          <p class="evidence-content">${chunk.content}</p>
        </div>
      `).join("");

    } catch (err) {
      console.error("Failed to fetch execution trace", err);
    }
  }

  // History Storage
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
    renderHistory();
  }

  function renderHistory() {
    let history = JSON.parse(localStorage.getItem("rag_history") || "[]");
    if (history.length === 0) {
      historyContainer.innerHTML = `<div class="empty-state"><p>No prior research queries recorded.</p></div>`;
      return;
    }

    historyContainer.innerHTML = history.map(h => `
      <div class="history-item panel" style="margin-bottom: 12px; padding: 14px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <strong>${h.query}</strong>
          <span class="badge ${h.status === 'SUCCESS' ? 'badge-green' : 'badge-yellow'}">${h.status}</span>
        </div>
        <div style="font-size:0.8rem; color:var(--text-muted); margin-top:6px;">
          Time: ${h.time} | Confidence: ${(h.confidence * 100).toFixed(0)}%
        </div>
      </div>
    `).join("");
  }
  renderHistory();

  btnClearHistory.addEventListener("click", () => {
    localStorage.removeItem("rag_history");
    renderHistory();
  });

  // Re-run Benchmark
  btnRunBenchmark.addEventListener("click", async () => {
    btnRunBenchmark.disabled = true;
    btnRunBenchmark.textContent = "Running Benchmark...";
    try {
      const res = await fetch(`${API_BASE}/v1/evaluation`);
      if (res.ok) {
        alert("Benchmark completed! Check reports/baseline_comparison.csv and reports/performance_report.md");
      }
    } catch {
      alert("Failed to run benchmark.");
    } finally {
      btnRunBenchmark.disabled = false;
      btnRunBenchmark.textContent = "Re-Run Benchmark";
    }
  });
});

