import { useState, useEffect, useRef, useCallback } from "react";

const API = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

// ── Normalise API response into consistent shape ──────────────
function normaliseRanking(raw) {
  if (!raw) return null;

  if (Array.isArray(raw.entities) && raw.entities.length > 0) {
    return {
      session_id: raw.session_id || "",
      ranking_rationale: raw.ranking_rationale || "",
      criteria: raw.criteria || [],
      rows: raw.entities.map(e => ({
        rank: e.rank,
        entity_name: e.name,
        entity_type: e.entity_type || "",
        description: e.description || "",
        composite_score: e.composite_score ?? 0,
        algorithm_scores: e.algorithm_scores || {},
        criterion_scores: e.criterion_scores || {},
        fields: e.properties || {},
        relationships: e.relationships || [],
        source_urls: Array.isArray(e.source_urls) ? e.source_urls : [],
        missing_keys: e.missing_criteria || [],
      })),
    };
  }

  if (Array.isArray(raw.rows) && raw.rows.length > 0) {
    return {
      ...raw,
      rows: raw.rows.map(r => ({
        rank: r.rank,
        entity_name: r.entity_name || "",
        entity_type: r.entity_type || "",
        description: r.description || "",
        composite_score: r.composite_score ?? 0,
        criterion_scores: r.criterion_scores || {},
        fields: r.fields || {},
        relationships: r.relationships || [],
        source_urls: r.source_url
          ? r.source_url.split(",").map(s => s.trim()).filter(Boolean)
          : [],
        missing_keys: r.missing_keys || [],
      })),
    };
  }
  return null;
}

// ── Node definitions ──────────────────────────────────────────
const CRAWL_NODES = [
  { id: "intent_parser",    label: "Intent Parser",   icon: "◎" },
  { id: "url_discovery",    label: "URL Discovery",   icon: "⊕" },
  { id: "web_crawler",      label: "Web Crawler",     icon: "⊞" },
  { id: "source_verifier",  label: "Source Verifier", icon: "⊛" },
  { id: "mongo_logger",     label: "Mongo Logger",    icon: "⊟" },
];
const GRAPH_NODES = [
  { id: "entity_extractor", label: "Entity Extractor", icon: "⊡" },
  { id: "neo4j_ingester",   label: "Neo4j Ingester",   icon: "⊠" },
  { id: "graph_structurer", label: "Graph Structurer", icon: "⋈" },
  { id: "insights_generator",label: "Insights Generator", icon: "✶" },
  { id: "metrics_evaluator",label: "Metrics Eval",     icon: "⊗" },
];
const RANK_NODES = [
  { id: "ranking_engine", label: "Ranking Engine", role: "LLM criteria selection + min-max composite scoring" },
];

// ── Styles ────────────────────────────────────────────────────
const css = `
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

  :root {
    --bg:      #060810;
    --surface: #0d1117;
    --border:  #1e2535;
    --accent:  #00e5ff;
    --accent2: #7c3aed;
    --accent3: #f59e0b;
    --text:    #e2e8f0;
    --muted:   #4a5568;
    --crawl:   #06b6d4;
    --graph:   #8b5cf6;
    --agents:  #10b981;
    --rank:    #f59e0b;
    --error:   #ef4444;
    --mono:    'JetBrains Mono', monospace;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    min-height: 100vh;
    overflow-x: hidden;
  }

  .grain {
    position: fixed; inset: 0; pointer-events: none; z-index: 100;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
    opacity: 0.6;
  }

  .app { max-width: 1400px; margin: 0 auto; padding: 2rem; position: relative; z-index: 1; }

  /* Header */
  .header {
    display: flex; align-items: center; gap: 1.5rem;
    margin-bottom: 3rem; padding-bottom: 1.5rem; border-bottom: 1px solid var(--border);
  }
  .logo-mark {
    width: 48px; height: 48px; border: 2px solid var(--accent);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.1rem;
    color: var(--accent); position: relative;
  }
  .logo-mark::after { content:''; position:absolute; inset:3px; border:1px solid var(--accent); opacity:0.3; }
  .header-title { font-family:'Syne',sans-serif; font-weight:800; font-size:1.5rem; letter-spacing:-0.03em; }
  .header-sub   { font-size:0.7rem; color:var(--muted); letter-spacing:0.15em; text-transform:uppercase; margin-top:0.2rem; }

  /* Query */
  .query-label { font-size:0.65rem; letter-spacing:0.2em; text-transform:uppercase; color:var(--muted); margin-bottom:0.75rem; }
  .query-row   { display:flex; gap:0.75rem; margin-bottom:2.5rem; }
  .topn-wrap   { display:flex; flex-direction:column; gap:0.3rem; flex-shrink:0; }
  .topn-label  { font-size:0.55rem; color:var(--muted); letter-spacing:0.15em; text-transform:uppercase; }
  .topn-input  {
    width:72px; background:var(--surface); border:1px solid var(--border);
    color:var(--accent); font-family:var(--mono); font-size:1rem; font-weight:600;
    padding:0.85rem 0.5rem; text-align:center; outline:none;
    transition:border-color 0.2s; -moz-appearance:textfield;
  }
  .topn-input::-webkit-inner-spin-button,
  .topn-input::-webkit-outer-spin-button { -webkit-appearance:none; }
  .topn-input:focus { border-color:var(--accent); }
  .query-input {
    flex:1; background:var(--surface); border:1px solid var(--border);
    color:var(--text); font-family:var(--mono); font-size:0.9rem;
    padding:0.85rem 1.25rem; outline:none; transition:border-color 0.2s;
  }
  .query-input:focus { border-color:var(--accent); box-shadow:0 0 0 1px var(--accent),inset 0 0 20px rgba(0,229,255,0.03); }
  .query-input::placeholder { color:var(--muted); }
  .run-btn {
    background:var(--accent); color:#000;
    font-family:'Syne',sans-serif; font-weight:700; font-size:0.8rem;
    letter-spacing:0.1em; text-transform:uppercase;
    border:none; padding:0 2rem; cursor:pointer; transition:all 0.15s; white-space:nowrap;
  }
  .run-btn:hover { background:#fff; }
  .run-btn:disabled { background:var(--muted); cursor:not-allowed; color:#333; }

  /* Status */
  .status-bar {
    display:flex; align-items:center; gap:0.75rem;
    padding:0.6rem 1rem; background:var(--surface); border:1px solid var(--border);
    margin-bottom:1.5rem; font-size:0.7rem;
  }
  .status-dot { width:6px; height:6px; border-radius:50%; background:var(--muted); }
  .status-dot.running   { background:var(--accent); animation:blink 1s ease infinite; }
  .status-dot.completed { background:var(--agents); }
  .status-dot.failed    { background:var(--error); }
  @keyframes blink { 0%,100%{opacity:1}50%{opacity:0.3} }

  /* Grid */
  .main-grid { display:grid; grid-template-columns:300px 1fr; gap:1.5rem; margin-bottom:2rem; }
  @media(max-width:900px){ .main-grid{ grid-template-columns:1fr; } }

  /* Pipeline sidebar */
  .pipeline-panel { display:flex; flex-direction:column; gap:1rem; }
  .panel-title { font-size:0.65rem; letter-spacing:0.2em; text-transform:uppercase; color:var(--muted); padding-bottom:0.5rem; border-bottom:1px solid var(--border); }
  .phase-group { background:var(--surface); border:1px solid var(--border); padding:1rem; }
  .phase-label { font-size:0.6rem; letter-spacing:0.25em; text-transform:uppercase; margin-bottom:0.75rem; padding-bottom:0.5rem; border-bottom:1px solid var(--border); }
  .phase-label.crawl  { color:var(--crawl); }
  .phase-label.graph  { color:var(--graph); }
  .phase-label.agents { color:var(--agents); }
  .node-item {
    display:flex; align-items:flex-start; gap:0.6rem;
    padding:0.4rem 0 0.4rem 0.5rem; font-size:0.75rem; color:var(--muted);
    border-left:2px solid transparent; transition:color 0.2s;
  }
  .node-item.active { color:var(--text); border-left-color:var(--accent); animation:nodePulse 1.2s ease-in-out infinite; }
  .node-item.done   { color:var(--agents); border-left-color:var(--agents); }
  .node-item.error  { color:var(--error);  border-left-color:var(--error); }
  @keyframes nodePulse { 0%,100%{opacity:1}50%{opacity:0.6} }
  .node-icon   { font-size:0.9rem; width:1.2rem; text-align:center; flex-shrink:0; margin-top:1px; }
  .node-status { margin-left:auto; font-size:0.6rem; flex-shrink:0; }

  /* Right panel */
  .right-panel { display:flex; flex-direction:column; gap:1rem; }
  .log-panel {
    background:var(--surface); border:1px solid var(--border);
    min-height:300px; max-height:380px; overflow-y:auto; display:flex; flex-direction:column;
  }
  .log-header {
    padding:0.75rem 1rem; font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase;
    color:var(--muted); border-bottom:1px solid var(--border);
    position:sticky; top:0; background:var(--surface); z-index:2; flex-shrink:0;
  }
  .log-body { padding:0.75rem; display:flex; flex-direction:column; gap:0.3rem; }
  .log-entry { font-size:0.7rem; line-height:1.5; padding:0.25rem 0.5rem; border-left:2px solid var(--border); animation:fadeSlide 0.3s ease; }
  @keyframes fadeSlide { from{opacity:0;transform:translateY(4px)} to{opacity:1;transform:none} }
  .log-entry.phase_start    { border-color:var(--accent2); color:var(--accent2); }
  .log-entry.phase_complete { border-color:var(--agents);  color:var(--agents); }
  .log-entry.node_start     { border-color:var(--crawl);   color:#7dd3fc; }
  .log-entry.node_complete  { border-color:var(--agents);  color:#6ee7b7; }
  .log-entry.agent_message  { border-color:var(--rank);    color:#fcd34d; }
  .log-entry.warning        { border-color:var(--accent3); color:var(--accent3); }
  .log-entry.error          { border-color:var(--error);   color:var(--error); }
  .log-entry.done           { border-color:var(--accent);  color:var(--accent); }
  .log-time { color:var(--muted); margin-right:0.5rem; }

  /* Agent comms */
  .agent-panel { background:var(--surface); border:1px solid var(--border); max-height:280px; overflow-y:auto; }
  .agent-header { padding:0.75rem 1rem; font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase; color:var(--muted); border-bottom:1px solid var(--border); position:sticky; top:0; background:var(--surface); }
  .agent-msg    { padding:0.6rem 1rem; border-bottom:1px solid var(--border); font-size:0.7rem; animation:fadeSlide 0.3s ease; }
  .agent-route  { display:flex; align-items:center; gap:0.4rem; margin-bottom:0.25rem; font-size:0.62rem; color:var(--muted); }
  .agent-name   { padding:0.1rem 0.4rem; font-size:0.6rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; }
  .agent-name.orchestrator    { background:rgba(0,229,255,0.1);   color:var(--accent); }
  .agent-name.ranking_engine  { background:rgba(245,158,11,0.12); color:var(--rank); }
  .agent-name.metrics_evaluator{ background:rgba(139,92,246,0.1); color:var(--graph); }
  .agent-name.intent_parser   { background:rgba(99,102,241,0.1);  color:#818cf8; }
  .agent-name.crawler_agent   { background:rgba(6,182,212,0.1);   color:var(--crawl); }
  .agent-name.validator       { background:rgba(16,185,129,0.1);  color:var(--agents); }
  .agent-content { color:var(--text); line-height:1.4; }
  .chip-row { display:flex; flex-wrap:wrap; gap:0.3rem; margin-top:0.35rem; }
  .chip { padding:0.1rem 0.4rem; background:rgba(255,255,255,0.05); border:1px solid var(--border); font-size:0.6rem; color:var(--muted); }

  /* Ranking section */
  .ranking-section { margin-top:2.5rem; }
  .ranking-title { font-family:'Syne',sans-serif; font-weight:800; font-size:1.4rem; letter-spacing:-0.03em; }
  .ranking-meta  { font-size:0.65rem; color:var(--muted); }
  .ranking-rationale {
    font-size:0.72rem; color:var(--muted); max-width:700px; line-height:1.6;
    margin-bottom:1.25rem; padding:0.75rem 1rem;
    border-left:2px solid var(--accent2); background:rgba(124,58,237,0.05);
  }
  .criteria-row { display:flex; flex-wrap:wrap; gap:0.5rem; margin-bottom:1.5rem; }
  .criterion-chip { display:flex; align-items:center; gap:0.4rem; padding:0.35rem 0.75rem; background:var(--surface); border:1px solid var(--border); font-size:0.68rem; }
  .criterion-weight { color:var(--rank); font-weight:600; }
  .criterion-dir    { color:var(--muted); font-size:0.58rem; }

  .insights-panel {
    margin-top: 1.5rem;
    padding: 1rem;
    border: 1px solid var(--border);
    background: var(--surface);
  }
  .insights-title {
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent2);
    margin-bottom: 0.6rem;
  }
  .insights-summary {
    font-size: 0.74rem;
    line-height: 1.6;
    color: var(--text);
    margin-bottom: 0.85rem;
  }
  .insight-item {
    padding: 0.65rem 0.7rem;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.02);
    margin-bottom: 0.5rem;
  }
  .insight-head {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    margin-bottom: 0.3rem;
  }
  .insight-title {
    font-size: 0.72rem;
    color: var(--accent);
  }
  .insight-confidence {
    font-size: 0.58rem;
    color: var(--muted);
  }
  .insight-finding {
    font-size: 0.7rem;
    line-height: 1.45;
    color: var(--text);
    margin-bottom: 0.35rem;
  }
  .insight-evidence {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
  }
  .insight-source {
    font-size: 0.6rem;
    border: 1px solid var(--border);
    color: var(--crawl);
    text-decoration: none;
    padding: 0.12rem 0.4rem;
  }
  .insight-source:hover {
    border-color: var(--crawl);
  }

  /* Table */
  .table-wrap { overflow-x:auto; border:1px solid var(--border); }
  table  { width:100%; border-collapse:collapse; font-size:0.75rem; }
  thead  { background:#0a0e18; }
  th {
    padding:0.75rem 1rem; text-align:left; font-size:0.6rem;
    letter-spacing:0.15em; text-transform:uppercase; color:var(--muted);
    border-bottom:1px solid var(--border); white-space:nowrap;
  }
  th.rank-col   { color:var(--rank);   width:48px; }
  th.score-col  { color:var(--accent); }
  th.crit-col   { color:var(--accent); }
  tr:hover td   { background:rgba(255,255,255,0.02); }
  td { padding:0.75rem 1rem; border-bottom:1px solid var(--border); vertical-align:top; }

  .rank-badge { display:inline-flex; align-items:center; justify-content:center; width:28px; height:28px; font-family:'Syne',sans-serif; font-weight:800; font-size:0.85rem; }
  .rank-badge.r1 { background:var(--rank);   color:#000; }
  .rank-badge.r2 { background:#94a3b8;       color:#000; }
  .rank-badge.r3 { background:#92400e;       color:#fff; }
  .rank-badge.rn { background:var(--border); color:var(--muted); }

  .entity-name { font-family:'Syne',sans-serif; font-weight:700; font-size:0.85rem; }
  .entity-type { font-size:0.58rem; color:var(--accent2); margin-top:0.15rem; }
  .entity-desc { font-size:0.58rem; color:var(--muted); margin-top:0.15rem; max-width:220px; line-height:1.4; }

  .score-cell { min-width:120px; }
  .score-bar  { display:flex; align-items:center; gap:0.5rem; }
  .score-track{ flex:1; height:3px; background:var(--border); max-width:80px; }
  .score-fill { height:100%; background:var(--accent); transition:width 0.8s ease; }
  .score-val  { font-size:0.68rem; color:var(--accent); min-width:2.5rem; }

  .crit-bars { margin-top:0.35rem; display:flex; flex-direction:column; gap:0.2rem; }
  .cbar-row  { display:flex; align-items:center; gap:0.3rem; font-size:0.54rem; color:var(--muted); }
  .cbar-label{ width:56px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .cbar-track{ flex:1; height:2px; background:var(--border); border-radius:1px; }
  .cbar-fill { height:100%; background:var(--accent2); border-radius:1px; }
  .cbar-pct  { min-width:24px; text-align:right; }

  .metric-val  { color:var(--text); }
  .metric-crit { color:var(--accent); font-weight:500; }
  .metric-miss { color:var(--muted); font-size:0.65rem; }

  .source-link { color:var(--crawl); text-decoration:none; font-size:0.65rem; display:block; }
  .source-link:hover { text-decoration:underline; }
  .source-more { font-size:0.55rem; color:var(--muted); margin-top:0.15rem; }

  .sources-panel { margin-top:1.5rem; padding:1rem; background:var(--surface); border:1px solid var(--border); }
  .sources-title { font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase; color:var(--accent2); font-weight:600; margin-bottom:0.6rem; }
  .sources-chips { display:flex; flex-wrap:wrap; gap:0.4rem; }
  .source-chip { font-size:0.6rem; color:var(--accent); text-decoration:none; padding:0.2rem 0.5rem; border:1px solid var(--border); font-family:var(--mono); transition:border-color 0.15s; }
  .source-chip:hover { border-color:var(--accent); }

  .empty-state { display:flex; flex-direction:column; align-items:center; justify-content:center; padding:4rem 2rem; text-align:center; gap:1rem; border:1px dashed var(--border); }
  .empty-glyph { font-size:3rem; opacity:0.15; font-family:'Syne',sans-serif; font-weight:800; }
  .empty-text  { font-size:0.75rem; color:var(--muted); max-width:400px; line-height:1.6; }

  .spinner { display:inline-block; width:12px; height:12px; border:2px solid var(--border); border-top-color:var(--accent); border-radius:50%; animation:spin 0.7s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
`;

// ── Helpers ───────────────────────────────────────────────────
function fmtTime(ts) {
  if (!ts) return "";
  const d = new Date(ts);
  return d.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function logLabel(ev) {
  switch (ev.type) {
    case "phase_start":    return `▶ ${ev.label || ev.phase}`;
    case "phase_complete": return `✓ ${ev.label || ev.phase}`;
    case "node_start":     return `↻ ${ev.node} — ${ev.label || ""}`;
    case "node_complete":  return `✓ ${ev.node}${ev.count != null ? ` (${ev.count})` : ""} — ${ev.label || ""}`;
    case "agent_message":  return `⟶ ${ev.from} → ${ev.to}: ${ev.content}`;
    case "insight_ready":  return `✶ insights ready (${(ev.items || []).length})`;
    case "warning":        return `⚠ ${ev.message}`;
    case "error":          return `✗ ${ev.message}`;
    case "done":           return `✦ Pipeline complete`;
    default:               return JSON.stringify(ev);
  }
}

function getHostname(url) {
  try { return new URL(url).hostname.replace("www.", ""); } catch { return url; }
}

// ── App ───────────────────────────────────────────────────────
export default function App() {
  const [query,       setQuery]       = useState("");
  const [topN,        setTopN]        = useState(10);
  const [jobId,       setJobId]       = useState(null);
  const [jobStatus,   setJobStatus]   = useState("idle");
  const [events,      setEvents]      = useState([]);
  const [agentMsgs,   setAgentMsgs]   = useState([]);
  const [nodeStates,  setNodeStates]  = useState({});
  const [rankedTable, setRankedTable] = useState(null);
  const [insights,    setInsights]    = useState({ summary: "", items: [], metadata: {} });
  const [error,       setError]       = useState(null);

  const logRef   = useRef(null);
  const agentRef = useRef(null);
  const esRef    = useRef(null);

  useEffect(() => { if (logRef.current)   logRef.current.scrollTop   = logRef.current.scrollHeight; },   [events]);
  useEffect(() => { if (agentRef.current) agentRef.current.scrollTop = agentRef.current.scrollHeight; }, [agentMsgs]);

  const handleEvent = useCallback((ev) => {
    setEvents(prev => [...prev, ev]);
    if (ev.type === "node_start")    setNodeStates(p => ({ ...p, [ev.node]: "active" }));
    if (ev.type === "node_complete") setNodeStates(p => ({ ...p, [ev.node]: "done" }));
    if (ev.type === "insight_ready") {
      setInsights({
        summary: ev.summary || "",
        items: Array.isArray(ev.items) ? ev.items : [],
        metadata: ev.metadata || {},
      });
      setNodeStates(p => ({ ...p, insights_generator: "done" }));
    }
    if (ev.type === "agent_message") {
      setAgentMsgs(prev => [...prev, ev]);
      if (ev.from) setNodeStates(p => ({ ...p, [ev.from]: "done" }));
    }
    if (ev.type === "done")  setJobStatus("completed");
    if (ev.type === "error") { setJobStatus("failed"); setError(ev.message); }
  }, []);

  const startPipeline = async () => {
    if (!query.trim() || jobStatus === "running") return;
    setJobStatus("running");
    setEvents([]); setAgentMsgs([]); setNodeStates({});
    setRankedTable(null); setInsights({ summary: "", items: [], metadata: {} }); setError(null); setJobId(null);

    try {
      const res  = await fetch(`${API}/crawl/rank`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_n: topN }),
      });
      const data = await res.json();
      const id   = data.job_id;
      setJobId(id);

      if (esRef.current) esRef.current.close();
      const es = new EventSource(`${API}/crawl/rank/${id}/stream`);
      esRef.current = es;

      ["phase_start","phase_complete","node_start","node_complete",
       "agent_message","insight_ready","warning","error","done","status"].forEach(type => {
        es.addEventListener(type, e => handleEvent({ ...JSON.parse(e.data), type }));
      });

      const fetchResult = (finalId) =>
        fetch(`${API}/crawl/rank/${finalId}`)
          .then(r => r.json())
          .then(result => {
            setRankedTable(normaliseRanking(result.ranking_result || result.ranked_table));
            setInsights(result.insights || { summary: "", items: [], metadata: {} });
            setJobStatus(result.status || "completed");
          });

      es.addEventListener("status", e => {
        const ev = JSON.parse(e.data);
        if (ev.status === "completed" || ev.status === "failed") {
          fetchResult(id);
          es.close();
        }
      });

      es.onerror = () => { fetchResult(id); es.close(); };

    } catch (err) {
      setError(err.message);
      setJobStatus("failed");
    }
  };

  const ns = id => nodeStates[id] || "pending";

  // Compute display columns: criteria columns first, then extras, max 8
  const tableColumns = rankedTable ? (() => {
    const critKeys = (rankedTable.criteria || []).map(c => c.column);
    const extra    = new Set();
    rankedTable.rows.forEach(r => Object.keys(r.fields || {}).forEach(k => {
      if (!critKeys.includes(k)) extra.add(k);
    }));
    return [...critKeys, ...[...extra]].slice(0, 8);
  })() : [];

  const allSources = rankedTable
    ? [...new Set(rankedTable.rows.flatMap(r => r.source_urls || []))].filter(Boolean)
    : [];

  return (
    <>
      <style>{css}</style>
      <div className="grain" />
      <div className="app">

        {/* Header */}
        <header className="header">
          <div className="logo-mark">WC</div>
          <div>
            <div className="header-title">WebCrawler Intelligence</div>
            <div className="header-sub">Multi-agent ranking pipeline · Neo4j · LangGraph</div>
          </div>
        </header>

        {/* Query input */}
        <div className="query-label">Research Question</div>
        <div className="query-row">
          <input
            className="query-input"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === "Enter" && startPipeline()}
            placeholder="e.g. Rank top English movies by IMDB rating"
          />
          <div className="topn-wrap">
            <span className="topn-label">Top N</span>
            <input
              className="topn-input"
              type="number"
              min={1}
              max={100}
              value={topN}
              onChange={e => setTopN(Math.max(1, Math.min(100, parseInt(e.target.value) || 10)))}
              title="How many results to return (1–100)"
            />
          </div>
          <button
            className="run-btn"
            onClick={startPipeline}
            disabled={jobStatus === "running" || !query.trim()}
          >
            {jobStatus === "running" ? "Running…" : "▶ Run"}
          </button>
        </div>

        {/* Status bar */}
        {jobStatus !== "idle" && (
          <div className="status-bar">
            <span className={`status-dot ${jobStatus}`} />
            <span style={{ color: "var(--muted)", fontSize: "0.65rem", letterSpacing: "0.1em", textTransform: "uppercase" }}>
              {jobStatus === "running" ? "Pipeline executing" : jobStatus === "completed" ? "Complete" : "Failed"}
            </span>
            {jobId && <span style={{ marginLeft: "auto", color: "var(--muted)", fontSize: "0.6rem" }}>job #{jobId}</span>}
            {error  && <span style={{ color: "var(--error)", fontSize: "0.68rem" }}>⚠ {error}</span>}
          </div>
        )}

        {/* Pipeline grid */}
        {jobStatus !== "idle" && (
          <div className="main-grid">

            {/* Left: node list */}
            <div className="pipeline-panel">
              <div className="panel-title">Pipeline Nodes</div>

              <div className="phase-group">
                <div className="phase-label crawl">01 · Crawl Layer</div>
                {CRAWL_NODES.map(n => (
                  <div key={n.id} className={`node-item ${ns(n.id)}`}>
                    <span className="node-icon">{ns(n.id) === "active" ? <span className="spinner" /> : n.icon}</span>
                    <span>{n.label}</span>
                    <span className="node-status">{ns(n.id) === "done" ? "✓" : ns(n.id) === "active" ? "…" : "·"}</span>
                  </div>
                ))}
              </div>

              <div className="phase-group">
                <div className="phase-label graph">02 · Graph Layer</div>
                {GRAPH_NODES.map(n => (
                  <div key={n.id} className={`node-item ${ns(n.id)}`}>
                    <span className="node-icon">{ns(n.id) === "active" ? <span className="spinner" /> : n.icon}</span>
                    <span>{n.label}</span>
                    <span className="node-status">{ns(n.id) === "done" ? "✓" : ns(n.id) === "active" ? "…" : "·"}</span>
                  </div>
                ))}
              </div>

              <div className="phase-group">
                <div className="phase-label agents">03 · Ranking Engine</div>
                {RANK_NODES.map(n => (
                  <div key={n.id} className={`node-item ${ns(n.id)}`}>
                    <span className="node-icon">{ns(n.id) === "active" ? <span className="spinner" /> : "⟳"}</span>
                    <div>
                      <div style={{ fontSize: "0.72rem" }}>{n.label}</div>
                      <div style={{ fontSize: "0.56rem", color: "var(--muted)", lineHeight: 1.4, marginTop: "0.1rem" }}>{n.role}</div>
                    </div>
                    <span className="node-status">{ns(n.id) === "done" ? "✓" : ns(n.id) === "active" ? "…" : "·"}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Right: logs */}
            <div className="right-panel">

              <div className="log-panel">
                <div className="log-header">Pipeline Log ({events.length} events)</div>
                <div className="log-body" ref={logRef}>
                  {events.map((ev, i) => (
                    <div key={i} className={`log-entry ${ev.type}`}>
                      <span className="log-time">{fmtTime(ev.timestamp)}</span>
                      {logLabel(ev)}
                    </div>
                  ))}
                  {jobStatus === "running" && events.length === 0 && (
                    <div className="log-entry" style={{ color: "var(--muted)" }}>
                      <span className="spinner" style={{ marginRight: "0.5rem" }} />
                      connecting to pipeline…
                    </div>
                  )}
                </div>
              </div>

              <div className="agent-panel">
                <div className="agent-header">Agent Communication Log ({agentMsgs.length})</div>
                <div ref={agentRef}>
                  {agentMsgs.length === 0 && (
                    <div style={{ padding: "1.25rem 1rem", fontSize: "0.68rem", color: "var(--muted)" }}>
                      Agent messages will appear here as the pipeline runs…
                    </div>
                  )}
                  {agentMsgs.map((msg, i) => (
                    <div key={i} className="agent-msg">
                      <div className="agent-route">
                        <span className={`agent-name ${msg.from}`}>{msg.from}</span>
                        <span>→</span>
                        <span className={`agent-name ${msg.to}`}>{msg.to}</span>
                        <span style={{ marginLeft: "auto", color: "var(--muted)", fontSize: "0.58rem" }}>{fmtTime(msg.timestamp)}</span>
                      </div>
                      <div className="agent-content">{msg.content}</div>
                      {msg.columns?.length > 0 && (
                        <div className="chip-row">
                          {msg.columns.slice(0, 10).map(c => <span key={c} className="chip">{c}</span>)}
                          {msg.columns.length > 10 && <span className="chip">+{msg.columns.length - 10} more</span>}
                        </div>
                      )}
                      {msg.criteria?.length > 0 && (
                        <div className="chip-row">
                          {msg.criteria.map(c => (
                            <span key={c.column} className="chip">{c.column} {(c.weight * 100).toFixed(0)}%</span>
                          ))}
                        </div>
                      )}
                      {msg.missing?.length > 0 && (
                        <div className="chip-row">
                          {msg.missing.map(m => <span key={m} className="chip">missing: {m}</span>)}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

            </div>
          </div>
        )}

        {/* Idle empty state */}
        {jobStatus === "idle" && (
          <div className="empty-state">
            <div className="empty-glyph">WC</div>
            <div className="empty-text">
              Type a ranking question above and press Run. The pipeline crawls the web,
              builds a Neo4j knowledge graph, extracts entities and triples, then uses
              a weighted min-max algorithm to produce a scored comparison table.
            </div>
          </div>
        )}

        {(insights.summary || (Array.isArray(insights.items) && insights.items.length > 0)) && (
          <section className="insights-panel">
            <div className="insights-title">
              Explainability Insights ({insights.items?.length || 0})
            </div>
            {insights.summary && <div className="insights-summary">{insights.summary}</div>}
            {(insights.items || []).map((item, idx) => (
              <div className="insight-item" key={`${item.title || "insight"}-${idx}`}>
                <div className="insight-head">
                  <div className="insight-title">{item.title || "Insight"}</div>
                  {typeof item.confidence === "number" && (
                    <div className="insight-confidence">conf {(item.confidence * 100).toFixed(0)}%</div>
                  )}
                </div>
                <div className="insight-finding">{item.finding || ""}</div>
                {Array.isArray(item.evidence) && item.evidence.length > 0 && (
                  <div className="insight-evidence">
                    {item.evidence
                      .map(ev => ev?.source_url)
                      .filter(Boolean)
                      .map((url, i) => (
                        <a
                          key={`${url}-${i}`}
                          className="insight-source"
                          href={url}
                          target="_blank"
                          rel="noreferrer"
                          title={url}
                        >
                          ↗ {getHostname(url)}
                        </a>
                      ))}
                  </div>
                )}
              </div>
            ))}
          </section>
        )}

        {/* ── Ranking Results ── */}
        {rankedTable && rankedTable.rows?.length > 0 && (
          <section className="ranking-section">

            <div style={{ display: "flex", alignItems: "baseline", gap: "1rem", marginBottom: "1rem", paddingBottom: "1rem", borderBottom: "1px solid var(--border)" }}>
              <h2 className="ranking-title">Ranking Results</h2>
              <span style={{ fontSize: "0.6rem", color: "var(--rank)", border: "1px solid var(--rank)", padding: "0.15rem 0.5rem", letterSpacing: "0.08em" }}>
                TOPSIS 55% + BORDA 35% + COMPLETENESS 10%
              </span>
              <span className="ranking-meta">
                {rankedTable.rows.length} entities
                {rankedTable.session_id ? ` · session #${rankedTable.session_id.slice(0, 8)}` : ""}
              </span>
            </div>

            {rankedTable.ranking_rationale && (
              <div className="ranking-rationale">⟐ {rankedTable.ranking_rationale}</div>
            )}

            {rankedTable.criteria?.length > 0 && (
              <div className="criteria-row">
                {rankedTable.criteria.map(c => (
                  <div key={c.column} className="criterion-chip">
                    <span style={{ color: "var(--text)" }}>{c.column}</span>
                    <span className="criterion-weight">{(c.weight * 100).toFixed(0)}%</span>
                    <span className="criterion-dir">{c.higher_is_better ? "↑ higher=better" : "↓ lower=better"}</span>
                  </div>
                ))}
              </div>
            )}

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th className="rank-col">#</th>
                    <th>Entity</th>
                    <th className="score-col">Score</th>
                    {tableColumns.map(k => {
                      const isCrit = rankedTable.criteria?.some(c => c.column === k);
                      return <th key={k} className={isCrit ? "crit-col" : ""}>{k}</th>;
                    })}
                    <th>Sources</th>
                  </tr>
                </thead>
                <tbody>
                  {rankedTable.rows.map((row, i) => {
                    const rankClass = i === 0 ? "r1" : i === 1 ? "r2" : i === 2 ? "r3" : "rn";
                    const fields    = row.fields || {};
                    const sources   = row.source_urls || [];

                    return (
                      <tr key={row.entity_name + i}>

                        {/* Rank badge */}
                        <td><span className={`rank-badge ${rankClass}`}>{row.rank}</span></td>

                        {/* Entity info */}
                        <td>
                          <div className="entity-name">{row.entity_name}</div>
                          {row.entity_type && row.entity_type !== "Entity" && (
                            <div className="entity-type">{row.entity_type}</div>
                          )}
                          {row.description && (
                            <div className="entity-desc">
                              {row.description.length > 110
                                ? row.description.slice(0, 110) + "…"
                                : row.description}
                            </div>
                          )}
                        </td>

                        {/* Composite score + algorithm breakdown + criterion bars */}
                        <td className="score-cell">
                          <div className="score-bar">
                            <div className="score-track">
                              <div className="score-fill" style={{ width: `${(row.composite_score * 100).toFixed(1)}%` }} />
                            </div>
                            <span className="score-val">{row.composite_score.toFixed(3)}</span>
                          </div>
                          {/* Algorithm breakdown: TOPSIS / Borda / Completeness */}
                          {row.algorithm_scores && Object.keys(row.algorithm_scores).length > 0 && (
                            <div className="crit-bars" style={{ marginTop: "0.4rem" }}>
                              {[
                                { key: "topsis",       label: "TOPSIS",   color: "#00e5ff", pct: 55 },
                                { key: "borda",        label: "Borda",    color: "#7c3aed", pct: 35 },
                                { key: "completeness", label: "Complete", color: "#10b981", pct: 10 },
                              ].map(({ key, label, color, pct }) => {
                                const val = row.algorithm_scores[key] ?? 0;
                                return (
                                  <div key={key} className="cbar-row">
                                    <span className="cbar-label" style={{ color, fontSize: "0.52rem" }}>{label} {pct}%</span>
                                    <div className="cbar-track">
                                      <div className="cbar-fill" style={{ width: `${(val * 100).toFixed(0)}%`, background: color }} />
                                    </div>
                                    <span className="cbar-pct">{(val * 100).toFixed(0)}%</span>
                                  </div>
                                );
                              })}
                            </div>
                          )}
                          {/* Per-criterion scores */}
                          {Object.keys(row.criterion_scores || {}).length > 0 && (
                            <div className="crit-bars">
                              {Object.entries(row.criterion_scores).map(([col, score]) => (
                                <div key={col} className="cbar-row">
                                  <span className="cbar-label">{col}</span>
                                  <div className="cbar-track">
                                    <div className="cbar-fill" style={{ width: `${(score * 100).toFixed(0)}%` }} />
                                  </div>
                                  <span className="cbar-pct">{(score * 100).toFixed(0)}%</span>
                                </div>
                              ))}
                            </div>
                          )}
                        </td>

                        {/* Data columns */}
                        {tableColumns.map(k => {
                          const v      = fields[k];
                          const empty  = !v || v === "null" || v === "N/A" || v === "none" || v === "None";
                          const isCrit = rankedTable.criteria?.some(c => c.column === k);
                          return (
                            <td key={k}>
                              {empty
                                ? <span className="metric-miss">—</span>
                                : <span className={isCrit ? "metric-crit" : "metric-val"}>{String(v)}</span>
                              }
                            </td>
                          );
                        })}

                        {/* Sources per entity */}
                        <td>
                          {sources.length > 0 ? (
                            <>
                              {sources.slice(0, 3).map((url, si) => (
                                <a key={si} className="source-link" href={url} target="_blank" rel="noreferrer" title={url}>
                                  ↗ {getHostname(url)}
                                </a>
                              ))}
                              {sources.length > 3 && <div className="source-more">+{sources.length - 3} more</div>}
                            </>
                          ) : <span className="metric-miss">—</span>}
                        </td>

                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* All sources panel */}
            {allSources.length > 0 && (
              <div className="sources-panel">
                <div className="sources-title">Sources Crawled ({allSources.length})</div>
                <div className="sources-chips">
                  {allSources.map((url, i) => (
                    <a key={i} className="source-chip" href={url} target="_blank" rel="noreferrer" title={url}>
                      ↗ {getHostname(url)}
                    </a>
                  ))}
                </div>
              </div>
            )}

          </section>
        )}

        {/* Completed but empty */}
        {jobStatus === "completed" && (!rankedTable || !rankedTable.rows?.length) && (
          <div className="empty-state" style={{ marginTop: "2rem" }}>
            <div className="empty-glyph">∅</div>
            <div className="empty-text">
              Pipeline completed but produced no ranking data. The crawl may not have found
              enough structured entities. Try a more specific question.
            </div>
          </div>
        )}

      </div>
    </>
  );
}