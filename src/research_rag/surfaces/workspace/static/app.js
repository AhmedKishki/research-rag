"use strict";

const state = {
  profile: null,
  status: null,
  sources: [],
  excludedSources: [],
  sourcesReady: false,
  sourcePages: { sources: 1, excluded: 1 },
  projects: [],
  currentProject: "",
  agentEntry: "",
  memory: null,
  memoryRounds: new Map(),
  memoryStanding: new Map(),
  standingScope: null,
  busy: false,
  forceRecompute: false,
  settingsRevision: "",
  settings: new Map(),
  chunkExclusions: new Map(),
  pendingSettings: null,
  updates: { payload: null, declinedVersion: "" },
};

const byId = (id) => document.getElementById(id);

function hasCapability(name) {
  return Boolean(state.profile?.capabilities?.[name]);
}

function applyProfile(profile) {
  state.profile = profile;
  document.title = profile.application_name;
  byId("application-name").textContent = profile.application_name;
  byId("project-label").textContent = profile.project_label;
  byId("workspace-nav").setAttribute("aria-label", profile.navigation_label);
  byId("sidebar-project-label").textContent = profile.project_label;
  byId("ingest-intro").textContent = profile.ingest_intro;
  // Whether a build can be continued is the host's own fact about its own
  // pipeline, so the sentence that says so arrives with the rest of its wording
  // and a host that makes no claim shows none.
  const ingestNote = byId("ingest-note");
  ingestNote.textContent = profile.ingest_resume_note || "";
  ingestNote.hidden = !profile.ingest_resume_note;
  // An empty footer removes the element rather than leaving an empty band: the
  // quotation rule belongs in the documentation, not in every view.
  const footer = byId("footer-text");
  footer.textContent = profile.footer_text || "";
  footer.hidden = !profile.footer_text;
  byId("bundle-import-intro").textContent = profile.bundle_import_intro;
  byId("memory-tab-label").textContent = profile.memory_label;
  byId("memory-heading").textContent = profile.memory_label;
  const memoryNote = byId("memory-note");
  memoryNote.textContent = profile.memory_note || "";
  memoryNote.hidden = !profile.memory_note;
  const versionLabel = byId("version-label");
  versionLabel.textContent = profile.version_label || "";
  versionLabel.hidden = !profile.version_label;
  // Only a host that serves MCP clients knows how one of them is named, so the
  // note arrives with the rest of this host's wording.
  const clientNaming = byId("client-naming");
  clientNaming.textContent = profile.client_naming_hint || "";
  clientNaming.hidden = !profile.client_naming_hint;
  document.querySelectorAll("[data-capability]").forEach((element) => {
    element.hidden = !hasCapability(element.dataset.capability);
  });
  // The build button names the build the corpus needs, so it waits for the
  // status that says which one that is.
  byId("ingest-button").hidden = true;
  const visibleNavItems = [...document.querySelectorAll(".sidebar-item")].filter(
    (item) => !item.hidden,
  );
  // The filter section is a group of capability-gated fields, so it goes when
  // the last of them goes: a heading over nothing is a heading a reader stops on.
  const filterSection = byId("filter-section");
  filterSection.hidden = !filterSection.querySelector(
    "[data-capability]:not([hidden])",
  );
  byId("filter-fields").hidden = filterSection.hidden;
  // A nav item is one panel's way in, so a sidebar holding none is a rule above
  // nothing and goes with them. The active view is chosen from the items that
  // survived the profile, and a profile that leaves none shows no panel at all
  // rather than the panel of an item that is gone.
  byId("workspace-nav").hidden = !visibleNavItems.length;
  byId("nav-toggle").hidden = !visibleNavItems.length;
  const activeItem = visibleNavItems.find((item) => item.classList.contains("is-active"));
  switchView(activeItem ? activeItem.dataset.view : null);
}

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = String(text);
  return element;
}

function button(label, action, value, className = "action-button") {
  const element = node("button", className, label);
  element.type = "button";
  element.dataset.action = action;
  if (value !== undefined) element.dataset.value = value;
  return element;
}

function listValue(value) {
  return String(value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function readableText(value) {
  return String(value || "")
    .replace(/\r\n?/g, "\n")
    .replace(/\u00ad/g, "")
    .replace(/([A-Za-z])-\s*\n\s*([a-z])/g, "$1$2")
    .split(/\n\s*\n+/)
    .map((paragraph) => paragraph.replace(/\s*\n\s*/g, " ").replace(/[\t ]+/g, " ").trim())
    .filter(Boolean)
    .join("\n\n");
}

function inlineText(value) {
  return readableText(value).replace(/\s+/g, " ").trim();
}

function formatNumber(value) {
  return new Intl.NumberFormat().format(Number(value || 0));
}

function formatDate(value) {
  if (!value) return "Not yet created";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) return String(value);
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(parsed);
}

function compactId(value) {
  const text = String(value || "");
  if (text.length <= 18) return text;
  return `${text.slice(0, 9)}…${text.slice(-6)}`;
}

// A byline says who wrote a source and when, and only what it actually has: a
// source with no reviewed year shows the authors alone rather than an empty
// separator, and one with neither says so in words instead of printing a
// placeholder the reader has to recognise as missing.
function authorLine(source) {
  const authors = Array.isArray(source.authors)
    ? source.authors.map(inlineText).filter(Boolean)
    : [];
  const parts = [];
  if (authors.length) parts.push(authors.join("; "));
  const year = source.year ?? source.publication_year;
  if (year !== null && year !== undefined && String(year).trim()) {
    parts.push(String(year));
  }
  if (parts.length) return parts.join(" · ");
  return source.authors === undefined && source.year === undefined
    ? "Authorship and year not reviewed"
    : "Authorship not reviewed";
}

// A locator says where in a source a passage sits. The server may name its kind,
// may name only a page, or may hand over a string it already formatted; whatever
// it sends, the label reports what is there. It never guesses a format the
// payload did not claim, because a section index labelled for one format is a
// statement about the reader's source that nothing in the payload supports.
function locatorLabel(locator) {
  if (!locator) return "Source passage";
  if (typeof locator === "string") return locator.trim() || "Source passage";
  const pageData = locator.page_data || {};
  const page = locator.page_label ?? locator.page ?? pageData.page_label ?? pageData.page;
  const hasPage = page !== null && page !== undefined && String(page).trim() !== "";
  if (locator.type === "pdf_page" || (locator.type === undefined && hasPage)) {
    return `Page ${hasPage ? page : "?"}`;
  }
  const section = locator.section_title || pageData.section_title || locator.href;
  if (section) return inlineText(section);
  const index = locator.section_index ?? pageData.section_index;
  if (index !== null && index !== undefined && String(index).trim() !== "") {
    return `Section ${index}`;
  }
  return "Passage location not reported";
}

function sourceForDocument(documentId) {
  return state.sources.find((source) => source.document_id === documentId) || null;
}

function setConnection(kind, label, detail = "") {
  const element = byId("connection-state");
  element.dataset.state = kind;
  element.lastElementChild.textContent = label;
  element.title = detail;
}

// The header says whether the app answers and whether its own checks found
// anything to act on. A degraded project still answers, so "ready" alone hid a
// warning the status payload carried; the count sends the reader to Status.
function connectionHealth(status) {
  const blocked = (status?.blocked_by || []).length;
  const warnings = (status?.degraded || []).length;
  const reasons = [...(status?.blocked_by || []), ...(status?.degraded || [])]
    .map((entry) => inlineText(entry.reason))
    .filter(Boolean)
    .join("\n");
  if (blocked) return ["blocked", `Local · ${blocked} blocked`, reasons];
  if (warnings) {
    return ["warn", `Local · ${warnings} warning${warnings === 1 ? "" : "s"}`, reasons];
  }
  return ["ready", "Local · ready", ""];
}

function setBusy(active, message = "Working…") {
  state.busy = active;
  byId("busy-bar").hidden = !active;
  byId("busy-message").textContent = message;
  document.querySelectorAll("button[type='submit']").forEach((element) => {
    element.disabled = active;
  });
  byId("refresh-button").disabled = active;
  byId("ingest-button").disabled = active || !hasCapability("ingestion");
  byId("rebuild-button").disabled = active || !hasCapability("force_recompute");
  byId("export-button").disabled = active || !hasCapability("bundle_export");
  byId("import-button").disabled = active || !hasCapability("bundle_import");
  // The run button is a submit, so re-enabling every submit above would leave it
  // live with nothing typed in it. Its own gate is the scope and the statement.
  syncSqlControls();
  // The same applies to the settings submit: its gate is a changed value.
  syncSettingsSubmit();
}

function toast(message, isError = false) {
  const item = node("div", `toast${isError ? " is-error" : ""}`, message);
  item.setAttribute("role", isError ? "alert" : "status");
  byId("toast-region").append(item);
  window.setTimeout(() => item.remove(), isError ? 7000 : 4200);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch (_error) {
    // A non-JSON response is represented by the HTTP status below.
  }
  if (!response.ok) {
    throw new Error(payload?.error || `Request failed with status ${response.status}`);
  }
  return payload;
}

function changeSummary(changes) {
  if (!changes) return "The source collection differs from the selected generation.";
  const parts = [];
  if (changes.added?.length) parts.push(`${changes.added.length} added`);
  if (changes.removed?.length) parts.push(`${changes.removed.length} removed`);
  if (changes.modified?.length) parts.push(`${changes.modified.length} modified`);
  if (changes.metadata_changed) parts.push("metadata changed");
  if (changes.source_exclusions_changed) parts.push("source inclusion changed");
  return parts.length
    ? `Pending changes: ${parts.join(", ")}. The existing generation remains searchable.`
    : "The source collection differs from the selected generation.";
}

function configureRetrieval(status) {
  const available = new Set(status.available_retrieval_methods || []);
  const radios = [...document.querySelectorAll("input[name='retrieval_method']")];
  radios.forEach((radio) => {
    radio.disabled = status.ready && !available.has(radio.value);
  });
  const checked = radios.find((radio) => radio.checked && !radio.disabled);
  if (!checked) {
    const preferred = radios.find(
      (radio) => radio.value === status.default_retrieval_method && !radio.disabled,
    );
    (preferred || radios.find((radio) => !radio.disabled) || radios[0]).checked = true;
  }
  byId("search-button").disabled = !status.ready || state.busy;
}

// What the server says a client is, beside the name it gave itself: the program
// it runs in, whether it reached the app over ssh, where it was started, and
// which project it asked for. Each part carries its own label rather than being
// joined into one line of separators, because a reader looking for the
// directory was reading for the directory and not for the pid beside it. Every
// part is optional because a client may have declared none of it, and a host
// that sends no facts at all draws the same row it always did.
function clientFacts(client) {
  const identity = client.identity || {};
  const host = identity.host || {};
  const facts = [];
  if (host.program) facts.push(["Program", host.program]);
  if (host.ssh) facts.push(["Reached through", "ssh"]);
  if (host.tmux) facts.push(["Reached through", "tmux"]);
  // A stdio bridge reaches the app through a Python HTTP client, so its
  // user agent names that library rather than the agent behind the pipe. An
  // HTTP client that named nothing is left with its user agent as the one
  // thing that says what it is.
  if (client.transport !== "stdio" && client.user_agent) {
    facts.push(["User agent", client.user_agent]);
  }
  if (identity.cwd) facts.push(["Directory", identity.cwd]);
  if (identity.project) facts.push(["Project", identity.project]);
  if (identity.pid) facts.push(["Process", `pid ${identity.pid}`]);
  return facts;
}

function clientDetail(client) {
  if (client.attached) return `${client.requests || 0} calls`;
  return client.detached_reason || "idle";
}

function factList(pairs, className) {
  const list = node("dl", className);
  for (const [label, value] of pairs) {
    list.append(node("dt", "fact-label", label));
    list.append(node("dd", "fact-value", value));
  }
  return list;
}

function clientRow(client) {
  const card = node("article", "client-row");
  const head = node("div", "client-row-head");
  head.append(node("h4", "client-row-name", client.label || client.name));
  head.append(node("span", "state-badge", clientDetail(client)));
  if (client.attached) {
    const drop = node("button", "text-button", "Disconnect");
    drop.type = "button";
    drop.addEventListener("click", () => disconnectClient(client.session_id));
    head.append(drop);
  }
  card.append(head);
  // The session id is how the same client is named in `clients` and in
  // `disconnect`, and it is the last thing shown rather than the first: a
  // reader looking for an agent reads the agent, and a reader disconnecting one
  // reads the id. One client holds every session it opened, so the count is
  // beside the id rather than one row per session.
  const facts = clientFacts(client);
  facts.push([
    "Session",
    `${client.sessions || 1} open · id ${client.session_id || ""}`,
  ]);
  card.append(factList(facts, "fact-list client-facts"));
  return card;
}

function renderClients(clients) {
  const container = byId("client-chips");
  container.replaceChildren();
  const attached = clients.filter((client) => client.attached).length;
  byId("client-count").textContent = String(clients.length);
  byId("client-detail").textContent = clients.length
    ? `${attached} attached · ${clients.length - attached} idle`
    : "none attached";
  if (!clients.length) {
    const empty = node("p", "form-note", "No agent is attached to this app.");
    container.append(empty);
    return;
  }
  for (const client of clients) {
    container.append(clientRow(client));
  }
}

async function disconnectClient(sessionId) {
  try {
    setBusy(true, "Disconnecting the client…");
    await api(`/api/clients/${encodeURIComponent(sessionId)}/disconnect`, {
      method: "POST",
      body: JSON.stringify({ reason: "Disconnected from the workspace." }),
    });
    toast("Client disconnected.");
    await loadWorkspace();
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function loadClients() {
  if (!hasCapability("clients")) return;
  try {
    const payload = await api("/api/clients");
    renderClients(payload.clients || []);
  } catch (error) {
    // A host that advertises the capability and cannot answer is a workspace
    // fact, not a failure to show the knowledge base, so the panel says so and
    // the rest of the view carries on.
    const container = byId("client-chips");
    container.replaceChildren(node("p", "form-note", error.message));
    byId("client-count").textContent = "—";
    byId("client-detail").textContent = "";
  }
}

function projectOptionLabel(project, isCurrent) {
  const running = project.running ? "app up" : "app not running";
  return isCurrent
    ? `${project.project_name} · ${running} · this project`
    : `${project.project_name} · ${running}`;
}

function selectedProject() {
  return (
    state.projects.find(
      (project) => project.project_name === byId("project-select").value,
    ) || null
  );
}

function showProjectActions() {
  const project = selectedProject();
  const url = project?.url || "";
  const open = byId("project-open");
  open.hidden = !url;
  // The command is the host's own, supplied through the profile, because a
  // project with no app has no address and an invented one would be a command
  // the reader pastes and fails on.
  const template = state.profile?.project_start_command || "";
  const command = template && project
    ? template.replace("{project}", project.project_name)
    : "";
  byId("project-start-command").hidden = !command;
  byId("project-start-command").textContent = command;
  const copy = byId("project-copy-command");
  copy.hidden = !command;
  copy.disabled = !command;
}

function renderProjects(payload) {
  state.projects = payload.projects || [];
  state.currentProject = payload.current || "";
  const select = byId("project-select");
  select.replaceChildren();
  for (const project of state.projects) {
    const option = node(
      "option",
      "",
      projectOptionLabel(project, project.project_name === state.currentProject),
    );
    option.value = project.project_name;
    select.append(option);
  }
  // A host that registered no project leaves the selector disabled rather than
  // offering an empty one a reader could choose from.
  select.disabled = !state.projects.length;
  if (state.projects.some((project) => project.project_name === state.currentProject)) {
    select.value = state.currentProject;
  }
  const message = byId("project-selector-message");
  message.hidden = !payload.message;
  message.textContent = payload.message || "";
  showProjectActions();
}

async function loadProjects() {
  if (!hasCapability("projects")) return;
  renderProjects(await api("/api/projects"));
}

function renderAgentEndpoint(status) {
  const url = String(status.mcp_url || "");
  byId("agent-url").textContent =
    url || "This host reports no MCP endpoint, so it serves no agent surface.";
  byId("agent-url-copy").disabled = !url;
}

// The client entry is the server's bytes, and the copy button sends them exactly
// as received. On screen they are indented, because a configuration pasted from
// a wall of escaped quotes is one a reader cannot check against what it names.
function readableEntry(entry) {
  try {
    return JSON.stringify(JSON.parse(entry), null, 2);
  } catch (_error) {
    return entry;
  }
}

async function loadAgentEntry() {
  if (!hasCapability("agent_entry")) return;
  const payload = await api("/api/agent-entry");
  state.agentEntry = payload.entry || "";
  byId("agent-entry").textContent = readableEntry(state.agentEntry);
  byId("agent-entry-copy").disabled = !state.agentEntry;
}

const METHOD_LABELS = { hybrid: "Hybrid", bm25: "BM25", dense: "Dense" };

function methodLabels(status) {
  return (status.available_retrieval_methods || []).map(
    (method) => METHOD_LABELS[method] || String(method),
  );
}

// The counts and identifiers a reader needs to act are in the cards above. The
// rest — the directory this page serves, the generation identifier in full, the
// methods a search may ask for — is here, labelled and selectable, because it is
// what a reader copies into a report and it was previously squeezed into a
// twelve-pixel line under a number.
function renderStatusFacts(status) {
  const rows = [["Project directory", status.project_root || ""]];
  if (status.source_root) rows.push(["Source directory", status.source_root]);
  if (status.generation_id) {
    rows.push(["Generation identifier", status.generation_id]);
    rows.push(["Built", formatDate(status.created_at)]);
  }
  const methods = methodLabels(status);
  if (methods.length) {
    rows.push([
      "Retrieval methods",
      `${methods.join(", ")} (default ${status.default_retrieval_method || "bm25"})`,
    ]);
  }
  rows.push([
    "Sources on disk",
    `${formatNumber(status.selected_source_count)} selected · ${formatNumber(
      status.excluded_source_count,
    )} excluded`,
  ]);
  rows.push([
    "Excluded passages",
    `${formatNumber(status.excluded_chunk_count)} of ${formatNumber(
      status.chunk_count,
    )} in this generation`,
  ]);
  const formats = status.allowed_formats || [];
  if (formats.length) rows.push(["Accepted formats", formats.join(", ")]);
  byId("status-facts").replaceChildren(factList(rows, "fact-list"));
}

// The build a reader is offered is decided by the corpus, not by the button. An
// ingestion reuses everything whose bytes, policies, and model are unchanged, so
// it is the right build for a new, changed, or outdated corpus and does nothing
// to a current one. A current corpus is offered no build in the header: the
// only one left is a build that reuses nothing, and that is named for its cost
// on the Status view rather than placed in every view as the primary action.
//
// A build that saved a checkpoint resumes only under the parameters it started
// with, so its own force flag is sent back rather than one read off the corpus.
function ingestPlan(status) {
  const progress = status.ingestion_progress;
  if (progress) {
    return { label: "Resume build", force: Boolean(progress.parameters?.force_recompute) };
  }
  if (!status.ready) return { label: "Create generation", force: false };
  if (status.generation_upgrade_required) return { label: "Upgrade generation", force: false };
  if (status.stale) return { label: "Ingest changes", force: false };
  return null;
}

const REBUILD_PLAN = { label: "Rebuild from scratch", force: true };

// The cost of a build that reuses nothing, stated from the counts the server
// reported rather than estimated from them.
function rebuildCost(status) {
  const sources = formatNumber(status?.indexed_source_count ?? status?.selected_source_count);
  const sentences = [
    `Nothing is reused: all ${sources} sources are extracted again and every passage is embedded again.`,
  ];
  const metrics = status?.last_build_metrics || {};
  if (Number.isFinite(metrics.reused_vector_count) && Number.isFinite(metrics.created_vector_count)) {
    sentences.push(
      `The last ingestion reused ${formatNumber(metrics.reused_vector_count)} vectors and created ${formatNumber(metrics.created_vector_count)}.`,
    );
  }
  sentences.push("An ordinary ingestion already rebuilds whatever changed.");
  return sentences.join(" ");
}

function progressSummary(status) {
  const progress = status.ingestion_progress || {};
  const phase = String(progress.phase || "unknown").replaceAll("_", " ");
  const counts = progress.progress || {};
  const reached = Number.isFinite(counts.total) && counts.total > 0
    ? ` at ${formatNumber(counts.completed)} of ${formatNumber(counts.total)} ${String(counts.unit || "").replaceAll("_", " ")}`
    : "";
  const sentences = [
    `It reached the ${phase} phase${reached}.`,
    "Resuming continues from that checkpoint.",
  ];
  if (status.ready) sentences.push("The generation in use stays searchable until the build succeeds.");
  return sentences.join(" ");
}

function openIngest(plan) {
  if (!plan) return;
  state.forceRecompute = plan.force;
  byId("ingest-dialog").querySelector("h2").textContent = plan.label;
  byId("ingest-submit").textContent = plan.label;
  const cost = byId("ingest-cost");
  cost.textContent = plan.force ? rebuildCost(state.status) : "";
  cost.hidden = !plan.force;
  byId("ingest-dialog").showModal();
}

const HEALTH_LABELS = { ok: "Passed", warn: "Warning", blocked: "Blocked", unknown: "Not checked" };

// The two badges a reader must act on are coloured; a passed or unchecked check
// stays grey, as every other badge that says nothing alarming does.
const HEALTH_BADGES = {
  warn: "state-badge state-badge-warning",
  blocked: "state-badge state-badge-blocked",
};

function checkLabel(check) {
  const words = String(check || "check").replaceAll(/[_.]+/g, " ").trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

function healthBadge(stateName) {
  return node(
    "span",
    HEALTH_BADGES[stateName] || "state-badge",
    HEALTH_LABELS[stateName] || HEALTH_LABELS.unknown,
  );
}

// A condition carries its reason and the command that clears it. The command is
// the server's own, shown whole and copied exactly, because a reader pastes it
// into a terminal; the page never composes one.
function healthCondition(entry) {
  const item = node("article", `health-condition health-condition-${entry.state}`);
  const head = node("div", "health-condition-head");
  head.append(healthBadge(entry.state), node("strong", "", checkLabel(entry.check)));
  item.append(head);
  if (entry.reason) item.append(node("p", "health-reason", inlineText(entry.reason)));
  const remedy = entry.remedy || entry.remedy_command;
  if (remedy) {
    const row = node("div", "update-command-row");
    row.append(node("code", "update-command", remedy));
    row.append(button("Copy command", "copy-remedy", remedy, "button"));
    item.append(row);
  }
  return item;
}

// The server's checks are read-only and run on every status read, so the page
// shows what they found: first what needs action, from `blocked_by` and
// `degraded`, then every check behind a disclosure. A host whose status carries
// no checks shows no section.
function renderHealth(status) {
  const section = byId("health-section");
  const checks = Array.isArray(status.checks) ? status.checks : [];
  const conditions = [
    ...(status.blocked_by || []).map((entry) => ({ ...entry, state: "blocked" })),
    ...(status.degraded || []).map((entry) => ({ ...entry, state: "warn" })),
  ];
  const notChecked = (status.not_checked || []).map((entry) =>
    typeof entry === "string" ? { check: entry, state: "unknown" } : { ...entry, state: "unknown" },
  );
  section.hidden = !checks.length && !conditions.length && !notChecked.length;
  if (section.hidden) return;

  byId("health-summary").textContent = conditions.length
    ? `${conditions.length} to act on`
    : "All passed";
  const list = byId("health-conditions");
  list.replaceChildren();
  conditions.forEach((entry) => list.append(healthCondition(entry)));
  if (!conditions.length) {
    list.append(node("p", "form-note", "Every check passed."));
  }

  const rows = [
    ...checks.map((entry) => ({ ...entry, state: entry.state || "unknown" })),
    ...notChecked,
  ];
  const passed = rows.filter((entry) => entry.state === "ok").length;
  byId("health-check-note").textContent = `${passed} of ${rows.length} passed`;
  const all = byId("health-checks");
  all.replaceChildren();
  rows.forEach((entry) => {
    const row = node("li", "health-check");
    row.append(healthBadge(entry.state));
    const body = node("div", "health-check-body");
    body.append(node("strong", "", checkLabel(entry.check)));
    if (entry.reason) body.append(node("p", "health-reason", inlineText(entry.reason)));
    row.append(body);
    all.append(row);
  });
}

function renderStatus(status) {
  state.status = status;
  const projectPath = status.project_root || "";
  const projectName = status.project_name || projectPath.split(/[\\/]/).filter(Boolean).pop()
    || state.profile?.project_fallback_name
    || "Knowledge base";
  byId("project-name").textContent = projectName;
  // The sidebar repeats the project as its quiet footer, so a reader who has
  // scrolled past the header still knows which project the page serves.
  byId("sidebar-project-name").textContent = projectName;
  byId("project-path").textContent = projectPath;
  byId("project-path").title = projectPath;
  byId("searchable-count").textContent = formatNumber(
    status.searchable_source_count ?? status.selected_source_count,
  );
  byId("source-detail").textContent = status.ready
    ? `${formatNumber(status.indexed_source_count)} indexed · ${formatNumber(status.excluded_source_count)} excluded`
    : `${formatNumber(status.selected_source_count)} ready to ingest · ${formatNumber(status.excluded_source_count)} excluded`;
  byId("chunk-count").textContent = status.ready ? formatNumber(status.chunk_count) : "0";
  // The identifier is shown whole. It wraps rather than shortening, because the
  // whole of it is what identifies the generation in a report.
  byId("generation-label").textContent = status.generation_id || "None yet";
  byId("generation-label").title = status.generation_id || "";
  byId("generation-date").textContent = status.created_at
    ? `Built ${formatDate(status.created_at)}`
    : "Not yet created";

  const plan = ingestPlan(status);
  state.ingestPlan = plan;
  const ingestButton = byId("ingest-button");
  ingestButton.hidden = !hasCapability("ingestion") || !plan;
  ingestButton.textContent = plan ? plan.label : "";
  // A rebuild replaces a generation, so there is none to offer before the first
  // build, nor while a checkpointed build is waiting to be resumed.
  byId("rebuild-button").hidden =
    !hasCapability("force_recompute") || !status.ready || Boolean(status.ingestion_progress);

  let indexState = "Not built";
  if (status.ready && status.stale) indexState = "Stale";
  else if (status.hybrid_ready) indexState = "Hybrid ready";
  else if (status.ready) indexState = "BM25 only";
  byId("index-state").textContent = indexState;
  byId("retrieval-detail").textContent = status.ready
    ? methodLabels(status).join(" · ") || "No method available"
    : state.profile?.source_types_label || "document sources";

  const notice = byId("status-notice");
  const noticeAction = byId("notice-action");
  // Every notice that asks for a build offers the build the header offers, so
  // the two controls cannot name different work.
  noticeAction.hidden = !hasCapability("ingestion") || !plan;
  noticeAction.textContent = plan ? plan.label : "";
  noticeAction.dataset.action = "ingest";
  if (status.ingestion_progress) {
    notice.hidden = false;
    byId("status-notice-title").textContent = "A build has a saved checkpoint";
    byId("status-notice-text").textContent = progressSummary(status);
  } else if (!status.ready) {
    notice.hidden = false;
    byId("status-notice-title").textContent = "No generation exists";
    byId("status-notice-text").textContent = status.message || "Create the first knowledge-base generation.";
  } else if (status.generation_upgrade_required) {
    notice.hidden = false;
    byId("status-notice-title").textContent = "This generation needs an upgrade";
    const reasons = (status.upgrade_reasons || []).join(", ").replaceAll("_", " ");
    byId("status-notice-text").textContent = reasons
      ? `Upgrade to apply: ${reasons}. The existing generation remains searchable.`
      : "Upgrade to apply the current extraction and retrieval policies.";
  } else if (status.stale) {
    notice.hidden = false;
    byId("status-notice-title").textContent = "The current generation is stale";
    byId("status-notice-text").textContent = changeSummary(status.changes);
  } else {
    notice.hidden = true;
  }
  // The server's own sentence about the project is always shown. It carries the
  // sentences a card cannot — that reviewed metadata is applied at read time,
  // that some passage exclusions were recorded against another generation — and
  // a panel that drops it when nothing is urgent hides a warning.
  const message = byId("status-message");
  message.textContent = status.message || "";
  message.hidden = !status.message;
  renderHealth(status);
  renderStatusFacts(status);
  configureRetrieval(status);
  renderPartitions(status);
  renderProjectTags(status);
  renderLanguages(status);
  syncFilterPicks();
  renderAgentEndpoint(status);
  if (hasCapability("generations")) renderGenerations(status.generations || []);
  if (hasCapability("sql_console")) renderSqlConsole(status);
}

function tagList(values, className = "tag") {
  const fragment = document.createDocumentFragment();
  (values || []).forEach((value) => fragment.append(node("span", className, value)));
  return fragment;
}

const FILTER_FIELDS = [
  "category-filter",
  "keyword-filter",
  "category-any-filter",
  "project-any-filter",
  "author-filter",
  "title-filter",
  "language-filter",
  "include-source-filter",
  "exclude-source-filter",
];

// The count on the section heading and the sentence on the disclosure summary
// are the same two facts, so both are computed here rather than in one place
// and left to disagree with the other.
function syncFilterSummary() {
  const set = FILTER_FIELDS.filter((field) => listValue(byId(field).value).length);
  byId("filter-count").textContent = String(set.length);
  byId("filter-summary-note").textContent = set.length
    ? `${set.length} filter${set.length === 1 ? "" : "s"} set`
    : "No filter set";
}

// A value typed into a field the disclosure has closed is invisible, so adding a
// value from a list opens the drawer that holds it. A reader who selected one of
// the partitions below then sees where it went.
function revealFilterField(field) {
  const drawer = byId("filter-fields");
  if (drawer && !drawer.open) drawer.open = true;
  const group = byId(field).closest(".filter-group");
  if (group) group.dataset.filled = "true";
}

function addSearchFilter(field, value, label = value) {
  const input = byId(field);
  if (!input || !value) return;
  const values = listValue(input.value);
  if (!values.includes(value)) values.push(value);
  input.value = values.join(", ");
  revealFilterField(field);
  syncFilterSummary();
  input.focus();
  toast(`Added to the search filter: ${label}`);
}

// A source chosen on the Sources view narrows a search, so the reader is taken
// to the search with the filter in view. A toast on a view the filter is not on
// told the reader about a change they could not see.
function searchWithSource(field, sourceId) {
  const source = state.sources.find((entry) => entry.source_id === sourceId);
  switchView("search", { moveFocus: true });
  addSearchFilter(field, sourceId, inlineText(source?.title) || sourceId);
}

// A list offers a choice only when picking from it narrows the search. One value
// every searchable source carries narrows nothing, and a list of none offers
// nothing, so neither is drawn: a heading over a list that cannot change a
// result is a heading a reader stops on.
function inventoryNarrows(entries) {
  if (entries.length !== 1) return entries.length > 1;
  const total = Number(state.status?.searchable_source_count ?? state.status?.selected_source_count);
  const carrying = Number(entries[0].searchable_source_count);
  return !Number.isFinite(total) || !Number.isFinite(carrying) || carrying < total;
}

function renderInventory(containerId, entries, key, action) {
  const container = byId(containerId);
  if (!container) return;
  container.replaceChildren();
  const group = container.closest(".pick-group");
  if (group) {
    group.hidden = !hasCapability(group.dataset.capability) || !inventoryNarrows(entries);
  }
  entries.forEach((item) => {
    const control = button(item[key], action, item[key], "pick-button");
    const count = node("span", "pick-count", formatNumber(item.searchable_source_count));
    count.append(node("span", "visually-hidden", "searchable sources"));
    control.append(count);
    control.title = `Search ${item[key]}`;
    container.append(control);
  });
}

// The lists sit at the foot of the filter drawer under a rule of their own, so
// the rule goes when the last list goes.
function syncFilterPicks() {
  const picks = byId("filter-picks");
  picks.hidden = ![...picks.querySelectorAll(".pick-group")].some((group) => !group.hidden);
}

function renderPartitions(status) {
  renderInventory(
    "partition-chips",
    status.categories || [],
    "category",
    "partition-filter",
  );
}

// The reviewed project tags on the status view, which narrow this project's own
// corpus. They are not the account's projects: those are the header selector,
// and a tag on a source is not a project this installation serves.
function renderProjectTags(status) {
  renderInventory("project-chips", status.projects || [], "project", "project-filter");
}

function renderLanguages(status) {
  renderInventory("language-chips", status.languages || [], "language", "language-filter");
}

const MEMORY_SCOPE_LIMIT = 10;

function memoryScopeLine(entry) {
  return [
    entry.directory,
    `${formatNumber(entry.round_count)} recorded round${entry.round_count === 1 ? "" : "s"}`,
    entry.latest_round_date ? `latest ${entry.latest_round_date}` : "no rounds yet",
  ]
    .filter(Boolean)
    .join(" · ");
}

function memorySubheading(label, id) {
  const row = node("div", "section-heading-row");
  row.append(node("h4", "memory-subheading", label));
  if (id) row.append(node("span", "count-badge", id));
  return row;
}

function memoryRoundCard(round) {
  const card = node("article", "memory-round");
  card.dataset.user = round.user;
  card.dataset.assistant = round.assistant;
  card.dataset.searchText = [round.date, round.time, round.user, round.assistant]
    .map(inlineText)
    .join(" ")
    .toLocaleLowerCase();

  const header = node("div", "result-card-header");
  header.append(node("h4", "result-title", `${round.date} ${round.time}`));
  header.append(node("span", "locator-badge", round.source_file || "Memory round"));
  card.append(header);

  const lines = node("div", "memory-lines");
  [
    ["user", round.user],
    ["assistant", round.assistant],
  ].forEach(([speaker, text]) => {
    const line = node("p", `memory-line memory-line-${speaker}`);
    line.append(node("span", "memory-speaker", speaker));
    line.append(node("span", "memory-text", readableText(text)));
    lines.append(line);
  });
  card.append(lines);

  const actions = node("div", "result-actions");
  actions.append(button("Copy round", "copy-round"));
  card.append(actions);
  return card;
}

function memoryAppendForm(scope) {
  const form = node("form", "memory-append");
  form.dataset.scope = scope;
  form.append(
    node("h4", "memory-subheading", state.profile?.memory_add_label || "Add a round"),
  );
  [
    ["user", "User line"],
    ["assistant", "Assistant line"],
  ].forEach(([field, label]) => {
    const wrapper = node("label", "memory-append-field");
    wrapper.append(node("span", "field-label", label));
    const input = node("textarea", "memory-append-input");
    input.rows = 2;
    input.required = true;
    input.dataset.field = field;
    wrapper.append(input);
    form.append(wrapper);
  });
  const actions = node("div", "dialog-actions");
  const submit = node("button", "button button-primary", "Save round");
  submit.type = "submit";
  actions.append(submit);
  form.append(actions);
  form.addEventListener("submit", appendMemory);
  return form;
}

function memoryRoundList(scope, rounds) {
  const list = node("div", "memory-rounds");
  if (!rounds.rounds?.length) {
    list.append(node("div", "no-records", "No rounds have been recorded in this scope."));
    return list;
  }
  rounds.rounds.forEach((round) => list.append(memoryRoundCard(round)));
  if (rounds.truncated) {
    list.append(
      node(
        "div",
        "form-note",
        `Showing the newest ${formatNumber(rounds.rounds.length)} of ${formatNumber(rounds.round_count)} rounds.`,
      ),
    );
  }
  return list;
}

function memoryScopeCard(entry, rounds, standing) {
  state.memoryRounds.set(entry.scope, rounds);
  state.memoryStanding.set(entry.scope, standing);

  const card = node("article", "memory-scope");
  card.dataset.scope = entry.scope;

  const header = node("div", "memory-scope-header");
  header.append(node("h3", "memory-scope-title", entry.label));
  header.append(
    node("span", "locator-badge", `${formatNumber(entry.round_count)} rounds`),
  );
  card.append(header);
  card.append(node("p", "result-meta", memoryScopeLine(entry)));

  card.append(
    memorySubheading(
      state.profile?.memory_standing_label || "Standing memory",
      standing.sha256 ? `sha256 ${compactId(standing.sha256)}` : "not created yet",
    ),
  );
  const standingActions = node("div", "result-actions");
  standingActions.append(button("Copy standing memory", "copy-standing"));
  if (hasCapability("memory_writes")) {
    standingActions.append(button("Edit standing memory", "edit-standing"));
  }
  card.append(standingActions);
  card.append(node("pre", "standing-document", standing.content || ""));

  card.append(
    memorySubheading(state.profile?.memory_rounds_label || "Recorded rounds"),
  );
  const filter = node("input", "memory-round-filter");
  filter.type = "search";
  filter.placeholder = "Filter by date or text…";
  filter.setAttribute("aria-label", "Filter rounds");
  card.append(filter);

  const list = memoryRoundList(entry.scope, rounds);
  filter.addEventListener("input", () => filterRounds(list, filter.value));
  card.append(list);

  if (hasCapability("memory_writes")) card.append(memoryAppendForm(entry.scope));
  return card;
}

async function loadMemory() {
  if (!hasCapability("memory")) return;
  const status = await api("/api/memory");
  const scopes = status.scopes || [];
  byId("memory-nav-count").textContent = String(scopes.length);
  const shown = scopes.slice(0, MEMORY_SCOPE_LIMIT);
  const loaded = await Promise.all(
    shown.map((entry) =>
      Promise.all([
        api(`/api/memory/rounds?scope=${encodeURIComponent(entry.scope)}&limit=20`),
        api(`/api/memory/standing?scope=${encodeURIComponent(entry.scope)}`),
      ]),
    ),
  );
  const container = byId("memory-scopes");
  container.replaceChildren();
  shown.forEach((entry, index) => {
    container.append(memoryScopeCard(entry, loaded[index][0], loaded[index][1]));
  });
  if (scopes.length > shown.length) {
    container.append(
      node(
        "p",
        "form-note",
        `Showing the first ${formatNumber(shown.length)} of ${formatNumber(scopes.length)} memory scopes.`,
      ),
    );
  }
}

async function refreshMemory({ announce = false } = {}) {
  if (!hasCapability("memory")) return;
  setBusy(true, "Reading memory…");
  try {
    await loadMemory();
    if (announce) toast("Memory refreshed.");
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function appendMemory(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const scope = form.dataset.scope;
  const userMessage = form.querySelector("[data-field='user']").value.trim();
  const assistantMessage = form.querySelector("[data-field='assistant']").value.trim();
  if (!scope || !userMessage || !assistantMessage) {
    toast("A user line and an assistant line are required.", true);
    return;
  }
  setBusy(true, "Saving the round…");
  try {
    await api("/api/memory/append", {
      method: "POST",
      body: JSON.stringify({
        scope,
        user_message: userMessage,
        assistant_message: assistantMessage,
      }),
    });
    await loadMemory();
    toast("Round saved.");
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
  }
}

function openStandingEditor(scope) {
  const standing = state.memoryStanding.get(scope);
  state.standingScope = scope;
  byId("memory-standing-content").value = standing?.content || "";
  byId("memory-standing-dialog").showModal();
}

async function saveStanding(event) {
  event.preventDefault();
  const scope = state.standingScope;
  const content = byId("memory-standing-content").value;
  if (!scope || !content.trim()) {
    toast("Standing memory must not be empty.", true);
    return;
  }
  const body = { scope, content };
  const standing = state.memoryStanding.get(scope);
  if (standing?.sha256) body.expected_sha256 = standing.sha256;
  setBusy(true, "Saving standing memory…");
  try {
    await api("/api/memory/standing", {
      method: "POST",
      body: JSON.stringify(body),
    });
    byId("memory-standing-dialog").close();
    state.standingScope = null;
    await loadMemory();
    toast("Standing memory saved.");
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
  }
}

function filterRounds(list, value) {
  const query = value.trim().toLocaleLowerCase();
  list.querySelectorAll(".memory-round").forEach((card) => {
    card.hidden = Boolean(query) && !card.dataset.searchText.includes(query);
  });
}

function sourceSearchText(source) {
  return [
    inlineText(source.title),
    ...(source.authors || []),
    ...(source.categories || []),
    ...(source.keywords || []),
    ...(source.project || []),
    source.source_relative_path,
  ].join(" ").toLocaleLowerCase();
}

function sourceCard(source) {
  const card = node("article", "source-card");
  const body = node("div", "source-card-body");
  const titleRow = node("div", "source-title-row");
  titleRow.append(node("span", "format-badge", source.format || "source"));
  titleRow.append(node("span", "source-title", inlineText(source.title) || source.source_relative_path));
  body.append(titleRow);
  body.append(node("p", "source-byline", authorLine(source)));
  if (source.doi) body.append(node("p", "source-doi", `doi:${inlineText(source.doi).replace(/^doi:/i, "")}`));
  if (
    (source.categories || []).length ||
    (source.keywords || []).length ||
    (source.project || []).length
  ) {
    const tags = node("div", "source-tags");
    tags.append(tagList(source.project, "tag tag-project"));
    tags.append(tagList(source.categories, "tag"));
    tags.append(tagList(source.keywords, "tag tag-keyword"));
    body.append(tags);
  }
  // The path and the identifier are how a reader names this source in a report
  // and in a filter, so they are kept whole and selectable behind one labelled
  // disclosure rather than compressed under the title.
  const identifiers = node("details", "identifier-drawer");
  identifiers.append(node("summary", "drawer-note", "Path and identifier"));
  identifiers.append(factList([
    ["Path", source.source_relative_path],
    ["Identifier", source.source_id],
  ], "fact-list identifier-list"));
  if (source.source_path && source.source_path !== source.source_relative_path) {
    identifiers.lastElementChild.append(
      node("dt", "fact-label", "Absolute path"),
      node("dd", "fact-value", source.source_path),
    );
  }
  body.append(identifiers);
  card.append(body);

  const actions = node("div", "source-card-actions");
  if (hasCapability("source_files")) {
    actions.append(button("Open original", "open-source", source.source_relative_path));
  }
  if (hasCapability("metadata")) {
    actions.append(button("Edit metadata", "edit-metadata", source.document_id));
  }
  // Two different decisions sit here, and their names keep them apart. The
  // search pair fills a filter for the next query and opens the search; the
  // project exclusion is recorded, read by every surface and agent, and kept
  // until it is restored, so it is named for its reach and coloured for it.
  if (hasCapability("source_selection")) {
    actions.append(button("Search only this source", "only-source", source.source_id));
    actions.append(button("Search without this source", "exclude-from-search", source.source_id));
  }
  if (hasCapability("source_inclusion")) {
    actions.append(
      button(
        "Exclude from project…",
        "exclude-source",
        source.document_id,
        "action-button action-button-danger",
      ),
    );
  }
  if (actions.childElementCount) card.append(actions);
  return card;
}

function excludedCard(source) {
  const card = node("article", "excluded-item");
  const body = node("div");
  body.append(node("strong", "", source.source_relative_path));
  body.append(node("p", "excluded-reason", inlineText(source.reason) || "No reason recorded"));
  const status = source.exists === false
    ? "File missing"
    : source.indexed_in_current_generation
      ? "Blocked from current search"
      : "Not in current index";
  body.append(node("span", "state-badge state-badge-blocked", status));
  card.append(body);
  if (hasCapability("source_inclusion")) {
    card.append(button("Restore source", "restore-source", source.source_relative_path));
  }
  return card;
}

// A page of sources holds ten. A collection drawn whole was one column of every
// card, so a reader looking for one source scrolled past all the others. The
// filter still reads every source, and the pages divide what it matched.
const SOURCES_PER_PAGE = 10;

function pageCount(total) {
  return Math.max(1, Math.ceil(total / SOURCES_PER_PAGE));
}

// The page numbers a pager shows: the first, the last, and the current page
// with its neighbours, with a gap where numbers are left out, so a pager over a
// thousand sources is as wide as one over thirty.
function pagerNumbers(page, pages) {
  const shown = [...new Set([1, page - 1, page, page + 1, pages])]
    .filter((number) => number >= 1 && number <= pages)
    .sort((left, right) => left - right);
  const items = [];
  shown.forEach((number, index) => {
    if (index && number - shown[index - 1] > 1) items.push(null);
    items.push(number);
  });
  return items;
}

function pagerButton(label, list, page, className) {
  const control = button(label, "source-page", String(page), `button button-quiet ${className}`);
  control.dataset.list = list;
  return control;
}

function renderPager(pagerId, list, page, pages) {
  const pager = byId(pagerId);
  pager.replaceChildren();
  pager.hidden = pages <= 1;
  if (pager.hidden) return;
  const previous = pagerButton("Previous", list, page - 1, "pager-step");
  previous.disabled = page <= 1;
  pager.append(previous);
  pagerNumbers(page, pages).forEach((number) => {
    if (number === null) {
      pager.append(node("span", "pager-gap", "…"));
      return;
    }
    const control = pagerButton(String(number), list, number, "pager-number");
    control.setAttribute("aria-label", `Page ${number} of ${pages}`);
    if (number === page) control.setAttribute("aria-current", "page");
    pager.append(control);
  });
  const next = pagerButton("Next", list, page + 1, "pager-step");
  next.disabled = page >= pages;
  pager.append(next);
}

// The page a reader is on survives a refresh, so a metadata edit on page five
// returns to page five; it is clamped because an exclusion can remove the last
// page.
function currentPage(list, total) {
  const pages = pageCount(total);
  const page = Math.min(Math.max(1, state.sourcePages[list] || 1), pages);
  state.sourcePages[list] = page;
  return [page, pages];
}

function renderSourcePage() {
  const list = byId("source-list");
  list.replaceChildren();
  const query = byId("source-filter").value.trim().toLocaleLowerCase();
  const matched = query
    ? state.sources.filter((source) => sourceSearchText(source).includes(query))
    : state.sources;
  const [page, pages] = currentPage("sources", matched.length);
  const start = (page - 1) * SOURCES_PER_PAGE;
  const shown = matched.slice(start, start + SOURCES_PER_PAGE);
  shown.forEach((source) => list.append(sourceCard(source)));

  const range = byId("source-range");
  range.hidden = !shown.length;
  if (!state.sources.length) {
    list.append(node("div", "no-records", state.sourcesReady ? "No sources match this collection." : "Create a generation to inspect indexed sources."));
  } else if (!shown.length) {
    list.append(node("div", "no-records", "No source matches this filter."));
  } else {
    const span = shown.length === 1
      ? formatNumber(start + 1)
      : `${formatNumber(start + 1)}–${formatNumber(start + shown.length)}`;
    const plural = matched.length === 1 ? "" : "s";
    range.textContent = matched.length === state.sources.length
      ? `Showing ${span} of ${formatNumber(matched.length)} source${plural}`
      : `Showing ${span} of ${formatNumber(matched.length)} matching source${plural}, ${formatNumber(state.sources.length)} in all`;
  }
  renderPager("source-pager", "sources", page, pages);
}

function renderExcludedPage() {
  const list = byId("excluded-list");
  list.replaceChildren();
  const [page, pages] = currentPage("excluded", state.excludedSources.length);
  const start = (page - 1) * SOURCES_PER_PAGE;
  state.excludedSources
    .slice(start, start + SOURCES_PER_PAGE)
    .forEach((source) => list.append(excludedCard(source)));
  renderPager("excluded-pager", "excluded", page, pages);
}

// A new page starts where its list starts, and a keyboard reader lands on the
// list's heading rather than on a pager button that may have moved.
function goToSourcePage(list, page) {
  state.sourcePages[list] = page;
  if (list === "excluded") renderExcludedPage();
  else renderSourcePage();
  const heading = byId(list === "excluded" ? "excluded-heading" : "source-list-heading");
  heading.tabIndex = -1;
  heading.focus({ preventScroll: true });
  heading.scrollIntoView({ block: "start", behavior: "smooth" });
}

function renderSources(payload) {
  state.sources = payload.sources || [];
  state.excludedSources = payload.excluded_sources || [];
  state.sourcesReady = Boolean(payload.ready);
  byId("source-nav-count").textContent = String(state.sources.length);
  byId("excluded-count").textContent = String(state.excludedSources.length);
  renderSourcePage();
  byId("excluded-section").hidden = !state.excludedSources.length;
  renderExcludedPage();
}

async function loadWorkspace({ announce = false } = {}) {
  setBusy(true, "Reading the current generation…");
  setConnection("loading", "Connecting");
  try {
    if (!state.profile) applyProfile(await api("/api/ui"));
    await loadClients();
    const [status, sources] = await Promise.all([
      api("/api/status"),
      hasCapability("sources")
        ? api("/api/sources")
        : Promise.resolve({ ready: false, sources: [], excluded_sources: [] }),
    ]);
    renderStatus(status);
    renderSources(sources);
    if (hasCapability("projects")) {
      try {
        await loadProjects();
      } catch (error) {
        toast(error.message, true);
      }
    }
    if (hasCapability("agent_entry")) {
      try {
        await loadAgentEntry();
      } catch (error) {
        toast(error.message, true);
      }
    }
    if (hasCapability("memory")) {
      try {
        await loadMemory();
      } catch (error) {
        toast(error.message, true);
      }
    }
    if (hasCapability("chunk_exclusion")) {
      try {
        await loadChunkExclusions();
      } catch (error) {
        toast(error.message, true);
      }
    }
    if (hasCapability("settings")) {
      try {
        await loadSettings();
      } catch (error) {
        toast(error.message, true);
      }
    }
    setConnection(...connectionHealth(state.status));
    // The update check runs after the workspace has drawn and is never awaited,
    // so a slow or unreachable release feed cannot hold up a search. It asks one
    // question and downloads nothing.
    if (hasCapability("updates")) {
      void checkUpdates();
    }
    if (announce) toast("Workspace refreshed.");
  } catch (error) {
    setConnection("error", "Connection failed");
    toast(error.message, true);
  } finally {
    setBusy(false);
    if (state.status) configureRetrieval(state.status);
  }
}

function switchView(name, { moveFocus = false } = {}) {
  let opened = null;
  document.querySelectorAll(".sidebar-item").forEach((item) => {
    const active = item.dataset.view === name;
    item.classList.toggle("is-active", active);
    item.querySelector(".nav-item")?.classList.toggle("is-active", active);
    // This is navigation rather than a tab set, so the current view is marked
    // the way a link to the current page is marked.
    if (active) item.querySelector(".nav-item")?.setAttribute("aria-current", "page");
    else item.querySelector(".nav-item")?.removeAttribute("aria-current");
  });
  document.querySelectorAll(".view-panel").forEach((panel) => {
    const active = panel.dataset.panel === name;
    panel.classList.toggle("is-active", active);
    panel.hidden = !active;
    // The panel is given a focus stop of its own so a keyboard reader who chose
    // a view lands on that view rather than on the sidebar they just left. The
    // stop is added only on a deliberate choice, so the first render does not
    // take the focus from wherever the page was opened.
    if (active && moveFocus) {
      panel.tabIndex = -1;
      opened = panel;
    }
  });
  if (opened) opened.focus({ preventScroll: true });
  closeNav();
}

// Below 992px the sidebar is a drawer behind the header button. It traps
// nothing: Escape and the scrim both close it, and the toggle keeps the focus
// so a keyboard reader is never lost when it goes.
function setNav(open) {
  byId("workspace-nav").classList.toggle("is-open", open);
  byId("nav-scrim").hidden = !open;
  byId("nav-toggle").setAttribute("aria-expanded", String(open));
}

function navIsOpen() {
  return byId("workspace-nav").classList.contains("is-open");
}

function closeNav({ restoreFocus = false } = {}) {
  if (!navIsOpen()) return;
  setNav(false);
  if (restoreFocus) byId("nav-toggle").focus();
}

function toggleNav() {
  if (navIsOpen()) {
    closeNav({ restoreFocus: true });
    return;
  }
  setNav(true);
  byId("workspace-nav").querySelector(".nav-item")?.focus();
}

// A score is a pair, because the two halves of one are a name and a number and
// neither reads beside the other: a bare 0.0321 is the fusion score to one reader
// and the cosine to another.
function scorePair(label, value) {
  if (value === null || value === undefined) return null;
  return [label, String(value)];
}

function resultCard(hit) {
  const card = node("article", "result-card");
  card.append(node("div", "result-rank", String(hit.rank).padStart(2, "0")));
  const content = node("div", "result-content");

  const header = node("div", "result-card-header");
  header.append(node("h3", "result-title", inlineText(hit.title) || hit.source_path));
  header.append(node("span", "locator-badge", locatorLabel(hit.locator)));
  content.append(header);
  content.append(node("div", "result-byline", `${authorLine(hit)} · ${hit.source_path}`));
  if (hit.doi) content.append(node("div", "result-doi", `doi:${inlineText(hit.doi).replace(/^doi:/i, "")}`));
  content.append(node("p", "result-citation", inlineText(hit.citation) || "Citation unavailable"));
  content.append(node("div", "semantic-text-label", state.profile?.result_text_label || "Retrieved passage"));
  content.append(node("p", "passage-text", readableText(hit.text)));

  if ((hit.categories || []).length || (hit.keywords || []).length) {
    const tags = node("div", "tag-row");
    tags.append(tagList(hit.categories, "tag"));
    tags.append(tagList(hit.keywords, "tag tag-keyword"));
    content.append(tags);
  }

  const footer = node("div", "result-footer");
  const actions = node("div", "result-actions");
  actions.append(button(state.profile?.copy_text_label || "Copy passage", "copy-passage", hit.chunk_id));
  actions.append(button("Copy citation", "copy-citation", hit.chunk_id));
  if (hasCapability("passage_context")) {
    actions.append(button("Nearby context", "show-context", hit.chunk_id));
  }
  if (hasCapability("chunk_exclusion")) {
    actions.append(chunkAction(hit));
  }
  const source = sourceForDocument(hit.document_id);
  if (source && hasCapability("source_files")) {
    actions.append(button("Open original", "open-source", source.source_relative_path));
  }
  footer.append(actions);
  content.append(footer);

  // The rank each component gave this passage and the passage identifier are
  // what a reader checks when a result looks wrong, and they are not what a
  // reader scans the list for. They are labelled, full, and selectable here
  // rather than compressed into a row of separators under every card.
  const scores = [
    scorePair("BM25 rank", hit.component_ranks?.bm25),
    scorePair("Dense rank", hit.component_ranks?.dense),
    scorePair("Cosine", hit.component_scores?.dense_cosine_similarity),
    scorePair("Fusion score", hit.fusion_score),
    scorePair("Rerank score", hit.rerank_score),
  ].filter(Boolean);
  const details = node("details", "identifier-drawer");
  const summary = node("summary", "drawer-note", "Scores and identifiers");
  summary.append(node("span", "count-badge", String(scores.length)));
  details.append(summary);
  const rows = scores.map(([label, value]) => [label, value]);
  rows.push(["Passage identifier", hit.chunk_id]);
  if (hit.document_id) rows.push(["Source identifier", hit.document_id]);
  if (hit.source_path) rows.push(["Path", hit.source_path]);
  details.append(factList(rows, "fact-list score-facts"));
  content.append(details);
  card.append(content);
  return card;
}

function renderResults(payload) {
  const results = byId("results");
  results.replaceChildren();
  byId("search-empty").hidden = true;
  byId("search-summary").hidden = false;
  const count = payload.result_count || 0;
  byId("result-heading").textContent = `${count} passage${count === 1 ? "" : "s"}`;
  const details = [
    payload.retrieval_method?.toUpperCase(),
    payload.reranked ? "CPU reranked" : null,
    payload.relevance_limited ? `relevance limited · requested ${payload.requested_top_k}` : null,
    compactId(payload.generation_id),
  ].filter(Boolean);
  byId("result-meta").textContent = details.join(" · ");
  if (!count) {
    results.append(node("div", "no-records", "No passages matched. Broaden the query or remove metadata filters."));
    return;
  }
  (payload.hits || []).forEach((hit) => {
    state.hits.set(hit.chunk_id, hit);
    results.append(resultCard(hit));
  });
  if (payload.stale) toast("Results come from the current generation, which is marked stale.");
}

function clearResults() {
  state.hits = new Map();
  byId("results").replaceChildren();
  byId("search-summary").hidden = true;
  byId("search-empty").hidden = false;
}

async function search(event) {
  event.preventDefault();
  const query = byId("query").value.trim();
  if (!query) return;
  const method = document.querySelector("input[name='retrieval_method']:checked")?.value || "hybrid";
  const payload = {
    query,
    top_k: Number(byId("top-k").value),
  };
  const optional = {
    categories: hasCapability("metadata_filters") ? listValue(byId("category-filter").value) : null,
    categories_any: hasCapability("category_partitions") ? listValue(byId("category-any-filter").value) : null,
    projects_any: hasCapability("project_metadata") ? listValue(byId("project-any-filter").value) : null,
    keywords: hasCapability("metadata_filters") ? listValue(byId("keyword-filter").value) : null,
    authors_any: hasCapability("bibliographic_filters") ? listValue(byId("author-filter").value) : null,
    titles_any: hasCapability("bibliographic_filters") ? listValue(byId("title-filter").value) : null,
    languages_any: hasCapability("bibliographic_filters") ? listValue(byId("language-filter").value) : null,
    source_ids: hasCapability("source_selection") ? listValue(byId("include-source-filter").value) : null,
    exclude_source_ids: hasCapability("source_selection") ? listValue(byId("exclude-source-filter").value) : null,
  };
  // Send a filter only when the server supports it and the user selected one.
  for (const [field, value] of Object.entries(optional)) {
    if (value?.length) payload[field] = value;
  }
  if (hasCapability("retrieval_modes")) payload.retrieval_method = method;
  if (hasCapability("reranking")) payload.rerank = byId("rerank").checked;
  setBusy(true, payload.rerank ? "Searching and CPU reranking…" : "Searching evidence…");
  try {
    state.hits = new Map();
    renderResults(await api("/api/search", { method: "POST", body: JSON.stringify(payload) }));
    // The results sit under the query, so the page moves only when an open
    // filter drawer has pushed them out of sight. Moving it otherwise took the
    // query off the screen for no gain.
    const summary = byId("search-summary");
    if (summary.getBoundingClientRect().top > window.innerHeight * 0.75) {
      summary.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
    if (state.status) configureRetrieval(state.status);
  }
}

function openSource(path) {
  window.open(`/api/source-file?path=${encodeURIComponent(path)}`, "_blank", "noopener");
}

async function copyText(text, message) {
  try {
    await navigator.clipboard.writeText(text);
    toast(message);
  } catch (_error) {
    toast("Clipboard access was blocked by the browser.", true);
  }
}

async function showContext(chunkId) {
  setBusy(true, "Loading nearby passages…");
  try {
    const payload = await api(`/api/passages/${encodeURIComponent(chunkId)}?context_chunks=1`);
    const container = byId("context-content");
    container.replaceChildren();
    (payload.context || []).forEach((passage) => {
      const requested = passage.chunk_id === payload.requested_chunk_id;
      // The server marks a neighbour it left out of the corpus, so the dialog
      // says so rather than letting a passage that no query returns sit in a
      // list of search hits.
      const excluded = passage.excluded_from_search === true;
      const item = node(
        "article",
        `context-passage${requested ? " is-requested" : ""}${excluded ? " context-excluded" : ""}`,
      );
      const citation = node("div", "context-citation");
      citation.append(node("span", "", inlineText(passage.citation)));
      citation.append(node("span", "locator-badge", locatorLabel(passage.locator)));
      item.append(citation);
      if (excluded) {
        item.append(
          node(
            "p",
            "context-excluded-warning",
            "Excluded from search. This passage is beside the one you asked for, and no query returns it.",
          ),
        );
      }
      item.append(node("p", "", readableText(passage.text)));
      if (hasCapability("chunk_exclusion") && !excluded) {
        // Changing what a search reads stays a deliberate step taken in the
        // dialog. A passage already excluded is offered nothing here, because
        // restoring it belongs to the list that owns the exclusions.
        item.append(chunkExcludeButton(passage));
      }
      container.append(item);
    });
    if (!container.children.length) container.append(node("div", "no-records", "No context was returned."));
    byId("context-title").textContent = payload.context?.[0]?.title || "Passage context";
    byId("context-dialog").showModal();
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
    if (state.status) configureRetrieval(state.status);
  }
}

function openMetadata(documentId) {
  const source = sourceForDocument(documentId);
  if (!source) return;
  byId("metadata-source-path").value = source.source_relative_path;
  byId("metadata-title").value = source.title || "";
  byId("metadata-authors").value = (source.authors || []).join(", ");
  byId("metadata-year").value = source.year || "";
  byId("metadata-doi").value = source.doi || "";
  byId("metadata-categories").value = (source.categories || []).join(", ");
  byId("metadata-keywords").value = (source.keywords || []).join(", ");
  byId("metadata-project").value = (source.project || []).join(", ");
  byId("metadata-dialog").showModal();
}

function openExclusion(documentId) {
  const source = sourceForDocument(documentId);
  if (!source) return;
  byId("exclusion-source-path").value = source.source_relative_path;
  byId("exclusion-source-name").textContent = source.source_relative_path;
  byId("exclusion-reason").value = "";
  byId("exclusion-dialog").showModal();
}

function bytes(count) {
  if (typeof count !== "number" || !Number.isFinite(count)) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = count;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${unit === 0 ? value : value.toFixed(1)} ${units[unit]}`;
}

function generationRow(generation) {
  const row = node("article", "record-row");
  const head = node("div", "record-row-head");
  head.append(node("code", "record-id", generation.generation_id));
  if (generation.is_current) {
    head.append(node("span", "state-badge state-badge-current", "In use"));
  } else {
    head.append(node("span", "state-badge", "Retained"));
  }
  row.append(head);
  // Every number here is a fact about what a build kept, and none of them is a
  // headline: the identifier above is what identifies the generation.
  const facts = [
    ["Built", generation.created_at ? formatDate(generation.created_at) : "Unknown"],
    ["Passages", formatNumber(generation.chunk_count ?? 0)],
    ["Sources", formatNumber(generation.document_count ?? 0)],
    ["Size on disk", bytes(generation.size_bytes)],
  ];
  if (generation.file_count) facts.push(["Files", formatNumber(generation.file_count)]);
  if (generation.schema_version) facts.push(["Schema", String(generation.schema_version)]);
  row.append(factList(facts, "fact-list"));
  // A generation whose manifest cannot be read is named rather than counted,
  // because a reader deciding what to remove needs to know which rows are facts.
  if (generation.manifest_error) {
    row.append(node("p", "record-warning", `Manifest unreadable: ${generation.manifest_error}`));
  }
  // The one a search reads cannot be removed, so the action is not offered
  // rather than offered and refused: a button that always fails is a button
  // that teaches a reader to click through the answers. It sits beside the
  // identifier because that is what it acts on.
  if (!generation.is_current) {
    const drop = node("button", "text-button", "Remove");
    drop.type = "button";
    drop.addEventListener("click", () => openGenerationRemoval(generation.generation_id));
    head.append(drop);
  }
  return row;
}

function renderGenerations(generations) {
  // The id stays the container hook other code and tests already read; the
  // contents are rows rather than chips, because an identifier and a size do not
  // belong in a badge.
  const container = byId("generation-chips");
  container.replaceChildren();
  byId("generation-count").textContent = formatNumber(generations.length);
  if (!generations.length) {
    container.append(node("p", "form-note", "This project has no build yet."));
    return;
  }
  for (const generation of generations) {
    container.append(generationRow(generation));
  }
}

function openGenerationRemoval(generationId) {
  byId("generation-remove-id").value = generationId;
  byId("generation-remove-name").textContent = generationId;
  byId("generation-confirm").value = "";
  byId("generation-submit").disabled = true;
  byId("generation-dialog").showModal();
}

async function removeGeneration(event) {
  event.preventDefault();
  const generationId = byId("generation-remove-id").value;
  if (byId("generation-confirm").value.trim() !== generationId) {
    toast("The typed id does not match the generation.", true);
    return;
  }
  setBusy(true, "Removing the generation…");
  try {
    const result = await api("/api/generations/remove", {
      method: "POST",
      body: JSON.stringify({ generation_id: generationId, confirm: generationId }),
    });
    byId("generation-dialog").close();
    toast(result.message || "Generation removed.");
    await loadWorkspace();
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
  }
}

function sqlScopes(status) {
  // A scope is an opaque identifier the server reported, so it is read here and
  // nowhere else, and a bare string is accepted as well as a described entry.
  return (status.sql_scopes || [])
    .map((entry) => (typeof entry === "string" ? { scope: entry } : entry))
    .filter((entry) => entry && typeof entry.scope === "string" && entry.scope.trim());
}

function syncSqlControls() {
  const ready = Boolean(
    byId("sql-scope").value && byId("sql-statement").value.trim() && !state.busy,
  );
  byId("sql-run-button").disabled = !ready;
  byId("sql-execute-button").disabled = !ready;
}

// --------------------------------------------------------------- updates --
//
// The server answers one question: whether a newer release has been published.
// This page asks it, shows what came back, and stops there. It downloads
// nothing, applies nothing, and starts no process, because a running app must
// not replace itself and a global installation may be serving other projects.
// Changing an installation is a command in a terminal, and the command is
// shown rather than run.

// Only a release page on the forge the project publishes from is turned into a
// link. The server validates the repository; the page checks the shape again
// before it renders anything a reader might click.
const RELEASE_HOSTS = ["github.com"];

function releaseHref(url) {
  try {
    const parsed = new URL(String(url || ""));
    if (parsed.protocol !== "https:") return "";
    if (!RELEASE_HOSTS.includes(parsed.hostname)) return "";
    return parsed.href;
  } catch (_error) {
    return "";
  }
}

// Three answers, and they are not the same answer. An update is available, the
// check succeeded and found no newer release, or the check could not be made —
// and the last one never claims the installation is current.
function updateOutcome(payload) {
  if (payload?.update_available) return "available";
  if (payload?.offline) return "offline";
  return payload?.release === null && payload?.message ? "unavailable" : "none";
}

function updateFacts(payload, includeRelease) {
  const rows = [["Installed version", payload.installed_version || "Not reported"]];
  if (includeRelease) {
    const release = payload.release || {};
    if (release.version) rows.push(["Latest release", release.version]);
    if (release.name) rows.push(["Release name", inlineText(release.name)]);
    if (release.tag_name) rows.push(["Tag", release.tag_name]);
    if (release.published_at) rows.push(["Published", formatDate(release.published_at)]);
  }
  const href = releaseHref(payload?.release?.html_url);
  if (href) rows.push(["Release page", href]);
  return rows;
}

function renderUpdates(payload) {
  state.updates.payload = payload || null;
  const outcome = updateOutcome(payload);
  const release = payload?.release || null;
  const message = byId("update-message");

  if (outcome === "available") {
    message.textContent = `${release?.name || release?.version || "A newer release"} is published and this app is behind.`;
  } else if (outcome === "offline") {
    message.textContent =
      "This machine could not reach the release feed, so whether a newer release exists is unknown.";
  } else if (outcome === "unavailable") {
    message.textContent = "The release feed could not be checked.";
  } else {
    message.textContent = "The release feed was checked and reported no newer release.";
  }
  // The server's own sentence is the reason for the two negative answers, and it
  // is shown rather than replaced, because it names what failed.
  if (outcome === "offline" || outcome === "unavailable") {
    const reason = node("span", "update-reason", inlineText(payload?.message));
    message.replaceChildren(document.createTextNode(`${message.textContent} `), reason);
  }

  const detail = byId("update-detail");
  detail.hidden = outcome !== "available";
  if (outcome === "available") {
    byId("update-release-tag").textContent = release?.tag_name || release?.version || "";
    byId("update-facts").replaceChildren(factList(updateFacts(payload, true), "fact-list"));
    // Release notes arrive as Markdown and are written as text. Nothing here
    // parses or sanitises markup: the notes are read, not rendered.
    byId("update-notes").textContent = release?.body || "The release reported no notes.";
  }

  const badge = byId("update-nav-badge");
  badge.hidden = outcome !== "available";
  badge.title = "A newer release is published";

  const declined = byId("update-declined");
  const isDeclined = Boolean(
    state.updates.declinedVersion && release?.version === state.updates.declinedVersion,
  );
  declined.hidden = !isDeclined;
  if (isDeclined) {
    declined.textContent = `You chose not to update to ${release.version} for now. Ask again by checking for updates.`;
  }
}

// A reader who declined a version is not asked about it again until they ask for
// a check themselves or a different version is published.
function shouldPromptUpdate(payload) {
  if (updateOutcome(payload) !== "available") return false;
  const version = payload?.release?.version || "";
  if (!version) return false;
  return version !== state.updates.declinedVersion;
}

function fillUpdateDialog(payload) {
  const release = payload.release || {};
  byId("update-dialog-title").textContent = `${release.name || release.version} is published`;
  byId("update-dialog-intro").textContent = `This app is running ${payload.installed_version || "an unreported version"}. Read the notes, then decide.`;
  byId("update-dialog-facts").replaceChildren(factList(updateFacts(payload, true), "fact-list"));
  byId("update-dialog-notes").textContent = release.body || "The release reported no notes.";
  // The choice is the first thing a reader sees; the command appears only after
  // they have chosen to go to a terminal, so a page cannot look like it is
  // already installing anything.
  byId("update-choice").hidden = false;
  byId("update-handoff").hidden = true;
  byId("update-command-copied").hidden = true;
  byId("update-command").textContent = payload.apply_command || "";
}

function openUpdateReview(payload) {
  fillUpdateDialog(payload);
  byId("update-dialog").showModal();
}

function declineUpdate() {
  const version = state.updates.payload?.release?.version || "";
  state.updates.declinedVersion = version;
  byId("update-dialog").close();
  renderUpdates(state.updates.payload);
  toast("Left for now. Nothing was downloaded and nothing was changed.");
}

async function checkUpdates({ manual = false } = {}) {
  if (!hasCapability("updates")) return;
  const message = byId("update-message");
  if (manual) {
    // A reader who asked to check again is no longer holding a decline against
    // the same version.
    state.updates.declinedVersion = "";
  }
  try {
    message.textContent = "Checking for a published release…";
    const payload = await api("/api/updates");
    renderUpdates(payload);
    if (shouldPromptUpdate(payload)) openUpdateReview(payload);
  } catch (error) {
    message.textContent = "The release feed could not be checked.";
    byId("update-detail").hidden = true;
    byId("update-nav-badge").hidden = true;
    if (manual) toast(error.message, true);
  }
}

function renderSqlConsole(status) {
  const select = byId("sql-scope");
  const scopes = sqlScopes(status);
  const chosen = select.value;
  select.replaceChildren();
  for (const entry of scopes) {
    const option = node("option", "", entry.label || entry.scope);
    option.value = entry.scope;
    select.append(option);
  }
  if (scopes.some((entry) => entry.scope === chosen)) select.value = chosen;
  select.disabled = !scopes.length;
  const message = byId("sql-message");
  message.hidden = scopes.length > 0;
  message.textContent = scopes.length
    ? ""
    : "This server reports no SQL scope, so there is nothing to run a statement against.";
  syncSqlControls();
}

function sqlCell(value) {
  return node(
    "td",
    "sql-cell",
    value === null || value === undefined ? "—" : String(value),
  );
}

function renderSqlResult(payload) {
  const container = byId("sql-results");
  container.replaceChildren();
  const columns = Array.isArray(payload.columns) ? payload.columns : [];
  const rows = Array.isArray(payload.rows) ? payload.rows : [];
  const count = payload.row_count ?? rows.length;
  container.append(node("p", "result-meta", `${formatNumber(count)} row${count === 1 ? "" : "s"}`));
  if (!columns.length) {
    container.append(node("div", "no-records", "The server returned no columns."));
    return;
  }

  const table = node("table", "sql-table");
  const header = node("tr");
  for (const column of columns) {
    const cell = node("th", "", column);
    cell.scope = "col";
    header.append(cell);
  }
  const head = node("thead");
  head.append(header);
  const body = node("tbody");
  for (const row of rows) {
    const line = node("tr");
    for (const value of row) line.append(sqlCell(value));
    body.append(line);
  }
  table.append(head, body);
  container.append(table);
  if (payload.truncated) {
    container.append(
      node(
        "p",
        "form-note",
        `The server truncated this result: ${formatNumber(rows.length)} rows are shown of ${formatNumber(count)}.`,
      ),
    );
  }
}

function sqlRequest() {
  const scope = byId("sql-scope").value;
  const statement = byId("sql-statement").value.trim();
  if (!scope || !statement) {
    toast("A scope and a statement are required.", true);
    return null;
  }
  return { scope, statement };
}

async function runSql(event) {
  event.preventDefault();
  const request = sqlRequest();
  if (!request) return;
  setBusy(true, "Running the statement…");
  try {
    const result = await api("/api/sql/query", {
      method: "POST",
      body: JSON.stringify(request),
    });
    byId("sql-affected").hidden = true;
    renderSqlResult(result);
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function executeSql() {
  const request = sqlRequest();
  if (!request) return;
  setBusy(true, "Applying the statement to stored records…");
  try {
    const result = await api("/api/sql/execute", {
      method: "POST",
      body: JSON.stringify(request),
    });
    const affected = result.rows_affected ?? 0;
    byId("sql-results").replaceChildren();
    const line = byId("sql-affected");
    line.hidden = false;
    line.textContent = `Statement applied. ${formatNumber(affected)} record${affected === 1 ? "" : "s"} affected.${result.reindexed ? " The server reindexed the change." : ""}`;
    toast("Statement applied.");
    // A reindexed write changed what a search reads, so the status beside the
    // panel would otherwise go on describing the records before it.
    if (result.reindexed) await loadWorkspace();
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
  }
}

// A value is parsed only as the kind the server declared. The range and the
// choices below come from the server too, and only ever narrow what the field
// offers; what it will accept stays the server's decision and its refusal is
// what the reader sees.
function parseSettingValue(kind, raw) {
  if (kind === "bool") {
    if (typeof raw === "boolean") return raw;
    const text = String(raw).trim().toLowerCase();
    if (["false", "0", "no", "off", ""].includes(text)) return false;
    return ["true", "1", "yes", "on"].includes(text);
  }
  const text = String(raw).trim();
  if (kind === "int") return /^-?\d+$/.test(text) ? Number.parseInt(text, 10) : null;
  if (kind === "float") {
    const parsed = Number.parseFloat(text);
    return Number.isNaN(parsed) ? null : parsed;
  }
  return text;
}

function settingValueLabel(value) {
  if (value === null || value === undefined || value === "") return "none";
  return String(value);
}

// A value the packaged default file supplies says nothing about where it came
// from, because every key is supplied by that file and the row already says
// what the default is. A value another layer supplied is the one that needs
// naming, so that is the only case where the origin is shown.
function settingOrigin(setting) {
  if (setting.defaulted) return null;
  return setting.origin || "default";
}

// The server may declare a list of the values it accepts. Where it does, the
// field is a select over exactly that list, so a reader cannot type a value the
// server never offered. Where it declares none, the field stays the plain
// control for its kind, which is what this page has always drawn.
function settingChoices(setting, kind) {
  const choices = Array.isArray(setting.choices) ? setting.choices : [];
  if (!choices.length || kind === "bool") return null;
  const control = node("select", "setting-input setting-select");
  const current = setting.value === null || setting.value === undefined ? "" : String(setting.value);
  for (const choice of choices) {
    const option = node("option", "", settingValueLabel(choice));
    option.value = String(choice);
    control.append(option);
  }
  if ([...control.options].some((option) => option.value === current)) control.value = current;
  else control.selectedIndex = 0;
  return control;
}

function settingInput(setting, controlId) {
  const kind = setting.kind || "str";
  let control = settingChoices(setting, kind);
  if (!control) {
    control = node("input", "setting-input");
    if (kind === "bool") {
      control.type = "checkbox";
      control.checked = Boolean(setting.value);
    } else if (kind === "int" || kind === "float") {
      control.type = "number";
      control.step = kind === "int" ? "1" : "any";
      control.value = setting.value === null || setting.value === undefined ? "" : String(setting.value);
      // A range the server declares constrains the field rather than the value:
      // the browser refuses a key outside it, and the server still decides.
      if (setting.minimum !== undefined && setting.minimum !== null) control.min = String(setting.minimum);
      if (setting.maximum !== undefined && setting.maximum !== null) control.max = String(setting.maximum);
    } else {
      control.type = "text";
      control.value = setting.value === null || setting.value === undefined ? "" : String(setting.value);
    }
  }
  control.id = controlId;
  control.dataset.settingKey = setting.key;
  control.dataset.settingKind = kind;
  // A setting the server set outside this project arrives read-only, and is
  // shown as it is rather than as an empty box a reader would try to fill.
  control.disabled = !setting.writable;
  return control;
}

// A row shows the label, the server's own description, the value the page
// loaded, the value the app starts from, where the value in force came from,
// what changing it costs, the variable that would override it, and the
// control. Every one of those is optional: a host that sends none of them draws
// the same row it always did, and a host that sends all of them gets all of them
// without a truncated line.
function settingRow(setting, index) {
  const row = node("article", "setting-row");
  const controlId = `setting-control-${index}`;
  const control = settingInput(setting, controlId);

  const field = node("label", "setting-field");
  field.htmlFor = controlId;
  field.append(node("span", "field-label setting-label", setting.label || setting.key));
  field.append(node("span", "setting-key", setting.key));
  if (setting.doc) field.append(node("p", "setting-doc", inlineText(setting.doc)));
  row.append(field);

  const facts = node("div", "setting-facts");
  facts.append(node("span", "setting-fact setting-value", `Value: ${settingValueLabel(setting.value)}`));
  if (setting.default !== undefined) {
    facts.append(
      node("span", "setting-fact setting-default", `Default: ${settingValueLabel(setting.default)}`),
    );
  }
  const origin = settingOrigin(setting);
  if (origin) facts.append(node("span", "locator-badge setting-origin", origin));
  const cost = setting.cost || {};
  if (cost.message) {
    const className = cost.level === "model" ? "setting-fact setting-cost setting-cost-model" : "setting-fact setting-cost";
    facts.append(node("span", className, cost.message));
  }
  if (setting.env) {
    facts.append(node("span", "setting-fact setting-env", `${setting.env} overrides this value`));
  }
  if (!setting.writable) {
    facts.append(
      node("span", "setting-fact setting-readonly", `Set by ${setting.origin || "the server"}; edit it there.`),
    );
  }
  row.append(facts);

  const holder = node("div", "setting-control");
  holder.append(control);
  row.append(holder);
  return row;
}

function renderSettings(payload) {
  state.settingsRevision = payload.revision || "";
  state.settings = new Map();
  const container = byId("settings-sections");
  container.replaceChildren();
  for (const section of payload.sections || []) {
    const block = node("section", "settings-section");
    block.append(node("h4", "", section.title || section.key));
    let index = 0;
    for (const setting of section.settings || []) {
      if (!setting?.key) continue;
      state.settings.set(setting.key, setting);
      block.append(settingRow(setting, `${index}-${slugify(setting.key)}`));
      index += 1;
    }
    container.append(block);
  }
  const message = byId("settings-message");
  message.hidden = !payload.message;
  message.textContent = payload.message || "";
  // The file every default comes from is named once here, because a row that
  // carried the path would repeat the same sentence on every row and read as
  // thirty-nine places to look for one fact.
  const defaults = byId("settings-defaults");
  defaults.hidden = !payload.default_file;
  defaults.textContent = payload.default_file
    ? `Every default below is the one ${payload.default_file} declares.`
    : "";
  syncSettingsSubmit();
}

function slugify(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "setting";
}

function settingsControls() {
  return [...document.querySelectorAll("#settings-sections [data-setting-key]")];
}

function controlValue(control) {
  return control.type === "checkbox" ? control.checked : control.value.trim();
}

// What the page loaded, as the control itself shows it. A setting the server
// reports with no value loads an empty control, so leaving that control empty
// is not a change to send: comparing the box against the string "null" would
// make every valueless setting look edited and leave Save live for ever.
function loadedSettingText(setting) {
  const value = setting.value;
  if (value === null || value === undefined) return "";
  return String(value).trim();
}

function settingChanged(setting, control) {
  if (control.type === "checkbox") return control.checked !== Boolean(setting.value);
  return String(controlValue(control)) !== loadedSettingText(setting);
}

function settingsDirty() {
  // A setting the page never loaded is not compared and not sent, because the
  // revision a write carries was computed against the values shown here.
  return settingsControls().some((control) => {
    const setting = state.settings.get(control.dataset.settingKey);
    if (!setting || !setting.writable) return false;
    return settingChanged(setting, control);
  });
}

function syncSettingsSubmit() {
  byId("settings-submit").disabled = !settingsDirty() || state.busy;
}

function changedSettings() {
  const values = {};
  for (const control of settingsControls()) {
    const key = control.dataset.settingKey;
    const setting = state.settings.get(key);
    if (!setting || !setting.writable) continue;
    const parsed = parseSettingValue(control.dataset.settingKind, controlValue(control));
    if (parsed === null) {
      toast(`${key} must be a ${control.dataset.settingKind} value.`, true);
      return null;
    }
    if (settingChanged(setting, control)) values[key] = parsed;
  }
  return values;
}

function expensiveSettings(values) {
  // The keys whose change is not free, named from the cost the server reported
  // for each, so the confirmation quotes the server rather than this page.
  return Object.keys(values).filter((key) => {
    const level = state.settings.get(key)?.cost?.level;
    return level === "regeneration" || level === "model";
  });
}

function openSettingsConfirmation(values, keys) {
  const list = byId("settings-confirm-list");
  list.replaceChildren();
  for (const key of keys) {
    const item = node("li", "");
    item.append(node("strong", "", key));
    const message = state.settings.get(key)?.cost?.message;
    if (message) item.append(document.createTextNode(` — ${message}`));
    list.append(item);
  }
  // A model change is the more expensive one, so it is the word asked for.
  const word = keys.some((key) => state.settings.get(key)?.cost?.level === "model")
    ? "model"
    : "ingest";
  state.pendingSettings = { values, word };
  byId("settings-confirm-label").textContent = `Type ${word} to confirm`;
  byId("settings-confirm-word").value = "";
  byId("settings-confirm-word").placeholder = word;
  byId("settings-confirm-submit").disabled = true;
  byId("settings-confirm-error").hidden = true;
  byId("settings-confirm-dialog").showModal();
}

function showSettingsResult(result) {
  const message = byId("settings-message");
  message.hidden = false;
  message.textContent = result.message || "Settings saved.";
  const notice = byId("settings-notice");
  notice.hidden = !result.requires_ingest;
  // A setting that changes what the corpus holds is applied by an ingestion.
  // The workspace says so and leaves the decision to the reader.
  notice.textContent = result.requires_ingest
    ? "These settings apply to the next generation. Run ingest to rebuild it; the generation in use stays searchable until you do."
    : "";
}

async function sendSettings(values) {
  setBusy(true, "Saving the settings…");
  try {
    const result = await api("/api/settings", {
      method: "POST",
      body: JSON.stringify({
        values,
        expected_revision: state.settingsRevision,
        confirm: true,
      }),
    });
    await loadSettings();
    showSettingsResult(result);
  } catch (error) {
    // The refusal is shown as the server worded it and is not sent again, because
    // the same revision would only be refused twice.
    const message = byId("settings-message");
    message.hidden = false;
    message.textContent = error.message;
    byId("settings-notice").hidden = true;
    toast(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function saveSettings(event) {
  event.preventDefault();
  const values = changedSettings();
  if (values === null) return;
  if (!Object.keys(values).length) {
    toast("No setting was changed.", true);
    return;
  }
  const keys = expensiveSettings(values);
  if (keys.length) {
    openSettingsConfirmation(values, keys);
    return;
  }
  await sendSettings(values);
}

async function confirmSettings(event) {
  event.preventDefault();
  const pending = state.pendingSettings;
  if (!pending) return;
  if (byId("settings-confirm-word").value.trim() !== pending.word) {
    const error = byId("settings-confirm-error");
    error.hidden = false;
    error.textContent = `Type ${pending.word} to confirm.`;
    return;
  }
  byId("settings-confirm-dialog").close();
  state.pendingSettings = null;
  await sendSettings(pending.values);
}

async function loadSettings() {
  if (!hasCapability("settings")) return;
  renderSettings(await api("/api/settings"));
}

async function reloadSettings() {
  if (!hasCapability("settings")) return;
  setBusy(true, "Reading the settings…");
  try {
    await loadSettings();
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
  }
}

function chunkExcludeButton(chunk) {
  const control = button("Exclude this chunk", "exclude-chunk", chunk.chunk_id);
  control.dataset.chunkSource = chunk.source_relative_path || chunk.source_path || "Unknown source";
  control.dataset.chunkLocator = chunk.locator ? locatorLabel(chunk.locator) : "No locator reported";
  return control;
}

function chunkAction(chunk) {
  // A chunk the server already lists as excluded is offered a restore instead,
  // so the reader is never asked to exclude what is already out.
  return state.chunkExclusions.has(chunk.chunk_id)
    ? button("Restore this chunk", "restore-chunk", chunk.chunk_id)
    : chunkExcludeButton(chunk);
}

function openChunkExclusion(control) {
  byId("chunk-id").value = control.dataset.value;
  byId("chunk-name").textContent = `${control.dataset.chunkSource} · ${control.dataset.chunkLocator}`;
  byId("chunk-reason").value = "";
  byId("chunk-error").hidden = true;
  byId("chunk-dialog").showModal();
}

async function writeChunkInclusion(payload, fromDialog = false) {
  setBusy(true, payload.included ? "Restoring the chunk…" : "Excluding the chunk…");
  try {
    const result = await api("/api/chunk-inclusion", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    if (fromDialog) byId("chunk-dialog").close();
    toast(result.message || (payload.included ? "Chunk restored." : "Chunk excluded."));
    clearResults();
    await loadChunkExclusions();
  } catch (error) {
    if (fromDialog) {
      // The dialog stays open so the refusal is read beside the reason that
      // caused it, and the same request is not sent again unchanged.
      const line = byId("chunk-error");
      line.hidden = false;
      line.textContent = error.message;
    } else {
      toast(error.message, true);
    }
  } finally {
    setBusy(false);
  }
}

async function saveChunkExclusion(event) {
  event.preventDefault();
  const chunkId = byId("chunk-id").value;
  const reason = byId("chunk-reason").value.trim();
  if (!chunkId || !reason) {
    const line = byId("chunk-error");
    line.hidden = false;
    line.textContent = "A reason is required.";
    return;
  }
  await writeChunkInclusion({ chunk_id: chunkId, included: false, reason }, true);
}

function chunkExclusionCard(entry) {
  const card = node("article", "excluded-item");
  const body = node("div");
  body.append(node("strong", "", entry.source_relative_path || entry.chunk_id));
  body.append(node("p", "source-path", entry.locator || "No locator reported"));
  body.append(node("p", "excluded-reason", inlineText(entry.reason) || "No reason recorded"));
  // A chunk this generation does not hold is named as such, because restoring it
  // changes the next generation rather than the passages already on screen.
  body.append(
    node(
      "span",
      "state-badge state-badge-blocked",
      entry.in_current_generation === false
        ? entry.message || "This chunk is not in the current generation."
        : "Blocked from current search",
    ),
  );
  card.append(body, button("Restore", "restore-chunk", entry.chunk_id));
  return card;
}

function renderChunkExclusions(payload) {
  const entries = payload.exclusions || [];
  state.chunkExclusions = new Map(entries.map((entry) => [entry.chunk_id, entry]));
  byId("chunk-exclusion-count").textContent = String(entries.length);
  const message = byId("chunk-exclusion-message");
  message.hidden = !payload.message;
  message.textContent = payload.message || "";

  const list = byId("chunk-exclusion-list");
  list.replaceChildren();
  if (!entries.length) {
    list.append(node("div", "no-records", "No chunk is excluded from retrieval."));
    return;
  }
  entries.forEach((entry) => list.append(chunkExclusionCard(entry)));
}

async function loadChunkExclusions() {
  if (!hasCapability("chunk_exclusion")) return;
  renderChunkExclusions(await api("/api/chunk-exclusions"));
}

async function saveMetadata(event) {
  event.preventDefault();
  const yearText = byId("metadata-year").value.trim();
  const metadata = {
    title: byId("metadata-title").value.trim(),
    authors: listValue(byId("metadata-authors").value),
    year: yearText ? Number(yearText) : null,
    doi: byId("metadata-doi").value.trim(),
    categories: listValue(byId("metadata-categories").value),
    keywords: listValue(byId("metadata-keywords").value),
  };
  if (hasCapability("project_metadata")) {
    metadata.project = listValue(byId("metadata-project").value);
  }
  const payload = {
    source_path: byId("metadata-source-path").value,
    metadata,
  };
  setBusy(true, "Saving reviewed metadata…");
  try {
    const result = await api("/api/source-metadata", { method: "POST", body: JSON.stringify(payload) });
    byId("metadata-dialog").close();
    toast(result.message || "Metadata saved. It applies to the next search.");
    await loadWorkspace();
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
    if (state.status) configureRetrieval(state.status);
  }
}

async function excludeSource(event) {
  event.preventDefault();
  const payload = {
    source_path: byId("exclusion-source-path").value,
    included: false,
    reason: byId("exclusion-reason").value.trim(),
  };
  setBusy(true, "Excluding source from retrieval…");
  try {
    const result = await api("/api/source-inclusion", { method: "POST", body: JSON.stringify(payload) });
    byId("exclusion-dialog").close();
    clearResults();
    toast(result.message || "Source excluded.");
    await loadWorkspace();
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
    if (state.status) configureRetrieval(state.status);
  }
}

async function restoreSource(path) {
  setBusy(true, "Restoring source…");
  try {
    const result = await api("/api/source-inclusion", {
      method: "POST",
      body: JSON.stringify({ source_path: path, included: true }),
    });
    clearResults();
    toast(result.message || "Source restored.");
    await loadWorkspace();
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
    if (state.status) configureRetrieval(state.status);
  }
}

async function ingest(event) {
  event.preventDefault();
  const chunkSize = Number(byId("chunk-size").value);
  const chunkOverlap = Number(byId("chunk-overlap").value);
  if (hasCapability("chunk_settings") && chunkOverlap >= chunkSize) {
    toast("Chunk overlap must be smaller than chunk size.", true);
    return;
  }
  setBusy(true, state.profile?.ingest_busy_message || "Building the indexes. This can take several minutes…");
  try {
    const request = {};
    if (hasCapability("chunk_settings")) {
      request.chunk_size = chunkSize;
      request.chunk_overlap = chunkOverlap;
    }
    if (hasCapability("force_recompute")) request.force_recompute = state.forceRecompute;
    const result = await api("/api/ingest", {
      method: "POST",
      body: JSON.stringify(request),
    });
    byId("ingest-dialog").close();
    clearResults();
    toast(`Generation ${compactId(result.generation_id)} is ready with ${formatNumber(result.chunk_count)} passages.`);
    await loadWorkspace();
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
    if (state.status) configureRetrieval(state.status);
  }
}

async function exportBundle() {
  const warning = state.profile?.bundle_export_warning
    || "Export this generation? The archive may contain complete original sources.";
  if (!window.confirm(warning)) return;
  setBusy(true, "Exporting the current generation and original sources…");
  try {
    const result = await api("/api/bundles/export", {
      method: "POST",
      body: JSON.stringify({}),
    });
    toast(`Bundle created: ${result.bundle_name}`);
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
    if (state.status) configureRetrieval(state.status);
  }
}

async function importBundle(event) {
  event.preventDefault();
  const payload = {
    bundle_name: byId("bundle-name").value.trim(),
    activate: byId("bundle-activate").checked,
  };
  setBusy(true, "Validating the bundle and rebuilding local indexes…");
  try {
    const result = await api("/api/bundles/import", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    byId("bundle-dialog").close();
    clearResults();
    toast(result.message || `Bundle ${result.generation_id} imported.`);
    await loadWorkspace();
  } catch (error) {
    toast(error.message, true);
  } finally {
    setBusy(false);
    if (state.status) configureRetrieval(state.status);
  }
}

function handleAction(event) {
  const target = event.target.closest("[data-action]");
  if (!target) return;
  const { action, value } = target.dataset;
  if (action === "open-source") openSource(value);
  else if (action === "edit-metadata") openMetadata(value);
  else if (action === "exclude-source") openExclusion(value);
  else if (action === "restore-source") restoreSource(value);
  else if (action === "show-context") showContext(value);
  else if (action === "exclude-chunk") openChunkExclusion(target);
  else if (action === "restore-chunk") writeChunkInclusion({ chunk_id: value, included: true });
  else if (action === "copy-passage") {
    const hit = state.hits.get(value);
    if (hit) copyText(readableText(hit.text), "Semantic text copied.");
  } else if (action === "copy-citation") {
    const hit = state.hits.get(value);
    if (hit) copyText(inlineText(hit.citation), "Citation copied.");
  } else if (action === "ingest") openIngest(state.ingestPlan);
  else if (action === "copy-remedy") copyText(value, "Command copied.");
  else if (action === "source-page") goToSourcePage(target.dataset.list, Number(value));
  else if (action === "partition-filter") addSearchFilter("category-any-filter", value);
  else if (action === "project-filter") addSearchFilter("project-any-filter", value);
  else if (action === "language-filter") addSearchFilter("language-filter", value);
  else if (action === "only-source") searchWithSource("include-source-filter", value);
  else if (action === "exclude-from-search") searchWithSource("exclude-source-filter", value);
  else if (action === "copy-standing" || action === "edit-standing") {
    const scope = target.closest(".memory-scope")?.dataset.scope;
    if (!scope) return;
    if (action === "copy-standing") {
      copyText(
        target.closest(".memory-scope")?.querySelector(".standing-document")?.textContent || "",
        "Standing memory copied.",
      );
    } else {
      openStandingEditor(scope);
    }
  } else if (action === "copy-round") {
    const round = target.closest(".memory-round");
    if (round) {
      copyText(`user: ${round.dataset.user}\nassistant: ${round.dataset.assistant}`, "Round copied.");
    }
  }
}

// A changed filter matches a different set, so it starts from that set's first
// page.
function filterSources() {
  state.sourcePages.sources = 1;
  renderSourcePage();
}

function initialize() {
  state.hits = new Map();
  document.querySelectorAll(".sidebar-item").forEach((item) => {
    item.querySelector(".nav-item")?.addEventListener("click", () => {
      switchView(item.dataset.view, { moveFocus: true });
    });
  });
  byId("nav-toggle").addEventListener("click", toggleNav);
  byId("nav-scrim").addEventListener("click", () => closeNav());
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeNav({ restoreFocus: true });
  });
  // Widening past the breakpoint turns the drawer back into a fixed column, so
  // it is left closed rather than reopening itself over the page.
  window.matchMedia("(min-width: 992px)").addEventListener("change", () => closeNav());
  document.querySelectorAll("dialog").forEach((dialog) => {
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close();
    });
  });
  document.querySelectorAll(".dialog-close").forEach((control) => {
    control.addEventListener("click", () => control.closest("dialog").close());
  });
  byId("refresh-button").addEventListener("click", () => loadWorkspace({ announce: true }));
  byId("ingest-button").addEventListener("click", () => openIngest(state.ingestPlan));
  byId("rebuild-button").addEventListener("click", () => openIngest(REBUILD_PLAN));
  byId("health-section").addEventListener("click", handleAction);
  byId("export-button").addEventListener("click", exportBundle);
  byId("import-button").addEventListener("click", () => byId("bundle-dialog").showModal());
  byId("notice-action").addEventListener("click", handleAction);
  byId("update-check-button").addEventListener("click", () => {
    void checkUpdates({ manual: true });
  });
  byId("update-not-now").addEventListener("click", declineUpdate);
  byId("update-continue").addEventListener("click", () => {
    // Choosing to continue reveals the command and takes the choice away, so
    // the dialog is left showing the one next step rather than both.
    byId("update-choice").hidden = true;
    byId("update-handoff").hidden = false;
    byId("update-command").focus();
  });
  byId("update-command-copy").addEventListener("click", () => {
    const command = byId("update-command").textContent;
    if (!command) return;
    byId("update-command-copied").hidden = false;
    void copyText(command, "Update command copied.");
  });
  byId("search-form").addEventListener("submit", search);
  byId("metadata-form").addEventListener("submit", saveMetadata);
  byId("exclusion-form").addEventListener("submit", excludeSource);
  byId("ingest-form").addEventListener("submit", ingest);
  byId("bundle-form").addEventListener("submit", importBundle);
  byId("source-filter").addEventListener("input", filterSources);
  byId("memory-scopes").addEventListener("click", handleAction);
  byId("memory-standing-form").addEventListener("submit", saveStanding);
  byId("results").addEventListener("click", handleAction);
  byId("source-list").addEventListener("click", handleAction);
  byId("source-pager").addEventListener("click", handleAction);
  byId("excluded-pager").addEventListener("click", handleAction);
  byId("partition-chips").addEventListener("click", handleAction);
  byId("project-chips").addEventListener("click", handleAction);
  // The language list fills the language box beside it, so selecting from it is
  // a filter action like every other one.
  byId("language-chips").addEventListener("click", handleAction);
  FILTER_FIELDS.forEach((field) => {
    byId(field).addEventListener("input", syncFilterSummary);
  });
  syncFilterSummary();
  byId("excluded-list").addEventListener("click", handleAction);
  byId("generation-form").addEventListener("submit", removeGeneration);
  byId("generation-confirm").addEventListener("input", (event) => {
    byId("generation-submit").disabled =
      event.target.value.trim() !== byId("generation-remove-id").value;
  });
  byId("sql-form").addEventListener("submit", runSql);
  byId("sql-execute-button").addEventListener("click", executeSql);
  byId("sql-statement").addEventListener("input", syncSqlControls);
  byId("settings-form").addEventListener("submit", saveSettings);
  byId("settings-sections").addEventListener("input", syncSettingsSubmit);
  // A select reports a chosen value on change; without this the submit gate
  // would stay closed on a setting the server drew as a list of choices.
  byId("settings-sections").addEventListener("change", syncSettingsSubmit);
  byId("settings-reload").addEventListener("click", reloadSettings);
  byId("settings-confirm-form").addEventListener("submit", confirmSettings);
  byId("settings-confirm-word").addEventListener("input", (event) => {
    const pending = state.pendingSettings;
    byId("settings-confirm-submit").disabled =
      !pending || event.target.value.trim() !== pending.word;
  });
  byId("chunk-form").addEventListener("submit", saveChunkExclusion);
  byId("chunk-exclusion-list").addEventListener("click", handleAction);
  byId("context-content").addEventListener("click", handleAction);
  byId("project-select").addEventListener("change", showProjectActions);
  byId("project-open").addEventListener("click", () => {
    const project = selectedProject();
    if (project?.url) window.open(project.url, "_blank", "noopener");
  });
  byId("project-copy-command").addEventListener("click", () => {
    copyText(byId("project-start-command").textContent, "Start command copied.");
  });
  byId("agent-url-copy").addEventListener("click", () => {
    copyText(byId("agent-url").textContent, "Endpoint copied.");
  });
  byId("agent-entry-copy").addEventListener("click", () => {
    copyText(state.agentEntry, "Client entry copied.");
  });
  loadWorkspace();
}

document.addEventListener("DOMContentLoaded", initialize);
