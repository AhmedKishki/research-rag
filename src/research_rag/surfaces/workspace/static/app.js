"use strict";

const state = {
  profile: null,
  status: null,
  sources: [],
  excludedSources: [],
  sourcesReady: false,
  searchedQuery: "",
  searchKey: "",
  // The passages on screen, by passage id, so a copy reads the text the server
  // sent rather than the text the card trimmed to show.
  hits: new Map(),
  restoringRoute: false,
  sourcePages: { sources: 1, excluded: 1 },
  projects: [],
  currentProject: "",
  agentEntry: "",
  memory: null,
  memoryRounds: new Map(),
  memoryStanding: new Map(),
  standingScope: null,
  generationActions: [],
  busy: false,
  forceRecompute: false,
  settingsRevision: "",
  settings: new Map(),
  chunkExclusions: new Map(),
  pendingSettings: null,
  updates: { payload: null, declinedVersion: "", checked: false },
  // A fold the reader opened or closed keeps that choice across a refresh, keyed
  // by the fold's id or its `data-fold` name.
  folds: new Map(),
  // The stats board: each card's scope, the answers the cards share, and the
  // panels drawn for them.
  statScopes: {},
  statData: new Map(),
  statPanels: new Map(),
  // The view on screen, which may be a detail view the sidebar has no item for,
  // the source being read, and the passage a link opened.
  view: "",
  sourceView: { id: "", page: 1, title: "" },
  passageId: "",
  passagePushed: false,
  closingForRoute: false,
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
  // The view a link or a reload names is drawn first, so the page does not show
  // the search for a moment before the route is applied.
  const routed = parseRoute(window.location.hash).view;
  const activeItem = visibleNavItems.find((item) => item.dataset.view === routed)
    || visibleNavItems.find((item) => item.classList.contains("is-active"));
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

// A list longer than this starts folded behind its heading and count, so a view
// opens on what it is about rather than on the length of one of its lists.
const FOLD_LIMIT = 5;

function foldKey(drawer) {
  return drawer.dataset.fold || drawer.id || "";
}

// Open a fold when its list is short, unless the reader already chose.
function foldList(drawer, count) {
  if (!drawer) return;
  const key = foldKey(drawer);
  drawer.open = key && state.folds.has(key) ? state.folds.get(key) : count <= FOLD_LIMIT;
}

// The actions a card offers sit behind one menu, so a list of ten cards is ten
// cards and not ten rows of the same five links. The menu is a disclosure: it
// opens on a click or a key, and the actions in it are ordinary buttons.
function actionMenu(name, actions) {
  const menu = node("details", "card-menu");
  const toggle = node("summary", "card-menu-toggle");
  toggle.title = "Actions";
  toggle.append(node("span", "card-menu-dots", "⋯"));
  toggle.append(node("span", "visually-hidden", `Actions for ${name}`));
  menu.append(toggle);
  const list = node("div", "card-menu-list");
  actions.forEach((action) => list.append(action));
  menu.append(list);
  return menu;
}

function closeMenus(except = null) {
  document.querySelectorAll("details.card-menu[open]").forEach((menu) => {
    if (menu !== except) menu.open = false;
  });
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
  document.querySelectorAll("button[type='submit']:not([data-gated])").forEach((element) => {
    element.disabled = active;
  });
  byId("refresh-button").disabled = active;
  byId("ingest-button").disabled = active || !hasCapability("ingestion");
  byId("rebuild-button").disabled = active || !hasCapability("force_recompute");
  byId("export-button").disabled = active || !hasCapability("bundle_export");
  byId("import-button").disabled = active || !hasCapability("bundle_import");
  // A generation's Load and Remove are ordinary buttons rather than gated
  // submits, so the loop above does not reach them. They are held here, where
  // the busy state is decided, so one operation cannot start while another is
  // changing which generation searches read.
  state.generationActions.forEach((control) => {
    control.disabled = active;
  });
  syncGatedSubmits();
}

// A submit marked `data-gated` is not switched by the loop above, because its
// own condition — a statement to run, a changed value, a typed confirmation —
// is what decides it. Each gate is re-read here instead, so a button that said
// nothing had been typed does not come back live because a refresh elsewhere
// finished.
function syncGatedSubmits() {
  syncSqlControls();
  syncSettingsSubmit();
  syncGenerationSubmit();
  syncSettingsConfirmSubmit();
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
    const error = new Error(payload?.error || `Request failed with status ${response.status}`);
    error.status = response.status;
    throw error;
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
  // The project on screen is this page, so opening it would only open a second
  // tab of the same workspace.
  open.hidden = !url || project.project_name === state.currentProject;
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
  // A selector over one project has nothing to switch to, so it is not drawn;
  // the sidebar footer still names the project.
  byId("project-selector").hidden =
    !hasCapability("projects") || (state.projects.length <= 1 && !payload.message);
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

// The counts and identifiers a reader needs to act are in the cards above, and
// the project directory is in the header. What is left — where the sources are
// read from, how many are on disk, and the formats accepted — is here, labelled
// and selectable. A fact the page already shows elsewhere is not repeated.
function renderStatusFacts(status) {
  const rows = [];
  if (status.source_root) rows.push(["Source directory", status.source_root]);
  rows.push([
    "Sources on disk",
    `${formatNumber(status.selected_source_count)} selected · ${formatNumber(
      status.excluded_source_count,
    )} excluded`,
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
  // With nothing to act on, the badge says so and the list is not drawn.
  const list = byId("health-conditions");
  list.replaceChildren();
  list.hidden = !conditions.length;
  conditions.forEach((entry) => list.append(healthCondition(entry)));

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
function revealFilterDrawer() {
  const drawer = byId("filter-fields");
  if (drawer && !drawer.open) drawer.open = true;
}

function addSearchFilter(field, value, label = value) {
  const input = byId(field);
  if (!input || !value) return;
  const values = listValue(input.value);
  if (!values.includes(value)) values.push(value);
  input.value = values.join(", ");
  revealFilterDrawer();
  syncFilterSummary();
  recordRoute({ replace: true });
  input.focus();
  toast(`Added to the search filter: ${label}`);
}

// A source chosen on the Sources view narrows a search, so the reader is taken
// to the search with the filter in view. A toast on a view the filter is not on
// told the reader about a change they could not see.
function searchWithSource(field, sourceId) {
  const source = state.sources.find((entry) => entry.source_id === sourceId);
  switchView("search", { moveFocus: true });
  recordRoute();
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
    group.querySelector(".count-badge").textContent = formatNumber(entries.length);
    foldList(group, entries.length);
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
  const title = inlineText(source.title) || source.source_relative_path;
  titleRow.append(node("span", "format-badge", source.format || "source"));
  const name = node("span", "source-title");
  name.append(source.source_id && hasCapability("sources") ? entityLink(title, sourceHref(source.source_id)) : title);
  titleRow.append(name);
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

  const actions = [];
  if (hasCapability("source_files")) {
    actions.push(button("Open original", "open-source", source.source_relative_path));
  }
  if (hasCapability("metadata")) {
    actions.push(button("Edit metadata", "edit-metadata", source.document_id));
  }
  // Two different decisions sit here, and their names keep them apart. The
  // search pair fills a filter for the next query and opens the search; the
  // project exclusion is recorded, read by every surface and agent, and kept
  // until it is restored, so it is named for its reach and coloured for it.
  if (hasCapability("source_selection")) {
    actions.push(button("Search only this source", "only-source", source.source_id));
    actions.push(button("Search without this source", "exclude-from-search", source.source_id));
  }
  if (hasCapability("source_inclusion")) {
    actions.push(
      button(
        "Exclude from project…",
        "exclude-source",
        source.document_id,
        "action-button action-button-danger",
      ),
    );
  }
  if (actions.length) titleRow.append(actionMenu(title, actions));
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

// A page of sources holds five, so one page fits the window. A collection drawn
// whole was one column of every card, so a reader looking for one source
// scrolled past all the others. The filter still reads every source, and the
// pages divide what it matched.
const SOURCES_PER_PAGE = 5;

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
  if (list === "chunks") {
    state.sourceView.page = page;
    recordRoute();
    void loadSourceView().then(() => {
      const heading = byId("source-chunks-heading");
      heading.tabIndex = -1;
      heading.focus({ preventScroll: true });
      heading.scrollIntoView({ block: "start", behavior: "smooth" });
    });
    return;
  }
  state.sourcePages[list] = page;
  if (list === "excluded") {
    byId("excluded-fold").open = true;
    renderExcludedPage();
  } else {
    renderSourcePage();
  }
  recordRoute();
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
  foldList(byId("excluded-fold"), state.excludedSources.length);
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
    // The update check runs once per page, after the workspace has drawn, and is
    // never awaited, so a slow or unreachable release feed cannot hold up a
    // search and a refresh after every save does not ask again. It asks one
    // question and downloads nothing.
    if (hasCapability("updates") && !state.updates.checked) {
      state.updates.checked = true;
      void checkUpdates();
    }
    if (hasCapability("stats") && activeView() === "stats") void loadStats();
    if (announce) toast("Workspace refreshed.");
  } catch (error) {
    setConnection("error", "Not answered", error.message);
    toast(error.message, true);
  } finally {
    setBusy(false);
    if (state.status) configureRetrieval(state.status);
  }
}

// ------------------------------------------------------------------- stats --
//
// The server counts each search's first five ranks and reports them beside the
// facts its selected generation records. Every figure is a card of its own with
// its own scope, because a reader comparing last week's searches with the whole
// history wants both at once. A card asks the server for exactly its scope, and
// the answers are shared between cards that chose the same one. The page draws
// what came back and never a figure the payload did not carry.

const STAT_TIME_SCOPES = [
  { label: "All time", days: null },
  { label: "30 days", days: 30 },
  { label: "7 days", days: 7 },
  { label: "24 hours", days: 1 },
];
const STAT_SIZES = [5, 10, 20, 50];
// What the largest sources are ranked by.
const STAT_LARGEST_BY = [
  { value: "passages", label: "By passages" },
  { value: "size", label: "By text size" },
];
const SCOPE_STORAGE_KEY = "research-rag.stat-scopes";

// A reader's scopes survive a reload where the browser allows it, and the board
// draws with its defaults where it does not.
function readSavedScopes() {
  try {
    const saved = JSON.parse(window.localStorage.getItem(SCOPE_STORAGE_KEY) || "{}");
    return saved && typeof saved === "object" ? saved : {};
  } catch (_error) {
    return {};
  }
}

function saveScopes() {
  try {
    window.localStorage.setItem(SCOPE_STORAGE_KEY, JSON.stringify(state.statScopes));
  } catch (_error) {
    // Storage is a convenience; the scopes stay in this page.
  }
}

function cardScope(card) {
  const saved = state.statScopes[card.id] || {};
  const timeScopes = card.timeScopes || STAT_TIME_SCOPES;
  const days = timeScopes.some((scope) => scope.days === saved.days) ? saved.days : null;
  const top = STAT_SIZES.includes(saved.top) ? saved.top : card.top || 10;
  const by = STAT_LARGEST_BY.some((option) => option.value === saved.by) ? saved.by : "passages";
  return { days, top, by };
}

function statRequest(source, scope) {
  // Only the largest-sources ranking depends on `by`, so a request for anything
  // else names it the default and shares an answer with every other card.
  const by = source === "stats" ? scope.by : "passages";
  const key = `${source}|${scope.days ?? ""}|${scope.top}|${by}`;
  if (!state.statData.has(key)) {
    const params = new URLSearchParams();
    if (scope.days !== null) params.set("days", String(scope.days));
    params.set(source === "history" ? "limit" : "top", String(scope.top));
    if (source === "stats" && by !== "passages") params.set("largest_by", by);
    const path = source === "history" ? "/api/stats/history" : "/api/stats";
    state.statData.set(
      key,
      api(`${path}?${params}`).catch((error) => {
        state.statData.delete(key);
        throw error;
      }),
    );
  }
  return state.statData.get(key);
}

// A bar for scale beside a count the row already states in text.
function statBar(value, maximum) {
  const track = node("span", "stat-bar");
  track.setAttribute("aria-hidden", "true");
  const fill = node("span", "stat-bar-fill");
  const share = maximum > 0 ? Math.max(0, Math.min(1, Number(value) / maximum)) : 0;
  fill.style.width = `${Math.round(share * 1000) / 10}%`;
  track.append(fill);
  return track;
}

// A table of labelled counts. Each column names its header and how to read the
// row, and the first numeric column also draws the bar.
function statTable(columns, rows, { bar = null } = {}) {
  if (!rows.length) return node("p", "form-note", "Nothing to count yet.");
  const table = node("table", "record-table stats-table");
  const header = node("tr");
  columns.forEach((column) => {
    const cell = node("th", column.numeric ? "number-cell" : "", column.label);
    cell.scope = "col";
    header.append(cell);
  });
  if (bar) {
    const cell = node("th", "bar-cell");
    cell.scope = "col";
    cell.append(node("span", "visually-hidden", "Scale"));
    header.append(cell);
  }
  const head = node("thead");
  head.append(header);
  const body = node("tbody");
  const maximum = bar ? Math.max(...rows.map((row) => Number(row[bar]) || 0)) : 0;
  rows.forEach((row) => {
    const line = node("tr");
    columns.forEach((column, index) => {
      const value = column.value(row);
      const cell = index === 0 ? node("th", "stat-row-label") : node("td", column.numeric ? "number-cell" : "");
      if (index === 0) cell.scope = "row";
      if (value !== null && typeof value === "object") cell.append(value);
      else if (column.numeric && typeof value === "number") cell.textContent = formatNumber(value);
      else cell.textContent = String(value ?? "");
      line.append(cell);
    });
    if (bar) {
      const cell = node("td", "bar-cell");
      cell.append(statBar(row[bar], maximum));
      line.append(cell);
    }
    body.append(line);
  });
  table.append(head, body);
  const scroller = node("div", "table-scroll");
  scroller.append(table);
  return scroller;
}

function milliseconds(value) {
  if (value === null || value === undefined) return "—";
  return value >= 1000 ? `${(value / 1000).toFixed(1)} s` : `${Math.round(value)} ms`;
}

// A name that goes somewhere: a source to its passages, a passage to the text
// around it. The click is handled by one listener, so the same link works from
// every view, and a copied address returns to the same place.
function entityLink(text, href, className = "") {
  const link = node("a", `entity-link ${className}`.trim(), text);
  link.href = href;
  return link;
}

function sourceHref(sourceId) {
  return `#/source?id=${encodeURIComponent(sourceId)}`;
}

function passageHref(chunkId) {
  return `#/passage?id=${encodeURIComponent(chunkId)}`;
}

function sourceCell(entry) {
  const cell = document.createDocumentFragment();
  const title = inlineText(entry.title) || entry.source_relative_path || entry.source_id || "Unknown source";
  cell.append(
    entry.source_id && entry.in_corpus !== false
      ? entityLink(title, sourceHref(entry.source_id), "stat-title")
      : node("span", "stat-title", title),
  );
  if (entry.in_corpus === false) cell.append(node("span", "state-badge", "Not in this generation"));
  return cell;
}

// A passage is two facts, the source it is in and the place in it, and each has
// a column of its own: the source goes to all of its passages, the place to the
// text around it.
function passageSourceCell(entry) {
  const title = inlineText(entry.title) || "Unknown source";
  const cell = document.createDocumentFragment();
  cell.append(
    entry.source_id && entry.in_corpus !== false
      ? entityLink(title, sourceHref(entry.source_id), "stat-title")
      : node("span", "stat-title", title),
  );
  return cell;
}

function passagePlaceCell(entry) {
  return entry.in_current_generation
    ? entityLink(locatorLabel(entry.locator), passageHref(entry.chunk_id), "locator-badge")
    : node("span", "locator-badge", "Not in this generation");
}

const RANK_COLUMNS = [
  { label: "Top five", numeric: true, value: (row) => row.top_five },
  { label: "Rank one", numeric: true, value: (row) => row.rank_one },
];

function countedTable(label, entries) {
  return statTable(
    [{ label, value: (row) => row.value }, { label: "Sources", numeric: true, value: (row) => row.count }],
    entries || [],
    { bar: "count" },
  );
}

function statGroup(title, content) {
  const group = node("section", "stats-group");
  group.append(node("h4", "stats-group-title", title));
  group.append(content);
  return group;
}

function numberBody(value, detail) {
  const box = document.createDocumentFragment();
  box.append(node("strong", "stat-value", value));
  if (detail) box.append(node("span", "stat-detail", detail));
  return box;
}

// The address that runs a kept search again: its question, its size, and each
// filter in the field the search page reads it from.
function searchHref(entry) {
  const params = new URLSearchParams();
  params.set("q", entry.query);
  if (entry.requested_top_k && String(entry.requested_top_k) !== DEFAULT_TOP_K) {
    params.set("k", String(entry.requested_top_k));
  }
  FILTER_PARAMS.forEach(([name]) => {
    const values = entry.filters?.[name];
    if (Array.isArray(values) && values.length) params.set(name, values.join(", "));
  });
  return `#/search?${params}`;
}

function historyBody(payload) {
  const searches = payload.searches || [];
  if (!searches.length) {
    return node(
      "p",
      "form-note",
      payload.recording === false
        ? "History is off, so only counts are kept. Turn on runtime.search_history in Config to keep questions."
        : "No search has kept its question yet.",
    );
  }
  const list = node("ul", "history-list");
  searches.forEach((entry) => {
    const item = node("li", "history-item");
    const head = node("div", "history-head");
    head.append(entityLink(entry.query, searchHref(entry), "history-query"));
    head.append(node("span", "state-badge", entry.caller || "unknown"));
    item.append(head);
    const filters = Object.entries(entry.filters || {}).filter(([, values]) => values.length);
    const meta = [
      formatDate(entry.searched_at),
      `${formatNumber(entry.result_count)} of ${formatNumber(entry.requested_top_k)} passages`,
      milliseconds(entry.elapsed_ms),
    ];
    item.append(node("p", "result-meta", meta.join(" · ")));
    if (filters.length) {
      const tags = node("div", "tag-row");
      filters.forEach(([name, values]) =>
        tags.append(node("span", "tag", `${name.replaceAll("_", " ")}: ${values.join(", ")}`)),
      );
      item.append(tags);
    }
    list.append(item);
  });
  return list;
}

function figure(label, value, detail) {
  const box = node("div", "stat-figure");
  box.append(node("span", "stat-label", label));
  box.append(node("strong", "stat-value", value));
  if (detail) box.append(node("span", "stat-detail", detail));
  return box;
}

// The four headline figures of the searches in a scope share one card and one
// scope: each is a line or two, and four cards for them were four mostly empty
// boxes.
function renderSearchFigures(payload) {
  const searches = payload.searches || {};
  const count = Number(searches.search_count || 0);
  const mean = searches.mean_result_count;
  const slow = searches.p95_elapsed_ms;
  const grid = node("div", "stat-figures");
  grid.append(
    figure(
      "Searches",
      formatNumber(count),
      mean === null || mean === undefined ? "None counted yet" : `${mean} passages on average`,
    ),
    figure(
      "With no results",
      formatNumber(searches.zero_result_count),
      count ? `${Math.round((100 * searches.zero_result_count) / count)}% of searches` : "—",
    ),
    figure(
      "Median time",
      milliseconds(searches.median_elapsed_ms),
      slow === null || slow === undefined ? "—" : `95th percentile ${milliseconds(slow)}`,
    ),
    figure(
      "Never in a top five",
      count ? formatNumber(payload.unreached_source_count) : "—",
      payload.corpus ? `of ${formatNumber(payload.corpus.source_count)} searchable sources` : "No generation yet",
    ),
  );
  return grid;
}

// Count every UTC interval between the first and last recorded day, including
// quiet intervals. Long histories use weeks or months rather than tiny bars.
function usageSeries(entries, scope = {}, now = new Date()) {
  const days = entries.filter((entry) => /^\d{4}-\d{2}-\d{2}$/.test(entry.day))
    .sort((left, right) => left.day.localeCompare(right.day));
  if (!days.length) return { interval: "day", buckets: [] };
  const end = scope.days ? new Date(now) : new Date(`${days.at(-1).day}T00:00:00Z`);
  const start = scope.days ? new Date(now - scope.days * 86400000) : new Date(`${days[0].day}T00:00:00Z`);
  start.setUTCHours(0, 0, 0, 0);
  end.setUTCHours(0, 0, 0, 0);
  const span = Math.round((end - start) / 86400000) + 1;
  const interval = span <= 60 ? "day" : span <= 420 ? "week" : "month";
  const key = (date) => {
    const day = new Date(date);
    if (interval === "week") day.setUTCDate(day.getUTCDate() - (day.getUTCDay() + 6) % 7);
    if (interval === "month") day.setUTCDate(1);
    return day.toISOString().slice(0, 10);
  };
  const totals = new Map();
  days.forEach((entry) => {
    const day = key(new Date(`${entry.day}T00:00:00Z`));
    totals.set(day, (totals.get(day) || 0) + Number(entry.count || 0));
  });
  const buckets = [];
  const cursor = new Date(`${key(start)}T00:00:00Z`);
  while (cursor <= end) {
    const day = key(cursor);
    buckets.push({ day, count: totals.get(day) || 0 });
    if (interval === "month") cursor.setUTCMonth(cursor.getUTCMonth() + 1);
    else cursor.setUTCDate(cursor.getUTCDate() + (interval === "week" ? 7 : 1));
  }
  return { interval, buckets };
}

function renderUsage(payload, scope) {
  const { interval, buckets } = usageSeries(payload.searches?.searches_by_day || [], scope);
  if (!buckets.length) return node("p", "form-note", "No search invocations in this time frame.");
  const chart = node("figure", "usage-chart");
  const total = buckets.reduce((sum, bucket) => sum + bucket.count, 0);
  const caption = `${formatNumber(total)} search invocations · per ${interval} · UTC`;
  chart.append(node("figcaption", "stat-detail", caption));
  const svgNode = (tag, attributes, text) => {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, String(value)));
    if (text !== undefined) element.textContent = text;
    return element;
  };
  const svg = svgNode("svg", {
    viewBox: "0 0 800 260", role: "img", "aria-label": caption,
    "aria-describedby": "usage-chart-description",
  });
  svg.append(svgNode("title", {}, caption));
  svg.append(svgNode("desc", { id: "usage-chart-description" },
    buckets.map((bucket) => `${bucket.day}: ${formatNumber(bucket.count)} invocations`).join("; ")));
  const maximum = Math.ceil(Math.max(1, ...buckets.map((bucket) => bucket.count)) / 2) * 2;
  const step = 730 / buckets.length;
  [0, 0.5, 1].forEach((share) => {
    const y = 220 - share * 190;
    svg.append(svgNode("line", { x1: 50, y1: y, x2: 780, y2: y, class: "usage-gridline" }));
    svg.append(svgNode("text", { x: 42, y: y + 4, "text-anchor": "end" }, formatNumber(maximum * share)));
  });
  buckets.forEach((bucket, index) => {
    const height = bucket.count / maximum * 190;
    const bar = svgNode("rect", {
      x: 50 + index * step + step * 0.1, y: 220 - height,
      width: step * 0.8, height, class: "usage-bar",
    });
    bar.append(svgNode("title", {}, `${bucket.day}: ${formatNumber(bucket.count)} invocations`));
    svg.append(bar);
  });
  [...new Set([0, Math.floor((buckets.length - 1) / 2), buckets.length - 1])].forEach((index) => {
    svg.append(svgNode("text", {
      x: 50 + (index + 0.5) * step, y: 248,
      "text-anchor": index === 0 ? "start" : index === buckets.length - 1 ? "end" : "middle",
    }, buckets[index].day));
  });
  chart.append(svg);
  const data = node("details", "usage-data");
  data.append(node("summary", "", "View counts"));
  data.append(statTable([
    { label: `${checkLabel(interval)} starting (UTC)`, value: (row) => row.day },
    { label: "Invocations", numeric: true, value: (row) => row.count },
  ], buckets));
  chart.append(data);
  return chart;
}

// Every card on the board: its title, where its numbers come from, the scopes a
// reader may change on it, and how it draws what came back. A card that would
// hold a line or two shares one with its neighbours, so the board is a few full
// cards and no scatter of small ones.
const STAT_CARDS = [
  {
    id: "searches",
    title: "Searches",
    source: "stats",
    scopes: ["time"],
    render: renderSearchFigures,
  },
  {
    id: "sources",
    title: "Sources by appearances",
    source: "stats",
    scopes: ["time", "size"],
    top: 10,
    count: (payload) => (payload.sources || []).length,
    render: (payload) =>
      statTable([{ label: "Source", value: sourceCell }, ...RANK_COLUMNS], payload.sources || [], { bar: "top_five" }),
  },
  {
    id: "passages",
    title: "Passages by appearances",
    source: "stats",
    scopes: ["time", "size"],
    top: 10,
    count: (payload) => (payload.passages || []).length,
    render: (payload) =>
      statTable(
        [
          { label: "Source", value: passageSourceCell },
          { label: "Passage", value: passagePlaceCell },
          ...RANK_COLUMNS,
        ],
        payload.passages || [],
        { bar: "top_five" },
      ),
  },
  {
    id: "history",
    title: "Search history",
    source: "history",
    scopes: ["time", "size"],
    top: 10,
    count: (payload) => payload.count || 0,
    clearable: true,
    render: (payload) => historyBody(payload),
  },
  {
    id: "usage",
    title: "Search usage",
    source: "stats",
    scopes: ["time"],
    timeScopes: STAT_TIME_SCOPES.filter((scope) => scope.days !== 1),
    render: renderUsage,
  },
  {
    id: "largest",
    title: "Largest sources",
    source: "stats",
    scopes: ["size", "by"],
    top: 5,
    count: (payload) => (payload.corpus?.largest_sources || []).length,
    render: (payload) => renderLargest(payload.corpus),
  },
  {
    id: "people",
    title: "Categories and authors",
    source: "stats",
    scopes: ["size"],
    top: 10,
    render: (payload, scope) => renderPeople(payload.corpus, scope.top),
  },
  {
    id: "corpus",
    title: "Corpus",
    source: "stats",
    scopes: ["size"],
    top: 5,
    render: (payload, scope) => renderCorpus(payload.corpus, scope.top),
  },
  {
    id: "build",
    title: "Last build",
    source: "stats",
    scopes: [],
    render: (payload) => renderBuild(payload),
  },
];

const NO_GENERATION = "This project has no generation yet.";

// Corpus size, missing metadata, and publication decades.
function renderCorpus(corpus, top) {
  if (!corpus) return node("p", "form-note", NO_GENERATION);
  const spread = corpus.passages_per_source || {};
  const missing = corpus.missing_metadata || {};
  const groups = node("div", "stats-groups");
  groups.append(
    statGroup(
      "Size",
      factList(
        [
          ["Searchable sources", formatNumber(corpus.source_count)],
          ["Passages", formatNumber(corpus.passage_count)],
          ["PDF pages", formatNumber(corpus.pdf_page_count)],
          [
            "Passages per source",
            `${formatNumber(spread.minimum)} fewest · ${formatNumber(spread.median)} median · ${formatNumber(spread.maximum)} most`,
          ],
        ],
        "fact-list",
      ),
    ),
    statGroup(
      "Reviewed metadata missing",
      factList(
        [
          ["Authors", `${formatNumber(missing.authors)} sources`],
          ["Year", `${formatNumber(missing.year)} sources`],
          ["Categories", `${formatNumber(missing.categories)} sources`],
        ],
        "fact-list",
      ),
    ),
    statGroup("Decades", countedTable("Decade", (corpus.decades || []).slice(0, top))),
  );
  return groups;
}

function renderPeople(corpus, top) {
  if (!corpus) return node("p", "form-note", NO_GENERATION);
  const groups = node("div", "stats-groups");
  groups.append(
    statGroup("Categories", countedTable("Category", (corpus.categories || []).slice(0, top))),
    statGroup("Authors", countedTable("Author", (corpus.authors || []).slice(0, top))),
  );
  return groups;
}

// The largest sources, ranked by what the card says. Text size is the measure
// every format has, where pages are a PDF's alone, so an EPUB shows no pages
// rather than none counted, and the bar follows the ranking.
function renderLargest(corpus) {
  if (!corpus) return node("p", "form-note", NO_GENERATION);
  if (!STAT_LARGEST_BY.some((option) => option.value === corpus.largest_by)) {
    return node("p", "form-note",
      "The running app has an older statistics API. Restart it from its starting terminal to load text-size rankings. Refreshing the browser alone does not reload the app.");
  }
  const bySize = corpus.largest_by === "size";
  return statTable(
    [
      { label: "Source", value: sourceCell },
      { label: "Passages", numeric: true, value: (row) => row.passage_count },
      { label: "Text", numeric: true, value: (row) => bytes(row.text_bytes) || "—" },
      {
        label: "Pages",
        numeric: true,
        value: (row) => (row.physical_pages ? formatNumber(row.physical_pages) : "—"),
      },
    ],
    corpus.largest_sources || [],
    { bar: bySize ? "text_bytes" : "passage_count" },
  );
}

function renderBuild(payload) {
  const build = payload.last_build;
  if (!build) return node("p", "form-note", NO_GENERATION);
  const phases = Object.entries(build.phase_seconds || {}).map(([phase, seconds]) => ({
    phase: checkLabel(phase),
    seconds,
  }));
  const facts = [["Built", formatDate(build.created_at)]];
  if (build.seconds !== null && build.seconds !== undefined) facts.push(["Build time", `${formatNumber(build.seconds)} s`]);
  if (build.created_vector_count !== null && build.created_vector_count !== undefined) {
    facts.push([
      "Vectors",
      `${formatNumber(build.reused_vector_count)} reused · ${formatNumber(build.created_vector_count)} embedded`,
    ]);
  }
  if (build.rebuilt_document_count !== null && build.rebuilt_document_count !== undefined) {
    facts.push([
      "Sources",
      `${formatNumber(build.reused_document_count)} reused · ${formatNumber(build.rebuilt_document_count)} rebuilt`,
    ]);
  }
  if (build.excluded_corrupt_unit_count) facts.push(["Corrupt units left out", formatNumber(build.excluded_corrupt_unit_count)]);
  if (build.dense_truncated_chunk_count) facts.push(["Passages truncated for embedding", formatNumber(build.dense_truncated_chunk_count)]);
  const generations = payload.generations || {};
  facts.push(["Generations on disk", `${formatNumber(generations.count)} · ${bytes(generations.bytes)}`]);
  const groups = node("div", "stats-groups");
  groups.append(
    statGroup("Facts", factList(facts, "fact-list")),
    statGroup(
      "Time by phase",
      statTable(
        [{ label: "Phase", value: (row) => row.phase }, { label: "Seconds", numeric: true, value: (row) => row.seconds }],
        phases,
        { bar: "seconds" },
      ),
    ),
  );
  return groups;
}

const SCOPE_NAMES = { days: "Time scope", top: "List size", by: "Ranking" };

function scopeSelect(card, field, options) {
  const label = node("label", "stat-scope-field");
  label.append(node("span", "visually-hidden", `${SCOPE_NAMES[field]} for ${card.title}`));
  const select = node("select", "stat-scope-select");
  select.dataset.scope = field;
  const current = cardScope(card)[field];
  options.forEach((option) => {
    const item = node("option", "", field === "top" ? `Top ${option}` : option.label);
    item.value = String(field === "days" ? (option.days ?? "") : field === "by" ? option.value : option);
    select.append(item);
  });
  select.value = String(current ?? "");
  label.append(select);
  return label;
}

function statPanel(card) {
  const panel = node("article", "stat-panel");
  panel.dataset.card = card.id;
  const head = node("header", "stat-panel-head");
  head.append(node("h3", "stat-panel-title", card.title));
  const count = node("span", "count-badge");
  count.hidden = true;
  head.append(count);
  const controls = node("div", "stat-scope");
  if (card.scopes.includes("time")) controls.append(scopeSelect(card, "days", card.timeScopes || STAT_TIME_SCOPES));
  if (card.scopes.includes("size")) controls.append(scopeSelect(card, "top", STAT_SIZES));
  if (card.scopes.includes("by")) controls.append(scopeSelect(card, "by", STAT_LARGEST_BY));
  if (card.clearable && hasCapability("stats")) {
    controls.append(button("Clear", "clear-history", "", "text-button"));
  }
  head.append(controls);
  const body = node("div", "stat-panel-body");
  panel.append(head, body);
  state.statPanels.set(card.id, { panel, body, count, card });
  return panel;
}

async function refreshCard(card) {
  const parts = state.statPanels.get(card.id);
  if (!parts) return;
  const request = (parts.request || 0) + 1;
  parts.request = request;
  parts.panel.setAttribute("aria-busy", "true");
  parts.count.hidden = true;
  parts.body.replaceChildren(node("p", "form-note", "Loading…"));
  try {
    const scope = cardScope(card);
    const payload = await statRequest(card.source, scope);
    if (parts.request !== request) return;
    parts.body.replaceChildren(card.render(payload, scope));
    parts.count.hidden = !card.count;
    if (card.count) parts.count.textContent = formatNumber(card.count(payload));
  } catch (error) {
    if (parts.request !== request) return;
    parts.body.replaceChildren(node("p", "form-note", error.message));
  } finally {
    if (parts.request === request) parts.panel.setAttribute("aria-busy", "false");
  }
}

function buildStatsBoard() {
  const board = byId("stats-board");
  board.replaceChildren();
  state.statPanels = new Map();
  const cards = node("div", "stats-lists");
  STAT_CARDS.forEach((card) => cards.append(statPanel(card)));
  board.append(cards);
}

async function clearHistory() {
  try {
    const result = await api("/api/stats/history/clear", { method: "POST", body: "{}" });
    toast(result.message || "History cleared.");
    state.statData = new Map();
    await refreshCard(STAT_CARDS.find((card) => card.id === "history"));
  } catch (error) {
    toast(error.message, true);
  }
}

async function loadStats() {
  // Counts change with every search, so each visit asks again.
  state.statData = new Map();
  if (!state.statPanels.size) buildStatsBoard();
  await Promise.all(STAT_CARDS.map((card) => refreshCard(card)));
}

// A card's scope changes only that card, and is remembered for the next visit.
function changeScope(event) {
  const select = event.target.closest?.(".stat-scope-select");
  const panel = event.target.closest?.(".stat-panel");
  if (!select || !panel) return;
  const card = STAT_CARDS.find((entry) => entry.id === panel.dataset.card);
  if (!card) return;
  const scope = { ...cardScope(card) };
  if (select.dataset.scope === "days") scope.days = select.value === "" ? null : Number(select.value);
  else if (select.dataset.scope === "by") scope.by = select.value;
  else scope.top = Number(select.value);
  state.statScopes[card.id] = scope;
  saveScopes();
  void refreshCard(card);
}

// ------------------------------------------------------------- source view --
//
// One source and its passages, a page at a time: where each sits, its text
// trimmed, how big it is, whether it is excluded, and how often a search put it
// in a top five. The counts are the server's own and come with the page.

const SOURCE_CHUNKS_PER_PAGE = 20;

function chunkCard(chunk) {
  const card = node("article", "source-card chunk-card");
  const head = node("div", "source-title-row");
  head.append(node("span", "format-badge", `#${formatNumber(chunk.ordinal)}`));
  head.append(entityLink(locatorLabel(chunk.locator), passageHref(chunk.chunk_id), "locator-badge"));
  if (chunk.excluded) head.append(node("span", "state-badge state-badge-blocked", "Excluded from search"));
  const returned = chunk.top_five
    ? `Top five ${formatNumber(chunk.top_five)} · rank one ${formatNumber(chunk.rank_one)}`
    : "Not returned by any search";
  head.append(node("span", "state-badge", returned));
  card.append(head);
  card.append(node("p", "passage-text", `${chunk.text}${chunk.truncated ? "…" : ""}`));
  const facts = [`${formatNumber(chunk.characters)} characters`];
  if (chunk.embedding_token_count !== null && chunk.embedding_token_count !== undefined) {
    facts.push(`${formatNumber(chunk.embedding_token_count)} tokens`);
  }
  if (chunk.dense_truncated) facts.push("truncated for embedding");
  if (chunk.content_kind && chunk.content_kind !== "prose") facts.push(chunk.content_kind);
  (chunk.quality_flags || []).forEach((flag) => facts.push(String(flag).replaceAll("_", " ")));
  card.append(node("p", "result-meta", facts.join(" · ")));
  if (hasCapability("passage_context")) {
    const actions = node("div", "result-actions");
    actions.append(button("Read in context", "show-context", chunk.chunk_id));
    card.append(actions);
  }
  return card;
}

function renderSourceView(payload) {
  const source = payload.source || {};
  state.sourceView.title = inlineText(source.title) || source.source_relative_path || "Source";
  byId("source-view-title").textContent = state.sourceView.title;
  document.title = routeTitle("source");
  byId("source-view-message").hidden = true;

  const summary = byId("source-summary");
  summary.replaceChildren();
  summary.append(node("p", "source-byline", authorLine(source)));
  if (source.doi) summary.append(node("p", "source-doi", `doi:${inlineText(source.doi).replace(/^doi:/i, "")}`));
  if ((source.categories || []).length || (source.keywords || []).length || (source.project || []).length) {
    const tags = node("div", "source-tags");
    tags.append(tagList(source.project, "tag tag-project"));
    tags.append(tagList(source.categories, "tag"));
    tags.append(tagList(source.keywords, "tag tag-keyword"));
    summary.append(tags);
  }
  const actions = node("div", "result-actions");
  if (hasCapability("source_files") && source.source_relative_path) {
    actions.append(button("Open original", "open-source", source.source_relative_path));
  }
  const listed = state.sources.find((entry) => entry.source_id === source.source_id);
  if (hasCapability("metadata") && listed) {
    actions.append(button("Edit metadata", "edit-metadata", listed.document_id));
  }
  if (hasCapability("source_selection")) {
    actions.append(button("Search only this source", "only-source", source.source_id));
  }
  if (actions.childElementCount) summary.append(actions);
  const facts = [
    ["Path", source.source_relative_path || ""],
    ["Identifier", source.source_id || ""],
    ["Format", source.format || ""],
  ];
  if (source.physical_pages) facts.push(["Pages", formatNumber(source.physical_pages)]);
  if (source.withheld_units) {
    facts.push([
      "Text withheld",
      `${formatNumber(source.withheld_units)} unreadable units${
        source.unclean_character_rate ? `, ${(source.unclean_character_rate * 100).toFixed(1)}% of the text` : ""
      }`,
    ]);
  }
  if (source.excluded) facts.push(["Search", "This source is excluded from search"]);
  summary.append(factList(facts.filter(([, value]) => value), "fact-list identifier-list"));

  byId("source-stat-cards").replaceChildren(
    statCardFor("Passages", formatNumber(source.passage_count), "In the generation in use"),
    statCardFor("In a top five", formatNumber(source.top_five), "Times any of its passages was returned"),
    statCardFor("At rank one", formatNumber(source.rank_one), "Times it was the first passage"),
  );
  byId("source-chunks-count").textContent = formatNumber(source.passage_count);
  const list = byId("source-chunks");
  list.replaceChildren();
  (payload.chunks || []).forEach((chunk) => list.append(chunkCard(chunk)));
  if (!(payload.chunks || []).length) list.append(node("div", "no-records", "This source holds no passages."));
  state.sourceView.page = payload.page || 1;
  renderPager("source-chunk-pager", "chunks", payload.page || 1, payload.pages || 1);
}

function statCardFor(label, value, detail) {
  const card = node("article", "stat-card");
  card.append(node("span", "stat-label", label));
  card.append(node("strong", "stat-value", value));
  if (detail) card.append(node("span", "stat-detail", detail));
  return card;
}

async function loadSourceView() {
  const { id, page } = state.sourceView;
  const message = byId("source-view-message");
  if (!id) return;
  byId("source-chunks").replaceChildren(node("div", "no-records", "Reading the passages…"));
  try {
    const params = new URLSearchParams({
      source_id: id,
      page: String(page),
      page_size: String(SOURCE_CHUNKS_PER_PAGE),
    });
    renderSourceView(await api(`/api/source-chunks?${params}`));
  } catch (error) {
    message.hidden = false;
    message.textContent = error.message;
    byId("source-chunks").replaceChildren();
    byId("source-chunk-pager").hidden = true;
  }
}

// ------------------------------------------------------------------ routes --
//
// Each view is a page in the browser's history, addressed by the fragment:
// `#/search?q=…&k=…` with the search's filters, `#/sources?page=3&filter=…`,
// and `#/status` and the rest bare. Back, Forward, and a reload return a reader
// to the view, the search, and the page of sources they were on. The fragment
// is never sent to the server, so a query stays in this browser.

// A search asks for any number of passages the server allows; ten unless the
// reader or the address says otherwise. The field's own maximum is the server's.
const DEFAULT_TOP_K = "10";

function topKLimit() {
  const maximum = Number(byId("top-k").max);
  return Number.isInteger(maximum) && maximum > 0 ? maximum : 50;
}

function validTopK(value) {
  const text = String(value ?? "").trim();
  if (!/^\d+$/.test(text)) return false;
  const number = Number(text);
  return number >= 1 && number <= topKLimit();
}

// The search filters by the names the search operation takes, each holding the
// text of its field as typed.
const FILTER_PARAMS = [
  ["categories", "category-filter"],
  ["keywords", "keyword-filter"],
  ["categories_any", "category-any-filter"],
  ["projects_any", "project-any-filter"],
  ["authors_any", "author-filter"],
  ["titles_any", "title-filter"],
  ["languages_any", "language-filter"],
  ["source_ids", "include-source-filter"],
  ["exclude_source_ids", "exclude-source-filter"],
];

function parseRoute(hash) {
  const text = String(hash || "").replace(/^#\/?/, "");
  const split = text.indexOf("?");
  const view = split < 0 ? text : text.slice(0, split);
  return {
    // A view name is a plain word; anything else names no view.
    view: /^[a-z]+$/.test(view) ? view : "",
    params: new URLSearchParams(split < 0 ? "" : text.slice(split + 1)),
  };
}

function navItems() {
  return [...document.querySelectorAll(".sidebar-item")].filter((item) => !item.hidden);
}

// A view the sidebar has no item of its own for. It is drawn in a panel and its
// parent item stays marked, so a source read from the Sources list is still "in"
// Sources, and a passage opened from a result is still "in" Search.
const DETAIL_VIEWS = {
  source: { panel: "source", nav: "sources", capability: "sources" },
  passage: { panel: "search", nav: "search", capability: "passage_context" },
};

function activeView() {
  return state.view
    || navItems().find((item) => item.classList.contains("is-active"))?.dataset.view
    || "search";
}

function routeHash(view = activeView()) {
  const params = new URLSearchParams();
  if (view === "search") {
    if (state.searchedQuery) params.set("q", state.searchedQuery);
    const topK = byId("top-k").value;
    if (topK && topK !== DEFAULT_TOP_K) params.set("k", topK);
    FILTER_PARAMS.forEach(([name, field]) => {
      const value = byId(field).value.trim();
      if (value) params.set(name, value);
    });
  } else if (view === "sources") {
    if (state.sourcePages.sources > 1) params.set("page", String(state.sourcePages.sources));
    const filter = byId("source-filter").value.trim();
    if (filter) params.set("filter", filter);
    if (state.sourcePages.excluded > 1) params.set("excluded", String(state.sourcePages.excluded));
  } else if (view === "source") {
    params.set("id", state.sourceView.id);
    if (state.sourceView.page > 1) params.set("page", String(state.sourceView.page));
  } else if (view === "passage") {
    params.set("id", state.passageId);
  }
  const query = params.toString();
  return `#/${view}${query ? `?${query}` : ""}`;
}

// The page title names the view, the query, or the page of sources, so the
// browser's history list tells its entries apart.
function routeTitle(view) {
  const name = state.profile?.application_name || document.title;
  const item = navItems().find((entry) => entry.dataset.view === view);
  const label = [...(item?.querySelector(".nav-item-name")?.childNodes || [])]
    .map((child) => child.textContent.trim())
    .find(Boolean) || view;
  if (view === "source") {
    const page = state.sourceView.page > 1 ? ` · page ${state.sourceView.page}` : "";
    return `${state.sourceView.title || "Source"}${page} · ${name}`;
  }
  if (view === "passage") return `Passage in context · ${name}`;
  let detail = "";
  if (view === "search" && state.searchedQuery) detail = ` “${state.searchedQuery}”`;
  if (view === "sources" && state.sourcePages.sources > 1) detail = ` · page ${state.sourcePages.sources}`;
  return `${label}${detail} · ${name}`;
}

function recordRoute({ replace = false } = {}) {
  if (state.restoringRoute) return;
  const view = activeView();
  const hash = routeHash(view);
  document.title = routeTitle(view);
  if (hash === window.location.hash) return;
  if (replace) window.history.replaceState(null, "", hash);
  else window.history.pushState(null, "", hash);
}

function searchKey() {
  return JSON.stringify([
    byId("query").value.trim(),
    byId("top-k").value,
    FILTER_PARAMS.map(([, field]) => byId(field).value.trim()),
  ]);
}

function pageParam(value) {
  const page = Number.parseInt(value || "1", 10);
  return Number.isFinite(page) && page > 0 ? page : 1;
}

// Draw the page the fragment names. A search runs again only when what the
// fragment asks for differs from the results already on screen, so returning
// from the sources to the search costs nothing.
async function applyRoute() {
  const { view, params } = parseRoute(window.location.hash);
  const available = navItems();
  const detail = DETAIL_VIEWS[view];
  const detailId = params.get("id") || "";
  const wantsDetail = Boolean(detail && detailId && hasCapability(detail.capability));
  // A detail view named without what it shows falls back to the list it belongs to.
  const named = detail && !wantsDetail ? detail.nav : view;
  const target = wantsDetail
    ? view
    : available.some((item) => item.dataset.view === named)
      ? named
      : available[0]?.dataset.view || "search";
  // A dialog belongs to the page being left, so it closes without that closing
  // being read as a request to leave a passage the route is still showing.
  state.closingForRoute = true;
  document.querySelectorAll("dialog[open]").forEach((dialog) => {
    if (!(target === "passage" && dialog.id === "context-dialog")) dialog.close();
  });
  state.closingForRoute = false;
  let rerun = false;
  let showPassage = "";
  state.restoringRoute = true;
  try {
    if (target === "source") {
      state.sourceView = { id: detailId, page: pageParam(params.get("page")), title: "" };
    } else if (target === "passage") {
      state.passageId = detailId;
      showPassage = detailId;
    }
    if (target === "search") {
      const query = params.get("q") || "";
      byId("query").value = query;
      const wanted = params.get("k") || DEFAULT_TOP_K;
      byId("top-k").value = validTopK(wanted) ? String(Number(wanted)) : DEFAULT_TOP_K;
      FILTER_PARAMS.forEach(([name, field]) => {
        byId(field).value = params.get(name) || "";
      });
      syncFilterSummary();
      state.searchedQuery = query;
      if (!query) clearResults();
      else rerun = searchKey() !== state.searchKey;
    } else if (target === "sources") {
      byId("source-filter").value = params.get("filter") || "";
      state.sourcePages.sources = pageParam(params.get("page"));
      state.sourcePages.excluded = pageParam(params.get("excluded"));
      renderSourcePage();
      renderExcludedPage();
    }
    switchView(target);
  } finally {
    state.restoringRoute = false;
  }
  // A fragment that named no view, a hidden one, or a page past the end is
  // replaced by the page actually drawn.
  recordRoute({ replace: true });
  if (target === "source") void loadSourceView();
  if (showPassage) void showContext(showPassage);
  if (rerun) await runSearch();
}

// Following a name goes through the router rather than the browser's own
// fragment jump, so a view the page draws itself is drawn in the same pass and
// Back returns to where the name was clicked.
function navigateTo(href) {
  if (!href || href === window.location.hash) return;
  state.passagePushed = href.startsWith("#/passage");
  window.history.pushState(null, "", href);
  void applyRoute();
}

// The route of a passage opened from a link is the dialog's own: closing it
// returns to the page it was opened from, or to the search when the address was
// opened cold.
function leavePassageRoute() {
  if (state.closingForRoute || state.restoringRoute || state.view !== "passage") return;
  if (state.passagePushed) {
    state.passagePushed = false;
    window.history.back();
    return;
  }
  state.passageId = "";
  window.history.replaceState(null, "", "#/search");
  void applyRoute();
}

function switchView(name, { moveFocus = false } = {}) {
  let opened = null;
  state.view = name || "";
  const detail = DETAIL_VIEWS[name];
  const panelName = detail ? detail.panel : name;
  const navName = detail ? detail.nav : name;
  document.querySelectorAll(".sidebar-item").forEach((item) => {
    const active = item.dataset.view === navName;
    item.classList.toggle("is-active", active);
    item.querySelector(".nav-item")?.classList.toggle("is-active", active);
    // This is navigation rather than a tab set, so the current view is marked
    // the way a link to the current page is marked.
    if (active) item.querySelector(".nav-item")?.setAttribute("aria-current", "page");
    else item.querySelector(".nav-item")?.removeAttribute("aria-current");
  });
  document.querySelectorAll(".view-panel").forEach((panel) => {
    const active = panel.dataset.panel === panelName;
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
  // The counts change with every search, so the view reads them when it opens.
  if (name === "stats" && hasCapability("stats")) void loadStats();
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
  const title = inlineText(hit.title) || hit.source_path;

  // The rank, the title, the place in the source, and the menu share one row,
  // so the passage below spans the whole card and sits centred in it. The
  // citation is not printed: it repeats the title, the byline, and the locator,
  // and "Copy citation" still copies it whole.
  const header = node("div", "result-card-header");
  header.append(node("span", "result-rank", String(hit.rank).padStart(2, "0")));
  // The title names the source and goes to all of its passages; the place in it
  // names this passage and goes to the text around it.
  const heading = node("h3", "result-title");
  heading.append(hit.source_id && hasCapability("sources") ? entityLink(title, sourceHref(hit.source_id)) : title);
  header.append(heading);
  header.append(
    hit.chunk_id && hasCapability("passage_context")
      ? entityLink(locatorLabel(hit.locator), passageHref(hit.chunk_id), "locator-badge")
      : node("span", "locator-badge", locatorLabel(hit.locator)),
  );
  const actions = [
    button(state.profile?.copy_text_label || "Copy passage", "copy-passage", hit.chunk_id),
    button("Copy citation", "copy-citation", hit.chunk_id),
  ];
  if (hasCapability("passage_context")) {
    actions.push(button("Nearby context", "show-context", hit.chunk_id));
  }
  const source = sourceForDocument(hit.document_id);
  if (source && hasCapability("source_files")) {
    actions.push(button("Open original", "open-source", source.source_relative_path));
  }
  if (hasCapability("chunk_exclusion")) {
    actions.push(chunkAction(hit));
  }
  header.append(actionMenu(title, actions));
  card.append(header);
  card.append(node("div", "result-byline", authorLine(hit)));
  card.append(node("p", "passage-text", readableText(hit.text)));

  if ((hit.categories || []).length || (hit.keywords || []).length) {
    const tags = node("div", "tag-row");
    tags.append(tagList(hit.categories, "tag"));
    tags.append(tagList(hit.keywords, "tag tag-keyword"));
    card.append(tags);
  }

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
  if (hit.doi) rows.push(["DOI", inlineText(hit.doi).replace(/^doi:/i, "")]);
  details.append(factList(rows, "fact-list score-facts"));
  card.append(details);
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
  // The label is a property of the whole results view, not of a passage, so it
  // is written once above the list rather than repeated on every card.
  const notice = byId("results-notice");
  notice.textContent = state.profile?.result_text_label || "Retrieved passage";
  notice.hidden = !count;
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
  state.searchKey = "";
  byId("results").replaceChildren();
  byId("search-summary").hidden = true;
  byId("results-notice").hidden = true;
  byId("search-empty").hidden = false;
}

// A submitted search is a page of its own in the browser's history, so Back
// returns to the search before it and a reload runs this one again.
async function search(event) {
  event.preventDefault();
  const query = byId("query").value.trim();
  if (!query) return;
  state.searchedQuery = query;
  recordRoute();
  await runSearch();
}

async function runSearch() {
  const query = byId("query").value.trim();
  if (!query) return;
  if (!validTopK(byId("top-k").value)) {
    toast(`Results must be a whole number from 1 to ${topKLimit()}.`, true);
    return;
  }
  const key = searchKey();
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
    state.searchKey = key;
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

// A source is already on this machine, so it opens in the desktop's own viewer
// and nothing is downloaded. A machine with no desktop to hand it to answers 501,
// and the file is shown in the browser instead.
async function openSource(path) {
  if (state.profile?.source_open_mode === "browser") {
    window.open(`/api/source-file?path=${encodeURIComponent(path)}`, "_blank", "noopener");
    return;
  }
  try {
    const result = await api("/api/open-source", {
      method: "POST",
      body: JSON.stringify({ source_path: path }),
    });
    toast(`Opened ${result.filename} in ${result.viewer}.`);
  } catch (error) {
    if (error.status !== 501) {
      toast(error.message, true);
      return;
    }
    window.open(`/api/source-file?path=${encodeURIComponent(path)}`, "_blank", "noopener");
  }
}

async function copyText(text, message) {
  try {
    if (!navigator.clipboard?.writeText) {
      copyTextWithoutClipboard(text, message);
      return;
    }
    await navigator.clipboard.writeText(text);
    toast(message);
  } catch (_error) {
    copyTextWithoutClipboard(text, message);
  }
}

function copyTextWithoutClipboard(text, message) {
  // HTTP LAN pages do not have the secure-context Clipboard API. Try the
  // user-gesture copy path, then expose selectable text if the browser refuses.
  const field = document.createElement("textarea");
  field.value = text;
  field.setAttribute("aria-label", "Text to copy");
  field.style.position = "fixed";
  field.style.opacity = "0";
  document.body.append(field);
  let copied = false;
  try {
    field.focus();
    field.select();
    field.setSelectionRange(0, text.length);
    copied = Boolean(document.execCommand?.("copy"));
  } catch (_error) {
    copied = false;
  } finally {
    field.remove();
  }
  if (copied) toast(message);
  else window.prompt("Select and copy this text:", text);
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
    const requestedPassage = (payload.context || []).find(
      (passage) => passage.chunk_id === payload.requested_chunk_id,
    ) || payload.context?.[0];
    const contextTitle = byId("context-title");
    const heading = requestedPassage?.title || "Passage context";
    // The dialog names the source, and the name goes to all of its passages.
    contextTitle.replaceChildren(
      requestedPassage?.source_id && hasCapability("sources")
        ? entityLink(heading, sourceHref(requestedPassage.source_id))
        : document.createTextNode(heading),
    );
    if (!byId("context-dialog").open) byId("context-dialog").showModal();
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

// A recorded fact is stated when the manifest carried it and named as missing
// when it did not. A generation that recorded no model must never be shown
// today's default beside it: the reader asked what this build used, and a
// substituted value would answer a different question.
const RECORDED_MISSING = "Not recorded";

function recordedObject(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : null;
}

function recordedText(value) {
  if (typeof value === "string" && value.trim()) return value.trim();
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  return RECORDED_MISSING;
}

function recordedList(value) {
  const items = Array.isArray(value)
    ? value.filter((item) => typeof item === "string" && item.trim())
    : [];
  return items.length ? items.join(", ") : RECORDED_MISSING;
}

// The facts a reader compares between builds, each read from the manifest this
// generation recorded. Every value is a string by the time it is drawn, so a
// manifest that carried markup is shown as the text it is.
function recordedConfigFacts(generation) {
  const chunking = recordedObject(generation.chunking);
  const retrieval = recordedObject(generation.retrieval);
  const dense = recordedObject(retrieval && retrieval.dense);
  const bm25 = recordedObject(retrieval && retrieval.bm25);
  const reranker = recordedObject(retrieval && retrieval.reranker);
  return [
    ["Embedding model", dense ? recordedText(dense.embedding_model) : RECORDED_MISSING],
    ["Embedding revision", dense ? recordedText(dense.embedding_model_revision) : RECORDED_MISSING],
    ["Embedding dimension", dense ? recordedText(dense.embedding_dimension) : RECORDED_MISSING],
    ["Embedding backend", dense ? recordedText(dense.backend) : RECORDED_MISSING],
    ["Reranker model", reranker ? recordedText(reranker.model) : RECORDED_MISSING],
    ["Reranker revision", reranker ? recordedText(reranker.model_revision) : RECORDED_MISSING],
    ["Chunk size", chunking ? recordedText(chunking.chunk_size) : RECORDED_MISSING],
    ["Chunk overlap", chunking ? recordedText(chunking.chunk_overlap) : RECORDED_MISSING],
    ["Chunk tokenizer", chunking ? recordedText(chunking.tokenizer) : RECORDED_MISSING],
    ["BM25 language", bm25 ? recordedText(bm25.language) : RECORDED_MISSING],
    ["Retrieval methods", retrieval ? recordedList(retrieval.available_methods) : RECORDED_MISSING],
  ];
}

function policyFacts(generation) {
  return [
    ["Retrieval policy fingerprint", recordedText(generation.retrieval_policy_fingerprint)],
    ["Extraction policy", recordedText(generation.extraction_policy_version)],
    ["Cleaning policy", recordedText(generation.cleaning_policy_version)],
    ["Artifact policy", recordedText(generation.artifact_policy_version)],
  ];
}

// The summary is one line so a table of builds stays comparable down a column,
// and the full blocks below it open on request.
function recordedConfigSummary(generation) {
  return node(
    "p",
    "recorded-config-summary",
    recordedConfigFacts(generation)
      .map(([label, value]) => `${label}: ${value}`)
      .join(" · "),
  );
}

// The expandable form states the whole recorded configuration: every fact
// above, the policy versions, and the raw chunking and retrieval blocks as
// text. It is labelled as the configuration this build recorded rather than
// the settings in force, because selection never restores a recorded config.
function recordedConfigBlock(generation) {
  const details = node("details", "recorded-config");
  details.append(node("summary", "recorded-config-title", "Recorded build configuration"));
  details.append(
    node(
      "p",
      "form-note",
      "Facts this build recorded. They are not the project's active settings; "
        + "loading a generation does not restore them.",
    ),
  );
  details.append(factList(recordedConfigFacts(generation), "fact-list recorded-config-facts"));
  details.append(node("p", "recorded-config-label", "Policy facts (recorded)"));
  details.append(factList(policyFacts(generation), "fact-list recorded-config-facts"));
  [
    ["Chunking", recordedObject(generation.chunking)],
    ["Retrieval", recordedObject(generation.retrieval)],
  ].forEach(([label, value]) => {
    details.append(node("p", "recorded-config-label", `${label} (recorded)`));
    details.append(node("pre", "recorded-config-json", JSON.stringify(value ?? null, null, 2)));
  });
  return details;
}

// One row per build, in a table: a reader comparing builds reads down a column,
// and ten builds as cards of six labelled lines each were a page of scrolling.
// The identifier is the row's head because it is what a removal or a load names.
function generationRow(generation) {
  const row = node("tr");
  const head = node("th", "record-id-cell");
  head.scope = "row";
  head.append(node("code", "record-id", generation.generation_id));
  if (generation.is_current) {
    head.append(node("span", "state-badge state-badge-current", "In use"));
  } else {
    head.append(node("span", "state-badge", "Retained"));
  }
  // A generation whose manifest cannot be read is named rather than counted,
  // because a reader deciding what to remove needs to know which rows are facts.
  if (generation.manifest_error) {
    head.append(node("p", "record-warning", `Manifest unreadable: ${generation.manifest_error}`));
    head.append(node("p", "record-warning", "Not possible: this generation cannot be loaded."));
  }
  head.append(recordedConfigSummary(generation));
  if (generation.partial) {
    head.append(node("p", "record-warning", `Partial: ${formatNumber(generation.skipped_source_count)} sources skipped. Retry ingestion, or use Load to select this generation manually.`));
  }
  head.append(recordedConfigBlock(generation));
  row.append(head);
  row.append(node("td", "", generation.created_at ? formatDate(generation.created_at) : "Unknown"));
  row.append(node("td", "number-cell", formatNumber(generation.chunk_count ?? 0)));
  row.append(node("td", "number-cell", formatNumber(generation.document_count ?? 0)));
  row.append(node("td", "number-cell", bytes(generation.size_bytes)));
  const action = node("td", "action-cell");
  // The one a search reads offers no action: a load would be a no-op and a
  // removal is refused, so neither is offered rather than offered and refused.
  // A damaged manifest row keeps its removal because that is how the reader
  // reclaims the space, but it is never offered a load it could not survive.
  if (!generation.is_current) {
    if (!generation.manifest_error) {
      const load = node("button", "text-button", "Load");
      load.type = "button";
      load.className = "text-button generation-action";
      load.setAttribute("aria-label", `Load ${generation.generation_id}`);
      load.disabled = state.busy;
      load.addEventListener("click", () => openGenerationLoad(generation.generation_id));
      state.generationActions.push(load);
      action.append(load);
    }
    const drop = node("button", "text-button", "Remove");
    drop.type = "button";
    drop.className = "text-button generation-action";
    drop.setAttribute("aria-label", `Remove ${generation.generation_id}`);
    drop.disabled = state.busy;
    drop.addEventListener("click", () => openGenerationRemoval(generation.generation_id));
    state.generationActions.push(drop);
    action.append(drop);
  }
  row.append(action);
  return row;
}

const GENERATION_COLUMNS = ["Generation", "Built", "Passages", "Sources", "Size"];

function renderGenerations(generations) {
  // The id stays the container hook other code and tests already read.
  const container = byId("generation-chips");
  container.replaceChildren();
  // The buttons the busy state holds are rebuilt with the rows, so a refresh
  // never leaves a stale control in the list.
  state.generationActions = [];
  byId("generation-count").textContent = formatNumber(generations.length);
  foldList(byId("generation-fold"), generations.length);
  if (!generations.length) {
    container.append(node("p", "form-note", "This project has no build yet."));
    return;
  }
  const total = state.status?.retained_generation_bytes
    ?? generations.reduce((sum, generation) => sum + (generation.size_bytes || 0), 0);
  container.append(
    node(
      "p",
      "form-note generation-total",
      `${formatNumber(generations.length)} build${generations.length === 1 ? "" : "s"} use ${bytes(total)} on disk.`,
    ),
  );
  const table = node("table", "record-table");
  const header = node("tr");
  GENERATION_COLUMNS.forEach((label, index) => {
    const cell = node("th", index >= 2 ? "number-cell" : "", label);
    cell.scope = "col";
    header.append(cell);
  });
  const actionHead = node("th", "action-cell");
  actionHead.scope = "col";
  actionHead.append(node("span", "visually-hidden", "Action"));
  header.append(actionHead);
  const head = node("thead");
  head.append(header);
  const body = node("tbody");
  generations.forEach((generation) => body.append(generationRow(generation)));
  table.append(head, body);
  const scroller = node("div", "table-scroll");
  scroller.append(table);
  container.append(scroller);
}

function syncGenerationSubmit() {
  const typed = byId("generation-confirm").value.trim();
  byId("generation-submit").disabled =
    state.busy || !typed || typed !== byId("generation-remove-id").value;
}

function openGenerationRemoval(generationId) {
  byId("generation-remove-id").value = generationId;
  byId("generation-remove-name").textContent = generationId;
  byId("generation-confirm").value = "";
  syncGenerationSubmit();
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

// Loading a generation is a deliberate choice about which indexed evidence a
// search reads, so it is confirmed before the request is sent. The dialog says
// what changes and what does not: the corpus on disk and the active settings
// are untouched.
function openGenerationLoad(generationId) {
  byId("generation-load-id").value = generationId;
  byId("generation-load-name").textContent = generationId;
  byId("generation-load-error").hidden = true;
  byId("generation-load-error").textContent = "";
  byId("generation-load-submit").disabled = state.busy;
  byId("generation-load-dialog").showModal();
}

async function loadGeneration(event) {
  event.preventDefault();
  const generationId = byId("generation-load-id").value;
  if (!generationId) return;
  setBusy(true, "Loading the generation…");
  try {
    const result = await api("/api/generations/use", {
      method: "POST",
      body: JSON.stringify({ generation_id: generationId }),
    });
    byId("generation-load-dialog").close();
    // Discard cached passages after selection, including a server no-op: another
    // client may already have switched the generation since this page searched.
    clearResults();
    toast(result.message || `Search now reads generation ${compactId(generationId)}.`);
    // The whole status is refreshed so the corpus counts, the retrieval methods,
    // and any BM25-only warning describe the generation now selected.
    await loadWorkspace();
  } catch (error) {
    // A generation that cannot be loaded is refused, not repaired. The dialog
    // stays open and carries the server's own reason, so the reader learns the
    // selection is not possible beside the choice that caused it. The current
    // generation, its results, and the settings are untouched.
    const line = byId("generation-load-error");
    line.hidden = false;
    line.textContent = `Not possible: ${error.message}`;
  } finally {
    setBusy(false);
    if (state.status) configureRetrieval(state.status);
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
  // The default is named only where it differs, because "Value: 8, Default: 8"
  // says one thing twice on most rows.
  if (
    setting.default !== undefined
    && settingValueLabel(setting.default) !== settingValueLabel(setting.value)
  ) {
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
    // A section is a fold like any other long list, and a control in a closed
    // fold is still read and sent, so folding hides nothing from a save.
    const block = node("details", "settings-section fold");
    block.dataset.fold = `settings-${slugify(section.key || section.title)}`;
    const summary = node("summary", "fold-summary");
    summary.append(node("h4", "", section.title || section.key));
    const count = node("span", "count-badge");
    summary.append(count);
    block.append(summary);
    let index = 0;
    for (const setting of section.settings || []) {
      if (!setting?.key) continue;
      state.settings.set(setting.key, setting);
      block.append(settingRow(setting, `${index}-${slugify(setting.key)}`));
      index += 1;
    }
    count.textContent = formatNumber(index);
    foldList(block, index);
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

function syncSettingsConfirmSubmit() {
  const pending = state.pendingSettings;
  byId("settings-confirm-submit").disabled =
    state.busy || !pending || byId("settings-confirm-word").value.trim() !== pending.word;
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
  syncSettingsConfirmSubmit();
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
  const control = button("Exclude this passage", "exclude-chunk", chunk.chunk_id);
  control.dataset.chunkSource = chunk.source_relative_path || chunk.source_path || "Unknown source";
  control.dataset.chunkLocator = chunk.locator ? locatorLabel(chunk.locator) : "No locator reported";
  return control;
}

function chunkAction(chunk) {
  // A chunk the server already lists as excluded is offered a restore instead,
  // so the reader is never asked to exclude what is already out.
  return state.chunkExclusions.has(chunk.chunk_id)
    ? button("Restore this passage", "restore-chunk", chunk.chunk_id)
    : chunkExcludeButton(chunk);
}

// What a passage decision did, from the fields the server returned. A passage
// this generation does not hold is excluded for later generations only, so the
// reader is told nothing on screen changed.
function passageDecisionMessage(result, included) {
  if (result.status === "unchanged") {
    return included ? "This passage is already in search." : "This passage is already excluded with this reason.";
  }
  if (included) return "Passage restored to search.";
  return result.in_current_generation === false
    ? "Passage exclusion saved. This generation does not hold the passage, so no current result changes."
    : "Passage excluded from search. The original file is unchanged.";
}

function openChunkExclusion(control) {
  byId("chunk-id").value = control.dataset.value;
  byId("chunk-name").textContent = `${control.dataset.chunkSource} · ${control.dataset.chunkLocator}`;
  byId("chunk-reason").value = "";
  byId("chunk-error").hidden = true;
  byId("chunk-dialog").showModal();
}

async function writeChunkInclusion(payload, fromDialog = false) {
  setBusy(true, payload.included ? "Restoring the passage…" : "Excluding the passage…");
  try {
    const result = await api("/api/chunk-inclusion", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    if (fromDialog) byId("chunk-dialog").close();
    toast(passageDecisionMessage(result, payload.included));
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
  // A passage this generation does not hold is named as such, because restoring
  // it changes a later generation rather than the passages already on screen.
  body.append(
    node(
      "span",
      "state-badge state-badge-blocked",
      entry.in_current_generation === false
        ? "Not in the current generation"
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
  foldList(byId("chunk-exclusion-fold"), entries.length);
  // The summary is the page's own sentence over the server's fields. The
  // server's message says "chunk", the word its agent tools use; this page calls
  // the same thing a passage everywhere. An empty list says so once, in the list.
  const withheld = entries.filter((entry) => entry.in_current_generation !== false).length;
  const message = byId("chunk-exclusion-message");
  message.hidden = !entries.length;
  message.textContent = withheld === entries.length
    ? `Every excluded passage is withheld from current search.`
    : `${formatNumber(withheld)} of ${formatNumber(entries.length)} excluded passages are withheld from current search; the others are not in this generation, and no ingestion restores them.`;

  const list = byId("chunk-exclusion-list");
  list.replaceChildren();
  if (!entries.length) {
    list.append(node("div", "no-records", "No passage is excluded from search."));
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

// Source estimates are coarse; passage estimates use the known phase total.
function ingestionPhaseCount(status) {
  return status.overall_progress?.unit === "sources"
    ? status.overall_progress : status.progress;
}

function renderIngestionProgress(status) {
  const bar = byId("ingestion-progress");
  const count = ingestionPhaseCount(status);
  const valid = count && Number.isFinite(count.completed) && Number.isFinite(count.total)
    && count.total > 0 && count.completed >= 0 && count.completed <= count.total;
  bar.hidden = !valid;
  if (valid) {
    bar.max = count.total;
    bar.value = count.completed;
  }
}

function ingestionEtaCount(status) {
  const sources = ["extraction", "chunking"].includes(status.phase);
  if (!sources && !["embedding", "dense_indexing", "qdrant_indexing"].includes(status.phase)) return null;
  const count = sources ? status.overall_progress : status.progress;
  return count?.unit === (sources ? "sources" : "chunks") && Number.isFinite(count.completed)
    && Number.isFinite(count.total) && count.total > 0
    && count.completed >= 0 && count.completed <= count.total ? count : null;
}

function phaseEtaEstimator() {
  let sample = null;
  return (status, now) => {
    const count = ingestionEtaCount(status);
    if (!count || !Number.isFinite(count.completed) || !Number.isFinite(count.total)
        || count.total <= 0 || count.completed < 0 || count.completed > count.total
        || !Number.isFinite(now)) {
      sample = null;
      return null;
    }
    const key = JSON.stringify([status.build_id, status.phase, count.unit, count.total]);
    if (!sample || sample.key !== key || count.completed < sample.completed || now < sample.time) {
      sample = { key, completed: count.completed, time: now,
        start: now, initial: count.completed, advances: 0, advancedAt: now,
        shown: null, shownAt: now, seconds: null };
      return null;
    }
    if (count.completed > sample.completed) {
      sample.advances++;
      sample.advancedAt = now;
    }
    sample.completed = count.completed;
    sample.time = now;
    const elapsed = (now - sample.start) / 1000;
    const delta = count.completed - sample.initial;
    const sources = count.unit === "sources";
    if ((sources ? delta < 3 : sample.advances < 2) || elapsed <= 0 || delta <= 0
        || count.completed === count.total) return null;
    // Idle polls do not inflate the measured rate or revise the displayed ETA.
    const seconds = (count.total - count.completed) * (sample.advancedAt - sample.start) / 1000 / delta;
    const minutes = Math.max(1, Math.round(seconds / 60));
    const step = minutes >= 10 ? 5 : 1;
    const lower = Math.max(step, Math.floor(minutes * 0.7 / step) * step);
    const upper = Math.max(lower + step, Math.ceil(minutes * 1.3 / step) * step);
    const candidate = sources ? `${lower}–${upper} min` : seconds;
    if (sample.shown === null || (now - sample.shownAt >= 60000
        && Math.abs(seconds - sample.seconds) >= Math.max(30, sample.seconds * 0.25))) {
      sample.shown = candidate;
      sample.seconds = seconds;
      sample.shownAt = now;
    }
    return sample.shown;
  };
}

function ingestionStallDetector() {
  let sample = null;
  return (status, now) => {
    const count = ingestionEtaCount(status) || ingestionPhaseCount(status);
    if (!count || !Number.isFinite(count.completed) || !Number.isFinite(now)) {
      sample = null;
      return false;
    }
    const key = JSON.stringify([status.build_id, status.phase, count.unit, count.total]);
    const activity = JSON.stringify([count.completed, status.source,
      status.source_progress?.completed, status.progress?.completed]);
    if (!sample || sample.key !== key || activity !== sample.activity || now < sample.time) {
      sample = { key, activity, time: now };
    }
    return count.completed < count.total && now - sample.time >= 60000;
  };
}

function formatPhaseEta(seconds) {
  if (seconds >= 60) return `${Math.max(1, Math.round(seconds / 60))} min`;
  const rounded = Math.max(5, Math.round(seconds / 5) * 5);
  const minutes = Math.floor(rounded / 60);
  const remainder = rounded % 60;
  return minutes ? `${minutes} min${remainder ? ` ${remainder} sec` : ""}` : `${rounded} sec`;
}

function liveIngestionSummary(progress, localEta = null, stalled = false) {
  const parts = [String(progress.phase || "Building indexes").replaceAll("_", " ")];
  const counts = (label, value) => {
    if (value && Number.isFinite(value.completed) && Number.isFinite(value.total) && value.total > 0) {
      parts.push(`${label}${formatNumber(value.completed)} / ${formatNumber(value.total)} ${String(value.unit || "").replaceAll("_", " ")}`.trim());
    }
  };
  if (progress.source) parts.push(String(progress.source));
  counts("", ingestionPhaseCount(progress));
  const etaCount = ingestionEtaCount(progress);
  if (etaCount && etaCount.completed < etaCount.total) {
    if (Number.isFinite(localEta) && localEta > 0) {
      parts.push(`ETA ${formatPhaseEta(localEta)}`);
    } else if (typeof localEta === "string") parts.push(`ETA ${localEta}`);
  }
  if (stalled) parts.push("No progress for 60s");
  return parts.join(" · ");
}

// One bounded status request at a time; never reload the source inventory while
// the single ingest POST owns the build. Late responses cannot overwrite the UI.
function pollIngestionProgress() {
  const estimate = phaseEtaEstimator();
  const stalled = ingestionStallDetector();
  let stopped = false;
  let timer;
  let controller;
  const poll = async () => {
    if (stopped) return;
    controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 10000);
    try {
      const status = await api("/api/status", { signal: controller.signal });
      if (!stopped && status.ingestion_progress) {
        const progress = status.ingestion_progress;
        renderIngestionProgress(progress);
        const now = performance.now();
        byId("busy-message").textContent = `${liveIngestionSummary(progress, estimate(progress, now), stalled(progress, now))} · Status checked ${new Date().toLocaleTimeString()}`;
      } else if (!stopped) {
        renderIngestionProgress({});
        estimate({}, performance.now());
        stalled({}, performance.now());
      }
    } catch (_error) {
      // Status failures do not cancel ingestion or replace its result.
    } finally {
      window.clearTimeout(timeout);
      if (!stopped) timer = window.setTimeout(poll, 2000);
    }
  };
  timer = window.setTimeout(poll, 2000);
  const stop = () => {
    stopped = true;
    renderIngestionProgress({});
    window.clearTimeout(timer);
    controller?.abort();
    window.removeEventListener("pagehide", stop);
  };
  window.addEventListener("pagehide", stop, { once: true });
  return stop;
}

async function ingest(event) {
  event.preventDefault();
  if (state.busy) return;
  const chunkSize = Number(byId("chunk-size").value);
  const chunkOverlap = Number(byId("chunk-overlap").value);
  if (hasCapability("chunk_settings") && chunkOverlap >= chunkSize) {
    toast("Chunk overlap must be smaller than chunk size.", true);
    return;
  }
  setBusy(true, state.profile?.ingest_busy_message || "Building the indexes. This can take several minutes…");
  const stopProgress = pollIngestionProgress();
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
    stopProgress();
    byId("ingest-dialog").close();
    if (result.status === "in_progress") {
      toast("Ingestion is still in progress. The generation is not ready yet.");
    } else if (result.status === "partial") {
      toast(`Partial generation ${compactId(result.generation_id)} retained, not selected. Existing results are unchanged. Retry ingestion, or use Load in the generation list to select it manually.`);
    } else {
      clearResults();
      toast(`Generation ${compactId(result.generation_id)} is ready with ${formatNumber(result.chunk_count)} passages.`);
    }
    await loadWorkspace();
  } catch (error) {
    toast(error.message, true);
  } finally {
    stopProgress();
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
  else if (action === "clear-history") void clearHistory();
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
  // Each keystroke refines one page rather than adding one to the history.
  recordRoute({ replace: true });
}

function initialize() {
  state.statScopes = readSavedScopes();
  // A name that goes somewhere is followed through the router, so the view it
  // names is drawn in the same pass, and a modified click keeps the browser's own
  // behaviour of opening it elsewhere.
  document.addEventListener("click", (event) => {
    const link = event.target.closest?.("a.entity-link");
    if (!link || event.defaultPrevented) return;
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    navigateTo(link.getAttribute("href"));
  });
  byId("context-dialog").addEventListener("close", leavePassageRoute);
  byId("stats-board").addEventListener("change", changeScope);
  byId("stats-board").addEventListener("click", handleAction);
  byId("source-summary").addEventListener("click", handleAction);
  byId("source-chunks").addEventListener("click", handleAction);
  byId("source-chunk-pager").addEventListener("click", handleAction);
  document.querySelectorAll(".sidebar-item").forEach((item) => {
    item.querySelector(".nav-item")?.addEventListener("click", () => {
      switchView(item.dataset.view, { moveFocus: true });
      recordRoute();
    });
  });
  byId("nav-toggle").addEventListener("click", toggleNav);
  byId("nav-scrim").addEventListener("click", () => closeNav());
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    const menu = document.activeElement?.closest?.("details.card-menu[open]");
    if (menu) {
      menu.open = false;
      menu.querySelector("summary")?.focus();
      return;
    }
    closeNav({ restoreFocus: true });
  });
  // One listener serves every fold and every card menu, including the ones a
  // render draws later. A fold remembers what the reader chose; a menu closes
  // when its action is taken or the reader clicks elsewhere.
  document.addEventListener("click", (event) => {
    const summary = event.target.closest?.("summary");
    const fold = summary?.parentElement;
    if (fold?.classList.contains("fold") && foldKey(fold)) {
      state.folds.set(foldKey(fold), !fold.open);
    }
    const menu = event.target.closest?.("details.card-menu");
    closeMenus(menu);
    if (menu && event.target.closest("[data-action]")) menu.open = false;
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
  // The query box is multi-line, so Enter keeps adding a line and Ctrl+Enter
  // (Cmd+Enter on a Mac) searches without reaching for the button.
  byId("query").addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || !(event.ctrlKey || event.metaKey)) return;
    event.preventDefault();
    byId("search-form").requestSubmit();
  });
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
  byId("generation-confirm").addEventListener("input", syncGenerationSubmit);
  byId("generation-load-form").addEventListener("submit", loadGeneration);
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
  byId("settings-confirm-word").addEventListener("input", syncSettingsConfirmSubmit);
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
  FILTER_FIELDS.forEach((field) => {
    byId(field).addEventListener("change", () => recordRoute({ replace: true }));
  });
  // Back and Forward move between the pages this workspace recorded. An open
  // dialog belongs to the page being left, so it closes.
  window.addEventListener("popstate", () => {
    state.closingForRoute = true;
    document.querySelectorAll("dialog[open]").forEach((dialog) => dialog.close());
    state.closingForRoute = false;
    void applyRoute();
  });
  loadWorkspace().then(() => applyRoute());
}

document.addEventListener("DOMContentLoaded", initialize);
