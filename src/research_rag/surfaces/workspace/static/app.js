"use strict";

const state = {
  profile: null,
  status: null,
  sources: [],
  excludedSources: [],
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
  document.querySelectorAll("[data-capability]").forEach((element) => {
    element.hidden = !hasCapability(element.dataset.capability);
  });
  const visibleNavItems = [...document.querySelectorAll(".sidebar-item")].filter(
    (item) => !item.hidden,
  );
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

function authorLine(source) {
  const authors = Array.isArray(source.authors)
    ? source.authors.map(inlineText).filter(Boolean)
    : [];
  const parts = [];
  if (authors.length) parts.push(authors.join("; "));
  if (source.year) parts.push(String(source.year));
  return parts.join(" · ") || "Authorship and year not reviewed";
}

function locatorLabel(locator) {
  if (!locator) return "Source passage";
  if (locator.type === "pdf_page") {
    return `Page ${locator.page_label || locator.page || "?"}`;
  }
  return locator.section_title || locator.href || `EPUB section ${locator.section_index || "?"}`;
}

function sourceForDocument(documentId) {
  return state.sources.find((source) => source.document_id === documentId) || null;
}

function setConnection(kind, label) {
  const element = byId("connection-state");
  element.dataset.state = kind;
  element.lastElementChild.textContent = label;
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

function renderClients(clients) {
  const container = byId("client-chips");
  container.replaceChildren();
  if (!clients.length) {
    const empty = node("p", "form-note", "No agent is attached to this app.");
    container.append(empty);
    return;
  }
  for (const client of clients) {
    const chip = node("div", "partition-chip");
    const label = node("span", "partition-chip-label", client.name || client.session_id);
    const detail = node(
      "span",
      "partition-chip-count",
      client.attached
        ? `${client.requests || 0} calls`
        : client.detached_reason || "idle",
    );
    chip.append(label, detail);
    if (client.attached) {
      const drop = node("button", "text-button", "Disconnect");
      drop.type = "button";
      drop.addEventListener("click", () => disconnectClient(client.session_id));
      chip.append(drop);
    }
    container.append(chip);
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

async function loadAgentEntry() {
  if (!hasCapability("agent_entry")) return;
  const payload = await api("/api/agent-entry");
  state.agentEntry = payload.entry || "";
  byId("agent-entry").textContent = state.agentEntry;
  byId("agent-entry-copy").disabled = !state.agentEntry;
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
  byId("generation-label").textContent = status.generation_id ? compactId(status.generation_id) : "None";
  byId("generation-label").title = status.generation_id || "";
  byId("generation-date").textContent = formatDate(status.created_at);

  let generationAction = "Regenerate";
  if (!status.ready) generationAction = "Create generation";
  else if (status.generation_upgrade_required) generationAction = "Regenerate";
  else if (status.stale) generationAction = "Re-ingest changes";
  state.forceRecompute = Boolean(
    status.ready && (!status.stale || status.generation_upgrade_required),
  );
  byId("ingest-button").textContent = generationAction;
  byId("ingest-dialog").querySelector("h2").textContent = generationAction;
  byId("ingest-submit").textContent = generationAction;

  let indexState = "Not built";
  if (status.ready && status.stale) indexState = "Stale";
  else if (status.hybrid_ready) indexState = "Hybrid ready";
  else if (status.ready) indexState = "BM25 only";
  byId("index-state").textContent = indexState;
  byId("retrieval-detail").textContent = status.ready
    ? (status.available_retrieval_methods || []).join(" · ").toUpperCase()
    : state.profile?.source_types_label || "document sources";

  const notice = byId("status-notice");
  const noticeAction = byId("notice-action");
  if (!status.ready) {
    notice.hidden = false;
    byId("status-notice-title").textContent = "No generation exists";
    byId("status-notice-text").textContent = status.message || "Create the first knowledge-base generation.";
    noticeAction.hidden = !hasCapability("ingestion");
    noticeAction.textContent = "Create generation";
    noticeAction.dataset.action = "ingest";
  } else if (status.generation_upgrade_required) {
    notice.hidden = false;
    byId("status-notice-title").textContent = "This generation needs an upgrade";
    const reasons = (status.upgrade_reasons || []).join(", ").replaceAll("_", " ");
    byId("status-notice-text").textContent = reasons
      ? `Regenerate to apply: ${reasons}. The existing generation remains searchable.`
      : "Regenerate to apply the current extraction and retrieval policies.";
    noticeAction.hidden = !hasCapability("ingestion");
    noticeAction.textContent = "Regenerate";
    noticeAction.dataset.action = "ingest";
  } else if (status.stale) {
    notice.hidden = false;
    byId("status-notice-title").textContent = "The current generation is stale";
    byId("status-notice-text").textContent = changeSummary(status.changes);
    noticeAction.hidden = !hasCapability("ingestion");
    noticeAction.textContent = "Create fresh generation";
    noticeAction.dataset.action = "ingest";
  } else {
    notice.hidden = true;
  }
  configureRetrieval(status);
  renderPartitions(status);
  renderProjectTags(status);
  renderLanguages(status);
  renderAgentEndpoint(status);
  if (hasCapability("generations")) renderGenerations(status.generations || []);
  if (hasCapability("sql_console")) renderSqlConsole(status);
}

function tagList(values, className = "tag") {
  const fragment = document.createDocumentFragment();
  (values || []).forEach((value) => fragment.append(node("span", className, value)));
  return fragment;
}

function addSearchFilter(field, value) {
  const input = byId(field);
  if (!input || !value) return;
  const values = listValue(input.value);
  if (!values.includes(value)) values.push(value);
  input.value = values.join(", ");
  input.focus();
  toast(`Added to the search filter: ${value}`);
}

function renderInventory(containerId, entries, key, action) {
  const container = byId(containerId);
  if (!container) return;
  container.replaceChildren();
  if (!entries.length) {
    container.append(node("span", "form-note", "None recorded yet."));
    return;
  }
  entries.forEach((item) => {
    const chip = button(
      `${item[key]} · ${formatNumber(item.searchable_source_count)}`,
      action,
      item[key],
      "tag tag-button",
    );
    chip.title = `Search ${item[key]}`;
    container.append(chip);
  });
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

function sourceCard(source) {
  const card = node("article", "source-card");
  card.dataset.searchText = [
    inlineText(source.title),
    ...(source.authors || []),
    ...(source.categories || []),
    ...(source.keywords || []),
    ...(source.project || []),
    source.source_relative_path,
  ].join(" ").toLocaleLowerCase();

  const body = node("div", "source-card-body");
  const titleRow = node("div", "source-title-row");
  titleRow.append(node("span", "format-badge", source.format || "source"));
  titleRow.append(node("span", "source-title", inlineText(source.title) || source.source_relative_path));
  body.append(titleRow);
  body.append(node("p", "source-byline", authorLine(source)));
  if (source.doi) body.append(node("p", "source-doi", `doi:${inlineText(source.doi).replace(/^doi:/i, "")}`));
  const path = node("p", "source-path", source.source_relative_path);
  path.title = source.source_path || source.source_relative_path;
  body.append(path);
  body.append(node("p", "source-id", source.source_id));
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

  const actions = node("div", "source-card-actions");
  if (hasCapability("source_files")) {
    actions.append(button("Open original", "open-source", source.source_relative_path));
  }
  if (hasCapability("metadata")) {
    actions.append(button("Edit metadata", "edit-metadata", source.document_id));
  }
  if (hasCapability("source_inclusion")) {
    actions.append(button("Exclude", "exclude-source", source.document_id));
  }
  if (hasCapability("source_selection")) {
    actions.append(button("Only this source", "only-source", source.source_id));
    actions.append(button("Exclude from search", "exclude-from-search", source.source_id));
  }
  card.append(body, actions);
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
  body.append(node("span", "state-badge", status));
  card.append(body);
  if (hasCapability("source_inclusion")) {
    card.append(button("Restore source", "restore-source", source.source_relative_path));
  }
  return card;
}

function renderSources(payload) {
  state.sources = payload.sources || [];
  state.excludedSources = payload.excluded_sources || [];
  byId("source-nav-count").textContent = String(state.sources.length);
  byId("excluded-count").textContent = String(state.excludedSources.length);

  const list = byId("source-list");
  list.replaceChildren();
  if (!state.sources.length) {
    list.append(node("div", "no-records", payload.ready ? "No sources match this collection." : "Create a generation to inspect indexed sources."));
  } else {
    state.sources.forEach((source) => list.append(sourceCard(source)));
  }

  const excludedSection = byId("excluded-section");
  excludedSection.hidden = !state.excludedSources.length;
  const excludedList = byId("excluded-list");
  excludedList.replaceChildren();
  state.excludedSources.forEach((source) => excludedList.append(excludedCard(source)));
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
    setConnection("ready", "Local · ready");
    if (announce) toast("Workspace refreshed.");
  } catch (error) {
    setConnection("error", "Connection failed");
    toast(error.message, true);
  } finally {
    setBusy(false);
    if (state.status) configureRetrieval(state.status);
  }
}

function switchView(name) {
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
  });
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

function scoreLabel(label, value) {
  if (value === null || value === undefined) return null;
  const shown = typeof value === "number" && !Number.isInteger(value) ? value.toFixed(4) : value;
  return node("span", "score-label", `${label} ${shown}`);
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

  const scores = node("div", "score-row");
  [
    scoreLabel("BM25 #", hit.component_ranks?.bm25),
    scoreLabel("Dense #", hit.component_ranks?.dense),
    scoreLabel("Cosine", hit.component_scores?.dense_cosine_similarity),
    scoreLabel("RRF", hit.fusion_score),
    scoreLabel("Rerank", hit.rerank_score),
  ].filter(Boolean).forEach((item) => scores.append(item));
  const chunk = node("span", "score-label", compactId(hit.chunk_id));
  chunk.title = hit.chunk_id;
  scores.append(chunk);
  footer.append(scores);
  content.append(footer);
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
    byId("search-summary").scrollIntoView({ behavior: "smooth", block: "start" });
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
      const item = node("article", `context-passage${passage.chunk_id === payload.requested_chunk_id ? " is-requested" : ""}`);
      const citation = node("div", "context-citation");
      citation.append(node("span", "", inlineText(passage.citation)));
      citation.append(node("span", "locator-badge", locatorLabel(passage.locator)));
      item.append(citation, node("p", "", readableText(passage.text)));
      if (hasCapability("chunk_exclusion")) {
        item.append(chunkAction(passage));
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

function renderGenerations(generations) {
  const container = byId("generation-chips");
  container.replaceChildren();
  if (!generations.length) {
    container.append(node("p", "form-note", "This project has no build yet."));
    return;
  }
  for (const generation of generations) {
    const chip = node("div", "partition-chip");
    chip.append(
      node("span", "partition-chip-label", generation.generation_id),
      node(
        "span",
        "partition-chip-count",
        generation.is_current
          ? "in use"
          : `${bytes(generation.size_bytes)} · ${generation.chunk_count} passages`,
      ),
    );
    // The one a search reads cannot be removed, so the action is not offered
    // rather than offered and refused: a button that always fails is a button
    // that teaches a reader to click through the answers.
    if (!generation.is_current) {
      const drop = node("button", "text-button", "Remove");
      drop.type = "button";
      drop.addEventListener("click", () => openGenerationRemoval(generation.generation_id));
      chip.append(drop);
    }
    if (generation.manifest_error) {
      chip.append(node("span", "partition-chip-count", "manifest unreadable"));
    }
    container.append(chip);
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
// loaded, where that value came from, what changing it costs, the variable
// that would override it, and the control. Every one of those is optional: a
// host that sends none of them draws the same row it always did, and a host
// that sends all of them gets all of them without a truncated line.
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
  facts.append(node("span", "locator-badge setting-origin", setting.origin || "default"));
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
      "state-badge",
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
    toast(result.message || "Metadata saved. Create a generation to apply it.");
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
  } else if (action === "ingest") byId("ingest-dialog").showModal();
  else if (action === "partition-filter") addSearchFilter("category-any-filter", value);
  else if (action === "project-filter") addSearchFilter("project-any-filter", value);
  else if (action === "language-filter") addSearchFilter("language-filter", value);
  else if (action === "only-source") addSearchFilter("include-source-filter", value);
  else if (action === "exclude-from-search") addSearchFilter("exclude-source-filter", value);
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

function filterSources(event) {
  const query = event.target.value.trim().toLocaleLowerCase();
  document.querySelectorAll(".source-card").forEach((card) => {
    card.hidden = Boolean(query) && !card.dataset.searchText.includes(query);
  });
}

function initialize() {
  state.hits = new Map();
  document.querySelectorAll(".sidebar-item").forEach((item) => {
    item.querySelector(".nav-item")?.addEventListener("click", () => {
      switchView(item.dataset.view);
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
  byId("ingest-button").addEventListener("click", () => byId("ingest-dialog").showModal());
  byId("export-button").addEventListener("click", exportBundle);
  byId("import-button").addEventListener("click", () => byId("bundle-dialog").showModal());
  byId("notice-action").addEventListener("click", handleAction);
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
  byId("partition-chips").addEventListener("click", handleAction);
  byId("project-chips").addEventListener("click", handleAction);
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
