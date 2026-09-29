"use strict";

let ME = null, GUILDS = [], FIELDS = [], current = "overview";
let pendingGuild = null;  // "Details" on a server card -> scroll to its learned card once Overview has drawn
const edits = new Map();  // path -> new value
const timers = {};

// ------------------------------------------------------------ tabs

function show(tab) {
  current = tab;
  for (const b of $$("#tabs button")) b.classList.toggle("active", b.dataset.tab === tab);
  for (const v of $$("[data-view]")) v.classList.toggle("hidden", v.dataset.view !== tab);
  history.replaceState(null, "", `#${tab}`);
  ({ overview: loadOverview, platforms: loadPlatforms, live: loadLive, servers: loadGuilds, people: loadPeople, settings: loadConfig, logs: loadLogs, account: loadAccount })[tab]?.();
}
$("#tabs").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) show(b.dataset.tab); });

// ------------------------------------------------------------ header / me

async function loadMe() {
  ME = await api("/api/admin/me");
  const b = ME.bot;
  $("#bot-name").textContent = b.user || b.name;
  $("#dot").className = "dot" + (b.ready ? " on" : "");
  $("#presence").textContent = b.presence || (b.ready ? "online" : "starting up…");
  if (b.avatar) { $("#bot-avatar").src = b.avatar; $("#bot-avatar").classList.remove("hidden"); setIcon(b.avatar); }
  $("#me").textContent = ME.name;
  if (ME.avatar) { $("#me-avatar").src = ME.avatar; $("#me-avatar").classList.remove("hidden"); }
  showPlatform(b.platform);
}

// ------------------------------------------------------------ platforms (Discord / Fluxer)
// Two bots in one process. PLAT = what's running (from /me). The Discord/Fluxer switches ("scopes") at the top of
// Servers, Live, People and Settings pick which bot you're looking at; they only show when both run
// (Settings always has its own: Shared / Discord / Fluxer).

let PLAT = null;
const PNAME = { all: "All", shared: "Shared", discord: "Discord", fluxer: "Fluxer" };
const store = {
  get: (k, d) => { try { return localStorage.getItem(k) || d; } catch { return d; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* private window */ } },
};
const SCOPE = { main: store.get("scope.main", "all"), people: store.get("scope.people", "discord"),
  settings: store.get("scope.settings", "all") };
const bothOn = () => !PLAT || PLAT.mode === "both";

// Which platform a view shows: the picked one when both run, else the only one running.
function scopeOf(key) {
  if (key === "settings") return SCOPE.settings;
  if (!bothOn()) return PLAT.mode;
  return SCOPE[key];
}

const platTag = (p) => h("span", { class: `ptag ${p}` }, PNAME[p]);

function showPlatform(p) {
  if (!p) return;
  const changed = !PLAT || PLAT.mode !== p.mode;
  PLAT = p;
  const chip = (key, on, up) => h("span", { class: "plat-chip" + (on ? "" : " off"),
    title: `${PNAME[key]}: ${!on ? "turned off" : up ? "online" : "offline"}` },
    h("span", { class: "dot" + (on && up ? " on" : on ? " bad" : "") }), PNAME[key]);
  $("#plat-chips").replaceChildren(chip("discord", p.mode !== "fluxer", p.discord),
    chip("fluxer", p.mode !== "discord", p.fluxer));
  if (changed) mountScopes();
}
$("#plat-chips").addEventListener("click", () => show("platforms"));

function mountScopes() {
  for (const slot of $$(".scope-slot")) {
    const key = slot.dataset.key || "main";
    const opts = slot.dataset.opts.split(",");
    slot.classList.toggle("hidden", !slot.dataset.always && !bothOn());
    const cur = scopeOf(key);
    slot.replaceChildren(h("div", { class: "seg", role: "group", "aria-label": "Platform" },
      opts.map((o) => h("button", { class: o === cur ? "active" : null, "aria-pressed": String(o === cur),
        onclick: () => setScope(key, o) }, PNAME[o]))));
  }
}

function setScope(key, value) {
  SCOPE[key] = value;
  store.set(`scope.${key}`, value);
  mountScopes();
  if (key === "main") { renderGuilds(); renderLiveServers(); renderLive(); }
  if (key === "people") { PEOPLE.sel = null; loadPeople(); }
  if (key === "settings") applyCfgFilter();
}

// ------------------------------------------------------------ platforms tab (turn each bot on/off)

let PLATS = null, PLAN = null;  // PLATS = /api/admin/platforms, PLAN = the on/off switches (applied on restart)

async function loadPlatforms() {
  try {
    PLATS = await api("/api/admin/platforms");
    if (!PLAN) PLAN = { discord: PLATS.discord.enabled, fluxer: PLATS.fluxer.enabled };
    renderPlatforms();
  } catch (e) { toast(e.message, true); }
}

const planMode = () => (PLAN.discord && PLAN.fluxer ? "both" : PLAN.discord ? "discord" : "fluxer");

function renderPlatforms() {
  $("#plat-cards").replaceChildren(platCard("discord"), platCard("fluxer"));
  const changed = planMode() !== PLATS.mode;
  $("#plat-bar").classList.toggle("hidden", !changed);
  const on = ["discord", "fluxer"].filter((k) => PLAN[k]).map((k) => PNAME[k]);
  $("#plat-msg").textContent = changed ? `After the restart: ${on.join(" + ")} on.` : "";
}

function platCard(key) {
  const d = PLATS[key], on = PLAN[key], other = key === "discord" ? "fluxer" : "discord";
  const lastOne = on && !PLAN[other];
  const notSetUp = key === "fluxer" && !d.configured;
  const sw = h("input", { type: "checkbox", class: "switch", checked: on, disabled: lastOne || (notSetUp && !on),
    "aria-label": `${PNAME[key]} bot on` });
  sw.addEventListener("change", () => { PLAN[key] = sw.checked; renderPlatforms(); });
  const status = !d.enabled ? h("span", { class: "pill" }, "off")
    : d.online ? h("span", { class: "pill ok" }, "online") : h("span", { class: "pill bad" }, "offline");
  const pending = on !== d.enabled ? h("span", { class: "pill warn" }, on ? "turns on after restart" : "turns off after restart") : null;
  const icon = d.avatar ? h("img", { class: "icon", src: d.avatar, alt: "" }) : h("div", { class: `icon ${key}` }, PNAME[key][0]);
  const go = (tab, scope) => () => { if (scope) setScope(scope, key); show(tab); };
  const why = lastOne ? "At least one bot stays on: the models and this dashboard run in the same process."
    : notSetUp && !on ? "Set fluxer.api_url and fluxer.token in Settings (Fluxer) first." : null;
  return h("article", { class: `card plat-card ${key}` + (on ? "" : " is-off") },
    h("div", { class: "top" }, icon,
      h("div", { style: "min-width:0;flex:1" }, h("div", { class: "title" }, `${PNAME[key]} bot`),
        h("div", { class: "muted small mono" }, d.user ? `as ${d.user}` : d.where || "not connected")),
      h("label", { class: "switch-row", title: why || "" }, sw, h("span", {}, on ? "On" : "Off"))),
    h("div", { class: "chips" }, status, pending),
    why ? h("p", { class: "quiet small", style: "margin:0 0 8px" }, why) : null,
    h("dl", { class: "kv" },
      h("dt", {}, "Servers"), h("dd", {}, d.servers ?? "-"),
      h("dt", {}, "In voice now"), h("dd", {}, d.voice ? `${d.voice} call${d.voice > 1 ? "s" : ""}` : "no"),
      h("dt", {}, "People remembered"), h("dd", {}, d.people ?? "-"),
      h("dt", {}, "Commands"), h("dd", {}, d.commands),
      h("dt", {}, "Data"), h("dd", { class: "mono small" }, d.data),
      key === "fluxer" ? [h("dt", {}, "Server"), h("dd", { class: "mono small" }, d.where || "not set")] : null),
    h("div", { class: "row" },
      h("button", { class: "ghost", onclick: go("servers", "main"), disabled: !d.online }, "Servers"),
      h("button", { class: "ghost", onclick: go("people", "people") }, "People"),
      h("button", { class: "ghost", onclick: go("settings", "settings") }, "Settings")));
}

$("#plat-undo").addEventListener("click", () => {
  PLAN = { discord: PLATS.discord.enabled, fluxer: PLATS.fluxer.enabled };
  renderPlatforms();
});
$("#plat-apply").addEventListener("click", async () => {
  const mode = planMode();
  $("#plat-apply").disabled = true;
  try { await api("/api/admin/platform", { mode }); }
  catch (e) { $("#plat-apply").disabled = false; return toast(e.message, true, 10000); }
  PLAN = null;
  $("#plat-apply").disabled = false;
  await restartAndWait();
});

$("#logout").addEventListener("click", async () => { await api("/api/auth/logout", {}); location.href = "/login"; });

// ------------------------------------------------------------ overview

async function loadOverview() {
  const jump = pendingGuild;
  pendingGuild = null;
  try {
    const [d, l] = await Promise.all([api("/api/admin/snapshot"), api("/api/admin/learned")]);
    $("#ov-board").replaceChildren(...(d.starting || !d.reply  // the same board as the public page (board.js)
      ? [h("p", { class: "starting" }, "The bot is starting up. Stats appear in a few seconds.")] : renderBoard(d)));
    const groups = ["discord", "fluxer"].map((k) => [k, l.guilds.filter((g) => (g.platform || "discord") === k)])
      .filter(([, gs]) => gs.length);
    $("#ov-learned").replaceChildren(...groups.flatMap(([k, gs]) => groups.length > 1
      ? [h("div", { class: "plat-head wide" }, platTag(k)), ...gs.map(learnedCard)] : gs.map(learnedCard)));
    if (jump) document.getElementById(`learned-${jump}`)?.scrollIntoView({ block: "center" });
  } catch (e) { toast(e.message, true); }
}

// One server's adapted state: reply length (0-3 steps shorter), follow-up window, mood strategies.
function learnedCard(g) {
  const t = g.tuning;
  const icon = g.icon ? h("img", { class: "icon", src: g.icon, alt: "" }) : h("div", { class: "icon" }, g.name.slice(0, 1));
  const steps = h("div", { class: "steps", title: "0 = normal length, 3 = a few words" },
    [0, 1, 2, 3].map((i) => h("span", { class: t && i <= t.level ? "on" : null })));
  return h("div", { class: "card learned-card", id: `learned-${g.id}` },
    h("div", { class: "top" }, icon, h("div", { class: "title" }, g.name), g.voice && h("span", { class: "lamp on" }, "ON AIR")),
    t ? [
      h("div", { class: "meter-top" }, h("span", {}, "Reply length"), h("span", { class: "num" }, t.level_name)),
      steps,
      facts([
        ["Waits for a follow-up", `${fix(t.followup_s, 0)} s`, `How long it keeps listening after replying (starts at ${fix(t.base_followup_s, 0)} s)`],
        ["Uninterrupted replies in a row", int(t.streak), "After 6 in a row it relaxes one step back towards normal length"],
      ]),
      t.log.length > 0 && h("div", { class: "log" }, h("div", { class: "kicker" }, "Recent changes"),
        t.log.map((x) => h("div", { class: "log-row" }, h("span", { class: "dim num" }, ago(x.at)), h("span", {}, x.why)))),
    ] : h("p", { class: "quiet" }, "Hasn't adjusted anything here yet (it learns from voice conversations)."),
    h("hr"),
    h("div", { class: "kicker" }, "Answering moods"),
    g.strategies.length ? g.strategies.map((s) => h("div", { class: "strat" },
      h("div", { class: "strat-mood" }, s.mood),
      s.options.map((o, i) => {
        const best = o.rate >= s.options[1 - i].rate;
        return h("div", { class: "strat-opt" + (best ? " best" : "") },
          h("span", {}, o.text), h("span", { class: "num", title: "went well / tried" }, `${o.wins}/${o.tries}`));
      })))
      : h("p", { class: "quiet" }, "No mood reactions tried here yet."));
}

// ------------------------------------------------------------ servers

function guildCard(g) {
  const icon = g.icon ? h("img", { class: "icon", src: g.icon, alt: "" })
    : h("div", { class: "icon" }, g.name.split(/\s+/).map((w) => w[0]).join("").slice(0, 3));
  const fx = g.platform === "fluxer";
  const chips = [
    g.voice ? h("span", { class: "pill ok" }, `🔊 in ${g.voice.name}`) : null,
    g.admin ? h("span", { class: "pill" }, "administrator") : null,
    g.missing.length ? h("span", { class: "pill warn", title: g.missing.join(", ") }, `missing ${g.missing.length} perm${g.missing.length > 1 ? "s" : ""}`) : null,
    g.commands === "missing" ? h("span", { class: "pill warn" }, "slash commands off") : null,
    g.boost_tier ? h("span", { class: "pill" }, `boost ${g.boost_tier}`) : null,
  ];

  const joinable = g.voice_channels.filter((c) => c.can_join);
  const select = h("select", { "aria-label": `Voice channel in ${g.name}` },
    joinable.length ? joinable.map((c) => h("option", { value: c.id, selected: g.voice && g.voice.id === c.id },
      `🔊 ${c.name}${c.people ? ` (${c.people})` : ""}`)) : h("option", { value: "" }, "no voice channels I can join"));

  const voiceRow = h("div", { class: "row" }, select,
    h("button", { disabled: !joinable.length, onclick: (e) => voice(g, select.value, e.currentTarget) }, g.voice ? "Move" : "Join"),
    g.voice ? h("button", { class: "danger", onclick: (e) => voice(g, "", e.currentTarget) }, "Disconnect") : null);

  return h("article", { class: "card guild" },
    h("div", { class: "top" }, icon,
      h("div", { style: "min-width:0" }, h("div", { class: "title", title: g.name }, g.name), h("div", { class: "muted small mono" }, g.id))),
    h("div", { class: "chips" }, chips),
    h("dl", { class: "kv" },
      h("dt", {}, "Members"), h("dd", {}, `${g.members ?? "?"}${g.humans != null ? ` (${g.humans} people)` : ""}`),
      h("dt", {}, "Owner"), h("dd", {}, g.owner),
      h("dt", {}, "Channels"), h("dd", {}, `${g.text_channels} text · ${g.voice_channels.length} voice`),
      h("dt", {}, "Bot joined"), h("dd", {}, dateStr(g.joined)),
      h("dt", {}, "Created"), h("dd", {}, dateStr(g.created)),
      g.voice ? [h("dt", {}, "In voice"), h("dd", {}, g.voice.people.join(", ") || "nobody")] : null,
      g.missing.length ? [h("dt", {}, "Missing"), h("dd", { class: "small" }, g.missing.join(", "))] : null,
    ),
    voiceRow,
    h("div", { class: "divider" }),
    h("div", { class: "row" },
      g.commands === "missing" ? h("button", { onclick: (e) => syncCommands(g, e.currentTarget) }, "Enable commands") : null,
      fx ? null : h("button", { class: "ghost", onclick: () => {
        pendingGuild = g.id; show("overview");
      } }, "Details"),
      h("div", { class: "spacer" }),
      h("button", { class: "danger", onclick: () => askLeave(g) }, "Remove bot")));
}

function renderGuilds() {
  if (!PLAT) return;
  const q = $("#srv-filter").value.trim().toLowerCase();
  const sc = scopeOf("main");
  const out = [];
  for (const key of ["discord", "fluxer"]) {
    if (sc !== "all" && sc !== key) continue;
    const running = PLAT.mode === key || PLAT.mode === "both";
    const all = GUILDS.filter((g) => (g.platform || "discord") === key);
    const list = all.filter((g) => !q || g.name.toLowerCase().includes(q) || g.id.includes(q));
    if (!running) {
      if (sc === key) out.push(h("p", { class: "page-note" }, `The ${PNAME[key]} bot is turned off. `,
        h("a", { href: "#platforms", onclick: (e) => { e.preventDefault(); show("platforms"); } }, "Turn it on on the Platforms tab.")));
      continue;
    }
    out.push(h("div", { class: "plat-head" }, platTag(key), h("span", { class: "count" }, `${all.length} server${all.length === 1 ? "" : "s"}`),
      h("div", { class: "spacer" }),
      key === "discord" ? h("button", { class: "primary", onclick: openInvite }, "＋ Add to a server")
        : h("span", { class: "muted small" }, "Add it from your Fluxer server's settings (Integrations / bots)")));
    out.push(h("div", { class: "guilds" }, ...(list.length ? list.map(guildCard)
      : [h("p", { class: "muted" }, all.length ? "No servers match." : "Not in any servers yet.")])));
  }
  $("#srv-count").textContent = `(${GUILDS.length})`;
  $("#guilds").replaceChildren(...out);
}

let INVITE = null;
async function loadGuilds() {
  try {
    const data = await api("/api/admin/guilds");
    GUILDS = data.guilds;
    INVITE = data;
    renderGuilds();
  } catch (e) { toast(e.message, true); }
}
$("#srv-filter").addEventListener("input", renderGuilds);

async function voice(g, channelId, btn) {
  btn.disabled = true;
  try {
    await api(`/api/admin/guilds/${g.id}/voice`, { channel_id: channelId });
    toast(channelId ? `Joining voice in ${g.name}…` : `Left voice in ${g.name}`);
    setTimeout(loadGuilds, 1500);
  } catch (e) { toast(e.message, true); btn.disabled = false; }
}

async function syncCommands(g, btn) {
  btn.disabled = true;
  try { toast((await api(`/api/admin/guilds/${g.id}/commands`, {})).note); loadGuilds(); }
  catch (e) { toast(e.message, true); btn.disabled = false; }
}

let leaving = null;
function askLeave(g) {
  leaving = g;
  $("#leave-name").textContent = g.name;
  $("#leave-confirm").value = "";
  $("#leave-go").disabled = true;
  $("#dlg-leave").showModal();
  $("#leave-confirm").focus();
}
$("#leave-confirm").addEventListener("input", (e) => { $("#leave-go").disabled = e.target.value.trim() !== leaving?.name; });
$("#leave-cancel").addEventListener("click", () => $("#dlg-leave").close());
$("#leave-go").addEventListener("click", async () => {
  $("#leave-go").disabled = true;
  try {
    await api(`/api/admin/guilds/${leaving.id}/leave`, { confirm: $("#leave-confirm").value });
    $("#dlg-leave").close();
    toast(`Removed from ${leaving.name}`);
    loadGuilds();
  } catch (e) { toast(e.message, true); $("#leave-go").disabled = false; }
});

async function openInvite() {
  if (!INVITE) await loadGuilds();
  if (!INVITE?.invite) return toast("The bot isn't logged in yet, so there's no invite link.", true);
  $("#invite-url").value = INVITE.invite;
  $("#invite-open").href = INVITE.invite;
  $("#invite-admin-url").value = INVITE.invite_admin;
  $("#dlg-invite").showModal();
}
for (const [btn, input] of [["#invite-copy", "#invite-url"], ["#invite-admin-copy", "#invite-admin-url"]]) {
  $(btn).addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($(input).value); toast("Copied"); }
    catch { $(input).select(); toast("Press Ctrl+C to copy"); }
  });
}
$("#invite-close").addEventListener("click", () => $("#dlg-invite").close());

// ------------------------------------------------------------ people

const PEOPLE = { list: [], sel: null };

function personAvatar(p) {
  const name = p.username || p.display_name || "?";
  return p.avatar ? h("img", { class: "ppl-av", src: p.avatar, alt: "" }) : h("div", { class: "ppl-av" }, name.slice(0, 1).toUpperCase());
}

function personItem(p) {
  const bits = [`${p.messages} msg`, p.profile_chars ? "notes" : null, p.pending ? `${p.pending} queued` : null,
    p.opted_out ? "opted out" : null].filter(Boolean);
  return h("button", { class: "ppl-item" + (p.id === PEOPLE.sel ? " active" : ""), onclick: () => loadPerson(p.id) },
    personAvatar(p),
    h("div", { class: "info" }, h("div", { class: "name" }, p.username || p.display_name || p.id),
      h("div", { class: "s" }, `${ago(p.last_seen)} · ${bits.join(" · ")}`)));
}

function renderPeople() {
  const q = $("#ppl-filter").value.trim().toLowerCase();
  const list = PEOPLE.list.filter((p) => !q || `${p.username} ${p.display_name} ${p.id}`.toLowerCase().includes(q));
  $("#ppl-count").textContent = `(${PEOPLE.list.length})`;
  $("#ppl-list").replaceChildren(...(list.length ? list.map(personItem)
    : [h("p", { class: "muted" }, PEOPLE.list.length ? "Nobody matches." : "Nothing stored about anyone yet.")]));
}

async function loadPeople() {
  if (!PEOPLE.sel) $("#ppl-detail").replaceChildren(h("p", { class: "quiet" }, "Pick someone to see what's stored."));
  try {
    const fx = scopeOf("people") === "fluxer";
    const data = await api("/api/admin/people" + pplQuery());
    $("#ppl-db").textContent = fx ? "data/fluxer/profiles.db" : "data/profiles.db";
    $("#ppl-mood").textContent = fx ? "data/fluxer/mood.json" : "data/mood.json";
    const pre = fx ? (PLATS?.fluxer.commands || "! commands").split(" ")[0] : "/";
    for (const [i, el] of $$(".ppl-cmd").entries()) el.textContent = pre + ["profile", "forget", "profiling"][i];
    PEOPLE.list = data.people;
    $("#ppl-off").classList.toggle("hidden", data.enabled);
    renderPeople();
    if (!PEOPLE.sel && PEOPLE.list.length) loadPerson(PEOPLE.list[0].id);
  } catch (e) { toast(e.message, true); }
}
$("#ppl-filter").addEventListener("input", renderPeople);

// Discord and Fluxer keep separate people data: the scope switch picks which one.
const pplQuery = () => (scopeOf("people") === "fluxer" ? "?platform=fluxer" : "");

async function loadPerson(id) {
  PEOPLE.sel = id;
  renderPeople();
  try { $("#ppl-detail").replaceChildren(personCard(await api(`/api/admin/people/${id}${pplQuery()}`))); }
  catch (e) { $("#ppl-detail").replaceChildren(h("p", { class: "error" }, e.message)); }
}

function personCard(p) {
  const name = p.username || p.display_name || p.id;
  const th = Object.entries(p.thresholds || {});
  const tone = p.tone;
  return h("article", { class: "card" },
    h("div", { class: "top" }, personAvatar(p),
      h("div", { style: "min-width:0;flex:1" }, h("div", { class: "title" }, name),
        h("div", { class: "muted small mono" }, p.id)),
      p.opted_out ? h("span", { class: "pill warn" }, "opted out") : null),
    h("dl", { class: "kv" },
      h("dt", {}, "Username"), h("dd", {}, p.username || "-"),
      h("dt", {}, "Name it was called"), h("dd", {}, p.display_name || "-"),
      h("dt", {}, "Roles"), h("dd", {}, p.roles?.length ? p.roles.join(", ") : "-"),
      h("dt", {}, "First seen"), h("dd", {}, p.first_seen ? new Date(p.first_seen * 1000).toLocaleString() : "-"),
      h("dt", {}, "Last seen"), h("dd", {}, p.last_seen ? new Date(p.last_seen * 1000).toLocaleString() : "-"),
      h("dt", {}, "Messages"), h("dd", {}, p.messages ?? 0),
      h("dt", {}, "Profile picture"), h("dd", {}, p.avatar_desc || "-"),
      h("dt", {}, "Linked account"), h("dd", {}, p.linked
        ? [platTag(p.linked.platform), ` ${p.linked.name} `, h("span", { class: "dim mono small" }, p.linked.id)] : "-")),
    h("section", {}, h("div", { class: "kicker" },
      "What it remembers", p.profile_updated ? h("span", { class: "dim" }, ` · updated ${ago(p.profile_updated)}`) : null),
      p.profile ? h("div", { class: "notes" }, p.profile) : h("p", { class: "quiet" }, "No notes yet.")),
    h("section", {}, h("div", { class: "kicker" }, `Queued for the next update (${p.pending?.length || 0} lines)`),
      p.pending?.length ? h("div", { class: "lines" }, p.pending.map((l) => h("div", { class: l.own ? null : "bot" },
        h("span", { class: "t" }, new Date(l.ts * 1000).toLocaleString()), l.line)))
        : h("p", { class: "quiet" }, "Nothing waiting: raw lines are deleted once they're summarized into the notes.")),
    h("section", {}, h("div", { class: "kicker" }, "Voice and mood"),
      tone || th.length ? h("dl", { class: "kv" },
        tone ? [h("dt", {}, "Tone baseline"), h("dd", {}, `${tone.samples} utterances · ${fix(tone.loudness_db, 1)} dB · ${fix(tone.words_per_s, 1)} words/s`)] : null,
        th.length ? [h("dt", {}, "Mood thresholds"), h("dd", {}, th.map(([g, v]) => `${g} ${fix(v, 2)}`).join(" · "))] : null)
        : h("p", { class: "quiet" }, "Nothing learned yet.")),
    h("section", {}, h("div", { class: "kicker" }, "Reminders"),
      p.reminders.length ? h("dl", { class: "kv" }, p.reminders.map((r) => [
        h("dt", {}, new Date(r.due * 1000).toLocaleString()),
        h("dd", {}, r.text, h("span", { class: "dim" }, r.own ? ` (set by them, ${r.deliver})` : ` (from ${r.creator})`))]))
        : h("p", { class: "quiet" }, "None.")),
    h("section", {}, h("div", { class: "kicker" }, "Server memories they're in"),
      p.lore?.length ? h("dl", { class: "kv" }, p.lore.map((l) => [
        h("dt", {}, new Date(l.created * 1000).toLocaleDateString()), h("dd", {}, l.text)]))
        : h("p", { class: "quiet" }, "None.")),
    h("div", { class: "divider" }),
    h("div", { class: "row" },
      p.opted_out !== undefined ? h("button", { onclick: (e) => setProfiling(p, p.opted_out, e.currentTarget) },
        p.opted_out ? "Turn profiling back on" : "Turn profiling off") : null,
      h("div", { class: "spacer" }),
      h("button", { class: "danger", onclick: () => askForget(p) }, "Forget everything")));
}

async function setProfiling(p, enabled, btn) {
  if (!enabled && !confirm(`Stop profiling ${p.username || p.id}? This also deletes their notes, queued lines and the server memories they're in.`)) return;
  btn.disabled = true;
  try { await api(`/api/admin/people/${p.id}/profiling${pplQuery()}`, { enabled }); toast(enabled ? "Profiling on." : "Profiling off, notes deleted."); }
  catch (e) { toast(e.message, true); }
  await loadPeople(); loadPerson(p.id);
}

let forgetting = null;
function askForget(p) {
  forgetting = p;
  $("#forget-name").textContent = p.username || p.id;
  $("#dlg-forget").showModal();
}
$("#forget-cancel").addEventListener("click", () => $("#dlg-forget").close());
$("#forget-go").addEventListener("click", async () => {
  $("#dlg-forget").close();
  try {
    await api(`/api/admin/people/${forgetting.id}/forget${pplQuery()}`, {});
    toast(`Forgot ${forgetting.username || forgetting.id}.`);
    PEOPLE.sel = null;
    $("#ppl-detail").replaceChildren(h("p", { class: "quiet" }, "Pick someone to see what's stored."));
    loadPeople();
  } catch (e) { toast(e.message, true); }
});

// ------------------------------------------------------------ settings

function fieldInput(f) {
  const set = (v) => { edits.set(f.path, v); markChanged(f.path, true); };
  const same = (v) => { edits.delete(f.path); markChanged(f.path, false); };
  const onVal = (v, orig) => (JSON.stringify(v) === JSON.stringify(orig) ? same() : set(v));
  const id = `f-${f.path}`;
  switch (f.kind) {
    case "bool": {
      const el = h("input", { type: "checkbox", id, checked: f.value });
      el.addEventListener("change", () => onVal(el.checked, f.value));
      return el;
    }
    case "secret": {
      const el = h("input", { type: "password", id, autocomplete: "new-password",
        placeholder: f.set ? "•••••••• set (leave blank to keep)" : "not set" });
      el.addEventListener("input", () => (el.value ? set(el.value) : same()));
      return el;
    }
    case "list": {
      const el = h("textarea", { id, rows: Math.max(2, Math.min(8, f.value.length + 1)), placeholder: "one per line" });
      el.value = f.value.join("\n");
      el.addEventListener("input", () => onVal(el.value.split("\n").map((s) => s.trim()).filter(Boolean), f.value));
      return el;
    }
    case "text": case "json": {
      const el = h("textarea", { id, rows: Math.min(14, Math.max(3, String(f.value).split("\n").length + 1)) });
      el.value = f.value;
      el.addEventListener("input", () => onVal(el.value, f.value));
      return el;
    }
    default: {
      const el = h("input", { id, inputmode: f.kind === "int" || f.kind === "float" ? "decimal" : null });
      el.value = f.value ?? "";
      el.addEventListener("input", () => onVal(el.value, String(f.value ?? "")));
      return el;
    }
  }
}

function markChanged(path, on) {
  document.getElementById(`s-${path}`)?.classList.toggle("changed", on);
  $("#savebar").classList.toggle("hidden", edits.size === 0);
  $("#save-count").textContent = `${edits.size} unsaved change${edits.size === 1 ? "" : "s"}`;
}

// Which bot a settings section belongs to. Everything not listed is shared by both.
const CFG_SCOPE = { platform: "platform", discord: "discord", presence: "discord", fluxer: "fluxer" };
const CFG_GROUPS = [["platform", "Platforms", "Which bots run (also on the Platforms tab)."],
  ["discord", "Discord only", "The Discord bot: login, which channels it answers in, its profile status."],
  ["fluxer", "Fluxer only", "The Fluxer bot: server, token, ! commands, channels, its own data folder."],
  ["shared", "Shared by both bots", "Persona, models, voice behaviour, memory, reminders, dashboard - one setting, both bots."]];
const cfgScope = (section) => CFG_SCOPE[section] || "shared";

function renderConfig() {
  const sections = new Map();
  for (const f of FIELDS) {
    const [top] = f.path.split(".");
    if (!sections.has(top)) sections.set(top, []);
    sections.get(top).push(f);
  }
  const out = [];
  const order = [...sections.keys()].sort((a, b) =>
    CFG_GROUPS.findIndex((g) => g[0] === cfgScope(a)) - CFG_GROUPS.findIndex((g) => g[0] === cfgScope(b)));
  let lastGroup = null;
  for (const name of order) {
    const fields = sections.get(name);
    const grp = cfgScope(name);
    if (grp !== lastGroup) {
      lastGroup = grp;
      const g = CFG_GROUPS.find((x) => x[0] === grp);
      out.push(h("div", { class: "cfg-group", "data-scope": grp },
        grp === "discord" || grp === "fluxer" ? platTag(grp) : null, h("span", { class: "cfg-group-t" }, g[1]),
        h("span", { class: "muted small" }, g[2])));
    }
    const body = h("div", { class: "cfg-body" });
    let group = null;
    for (const f of fields) {
      const parts = f.path.split(".");
      const g = parts.slice(1, -1).join(".");
      if (g !== group) { group = g; if (g) body.append(h("div", { class: "sub-title" }, g)); }
      const badge = f.applies === "restart" ? h("span", { class: "pill warn" }, "restart")
        : f.applies === "next_join" ? h("span", { class: "pill" }, "next join") : null;
      body.append(h("div", { class: "setting", id: `s-${f.path}`, "data-search": `${f.path} ${f.help}`.toLowerCase() },
        h("div", {}, h("label", { class: "name", for: `f-${f.path}` }, parts[parts.length - 1]), " ", badge,
          f.help ? h("div", { class: "help" }, f.help) : null),
        h("div", {}, fieldInput(f))));
    }
    out.push(h("details", { class: "cfg", "data-section": name, "data-scope": grp }, h("summary", {}, name,
      h("span", { class: "muted small" }, `${fields.length} setting${fields.length === 1 ? "" : "s"}`)), body));
  }
  $("#cfg").replaceChildren(...out);
  applyCfgFilter();
}

async function loadConfig(force = false) {
  if (FIELDS.length && !force) return;
  try {
    const data = await api("/api/admin/config");
    FIELDS = data.fields;
    $("#cfg-path").textContent = data.path;
    edits.clear();
    renderConfig();
    markChanged("", false);
  } catch (e) { toast(e.message, true); }
}

function applyCfgFilter() {
  const q = $("#cfg-filter").value.trim().toLowerCase();
  const sc = scopeOf("settings");
  const groupsShown = new Set();
  for (const d of $$("details.cfg")) {
    const inScope = sc === "all" || d.dataset.scope === sc || d.dataset.scope === "platform";
    let any = false;
    for (const s of $$(".setting", d)) {
      const hit = inScope && (!q || s.dataset.search.includes(q) || d.dataset.section.includes(q));
      s.classList.toggle("hidden", !hit);
      any ||= hit;
    }
    d.classList.toggle("hidden", !any);
    if (q) d.open = any;
    if (any) groupsShown.add(d.dataset.scope);
  }
  for (const g of $$(".cfg-group")) g.classList.toggle("hidden", !groupsShown.has(g.dataset.scope));
}
$("#cfg-filter").addEventListener("input", applyCfgFilter);
$("#cfg-expand").addEventListener("click", (e) => {
  const open = e.target.textContent === "Expand all";
  for (const d of $$("details.cfg")) d.open = open;
  e.target.textContent = open ? "Collapse all" : "Expand all";
});
$("#cfg-discard").addEventListener("click", () => loadConfig(true));
$("#cfg-save").addEventListener("click", async () => {
  const btn = $("#cfg-save");
  btn.disabled = true;
  try {
    const r = await api("/api/admin/config", { changes: Object.fromEntries(edits) });
    await loadConfig(true);
    if (!r.saved.length) toast("Nothing changed.");
    else if (r.restart.length) {
      toast(h("div", {}, `Saved ${r.saved.length} setting(s). These need a restart: ${r.restart.join(", ")} `,
        h("button", { class: "primary", style: "margin-top:8px", onclick: () => $("#dlg-restart").showModal() }, "Restart now")), false, 15000);
    } else toast(`Saved ${r.saved.length} setting(s)` + (r.next_join.length ? ". Voice changes apply the next time it joins." : ", live now."));
  } catch (e) { toast(e.message, true, 10000); }
  finally { btn.disabled = false; }
});
window.addEventListener("beforeunload", (e) => { if (edits.size) { e.preventDefault(); e.returnValue = ""; } });

// ------------------------------------------------------------ live

const LIVE = { seq: 0, rows: [], servers: [] };
const LIVE_ICON = { heard: "🎙", reply: "🤖", cut: "🤖", event: "•", text_in: "💬", text_out: "🤖" };

function liveServerCard(s) {
  const v = s.voice;
  const state = s.muted ? h("span", { class: "pill bad" }, "muted")
    : v && v.speaking ? h("span", { class: "lamp on" }, "ON AIR")
    : v ? h("span", { class: "pill ok" }, "listening") : h("span", { class: "pill" }, "not in voice");
  return h("article", { class: "card live-server" + (s.muted ? " is-muted" : "") },
    h("div", { class: "info" },
      h("div", { class: "title", title: s.name }, s.name),
      h("div", { class: "s" }, v ? `🔊 ${v.channel} · ${v.people.length ? v.people.join(", ") : "nobody"}` : "text only")),
    state,
    h("button", { class: s.muted ? "primary" : "danger", onclick: (e) => setMute(s, !s.muted, e.currentTarget) },
      s.muted ? "Unmute" : "Mute"));
}

function liveRow(e, showServer, showPlat) {
  const cut = e.kind === "cut";
  return h("div", { class: `fl ${e.kind}` },
    h("span", { class: "t" }, new Date(e.ts * 1000).toLocaleTimeString()),
    h("span", {},
      showPlat ? [platTag(e.platform || "discord"), " "] : null,
      h("span", { class: "where" }, (showServer ? `${e.server} · ` : "") + e.where + "  "),
      `${LIVE_ICON[e.kind] || ""} `,
      e.who ? h("span", { class: "who" }, `${e.who}: `) : null,
      e.text, cut ? h("span", { class: "where" }, " (cut off)") : null));
}

const inScope = (p) => { const sc = PLAT ? scopeOf("main") : "all"; return sc === "all" || (p || "discord") === sc; };

function renderLive() {
  const gid = $("#live-server").value, text = $("#live-text").checked;
  const rows = LIVE.rows.filter((e) => inScope(e.platform) && (!gid || e.guild === gid) && (text || !e.kind.startsWith("text_")));
  const box = $("#live-feed");
  const follow = $("#live-follow").checked;
  const tags = bothOn() && scopeOf("main") === "all";
  box.replaceChildren(...(rows.length ? rows.map((e) => liveRow(e, !gid && LIVE.servers.length > 1, tags))
    : [h("div", { class: "empty" }, "Nothing yet. Lines show up here as people talk to Static.")]));
  if (follow) box.scrollTop = box.scrollHeight;
}

function renderLiveServers() {
  const servers = LIVE.servers.filter((s) => inScope(s.platform));
  const out = [];
  for (const key of ["discord", "fluxer"]) {
    const mine = servers.filter((s) => (s.platform || "discord") === key);
    if (!mine.length) continue;
    if (bothOn() && scopeOf("main") === "all") out.push(h("div", { class: "plat-head small-head" }, platTag(key)));
    out.push(...mine.map(liveServerCard));
  }
  $("#live-servers").replaceChildren(...out);
  const sel = $("#live-server"), keep = sel.value;
  sel.replaceChildren(h("option", { value: "" }, "All servers"),
    ...servers.map((s) => h("option", { value: s.id, selected: s.id === keep },
      bothOn() && scopeOf("main") === "all" ? `${s.name} (${PNAME[s.platform || "discord"]})` : s.name)));
}

async function loadLive() {
  try {
    const data = await api(`/api/admin/live?after=${LIVE.seq}`);
    if (data.reset) LIVE.rows = [];
    LIVE.rows.push(...data.entries);
    LIVE.rows.splice(0, Math.max(0, LIVE.rows.length - 400));
    LIVE.seq = data.seq;
    const serversChanged = JSON.stringify(data.servers) !== JSON.stringify(LIVE.servers);
    LIVE.servers = data.servers;
    if (serversChanged) renderLiveServers();
    if (data.entries.length || data.reset || !$("#live-feed").childElementCount) renderLive();
  } catch (e) { toast(e.message, true); }
}

async function setMute(s, muted, btn) {
  btn.disabled = true;
  try {
    await api(`/api/admin/guilds/${s.id}/mute`, { muted });
    toast(muted ? `Muted in ${s.name}: it won't reply there until you unmute.` : `Unmuted in ${s.name}.`);
    await loadLive();
  } catch (e) { toast(e.message, true); btn.disabled = false; }
}
for (const id of ["#live-server", "#live-text"]) $(id).addEventListener("change", renderLive);
$("#live-follow").addEventListener("change", () => { if ($("#live-follow").checked) renderLive(); });

// ------------------------------------------------------------ logs

function logLine(line) {
  const cls = /\b(ERROR|CRITICAL|Traceback)\b/.test(line) ? "e" : /\bWARNING\b/.test(line) ? "w"
    : line.includes("🎙") || line.includes("💬") ? "h" : line.includes("🤖") ? "b" : "";
  return h("div", { class: cls }, line);
}
async function loadLogs() {
  const box = $("#logs");
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  try {
    const q = encodeURIComponent($("#log-q").value);
    const data = await api(`/api/admin/logs?n=${$("#log-n").value}&q=${q}`);
    box.replaceChildren(...data.lines.map(logLine));
    if (atBottom || !box.dataset.loaded) box.scrollTop = box.scrollHeight;
    box.dataset.loaded = "1";
  } catch (e) { box.replaceChildren(h("div", { class: "e" }, e.message)); }
}
let logDebounce;
$("#log-q").addEventListener("input", () => { clearTimeout(logDebounce); logDebounce = setTimeout(loadLogs, 300); });
$("#log-n").addEventListener("change", loadLogs);

// ------------------------------------------------------------ account

async function loadAccount() {
  await loadMe();
  $("#acct-who").replaceChildren(h("b", {}, ME.name), ` via ${ME.via === "discord" ? "Discord" : "password"}`);
  $("#acct-admins").replaceChildren(...(ME.admins.length ? ME.admins.map((a) =>
    h("div", {}, a.name || "unknown user", " ", h("span", { class: "muted mono" }, a.id))) : [h("span", { class: "muted" }, "none")]));
  $("#pw-state").textContent = ME.password_user ? `Set up for user “${ME.password_user}”. Saving replaces it and signs out other password sessions.`
    : "Not set up. Set one as a fallback for when Discord login isn't available.";
  if (!$("#pw-user").value) $("#pw-user").value = ME.password_user || "admin";
  const d = $("#discord-state");
  if (ME.discord_ready) {
    d.replaceChildren(h("span", { class: "pill ok" }, "on"), " Admins can use “Log in with Discord”.",
      h("div", { class: "muted", style: "margin-top:6px" }, "Redirect URL: ", h("code", {}, ME.redirect_uri)));
  } else {
    d.replaceChildren(h("span", { class: "pill warn" }, "off"),
      h("ol", { class: "muted", style: "padding-left:18px;margin:8px 0 0" },
        h("li", {}, "Settings › dashboard › public_url: your HTTPS address, e.g. https://static.example.com"),
        h("li", {}, "Discord Developer Portal › your app › OAuth2: copy the Client Secret into dashboard › discord_client_secret"),
        h("li", {}, "Same page › Redirects: add ", h("code", {}, (ME.public_url || "https://your.domain").replace(/\/$/, "") + "/auth/callback")),
        h("li", {}, "Restart the bot")));
  }
}

$("#pw-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#pw-msg").className = "small";
  try {
    await api("/api/admin/password", { username: $("#pw-user").value, password: $("#pw-pass").value });
    $("#pw-pass").value = "";
    $("#pw-msg").className = "small okmsg";
    $("#pw-msg").textContent = "Saved.";
    loadAccount();
  } catch (err) { $("#pw-msg").className = "small error"; $("#pw-msg").textContent = err.message; }
});

$("#signout-all").addEventListener("click", async () => {
  if (!confirm("Sign out every browser, including this one?")) return;
  await api("/api/admin/signout-all", {});
  location.href = "/login";
});

// ------------------------------------------------------------ restart

$("#restart").addEventListener("click", () => $("#dlg-restart").showModal());
$("#restart-cancel").addEventListener("click", () => $("#dlg-restart").close());
$("#restart-go").addEventListener("click", () => { $("#dlg-restart").close(); restartAndWait(); });

async function restartAndWait() {
  try { await api("/api/admin/restart", {}); } catch (e) { return toast(e.message, true); }
  toast("Restarting… this page reconnects when the bot is back.", false, 120000);
  $("#dot").className = "dot";
  const started = Date.now();
  await new Promise((r) => setTimeout(r, 4000));
  for (;;) {
    try {
      const me = await api("/api/admin/me");
      if (me.bot.ready) break;
    } catch { /* still down */ }
    if (Date.now() - started > 180000) return toast("The bot hasn't come back after 3 minutes. Check the service on the server.", true, 60000);
    await new Promise((r) => setTimeout(r, 2000));
  }
  toast("Back online.");
  GUILDS = []; FIELDS = [];
  await loadMe();
  show(current);
}

// ------------------------------------------------------------ live refresh

timers.tick = setInterval(() => {
  if (document.hidden) return;
  if (current === "overview") loadOverview();
  if (current === "logs" && $("#log-live").checked) loadLogs();
}, 5000);
timers.live = setInterval(() => { if (!document.hidden && current === "live") loadLive(); }, 1500);
timers.me = setInterval(() => { if (!document.hidden) loadMe().catch(() => {}); }, 30000);

(async () => {
  await loadMe();
  const start = location.hash.slice(1);
  show(["overview", "platforms", "live", "servers", "people", "settings", "logs", "account"].includes(start) ? start : "overview");
})();
