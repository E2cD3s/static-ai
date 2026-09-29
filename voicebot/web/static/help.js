"use strict";
// Help page. The words live in help.html; this fills in what depends on the running bot (its name, wake word,
// timings), removes [data-if] parts for features that are off, and builds the lists that come from live data.

function secs(s) {
  if (s >= 60 && s % 60 === 0) { const m = s / 60; return m === 1 ? "a minute" : `${m} minutes`; }
  return `${Math.round(s)} seconds`;
}

function items(sel, list) {
  const el = $(sel);
  if (el) el.replaceChildren(...list.filter(Boolean).map((x) => h("li", {}, ...[].concat(x))));
}

function render(d) {
  const name = d.name || "Static", v = d.voice, t = d.text, f = d.features;
  const pl = d.platforms || { discord: true, fluxer: false, fluxer_prefix: "!" };
  const cap = (s) => s[0].toUpperCase() + s.slice(1);
  const wake = v.wake_words[0] ? cap(v.wake_words[0]) : name;
  const wakeMode = v.mode === "wake_word";
  const flags = {
    wake: wakeMode, always: !wakeMode, speaker: wakeMode && v.followup_scope === "speaker",
    wake_extra: wakeMode && v.wake_words.length > 1, barge_in: v.barge_in,
    leave_on_request: v.leave_on_request, auto_leave_when_empty: v.auto_leave_when_empty, boomerang: v.boomerang,
    context: t.context_messages > 0, extras: f.reminders || f.clips || f.quotes || f.fun || f.weather,
    discord: pl.discord, fluxer: pl.fluxer, fluxer_host: pl.fluxer && !!pl.fluxer_host,
    learns: f.tuning || f.mood, ...f,
    barge_speaker: v.barge_in && v.barge_in_scope !== "anyone", resume: v.barge_in && v.resume_s > 0,
  };

  // Show/hide by feature. "!flag" = only when it's off.
  for (const el of $$("[data-if]")) {
    const key = el.dataset.if, on = key.startsWith("!") ? !flags[key.slice(1)] : !!flags[key];
    if (!on) el.remove();
  }
  const values = {
    name, wake, followup: secs(v.followup_s), context: t.context_messages,
    clip_default: secs(f.clip_default_s), clip_buffer: secs(f.clip_buffer_s),
    wake_extra: v.wake_words.slice(1).map((w) => `"${w}"`).join(", "),
    barge: v.barge_in_s >= 1 ? `${+v.barge_in_s.toFixed(1)} seconds` : "half a second",
    resume: secs(v.resume_s), check_cooldown: secs(f.check_in_cooldown_min * 60),
    fluxer_host: pl.fluxer_host || "", fx_prefix: pl.fluxer_prefix,
  };
  for (const el of $$("[data-fill]")) if (el.dataset.fill in values) el.textContent = values[el.dataset.fill];
  // Example phrases use the configured wake word.
  for (const q of $$("q")) q.textContent = q.textContent.replace(/\bStatic\b/g, wake);
  // Fluxer-only examples use its configured prefix (written with "!" in the page).
  if (pl.fluxer_prefix !== "!") for (const c of $$(".cmd-fx")) c.textContent = c.textContent.replace(/^!/, pl.fluxer_prefix);

  document.title = `${name} Help`;
  if (d.avatar) { $("#avatar").src = d.avatar; $("#avatar").classList.remove("hidden"); setIcon(d.avatar); }

  TEXT_WHEN = (tt) => [
    tt.mentions && "@mention it anywhere",
    tt.replies && "reply to one of its messages",
    tt.name && ["say its name in a message: ", h("q", {}, `${wake}, is this take bad?`)],
    tt.threads && "in threads it started or joined, it answers every message",
    tt.dms && "in DMs, it answers every message",
    tt.channels > 0 && `in ${tt.channels} dedicated channel${tt.channels === 1 ? "" : "s"}, it answers everything`,
  ];

  items("#keeps", [
    "the current conversation, until it goes quiet for a while",
    f.profiles && "short notes about you, written from your conversations",
    f.lore && "a few memorable moments from the server's chats (one line each)",
    f.reminders && "reminders and polls, until they're done",
    f.mood && "how you usually sound (volume and pace, as numbers)",
    f.tuning && "how long this server likes its replies",
    f.quotes && "lines someone asked it to quote (who said it and when), until an admin or whoever saved it deletes them",
    f.links && "which Discord and Fluxer accounts belong together, only if you link them yourself",
  ]);
  items("#not-keeps", [
    f.clips ? `recordings of the call (the most recent ${secs(f.clip_buffer_s).replace(/^a /, "")} is held in memory for clips, then overwritten)`
      : "recordings of the call",
    "anything sent to a cloud AI: listening, thinking and speaking all happen on the owner's PC",
    f.search && "your voice or name in web searches (only the search words leave the machine)",
    f.weather && "who asked about the weather (only the place name goes to the weather service)",
  ]);

  const m = d.models || {};
  const stages = [
    ["s-hear", "Hear", "It listens to each person separately, notices when they stop talking, and turns speech into text.", m.stt],
    ["s-look", "Look", f.search ? "It decides whether the question needs fresh facts, and if so searches the web."
      : "It gathers what it knows: who's here and what's been said.", null],
    ["s-think", "Think", "A language model writes the reply in character, one sentence at a time.", m.llm],
    ["s-speak", "Speak", "Each sentence becomes speech as soon as it's written and plays into the call.", m.tts],
  ];
  $("#stages").replaceChildren(...stages.map(([cls, title, text, tech]) => h("li", {},
    h("h3", {}, h("i", { class: cls }), title), h("p", {}, text),
    tech && h("p", { class: "tech" }, tech.split(" · ")[0]))));

  // Which platform's details to show: ?p=fluxer, the last one picked, or whichever bot runs.
  const avail = ["discord", "fluxer"].filter((k) => pl[k]);
  if (!avail.length) avail.push("discord");
  let want = new URLSearchParams(location.search).get("p");
  try { want = want || localStorage.getItem("help.platform"); } catch { /* private window */ }
  const tabs = $("#plat-tabs");
  tabs.classList.toggle("hidden", avail.length < 2);
  for (const b of $$("button", tabs)) b.addEventListener("click", () => showPlatform(b.dataset.p, d, true));
  showPlatform(avail.includes(want) ? want : avail[0], d, false);
}

let TEXT_WHEN = null;

// Everything that differs between the Discord and the Fluxer bot: command prefix, text triggers, command list,
// and the [data-platform] bits of the page.
function showPlatform(p, d, remember) {
  const pl = d.platforms || {};
  const fx = p === "fluxer", prefix = fx ? (pl.fluxer_prefix || "!") : "/";
  document.body.dataset.platform = p;
  for (const b of $$("#plat-tabs button")) b.setAttribute("aria-selected", String(b.dataset.p === p));
  for (const el of $$("[data-platform]")) el.hidden = el.dataset.platform !== p;
  for (const c of $$("code.cmd")) c.textContent = prefix + c.dataset.cmd;
  items("#text-when", TEXT_WHEN(fx ? d.fluxer_text : d.text));
  if (remember) {
    try { localStorage.setItem("help.platform", p); } catch { /* private window */ }
    const url = new URL(location.href);
    url.searchParams.set("p", p);
    history.replaceState(null, "", url);
  }

  // Commands, grouped (owner-only ones sit under "Owner").
  const list = fx ? d.fluxer_commands : d.commands;
  const groups = new Map();
  for (const c of list) (groups.get(c.group) || groups.set(c.group, []).get(c.group)).push(c);
  $("#cmds").replaceChildren(...[...groups].map(([g, cmds]) => h("div", { class: "cmd-group" },
    h("h3", {}, g),
    h("dl", { class: "cmds" }, ...cmds.flatMap((c) => [
      h("dt", { title: c.params.length ? "Options: " + c.params.map((p) => p.required ? p.name : `${p.name} (optional)`).join(", ") : null },
        h("code", {}, `${prefix}${c.name}`), fx && c.args && c.args.length <= 14 ? h("span", { class: "cmd-args" }, ` ${c.args}`) : null),
      h("dd", {}, c.description, fx && c.args && c.args.length > 14
        ? h("div", { class: "cmd-eg" }, "e.g. ", h("code", {}, `${prefix}${c.name} ${c.args}`)) : null),
    ])))));
}

api("/api/help").then(render).catch((e) => {
  $("#cmds").replaceChildren(h("p", { class: "error" }, `Couldn't load the live details (${e.message}). Reload to try again.`));
});

// Contents: highlight the section being read.
const tocLinks = $$(".h-toc a");
const spy = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    for (const a of tocLinks) a.classList.toggle("here", a.getAttribute("href") === `#${e.target.id}`);
  }
}, { rootMargin: "-20% 0px -70% 0px" });
for (const s of $$(".h-doc section")) spy.observe(s);
