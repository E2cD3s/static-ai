"use strict";
// The live board drawn from /api/status (voicebot/web/snapshot.py): public page (status.js) and admin Overview
// (admin.js). Everything is built with h(), never innerHTML, since server/channel names come from Discord.

// ---------- formatting ----------
const isNum = (x) => typeof x === "number" && isFinite(x);
const ms = (x) => !isNum(x) ? "-" : x >= 1000 ? `${(x / 1000).toFixed(x >= 10000 ? 0 : 2)} s` : `${Math.round(x)} ms`;
const int = (x) => isNum(x) ? Math.round(x).toLocaleString() : "-";
const gb = (x, d = 1) => isNum(x) ? `${x.toFixed(d)} GB` : "-";
const pct = (x) => isNum(x) ? `${Math.round(x)}%` : "-";
const fix = (x, d = 1) => isNum(x) ? x.toFixed(d) : "-";
const plural = (n, word) => `${int(n)} ${word}${n === 1 ? "" : "s"}`;
const secs = (s) => !isNum(s) ? "-" : s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`;

function facts(rows) {
  return h("dl", { class: "facts" }, rows.filter(Boolean).flatMap(([k, v, tip]) =>
    [h("dt", { title: tip }, k), h("dd", {}, v)]));
}

// fullGood: a full bar is the healthy state (e.g. model 100% on GPU), so it never turns red.
function meter(label, value, used, total, tip, fullGood = false) {
  const frac = total ? Math.min(1, Math.max(0, used / total)) : 0;
  return h("div", { class: "meter", title: tip },
    h("div", { class: "meter-top" }, h("span", {}, label), h("span", { class: "num" }, value)),
    h("div", { class: "meter-track" },
      h("span", { class: fullGood ? (frac < 0.999 ? "hot" : "good") : frac > 0.92 ? "hot" : frac > 0.8 ? "warn" : null, style: `width:${(frac * 100).toFixed(1)}%` })));
}

function sparkline(values, colorVar) {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 100 40");
  svg.setAttribute("preserveAspectRatio", "none");
  if (values.length < 2) return svg;
  const max = Math.max(...values), min = Math.min(...values), span = max - min || 1;
  const pts = values.map((v, i) => `${(i / (values.length - 1) * 100).toFixed(2)},${(38 - (v - min) / span * 34).toFixed(2)}`);
  const line = document.createElementNS(NS, "polyline");
  line.setAttribute("points", pts.join(" "));
  line.setAttribute("fill", "none");
  line.setAttribute("stroke", `var(${colorVar})`);
  line.setAttribute("stroke-width", "1.6");
  line.setAttribute("vector-effect", "non-scaling-stroke");
  line.setAttribute("stroke-linejoin", "round");
  svg.append(line);
  return svg;
}

const none = (text = "no data yet") => h("span", { class: "none" }, text);

function section(title, sub, ...body) {
  return h("section", { class: "block" }, h("h2", {}, title), sub && h("p", { class: "sub" }, sub), ...body);
}

// ---------- hero: how fast it answers out loud ----------
function renderHero(r) {
  const hero = h("div", { class: "hero" }, h("p", { class: "kicker" }, "Time to answer out loud · median"));
  if (!r.count) {
    hero.append(
      h("div", { class: "big-none" }, "No replies yet"),
      h("p", { class: "lede" }, "No voice replies since the bot last started. Once someone talks to it in a voice " +
        "channel, this shows how long it takes from the moment they stop speaking to the moment it starts answering."));
    return hero;
  }
  const [val, unit] = r.p50 >= 1000 ? [(r.p50 / 1000).toFixed(2), "s"] : [Math.round(r.p50), "ms"];
  hero.append(h("div", { class: "hero-row" },
    h("div", { class: "big num" }, val, h("small", {}, unit)),
    h("dl", { class: "hero-side" },
      h("dt", { title: "9 in 10 replies start faster than this" }, "Slow ones (p90)"), h("dd", {}, ms(r.p90)),
      h("dt", {}, "Fastest"), h("dd", {}, ms(r.best)),
      h("dt", {}, "Slowest"), h("dd", {}, ms(r.worst)),
      h("dt", { title: "Replies it started writing before you'd finished, then kept" }, "Started early"),
      h("dd", {}, `${int(r.speculative)} of ${int(r.count)}`)),
    h("div", { class: "trend" }, sparkline(r.trend, "--signal"),
      h("div", { class: "cap" }, h("span", {}, `last ${r.trend.length} replies`), h("span", {}, "newest →")))));

  const total = r.stages.reduce((a, s) => a + s.ms, 0);
  if (total > 0) {
    hero.append(h("div", { class: "chain" },
      h("p", { class: "kicker" }, "Where the time goes · average"),
      h("div", { class: "chain-bar", role: "img", "aria-label": r.stages.map((s) => `${s.label} ${ms(s.ms)}`).join(", ") },
        r.stages.filter((s) => s.ms > 0).map((s) =>
          h("span", { class: `s-${s.key}`, style: `flex:${s.ms.toFixed(1)}`, title: `${s.label}: ${ms(s.ms)}` }))),
      h("div", { class: "chain-key" }, r.stages.map((s) =>
        h("div", {}, h("i", { class: `s-${s.key}` }), h("span", {}, s.label), h("b", {}, ms(s.ms)))))));
  }
  return hero;
}

// ---------- the signal chain: ears -> brain -> voice ----------
function renderChain(d) {
  const { hear, think, speak } = d;
  const ears = h("div", { class: "card" },
    h("h3", {}, "① Ears · speech to text"),
    h("div", { class: "what" }, hear.desc || "not loaded"),
    h("div", { class: "metric num" }, ...(isNum(hear.rtf) ? [`${fix(hear.rtf, 0)}×`, h("small", {}, "realtime")] : [none()])),
    h("div", { class: "metric-note" }, isNum(hear.rtf)
      ? `Transcribes a second of speech in ${ms(1000 / hear.rtf)}` : "Waiting for someone to talk"),
    facts([
      ["Clips transcribed", int(hear.calls)],
      ["Speech heard", secs(hear.audio_s)],
      ["Average per clip", ms(hear.avg_ms)],
    ]));

  const brain = h("div", { class: "card" },
    h("h3", {}, "② Brain · language model"),
    h("div", { class: "what" }, think.model || "-"),
    h("div", { class: "metric num" }, ...(isNum(think.tok_s) ? [fix(think.tok_s, 0), h("small", {}, "tokens / second")] : [none()])),
    h("div", { class: "metric-note" }, isNum(think.ttft_ms)
      ? `Starts writing ${ms(think.ttft_ms)} after it's asked` : "No requests yet"),
    think.tok_s_trend.length > 1 && h("div", { class: "trend", title: "Writing speed of recent requests" },
      sparkline(think.tok_s_trend, "--s-think")),
    facts([
      ["Requests", `${int(think.requests)}${think.errors ? ` · ${think.errors} failed` : ""}`],
      ["Tokens read / written", `${int(think.prompt_tokens)} / ${int(think.output_tokens)}`],
      think.cancelled && ["Cut short", int(think.cancelled), "Stopped early: someone talked over it, or a draft reply was dropped"],
    ]),
    think.context.map((c) => meter(`Memory in use · ${c.purpose}`, `${int(c.used)} / ${int(c.max)}`, c.used, c.max,
      "How much of the model's context window the latest request filled")),
    ...(think.ollama?.models || []).map((m) => meter(
      `${m.name.replace(/:latest$/, "")} · ${m.params} ${m.quant}`, `${pct(m.gpu_pct)} on GPU`, m.gpu_pct, 100,
      "Below 100% means part of the model runs on the CPU, which is several times slower", true)),
    think.by_purpose.length > 0 && h("table", { class: "mini" },
      h("thead", {}, h("tr", {}, h("th", {}, "Used for"), h("th", {}, "Requests"), h("th", {}, "Tok/s"), h("th", {}, "1st token"))),
      h("tbody", {}, think.by_purpose.map((p) => h("tr", {},
        h("td", {}, p.purpose), h("td", {}, int(p.requests)), h("td", {}, fix(p.tok_s, 0)), h("td", {}, ms(p.ttft_ms)))))));

  const voice = h("div", { class: "card" },
    h("h3", {}, "③ Voice · text to speech"),
    h("div", { class: "what" }, speak.desc || "not loaded"),
    h("div", { class: "metric num" }, ...(isNum(speak.rtf) ? [`${fix(speak.rtf, 0)}×`, h("small", {}, "realtime")] : [none()])),
    h("div", { class: "metric-note" }, isNum(speak.rtf)
      ? `Makes a second of speech in ${ms(1000 / speak.rtf)}` : "Hasn't spoken yet"),
    facts([
      ["Sentences spoken", int(speak.calls)],
      ["Speech made", secs(speak.audio_s)],
      ["Average per sentence", ms(speak.avg_ms)],
    ]));

  return section("Signal chain", "What happens to every voice reply, in order.",
    h("div", { class: "stations" }, ears, h("div", { class: "arrow", "aria-hidden": "true" }, "→"),
      brain, h("div", { class: "arrow", "aria-hidden": "true" }, "→"), voice));
}

// ---------- live voice ----------
function renderOnAir(list) {
  return section("On air now", null, list.length
    ? h("div", { class: "onair-list" }, list.map((v) => h("div", { class: "card onair-item" },
        h("span", { class: "pulse" }),
        h("div", {}, h("div", { class: "t" }, v.platform ? [h("span", { class: `ptag ${v.platform}` },
          v.platform === "fluxer" ? "Fluxer" : "Discord"), " "] : null, v.channel ? `🔊 ${v.channel}` : "🔊 In a voice channel"),
          h("div", { class: "s" }, [v.server, plural(v.people, "person").replace("persons", "people"),
            isNum(v.ws_ms) ? `voice ping ${ms(v.ws_ms)}` : null].filter(Boolean).join(" · "))))))
    : h("p", { class: "quiet" }, "Not in a voice channel right now."));
}

// ---------- the machine ----------
function renderMachine(m) {
  const cards = [];
  const g = m.gpu;
  if (g) cards.push(h("div", { class: "card" },
    h("h3", {}, "Graphics card"), h("div", { class: "what" }, `${g.name} · driver ${g.driver} · CUDA ${g.cuda}`),
    h("div", { class: "metric num" }, pct(g.util), h("small", {}, "busy")),
    meter("Video memory", `${fix(g.vram_used)} / ${gb(g.vram_total, 0)}`, g.vram_used, g.vram_total,
      "Speech models and the language model all share this"),
    meter("Power", `${fix(g.power, 0)} / ${fix(g.power_limit, 0)} W`, g.power, g.power_limit),
    h("hr"),
    facts([["Temperature", `${fix(g.temp, 0)} °C`], ["Fan", pct(g.fan)],
      ["Clocks (core / memory)", `${int(g.clock_core)} / ${int(g.clock_mem)} MHz`]])));

  const c = m.cpu;
  cards.push(h("div", { class: "card" },
    h("h3", {}, "Processor"), h("div", { class: "what" }, `${c.model} · ${c.cores} cores / ${c.threads} threads`),
    h("div", { class: "metric num" }, pct(c.util), h("small", {}, "busy")),
    h("div", { class: "cores", title: "Each bar is one CPU thread" },
      c.per_core.map((p) => h("div", {}, h("span", { style: `height:${Math.min(100, p)}%` })))),
    h("div", { class: "cores-cap" }, "per thread"),
    h("hr"),
    facts([["Temperature", isNum(c.temp) ? `${fix(c.temp, 0)} °C` : "-"],
      ["Clock", isNum(c.freq) ? `${(c.freq / 1000).toFixed(2)} GHz` : "-"],
      ["Load (1 / 5 / 15 min)", c.load.map((x) => x.toFixed(2)).join(" / "), "Threads waiting for the CPU, averaged"]])));

  cards.push(h("div", { class: "card" },
    h("h3", {}, "Memory & disk"), h("div", { class: "what" }, `${m.os} · machine up ${duration(m.host_up)}`),
    meter("RAM", `${fix(m.ram.used)} / ${gb(m.ram.total)}`, m.ram.used, m.ram.total),
    m.swap.total > 0 && meter("Swap", `${fix(m.swap.used)} / ${gb(m.swap.total)}`, m.swap.used, m.swap.total,
      "Memory moved to disk. Some is normal; lots of it slows things down"),
    meter("Disk", `${fix(m.disk.used, 0)} / ${gb(m.disk.total, 0)}`, m.disk.used, m.disk.total),
    h("hr"),
    facts([["The bot itself", `${gb(m.process.rss, 2)} RAM · ${pct(m.process.cpu)} CPU`],
      ["Bot threads", int(m.process.threads)]])));

  return section("The machine", "One home PC runs all of it: no cloud services.", h("div", { class: "machine" }, cards));
}

// ---------- tallies since start ----------
function renderActivity(a) {
  const items = [
    [a.heard, "things heard"], [a.voice_replies, "spoken replies"], [a.text_replies, "text replies"],
    [a.conversations, "text conversations"], [a.barge_ins, "times interrupted", "Someone talked over it and it stopped"],
    [a.backchannels, "“mm-hm”s ignored", "Short sounds like “yeah” or “mm” that it didn't treat as a turn"],
    [a.spec_used, "early starts kept", `Started writing before you finished and kept it (${int(a.spec_started)} tried)`],
    [a.clips, "clips saved"],
  ];
  return section("Since it started", null,
    h("div", { class: "tally" }, items.map(([n, label, tip]) =>
      h("div", { title: tip }, h("b", { class: "num" }, int(n)), h("span", {}, label)))));
}

// ---------- extras ----------
function renderFeatures(f) {
  const cards = [];
  const s = f.search;
  cards.push(h("div", { class: "card" },
    h("h3", {}, "Web search", !s.enabled && h("span", { class: "off" }, "OFF")),
    h("div", { class: "what" }, `Checked ${plural(s.checks, "question")} · searched ${plural(s.searches, "time")}`),
    s.backends.length > 0 && h("table", { class: "mini" },
      h("thead", {}, h("tr", {}, h("th", {}, "Source"), h("th", {}, "Found"), h("th", {}, "Empty"), h("th", {}, "Failed"), h("th", {}, "Avg"))),
      h("tbody", {}, s.backends.map((b) => h("tr", {},
        h("td", {}, b.name), h("td", {}, int(b.ok)), h("td", {}, int(b.empty)), h("td", {}, int(b.fail)), h("td", {}, ms(b.ms))))))));

  const r = f.reminders;
  cards.push(h("div", { class: "card" },
    h("h3", {}, "Reminders & polls", !r.enabled && h("span", { class: "off" }, "OFF")),
    h("div", { class: "what" }, "Set by voice, text or /remind"),
    facts([["Waiting to go off", int(r.pending)], ["Polls open", int(r.polls)],
      ["Set since start", int(r.set)], ["Delivered since start", int(r.sent)]])));

  const mood = f.mood;
  cards.push(h("div", { class: "card" },
    h("h3", {}, "Mood reading", !mood && h("span", { class: "off" }, "OFF")),
    h("div", { class: "what" }, "Reads the feeling behind each thing said, to pick how to answer"),
    mood ? [facts([["Reads", int(mood.reads)], ["Average time", ms(mood.avg_ms)]]),
      mood.top.length ? h("div", { class: "tags" }, mood.top.map(([name, n]) => h("span", {}, name, h("b", {}, int(n)))))
        : h("p", { class: "quiet" }, "Nothing read yet.")]
      : h("p", { class: "quiet" }, "Turned off.")));

  if (f.memory) cards.push(h("div", { class: "card" },
    h("h3", {}, "Long-term memory"),
    h("div", { class: "what" }, "What it remembers about people between conversations"),
    facts([["People it knows", int(f.memory.people)], ["With a written profile", int(f.memory.profiles)]])));

  return section("Extras", null, h("div", { class: "features" }, cards));
}

function boardFoot(d) {
  return [
    ...Object.entries(d.versions || {}).map(([k, v]) => h("span", {}, `${k} ${v}`)),
    d.think?.ollama?.version && h("span", {}, `Ollama ${d.think.ollama.version}`),
    isNum(d.collected_ms) && h("span", {}, `gathered in ${ms(d.collected_ms)}`),
    h("span", {}, "refreshes every 10 s · same numbers as /status in Discord")];
}

// The whole board, top to bottom. Used by the public page and the admin Overview.
function renderBoard(d) {
  return [renderHero(d.reply), renderOnAir(d.voice_now), renderChain(d),
    renderMachine(d.machine), renderActivity(d.activity), renderFeatures(d.features)];
}
