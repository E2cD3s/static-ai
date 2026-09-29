"use strict";
// Public status page: the header strip + the shared board (board.js).

let lastTs = 0, failures = 0;

// ---------- header ----------
function renderStrip(d) {
  const bot = d.bot || {};
  $("#name").textContent = bot.name || "Static";
  const live = (d.voice_now || []).length > 0;
  const lamp = $("#lamp");
  lamp.textContent = !bot.ready ? "OFF AIR" : live ? "ON AIR" : "STANDING BY";
  lamp.className = "lamp" + (!bot.ready ? "" : live ? " on" : " idle");
  lamp.title = live ? "Talking in a voice channel right now" : bot.ready ? "Online, not in a voice channel" : "Not connected";
  // Which bots run (Discord / Fluxer), lit when online.
  const pl = bot.platform;
  const plats = pl ? ["discord", "fluxer"].filter((k) => pl.mode === k || pl.mode === "both") : [];
  $("#plats").replaceChildren(...(plats.length > 1 || plats[0] === "fluxer" ? plats.map((k) => h("span", {
    class: "plat-chip", title: `${k === "fluxer" ? "Fluxer" : "Discord"} bot ${pl[k] ? "online" : "offline"}` },
    h("span", { class: "dot" + (pl[k] ? " on" : " bad") }), k === "fluxer" ? "Fluxer" : "Discord")) : []));
  $("#presence").textContent = bot.presence || (bot.ready ? "online" : "starting up…");
  $("#uptime").textContent = `up ${duration(bot.uptime || 0)}`;
  const servers = (bot.guilds || 0) + (pl && pl.fluxer ? pl.fluxer_servers || 0 : 0);
  $("#guilds").textContent = plural(servers, "server");
  $("#ping").textContent = isNum(bot.ping_ms) ? `Discord ${ms(bot.ping_ms)}` : "";
  $("#ping").title = "Round trip to Discord's gateway";
  if (bot.avatar) { $("#avatar").src = bot.avatar; $("#avatar").classList.remove("hidden"); setIcon(bot.avatar); }
  document.title = `${bot.name || "Static"} Status`;
}

// ---------- loop ----------
async function refresh() {
  try {
    const d = await api("/api/status");
    renderStrip(d);
    if (d.starting || !d.reply) {
      $("#main").replaceChildren(h("p", { class: "starting" }, "The bot is starting up. Stats appear in a few seconds."));
    } else {
      $("#main").replaceChildren(...renderBoard(d));
      $("#foot").replaceChildren(...boardFoot(d).filter(Boolean));
    }
    lastTs = d.ts;
    failures = 0;
  } catch (e) {
    failures++;
    $("#lamp").textContent = "OFF AIR";
    $("#lamp").className = "lamp";
    $("#presence").textContent = `can't reach the bot (${e.message})`;
  }
}

setInterval(() => {
  const u = $("#updated");
  u.textContent = lastTs ? `updated ${ago(lastTs)}` : "-";
  u.className = "upd" + (failures ? " bad" : "");
}, 1000);

refresh();
setInterval(refresh, 10000);
