// Shared helpers for the status page and the admin panel.
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "html") el.innerHTML = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const res = await fetch(path, { credentials: "same-origin", ...opts });
  let data = {};
  try { data = await res.json(); } catch { /* empty body */ }
  if (res.status === 401 && path.startsWith("/api/admin")) { location.href = "/login"; throw new Error("signed out"); }
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

function duration(s) {
  s = Math.max(0, Math.floor(s));
  const d = Math.floor(s / 86400), hh = Math.floor(s % 86400 / 3600), m = Math.floor(s % 3600 / 60);
  return [d && `${d}d`, hh && `${hh}h`, m && `${m}m`].filter(Boolean).join(" ") || `${s}s`;
}

function ago(ts) {
  const s = Date.now() / 1000 - ts;
  if (s < 60) return `${Math.max(0, Math.round(s))}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

function dateStr(ts) {
  return ts ? new Date(ts * 1000).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) : "-";
}

// <time data-ts> from the rendered embeds (Discord <t:…:T> timestamps) -> local time.
function fillTimes(root = document) {
  for (const t of $$("time[data-ts]", root)) t.textContent = new Date(+t.dataset.ts * 1000).toLocaleTimeString();
}

// Browser-tab icon = the bot's avatar.
function setIcon(url) {
  const link = $("link[rel=icon]");
  if (link && link.href !== url) link.href = url;
}

let toastTimer;
function toast(msg, bad = false, ms = 5000) {
  $(".toast")?.remove();
  const el = h("div", { class: "toast" + (bad ? " bad" : ""), role: "status" }, msg);
  document.body.append(el);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.remove(), ms);
}
