"use strict";

const ERRORS = {
  not_admin: "That Discord account isn't an admin of this bot.",
  state: "Login expired or was tampered with. Try again.",
  denied: "Discord login was cancelled.",
  discord: "Discord login failed. Try again, or use the password.",
  discord_off: "Discord login isn't set up on this bot.",
};

(async () => {
  const err = new URLSearchParams(location.search).get("error");
  if (err) $("#error").textContent = ERRORS[err] || "Login failed.";
  let opts;
  try { opts = await api("/api/auth/options"); } catch (e) { $("#error").textContent = e.message; return; }
  $("#title").textContent = `${opts.name} admin`;
  if (opts.avatar) { $("#avatar").src = opts.avatar; $("#avatar").classList.remove("hidden"); setIcon(opts.avatar); }
  $("#discord").classList.toggle("hidden", !opts.discord);
  $("#pw").classList.toggle("hidden", !opts.password);
  $("#or").classList.toggle("hidden", !(opts.discord && opts.password));
  $("#none").classList.toggle("hidden", opts.discord || opts.password);
})();

$("#pw").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const btn = $("#pw button");
  btn.disabled = true;
  $("#error").textContent = "";
  try {
    await api("/api/auth/login", { username: $("#username").value, password: $("#password").value });
    location.href = "/admin";
  } catch (e) {
    $("#error").textContent = e.message;
    $("#password").value = "";
    $("#password").focus();
  } finally {
    btn.disabled = false;
  }
});
