// Helpers shared by the dashboard's Usage page (stats_usage.js) and Tests
// page (stats_tests.js): number and time formatting that matches
// `ollama stats`, building page elements from text (never HTML), and
// talking to Ollama. Loaded first, as window.OllamaStats.
(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";


  // ---------- formatting (matches `ollama stats`) ----------

  function compact(n) {
    n = Number(n) || 0;
    if (n >= 1e9) return oneDecimal(n / 1e9) + "B";
    if (n >= 1e6) return oneDecimal(n / 1e6) + "M";
    if (n >= 1e3) return Math.round(n / 1e3) + "K";
    return String(Math.round(n));
  }
  function oneDecimal(x) { return x % 1 === 0 ? String(x) : x.toFixed(1); }
  function full(n) { return (Number(n) || 0).toLocaleString(); }
  function truncate(s, n) { s = String(s); return s.length > n ? s.slice(0, n - 1) + "…" : s; }

  // Durations from the API are nanoseconds.
  function seconds(ns) { return (Number(ns) || 0) / 1e9; }
  function dur(ns) {
    var s = seconds(ns);
    if (s <= 0) return "–";
    if (s < 0.1) return Math.round(s * 1000) + "ms";
    if (s < 60) return s.toFixed(1) + "s";
    return Math.floor(s / 60) + "m " + Math.floor(s % 60) + "s";
  }
  // Writing speed, from tokens and nanoseconds.
  function speedOf(tokens, ns) { var s = seconds(ns); return s > 0 ? tokens / s : 0; }
  function speedText(tokens, ns) { var v = speedOf(tokens, ns); return v > 0 ? v.toFixed(1) + " tok/s" : "–"; }
  // The wait before the first word: everything except writing.
  function waitOf(total, output) { return Math.max(0, (total || 0) - (output || 0)); }
  // Same rules as format.HumanBytes in Go, so both views show the same size.
  function bytes(b) {
    b = Number(b) || 0;
    if (b <= 0) return "–";
    var units = ["B", "KB", "MB", "GB", "TB"], i = 0;
    while (b >= 1000 && i < units.length - 1) { b /= 1000; i++; }
    if (i === 0) return b + " B";
    if (b >= 10 || b === Math.trunc(b)) return Math.trunc(b) + " " + units[i];
    return b.toFixed(1) + " " + units[i];
  }
  function ctxSize(n) { return n >= 1024 && n % 1024 === 0 ? (n / 1024) + "K" : compact(n); }
  function ago(iso) {
    if (!iso) return "–";
    var s = (Date.now() - new Date(iso).getTime()) / 1000;
    if (s < 5) return "just now";
    if (s < 60) return Math.round(s) + "s ago";
    if (s < 3600) return Math.round(s / 60) + " min ago";
    if (s < 86400) return Math.round(s / 3600) + " hr ago";
    return Math.round(s / 86400) + " days ago";
  }
  function when(iso, withSeconds) {
    var opts = { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" };
    if (withSeconds) opts.second = "2-digit";
    return new Date(iso).toLocaleString(undefined, opts);
  }
  function pct(x) { return x == null ? "–" : Math.round(x * 100) + "%"; }
  // Total tokens in = newly read + reused.
  function newRead(prompt, cached) { return Math.max(0, (prompt || 0) - (cached || 0)); }
  function reused(cached, prompt) { return prompt > 0 ? Math.min(1, (cached || 0) / prompt) : null; }

  // ---------- DOM (text only, never HTML from data) ----------

  function $(id) { return document.getElementById(id); }
  function el(tag, attrs, text) {
    var e = document.createElement(tag);
    for (var k in attrs || {}) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }
  function svgEl(tag, attrs, text) {
    var e = document.createElementNS(SVG_NS, tag);
    for (var k in attrs || {}) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }
  function row(cells, header) {
    var tr = el("tr");
    cells.forEach(function (c) {
      var td = el(header ? "th" : "td");
      if (c.avg != null) {
        td.appendChild(document.createTextNode(c.text + " "));
        td.appendChild(el("span", { class: "avg" }, "(avg " + c.avg + ")"));
      } else {
        td.textContent = c.text;
      }
      tr.appendChild(td);
    });
    return tr;
  }
  function headRow(names) { return row(names.map(function (h) { return { text: h }; }), true); }
  // Fills a table from a header and rows, replacing what was there.
  function fillTable(table, names, rows) {
    table.replaceChildren();
    if (!rows.length) return;
    var thead = el("thead"), tbody = el("tbody");
    thead.appendChild(headRow(names));
    rows.forEach(function (r) { tbody.appendChild(r); });
    table.append(thead, tbody);
  }
  function wrapTable(table) { var w = el("div", { class: "table-wrap" }); w.appendChild(table); return w; }

  // ---------- talking to Ollama ----------

  // httpError carries the server's own message, so pages can show it.
  function httpError(status, data, text) {
    var e = new Error((data && data.error) || text || ("server returned " + status));
    e.status = status;
    return e;
  }
  function readJSON(r) {
    return r.text().then(function (t) {
      var data = null;
      try { data = t ? JSON.parse(t) : {}; } catch (e) { /* not JSON */ }
      if (!r.ok) throw httpError(r.status, data, t);
      return data || {};
    });
  }
  function getJSON(url, signal) { return fetch(url, { cache: "no-store", signal: signal }).then(readJSON); }

  window.OllamaStats = {
    compact: compact,
    full: full,
    truncate: truncate,
    dur: dur,
    speedOf: speedOf,
    speedText: speedText,
    waitOf: waitOf,
    bytes: bytes,
    ctxSize: ctxSize,
    ago: ago,
    when: when,
    pct: pct,
    newRead: newRead,
    reused: reused,
    $: $,
    el: el,
    svgEl: svgEl,
    row: row,
    fillTable: fillTable,
    wrapTable: wrapTable,
    readJSON: readJSON,
    getJSON: getJSON
  };
})();
