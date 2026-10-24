// The dashboard's Usage page (/stats): totals, charts, tags, and the
// request log for a tag. It reads /api/stats and /api/stats/log and
// refreshes every few seconds while the page is open.
(function () {
  "use strict";

  var S = window.OllamaStats;
  var compact = S.compact;
  var full = S.full;
  var truncate = S.truncate;
  var dur = S.dur;
  var speedOf = S.speedOf;
  var speedText = S.speedText;
  var waitOf = S.waitOf;
  var bytes = S.bytes;
  var ctxSize = S.ctxSize;
  var ago = S.ago;
  var when = S.when;
  var pct = S.pct;
  var newRead = S.newRead;
  var reused = S.reused;
  var $ = S.$;
  var el = S.el;
  var svgEl = S.svgEl;
  var row = S.row;
  var fillTable = S.fillTable;
  var getJSON = S.getJSON;
  var Grading = window.OllamaGrading;

  var REFRESH_MS = 3000;
  var REQUEST_TIMEOUT_MS = 10000;
  var DAYS_SHOWN = 30;
  var CONTEXT_WARN_AT = 0.9;
  var TAGS_SHOWN = 10;


  var range = "all";
  try { range = localStorage.getItem("ollama-stats-range") === "today" ? "today" : "all"; } catch (e) { /* storage blocked */ }
  var last = null;
  var tagLog = null;

  // The selected tag lives in the address, like #tag=cloud-test, so a tag
  // view can be bookmarked or shared. Invalid tags are ignored.
  function tagFromHash() {
    var m = location.hash.match(/^#tag=(.+)$/);
    if (!m) return "";
    var t = "";
    try { t = decodeURIComponent(m[1]); } catch (e) { return ""; }
    return Grading.isValidTag(t) ? t : "";
  }
  var tag = tagFromHash();

  // ---------- per-model math (matches api.ModelStats methods) ----------

  function ok(m) { return Math.max(0, m.requests - m.errors); }
  function avgOf(m, total) { var n = ok(m); return n > 0 ? (total || 0) / n : 0; }
  function tps(m) { return speedOf(m.output_tokens, m.output_duration); }
  function avgWait(m) { return avgOf(m, waitOf(m.total_duration, m.output_duration)); }
  function avgTime(m) { return avgOf(m, m.total_duration); }
  // The window the fullest request ran in, or the current one if unknown.
  function fillWindow(m) { return m.max_context_window || m.context_length || 0; }
  // Same wording as `ollama ps`.
  function processor(m) {
    if (m.cloud) return "Cloud";
    if (m.memory_vram == null || !(m.memory_size > 0) || m.memory_vram > m.memory_size) return "–";
    var gpu = m.memory_vram / m.memory_size;
    if (gpu >= 1) return "100% GPU";
    if (gpu <= 0) return "100% CPU";
    var cpu = Math.round((1 - gpu) * 100);
    return cpu + "%/" + (100 - cpu) + "% CPU/GPU";
  }

  // ---------- cost estimate ----------

  var prices = { input: 1, output: 4 };
  try {
    var savedPrices = JSON.parse(localStorage.getItem("ollama-stats-prices") || "null");
    if (savedPrices && savedPrices.input >= 0 && savedPrices.output >= 0) prices = savedPrices;
  } catch (e) { /* storage blocked or bad value */ }
  function money(x) {
    if (x > 0 && x < 0.01) return "under $0.01";
    return "$" + x.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  // ---------- charts ----------

  // A bar with a 4px rounded data end and a square base.
  function barPath(x, y, w, h, horizontal) {
    var r = Math.min(4, horizontal ? w : h, horizontal ? h / 2 : w / 2);
    if (w <= 0 || h <= 0) return "";
    if (horizontal) {
      return "M" + x + "," + y + "H" + (x + w - r) + "Q" + (x + w) + "," + y + " " + (x + w) + "," + (y + r) +
        "V" + (y + h - r) + "Q" + (x + w) + "," + (y + h) + " " + (x + w - r) + "," + (y + h) + "H" + x + "Z";
    }
    return "M" + x + "," + (y + h) + "V" + (y + r) + "Q" + x + "," + y + " " + (x + r) + "," + y +
      "H" + (x + w - r) + "Q" + (x + w) + "," + y + " " + (x + w) + "," + (y + r) + "V" + (y + h) + "Z";
  }
  function niceMax(v) {
    if (v <= 0) return 1;
    var p = Math.pow(10, Math.floor(Math.log10(v))), f = v / p;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p;
  }

  var tip = $("tooltip");
  function showTip(evt, title, lines) {
    tip.replaceChildren(el("strong", null, title));
    lines.forEach(function (l) { tip.appendChild(el("div", null, l)); });
    tip.style.opacity = "1";
    var pad = 12, r = tip.getBoundingClientRect();
    var x = evt.clientX + pad, y = evt.clientY - r.height - pad;
    if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
    if (y < 8) y = evt.clientY + pad;
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }
  function hideTip() { tip.style.opacity = "0"; }
  function hover(hit, mark, title, lines) {
    hit.addEventListener("mousemove", function (e) { mark.classList.add("hover"); showTip(e, title, lines); });
    hit.addEventListener("mouseleave", function () { mark.classList.remove("hover"); hideTip(); });
  }

  function lastDays(today, saved) {
    var byDate = {};
    (saved || []).forEach(function (d) { byDate[d.date] = d; });
    var parts = today.split("-").map(Number);
    var out = [];
    for (var i = DAYS_SHOWN - 1; i >= 0; i--) {
      var d = new Date(parts[0], parts[1] - 1, parts[2] - i);
      var key = d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0") + "-" + String(d.getDate()).padStart(2, "0");
      out.push({ date: d, stats: byDate[key] || { requests: 0, errors: 0, output_tokens: 0 } });
    }
    return out;
  }

  function drawDays(st) {
    var box = $("chart-days");
    var days = lastDays(st.today, st.days);
    var W = Math.max(box.clientWidth, 280), H = 180;
    var left = 36, right = 4, top = 8, bottom = 22;
    var plotW = W - left - right, plotH = H - top - bottom;
    var max = niceMax(Math.max.apply(null, days.map(function (d) { return d.stats.requests; })));
    var slot = plotW / days.length, bw = Math.min(24, Math.max(2, slot - 2));
    var busy = days.filter(function (d) { return d.stats.requests > 0; }).map(function (d) {
      return d.date.toLocaleDateString(undefined, { month: "short", day: "numeric" }) + ": " + d.stats.requests;
    });

    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, height: H, role: "img",
      "aria-label": "Requests per day for the last " + DAYS_SHOWN + " days. " + (busy.length ? busy.join(", ") : "No requests.") });
    [0, 0.5, 1].forEach(function (f) {
      var y = top + plotH - f * plotH;
      svg.appendChild(svgEl("line", { x1: left, x2: W - right, y1: y, y2: y, stroke: f === 0 ? "var(--baseline)" : "var(--grid)", "stroke-width": 1 }));
      svg.appendChild(svgEl("text", { x: left - 8, y: y + 4, "text-anchor": "end" }, compact(max * f)));
    });
    days.forEach(function (d, i) {
      var x = left + i * slot + (slot - bw) / 2;
      var h = d.stats.requests / max * plotH;
      var hit = svgEl("rect", { class: "hit", x: left + i * slot, y: top, width: slot, height: plotH });
      var bar = svgEl("path", { class: "chart-bar", d: barPath(x, top + plotH - h, bw, h, false) });
      hover(hit, bar, d.date.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" }), [
        full(d.stats.requests) + " requests",
        full(d.stats.output_tokens) + " tokens written",
        d.stats.errors ? full(d.stats.errors) + " errors" : "No errors"
      ]);
      svg.appendChild(hit);
      svg.appendChild(bar);
      if (i === 0 || i === days.length - 1 || i === Math.floor(days.length / 2)) {
        svg.appendChild(svgEl("text", { x: x + bw / 2, y: H - 4, "text-anchor": i === 0 ? "start" : i === days.length - 1 ? "end" : "middle" },
          i === days.length - 1 ? "Today" : d.date.toLocaleDateString(undefined, { month: "short", day: "numeric" })));
      }
    });
    box.replaceChildren(svg);
  }

  // Horizontal bars, one per model, with the value at the tip.
  function drawModelBars(boxId, title, models, valueOf, labelOf, tipLines, emptyText) {
    var box = $(boxId);
    var rows = models.filter(function (m) { return valueOf(m) > 0; }).sort(function (a, b) { return valueOf(b) - valueOf(a); });
    if (!rows.length) { box.replaceChildren(el("p", { class: "hint" }, emptyText)); return; }

    var W = Math.max(box.clientWidth, 260), rowH = 34, bh = 18;
    var nameW = Math.min(170, Math.max(90, W * 0.36)), valueW = 76;
    var plotW = W - nameW - valueW, H = rows.length * rowH;
    var max = Math.max.apply(null, rows.map(valueOf));
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, height: H, role: "img",
      "aria-label": title + ": " + rows.map(function (m) { return m.model + " " + labelOf(m); }).join(", ") });
    svg.appendChild(svgEl("line", { x1: nameW, x2: nameW, y1: 0, y2: H, stroke: "var(--baseline)", "stroke-width": 1 }));
    rows.forEach(function (m, i) {
      var y = i * rowH, cy = y + rowH / 2;
      var w = Math.max(2, valueOf(m) / max * plotW);
      var hit = svgEl("rect", { class: "hit", x: 0, y: y, width: W, height: rowH });
      var bar = svgEl("path", { class: "chart-bar", d: barPath(nameW, cy - bh / 2, w, bh, true) });
      hover(hit, bar, m.model, tipLines(m));
      svg.appendChild(hit);
      svg.appendChild(svgEl("text", { class: "name", x: 0, y: cy + 4 }, truncate(m.model, 22)));
      svg.appendChild(bar);
      svg.appendChild(svgEl("text", { class: "val", x: nameW + w + 8, y: cy + 4 }, labelOf(m)));
    });
    box.replaceChildren(svg);
  }

  var TIME_PARTS = [
    { key: "load_duration", name: "Loading model", cls: "seg-1" },
    { key: "prompt_duration", name: "Reading prompt", cls: "seg-2" },
    { key: "output_duration", name: "Writing answer", cls: "seg-3" }
  ];

  // Stacked bars: loading, reading, writing. A 2px gap separates segments.
  function drawTimeChart(models) {
    var box = $("chart-time");
    var rows = models.filter(function (m) { return avgTime(m) > 0; }).sort(function (a, b) { return avgTime(b) - avgTime(a); });
    if (!rows.length) { box.replaceChildren(el("p", { class: "hint" }, "No finished requests yet.")); return; }

    var W = Math.max(box.clientWidth, 260), rowH = 34, bh = 18, gap = 2;
    var nameW = Math.min(170, Math.max(90, W * 0.3)), valueW = 60;
    var plotW = W - nameW - valueW, H = rows.length * rowH;
    var max = Math.max.apply(null, rows.map(function (m) {
      return TIME_PARTS.reduce(function (sum, p) { return sum + avgOf(m, m[p.key]); }, 0);
    }));
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, height: H, role: "img",
      "aria-label": "Average time per request by model, split into loading, reading, and writing. Values are in the table below." });
    svg.appendChild(svgEl("line", { x1: nameW, x2: nameW, y1: 0, y2: H, stroke: "var(--baseline)", "stroke-width": 1 }));
    rows.forEach(function (m, i) {
      var y = i * rowH, cy = y + rowH / 2, x = nameW, sum = 0;
      svg.appendChild(svgEl("text", { class: "name", x: 0, y: cy + 4 }, truncate(m.model, 22)));
      var visible = TIME_PARTS.filter(function (p) { return avgOf(m, m[p.key]) > 0; });
      visible.forEach(function (p, j) {
        var v = avgOf(m, m[p.key]), isLast = j === visible.length - 1;
        sum += v;
        var w = Math.max(1, v / max * plotW - (isLast ? 0 : gap));
        var hit = svgEl("rect", { class: "hit", x: x, y: y, width: w + gap, height: rowH });
        var seg = isLast
          ? svgEl("path", { class: "seg " + p.cls, d: barPath(x, cy - bh / 2, w, bh, true) })
          : svgEl("rect", { class: "seg " + p.cls, x: x, y: cy - bh / 2, width: w, height: bh });
        hover(hit, seg, m.model, [p.name + ": " + dur(v) + " average"]);
        svg.appendChild(hit);
        svg.appendChild(seg);
        x += w + (isLast ? 0 : gap);
      });
      svg.appendChild(svgEl("text", { class: "val", x: x + 8, y: cy + 4 }, dur(sum)));
    });
    box.replaceChildren(svg);
  }

  // ---------- tables and notes ----------

  function drawTimeTable(models) {
    fillTable($("time-table"), ["Model", "Avg loading model", "Avg reading prompt", "Avg writing answer", "Prompt reused", "Processor"],
      models.filter(function (m) { return ok(m) > 0; }).map(function (m) {
        return row([
          { text: m.model },
          { text: dur(avgOf(m, m.load_duration)) },
          { text: dur(avgOf(m, m.prompt_duration)) },
          { text: dur(avgOf(m, m.output_duration)) },
          { text: pct(reused(m.prompt_cached_tokens, m.prompt_tokens)) },
          { text: processor(m) }
        ]);
      }));
  }

  function drawCost(models, label) {
    var tin = 0, tout = 0;
    models.forEach(function (m) { tin += m.prompt_tokens; tout += m.output_tokens; });
    $("cost-value").textContent = money(tin / 1e6 * prices.input + tout / 1e6 * prices.output);
    $("cost-note").textContent = "Estimate for " + full(tin) + " tokens in and " + full(tout) + " tokens out " + label +
      ". Running locally costs nothing per token. Prices are examples; enter the ones for the service you'd compare against.";
  }

  function contextText(m) {
    var win = fillWindow(m);
    if (!m.max_context_used) return "–";
    return compact(m.max_context_used) + (win ? " / " + ctxSize(win) + " (" + Math.round(m.max_context_used / win * 100) + "%)" : "");
  }

  function drawModelsTable(models) {
    fillTable($("models-table"),
      ["Model", "Requests", "Errors", "Avg first word", "Avg speed", "Avg time", "Total tokens in", "Newly read", "Reused", "Total tokens out", "Max context", "Memory needed", "Last used"],
      models.map(function (m) {
        return row([
          { text: m.model },
          { text: full(m.requests) },
          { text: full(m.errors) },
          { text: dur(avgWait(m)) },
          { text: speedText(m.output_tokens, m.output_duration) },
          { text: dur(avgTime(m)) },
          { text: compact(m.prompt_tokens) },
          { text: compact(newRead(m.prompt_tokens, m.prompt_cached_tokens)) },
          { text: compact(m.prompt_cached_tokens) },
          { text: compact(m.output_tokens) },
          { text: contextText(m) },
          { text: bytes(m.memory_size) },
          { text: ago(m.last_used) }
        ]);
      }));
  }

  function note(title, text) {
    var li = el("li");
    li.appendChild(el("strong", null, title + ": "));
    li.appendChild(document.createTextNode(text));
    return li;
  }

  function drawNotes(models) {
    var notes = $("notes");
    notes.replaceChildren();
    models.forEach(function (m) {
      if (m.memory_vram != null && m.memory_size > 0 && m.memory_vram < m.memory_size) {
        var cpu = Math.round((1 - m.memory_vram / m.memory_size) * 100);
        notes.appendChild(note("Running partly on the CPU", m.model + " is " + cpu + "% on the CPU, which is much slower than the GPU. " +
          "A smaller model or a smaller context size may fit fully on the GPU."));
      }
      var win = fillWindow(m);
      if (win > 0 && m.max_context_used >= CONTEXT_WARN_AT * win) {
        notes.appendChild(note("Near context limit", m.model + " used " + compact(m.max_context_used) + " of its " +
          ctxSize(win) + " context window. In long chats, older messages may get cut off."));
      }
    });
  }

  function drawLastTable(models) {
    var withLast = models.filter(function (m) { return m.last; });
    $("last-card").hidden = !withLast.length;
    fillTable($("last-table"), ["Model", "When", "First word", "Speed", "Time", "Tokens in / out", "Context"],
      withLast.map(function (m) {
        var l = m.last, n = ok(m);
        return row([
          { text: m.model },
          { text: ago(l.at) },
          { text: dur(waitOf(l.total_duration, l.output_duration)), avg: dur(avgWait(m)) },
          { text: speedText(l.output_tokens, l.output_duration), avg: tps(m) ? tps(m).toFixed(1) : "–" },
          { text: dur(l.total_duration), avg: dur(avgTime(m)) },
          { text: compact(l.prompt_tokens) + " / " + compact(l.output_tokens),
            avg: n ? compact(m.prompt_tokens / n) + " / " + compact(m.output_tokens / n) : "–" },
          { text: compact(l.prompt_tokens + l.output_tokens) + " (max " + compact(m.max_context_used) + ")" }
        ]);
      }));
  }

  // The tag list is only redrawn when it changes, so a keyboard user tabbing
  // through it doesn't lose their place every refresh.
  var tagsDrawn = "";
  function drawTags(st) {
    var all = st.tags || [];
    var total = Math.max(st.tags_total || 0, all.length);
    var q = $("tag-search").value.trim().toLowerCase();
    var matches = q ? all.filter(function (t) { return t.tag.toLowerCase().indexOf(q) !== -1; }) : all;
    var shown = matches.slice(0, TAGS_SHOWN);
    var key = JSON.stringify([q, total, shown.map(function (t) { return [t.tag, t.requests, t.errors, ago(t.last_used)]; })]);
    if (key === tagsDrawn) return;
    tagsDrawn = key;

    fillTable($("tags-table"), ["Tag", "Requests", "Errors", "Last used"], shown.map(function (tg) {
      var tr = row([{ text: "" }, { text: full(tg.requests) }, { text: full(tg.errors) }, { text: ago(tg.last_used) }]);
      var btn = el("button", { type: "button", class: "tag-link" }, tg.tag);
      btn.addEventListener("click", function () { setTag(tg.tag); });
      tr.firstChild.appendChild(btn);
      return tr;
    }));

    var more = $("tags-more");
    if (!all.length) more.textContent = "No tags yet.";
    else if (q && !matches.length) more.textContent = "No recent tags match. Press Enter to open \"" + $("tag-search").value.trim() + "\" anyway.";
    else if (q) more.textContent = matches.length > TAGS_SHOWN ? "Showing " + TAGS_SHOWN + " of " + matches.length + " matches. Keep typing to narrow it down." : "";
    else more.textContent = total > TAGS_SHOWN ? "Showing the " + TAGS_SHOWN + " most recent of " + full(total) + " tags. Search to find others." : "";
  }

  function drawLogTable(entries) {
    fillTable($("log-table"), ["Time", "Model", "First word", "Speed", "Time taken", "Load / read / write", "Total in", "Newly read", "Reused", "Out", "Status"],
      entries.map(function (e) {
        var tr = row([
          { text: when(e.at, true) },
          { text: e.model },
          { text: e.error ? "–" : dur(waitOf(e.total_duration, e.output_duration)) },
          { text: e.error ? "–" : speedText(e.output_tokens, e.output_duration) },
          { text: e.error ? "–" : dur(e.total_duration) },
          { text: e.error ? "–" : dur(e.load_duration) + " / " + dur(e.prompt_duration) + " / " + dur(e.output_duration) },
          { text: compact(e.prompt_tokens) },
          { text: compact(newRead(e.prompt_tokens, e.prompt_cached_tokens)) },
          { text: compact(e.prompt_cached_tokens) },
          { text: compact(e.output_tokens) },
          { text: e.error ? "Error" : "OK" }
        ]);
        if (e.error) tr.lastChild.className = "status-error";
        return tr;
      }));
  }

  // ---------- rendering ----------

  function setStatus(text, isError) {
    $("live-text").textContent = text;
    $("live-text").classList.toggle("error", !!isError);
    if (isError) $("live-dot").classList.remove("on");
  }

  function render(st) {
    $("stats-off").hidden = !st.disabled;
    $("content").hidden = !!st.disabled;
    if (st.disabled) { setStatus("Stats are off"); return; }

    var tagged = !!tag;
    $("tag-banner").hidden = !tagged;
    $("tags-card").hidden = tagged;
    $("range-control").hidden = tagged;
    $("days-card").hidden = tagged;
    $("log-card").hidden = !tagged;
    $("last-card").hidden = tagged;
    if (!tagged) drawTags(st);

    var models, label;
    if (tagged) {
      if (!tagLog || tagLog.tag !== tag) return;
      models = tagLog.models;
      label = "with this tag";
      $("tag-name").textContent = tag;
      $("tag-count").textContent = full(tagLog.total) + (tagLog.total === 1 ? " request" : " requests");
      $("log-hint").textContent = tagLog.total > tagLog.requests.length
        ? "Newest " + tagLog.requests.length + " of " + full(tagLog.total) + ", newest first"
        : "Newest first. Numbers only, never prompts or answers.";
      drawLogTable(tagLog.requests);
    } else {
      models = range === "today" ? st.today_models : st.all_time_models;
      label = range === "today" ? "today" : "all time";
    }

    $("since").textContent = st.since ? "Tracking since " + new Date(st.since).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" }) : "";
    $("live-dot").classList.toggle("on", st.active > 0);
    setStatus(st.active === 1 ? "1 request running" : st.active > 1 ? st.active + " requests running" : "Idle");

    var tot = { requests: 0, errors: 0, tokens: 0, wait: 0, ok: 0 };
    models.forEach(function (m) {
      tot.requests += m.requests; tot.errors += m.errors; tot.tokens += m.output_tokens;
      tot.ok += ok(m); tot.wait += waitOf(m.total_duration, m.output_duration);
    });
    $("t-requests").textContent = full(tot.requests);
    $("t-requests-note").textContent = models.length + (models.length === 1 ? " model " : " models ") + label;
    $("t-tokens").textContent = compact(tot.tokens);
    $("t-tokens-note").textContent = tot.tokens >= 1000 ? full(tot.tokens) + " exactly" : "Across all models";
    $("t-wait").textContent = tot.ok ? dur(tot.wait / tot.ok) : "–";
    $("t-errors").textContent = full(tot.errors);
    $("t-errors-note").textContent = tot.requests ? (tot.errors / tot.requests * 100).toFixed(1) + "% of requests" : "";

    var empty = "No requests " + label + " yet.";
    if (!tagged) drawDays(st);
    drawModelBars("chart-speed", "Average speed by model", models, tps, function (m) { return tps(m).toFixed(1); },
      function (m) { return [tps(m).toFixed(1) + " tokens per second", full(m.output_tokens) + " tokens written"]; }, empty);
    drawModelBars("chart-wait", "Average wait before first word", models, avgWait, function (m) { return dur(avgWait(m)); },
      function (m) { return [dur(avgWait(m)) + " average wait", full(m.requests) + " requests"]; }, empty);
    drawTimeChart(models);
    drawTimeTable(models);
    drawCost(models, label);

    $("models-hint").hidden = !!models.length;
    $("models-hint").textContent = models.length ? "" :
      tagged ? "No requests with this tag yet. Tag one with: ollama run llama3.2 \"Hi\" --tag " + tag :
      empty + " Chat with a model, for example: ollama run llama3.2";
    drawModelsTable(models);
    drawNotes(tagged ? models : st.all_time_models);
    if (!tagged) drawLastTable(st.all_time_models);
  }

  function setRange(r) {
    range = r;
    try { localStorage.setItem("ollama-stats-range", r); } catch (e) { /* storage blocked */ }
    document.querySelectorAll("#range-control button").forEach(function (b) {
      b.setAttribute("aria-pressed", String(b.dataset.range === r));
    });
    if (last) render(last);
  }

  // ---------- refreshing ----------

  // One refresh at a time: the next starts only after the last one ends,
  // each gives up after a timeout, and refreshing pauses while the tab is
  // hidden. A refresh started for an older tag is ignored.
  var refreshTimer = null, refreshing = null;
  function refresh() {
    clearTimeout(refreshTimer);
    if (refreshing) refreshing.abort();
    var ctl = new AbortController();
    refreshing = ctl;
    var timeout = setTimeout(function () { ctl.abort(); }, REQUEST_TIMEOUT_MS);
    var want = tag;
    Promise.all([
      getJSON("/api/stats", ctl.signal),
      want ? getJSON("/api/stats/log?limit=200&tag=" + encodeURIComponent(want), ctl.signal) : null
    ]).then(function (res) {
      if (want !== tag) return;
      last = res[0];
      tagLog = res[1];
      try {
        render(last);
      } catch (e) {
        console.error(e);
        setStatus("Something went wrong showing the stats. Reload the page to try again.", true);
      }
    }, function (e) {
      if (ctl !== refreshing) return; // replaced by a newer refresh
      if (e.status) setStatus("Ollama returned an error: " + e.message, true);
      else setStatus("Can't reach Ollama. Is it running?", true);
    }).then(function () {
      clearTimeout(timeout);
      if (ctl !== refreshing) return;
      refreshing = null;
      if (!document.hidden) refreshTimer = setTimeout(refresh, REFRESH_MS);
    });
  }

  function setTag(t) {
    if (t === tag) return;
    tag = t;
    tagLog = null;
    tagsDrawn = "";
    if (t) $("tag-search").value = "";
    window.scrollTo(0, 0);
    history.replaceState(null, "", t ? "#tag=" + encodeURIComponent(t) : location.pathname);
    refresh();
  }

  function onPriceInput() {
    var inV = $("price-in").value.trim(), outV = $("price-out").value.trim();
    var a = Number(inV), b = Number(outV);
    var badIn = inV === "" || !isFinite(a) || a < 0, badOut = outV === "" || !isFinite(b) || b < 0;
    $("price-in").setAttribute("aria-invalid", String(badIn));
    $("price-out").setAttribute("aria-invalid", String(badOut));
    if (badIn || badOut) { $("price-error").textContent = "Enter a price of 0 or more for both fields."; return; }
    $("price-error").textContent = "";
    prices = { input: a, output: b };
    try { localStorage.setItem("ollama-stats-prices", JSON.stringify(prices)); } catch (e) { /* storage blocked */ }
    if (last) render(last);
  }

  function initUsage() {
    $("nav-usage").setAttribute("aria-current", "page");
    document.querySelectorAll("#range-control button").forEach(function (b) {
      b.addEventListener("click", function () { setRange(b.dataset.range); });
    });
    $("price-in").value = prices.input;
    $("price-out").value = prices.output;
    $("price-in").addEventListener("input", onPriceInput);
    $("price-out").addEventListener("input", onPriceInput);
    $("tag-search").addEventListener("input", function () {
      $("tag-error").textContent = "";
      $("tag-search").removeAttribute("aria-invalid");
      if (last) drawTags(last);
    });
    $("tag-form").addEventListener("submit", function (e) {
      e.preventDefault();
      var q = $("tag-search").value.trim();
      if (!q) return;
      if (!Grading.isValidTag(q)) {
        $("tag-search").setAttribute("aria-invalid", "true");
        $("tag-error").textContent = "Tags can only use letters, numbers, spaces, and - _ . : / @ (up to 64 characters).";
        return;
      }
      var all = (last && last.tags) || [];
      var exact = all.filter(function (t) { return t.tag === q; })[0];
      var matches = all.filter(function (t) { return t.tag.toLowerCase().indexOf(q.toLowerCase()) !== -1; });
      setTag(exact ? exact.tag : matches.length === 1 ? matches[0].tag : q);
    });
    $("tag-clear").addEventListener("click", function () { setTag(""); });
    window.addEventListener("hashchange", function () { setTag(tagFromHash()); });
    window.addEventListener("resize", function () { if (last) render(last); });
    document.addEventListener("visibilitychange", function () { if (!document.hidden) refresh(); });

    setRange(range);
    refresh();
  }

  if (!/\/stats\/tests\/?$/.test(location.pathname)) initUsage();
})();
