// The dashboard's Tests page (/stats/tests): runs a test suite on the
// models you pick, grades fact checks with stats_grading.js, and saves the
// results through /api/stats/suite-runs.
(function () {
  "use strict";

  var S = window.OllamaStats;
  var compact = S.compact;
  var full = S.full;
  var truncate = S.truncate;
  var dur = S.dur;
  var speedOf = S.speedOf;
  var bytes = S.bytes;
  var when = S.when;
  var pct = S.pct;
  var reused = S.reused;
  var $ = S.$;
  var el = S.el;
  var row = S.row;
  var fillTable = S.fillTable;
  var wrapTable = S.wrapTable;
  var readJSON = S.readJSON;
  var getJSON = S.getJSON;
  var Grading = window.OllamaGrading;


  var T = { suites: [], suite: null, models: [], runs: [], running: false, abort: null, shownRun: null, tab: "perf" };

  // Result kinds saved with each run. "acc" is the earlier name for fact
  // checks, still read so older runs load.
  function isFact(r) { return r.kind === "fact" || r.kind === "acc"; }
  // Older runs saved out of room as this text instead of a flag.
  function outOfRoom(r) { return !r.pass && (r.out_of_room || r.got === "(ran out of room before answering)"); }
  function runHas(run, section) {
    return run.sections.indexOf(section) !== -1 || (section === "fact_checks" && run.sections.indexOf("accuracy") !== -1);
  }

  // ---------- prompts for performance scenarios ----------

  var FILLER = [
    "The town sits at the bend of a slow river, where the water turns green in late summer.",
    "A stone bridge crosses the river near the old mill, which now sells bread on weekends.",
    "Most streets are narrow, and the houses are painted in faded yellows, blues, and reds.",
    "On market days, farmers bring apples, cheese, honey, and baskets of fresh herbs.",
    "The library opens at nine and keeps a shelf of local maps going back two hundred years.",
    "A bell in the square rings every hour, and children race to count the chimes.",
    "In winter, fog settles over the fields and the lamps along the river glow orange.",
    "The bakery's best seller is a small walnut loaf that usually sells out before noon.",
    "Cyclists follow the river path north for twelve kilometers to reach the lake.",
    "An old clock repair shop still fixes watches by hand, one tiny gear at a time."
  ];
  function runCode() { return String(1000 + Math.floor(Math.random() * 9000)); }
  function fillerText(n) {
    var out = [], count = 0, i = 0;
    while (count < n) { var line = FILLER[i % FILLER.length]; out.push(line); count += line.split(" ").length; i++; }
    return out.join(" ");
  }
  var REASONING = "A train leaves City A at 9:00 AM traveling at 60 km/h toward City B, which is 300 km away. " +
    "A second train leaves City B at 10:00 AM traveling toward City A at 90 km/h. At what time do the trains meet, " +
    "and how far from City A? Show your work.";

  // A unique code at the start of every message stops Ollama from reusing
  // earlier work, so each scenario measures reading from scratch.
  function scenarioMessages(sc, code) {
    var p = "[" + code + "] ";
    if (sc.input === "reasoning") p += REASONING;
    else if (sc.input_words) p += "Here are some notes about a town.\n\n" + fillerText(sc.input_words) + "\n\nReply with only the word OK.";
    else p += "Reply with only the word OK.";
    return [{ role: "user", content: p }];
  }

  // A chat of filler notes, used to fill the context window to a set size.
  function contextChat(code, pairs) {
    var m = [
      { role: "user", content: "[" + code + "] I'm going to send you some notes." },
      { role: "assistant", content: "Okay, send them." }
    ];
    for (var i = 1; i <= pairs; i++) {
      m.push({ role: "user", content: "[" + code + "] Note " + i + ": " + FILLER[i % FILLER.length] });
      m.push({ role: "assistant", content: "Noted, note " + i + " saved." });
    }
    m.push({ role: "user", content: "Reply with only the word OK." });
    return m;
  }

  // ---------- sending test requests ----------

  // Every model request from the Tests page carries this header, so Ollama
  // logs it but doesn't count it as usage.
  var TEST_HEADERS = { "Content-Type": "application/json", "X-Ollama-Stats": "test" };
  var SAMPLING = { temperature: 0, seed: 1 };

  function postJSON(url, body, signal, headers) {
    return fetch(url, { method: "POST", headers: headers || { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: signal })
      .then(readJSON);
  }
  // timed sends a model request and notes how long it took from here, in
  // nanoseconds. Cloud models don't report their own reading and writing
  // times, so this is the time they're judged by.
  function timed(url, body, signal) {
    var start = performance.now();
    return postJSON(url, body, signal, TEST_HEADERS).then(function (r) {
      r.elapsed = Math.round((performance.now() - start) * 1e6);
      return r;
    });
  }
  function chat(model, messages, opts, signal) {
    var body = { model: model, messages: messages, stream: false, options: Object.assign({}, SAMPLING, opts.options) };
    if (opts.tag) body.tag = opts.tag;
    if (opts.think !== undefined) body.think = opts.think;
    if (opts.format) body.format = opts.format;
    return timed("/api/chat", body, signal);
  }
  // Output scenarios continue a plain number list with no chat template, so
  // every model keeps writing until the token cap instead of stopping early.
  function generateRaw(model, prompt, opts, signal) {
    var body = { model: model, prompt: prompt, raw: true, stream: false, options: Object.assign({}, SAMPLING, opts.options) };
    if (opts.tag) body.tag = opts.tag;
    return timed("/api/generate", body, signal);
  }

  function stopped() { return new DOMException("Stopped", "AbortError"); }
  function sleep(ms, signal) {
    return new Promise(function (resolve, reject) {
      if (signal.aborted) { reject(stopped()); return; }
      function onAbort() { clearTimeout(t); reject(stopped()); }
      var t = setTimeout(function () { signal.removeEventListener("abort", onAbort); resolve(); }, ms);
      signal.addEventListener("abort", onAbort, { once: true });
    });
  }

  // Unloads a model and waits until it's out of memory, so a cold start
  // really starts cold.
  function unload(model, signal) {
    return postJSON("/api/generate", { model: model, keep_alive: 0 }, signal, TEST_HEADERS).then(function () {
      var tries = 0;
      function check() {
        return getJSON("/api/ps", signal).then(function (ps) {
          var still = (ps.models || []).some(function (m) { return m.name === model || m.model === model; });
          if (!still) return;
          if (++tries > 30) throw new Error("the model didn't unload, so a cold start couldn't be measured");
          return sleep(500, signal).then(check);
        });
      }
      return check();
    });
  }

  function metricsOf(r) {
    return {
      load: r.load_duration || 0, read: r.prompt_eval_duration || 0, write: r.eval_duration || 0, total: r.total_duration || 0,
      elapsed: r.elapsed || 0,
      tokens_in: r.prompt_eval_count || 0, tokens_cached: r.prompt_eval_cached_count || 0, tokens_out: r.eval_count || 0
    };
  }
  function average(list) {
    var out = {};
    Object.keys(list[0]).forEach(function (k) {
      if (typeof list[0][k] === "number") out[k] = list.reduce(function (s, x) { return s + x[k]; }, 0) / list.length;
    });
    return out;
  }

  // What a performance scenario measures, from how it's set up.
  function metricOf(sc) {
    if (sc.cold) return "load";
    if (sc.window) return "context";
    if (sc.output === "count") return "write";
    if (sc.reuse) return "reuse";
    if (sc.think !== undefined) return "think";
    return "read";
  }

  // Measures how many tokens the start of a context chat and each filler
  // exchange take with this model's tokenizer. Done once per model and
  // window size, then reused by every context scenario and repeat. A failed
  // measurement isn't kept, so the next scenario tries again.
  function fillerSize(model, window, ctx) {
    var key = model + "|" + window;
    if (!ctx.fillerSizes[key]) {
      var calib = { options: { num_predict: 1, num_ctx: window } };
      ctx.fillerSizes[key] = chat(model, contextChat(runCode(), 0), calib, ctx.signal).then(function (a) {
        return chat(model, contextChat(runCode(), 10), calib, ctx.signal).then(function (b) {
          if (!(a.prompt_eval_count > 0) || !(b.prompt_eval_count > a.prompt_eval_count)) {
            throw new Error("couldn't measure the model's token counts");
          }
          return { base: a.prompt_eval_count, per: (b.prompt_eval_count - a.prompt_eval_count) / 10 };
        });
      });
      ctx.fillerSizes[key].catch(function () { delete ctx.fillerSizes[key]; });
    }
    return ctx.fillerSizes[key];
  }

  function runPerformance(model, info, sc, ctx) {
    var signal = ctx.signal;
    if (sc.thinking_only && !info.thinking) return Promise.resolve({ skipped: "Not a thinking model" });
    // A cloud model can't be unloaded or given a smaller window.
    if (info.cloud && (sc.cold || sc.window)) return Promise.resolve({ skipped: "Doesn't apply to cloud models" });
    var opts = { tag: ctx.tag, think: sc.think, options: { num_predict: sc.output_tokens } };
    if (sc.window) opts.options.num_ctx = sc.window;

    // Load the model with the right window first, so loading time doesn't
    // leak into reading or writing (except for the cold start).
    function ready() {
      if (info.cloud) return Promise.resolve();
      if (sc.cold) return unload(model, signal).then(function () { ctx.loadedWindow[model] = 0; });
      var want = sc.window || 0;
      if (ctx.loadedWindow[model] === want) return Promise.resolve();
      return chat(model, [{ role: "user", content: "Hi" }], { options: { num_predict: 1, num_ctx: sc.window || undefined } }, signal)
        .then(function () { ctx.loadedWindow[model] = want; });
    }

    function once() {
      var code = runCode();
      if (sc.window) {
        return fillerSize(model, sc.window, ctx).then(function (size) {
          var pairs = Math.max(0, Math.round((sc.window * sc.fill - size.base) / size.per));
          return chat(model, contextChat(code, pairs), opts, signal).then(function (r) {
            var m = metricsOf(r);
            m.sent = Math.round(size.base + pairs * size.per);
            return m;
          });
        });
      }
      if (sc.output === "count") {
        var list = "Run " + code + ".\n";
        for (var k = 1; k <= 5; k++) list += k + "\n";
        return generateRaw(model, list, opts, signal).then(metricsOf);
      }
      var msgs = scenarioMessages(sc, code);
      var first = sc.reuse ? chat(model, msgs, { options: opts.options }, signal) : Promise.resolve();
      return first.then(function () { return chat(model, msgs, opts, signal); }).then(metricsOf);
    }

    var results = [];
    function loop(i) {
      if (i >= ctx.repeats) return Promise.resolve(Object.assign({ runs: ctx.repeats, cloud: !!info.cloud }, average(results)));
      return ready().then(once).then(function (m) { results.push(m); return loop(i + 1); });
    }
    return loop(0);
  }

  function runFactCheck(model, test, ctx) {
    var opts = { tag: ctx.tag, options: { num_predict: Grading.FACT_CHECK_TOKENS } };
    var prompt = test.prompt;
    if (test.answer) {
      opts.format = Grading.answerSchema(test.answer);
      prompt = Grading.answerPrompt(test.prompt, test.answer);
    }
    return chat(model, [{ role: "user", content: prompt }], opts, ctx.signal).then(function (r) {
      var text = Grading.cleanAnswer(r.message && r.message.content);
      var out = { tokens_out: r.eval_count || 0, total: r.total_duration || 0 };
      // Used the whole token cap: usually a thinking model that kept
      // thinking and never wrote its answer.
      var usedAll = out.tokens_out >= Grading.FACT_CHECK_TOKENS;
      if (test.answer) {
        // Known answers only keep the answer itself.
        var g = Grading.gradeReply(test.answer, text);
        out.pass = g.pass;
        out.got = g.got != null ? truncate(g.got, 80) : usedAll ? null : "(no usable answer)";
        out.out_of_room = g.got == null && usedAll;
      } else {
        // For free-form tests the reply is the answer (the sentence, the code).
        out.pass = Grading.check(test, text);
        out.got = truncate(text.replace(/\s+/g, " "), 40);
        out.answer = text.slice(0, 800);
        out.out_of_room = !text && usedAll;
      }
      return out;
    });
  }

  function setProgress(done, total, text) {
    var p = total ? Math.round(done / total * 100) : 0;
    $("progress-fill").style.width = p + "%";
    $("progress-bar").setAttribute("aria-valuenow", String(p));
    $("progress-text").textContent = text;
  }

  function startRun() {
    var suite = T.suite;
    if (!suite) { $("run-error").textContent = "No test suite is loaded. Reload the page once Ollama is running."; return; }
    var models = Array.prototype.slice.call(document.querySelectorAll("#model-list input:checked")).map(function (i) { return i.value; });
    var doPerf = $("run-perf").checked && suite.performance.length > 0;
    var doFacts = $("run-facts").checked && suite.fact_checks.length > 0;
    if (!models.length) { $("run-error").textContent = "Pick at least one model."; return; }
    if (!doPerf && !doFacts) { $("run-error").textContent = "Pick Performance, Fact checks, or both."; return; }
    $("run-error").textContent = "";
    try { localStorage.setItem("ollama-tests-models", JSON.stringify(models)); } catch (e) { /* storage blocked */ }

    var now = new Date();
    var stamp = now.getFullYear() + String(now.getMonth() + 1).padStart(2, "0") + String(now.getDate()).padStart(2, "0") +
      "-" + String(now.getHours()).padStart(2, "0") + String(now.getMinutes()).padStart(2, "0") + String(now.getSeconds()).padStart(2, "0");
    var id = "run-" + stamp;
    var ctl = new AbortController();
    var ctx = { tag: suite.name + ":" + id, signal: ctl.signal, repeats: Number($("repeats").value) || 1, loadedWindow: {}, fillerSizes: {} };
    var run = { id: id, suite: suite.name, tag: ctx.tag, started_at: now.toISOString(), models: models,
      sections: [].concat(doFacts ? ["fact_checks"] : [], doPerf ? ["performance"] : []), results: [] };

    var infoOf = {};
    T.models.forEach(function (m) { infoOf[m.name] = m; });
    var steps = [];
    models.forEach(function (model) {
      if (doFacts) suite.fact_checks.forEach(function (t) { steps.push({ kind: "fact", model: model, item: t }); });
      if (doPerf) suite.performance.forEach(function (sc) { steps.push({ kind: "perf", model: model, item: sc }); });
    });

    T.running = true; T.abort = ctl;
    $("run-btn").disabled = true; $("stop-btn").hidden = false; $("progress").hidden = false;

    var i = 0;
    function next() {
      if (i >= steps.length) return Promise.resolve();
      var st = steps[i];
      setProgress(i, steps.length, "Running " + (i + 1) + " of " + steps.length + ": " + st.model + " · " +
        (st.kind === "fact" ? st.item.category + ": " + st.item.id : st.item.title));
      var work = st.kind === "fact" ? runFactCheck(st.model, st.item, ctx) : runPerformance(st.model, infoOf[st.model] || {}, st.item, ctx);
      return work.then(function (res) {
        run.results.push(Object.assign({ kind: st.kind, id: st.item.id, model: st.model }, res));
      }, function (err) {
        if (err && err.name === "AbortError") throw err;
        // If Ollama can't be reached at all, stop instead of failing every step.
        if (err && !err.status) throw err;
        run.results.push({ kind: st.kind, id: st.item.id, model: st.model, error: String(err && err.message || err) });
      }).then(function () { i++; return next(); });
    }

    next().catch(function (err) {
      if (err && err.name === "AbortError") run.cancelled = true;
      else { run.cancelled = true; $("run-error").textContent = "The run stopped early: can't reach Ollama. Is it still running?"; }
    }).then(function () {
      run.finished_at = new Date().toISOString();
      setProgress(i, steps.length, run.cancelled ? "Stopped after " + i + " of " + steps.length + "." : "Done. " + steps.length + " steps finished.");
      return postJSON("/api/stats/suite-runs", run).then(function () { return null; }, function (e) {
        // Keep the results on screen even when they can't be saved.
        run.unsaved = true;
        $("run-error").textContent = "These results couldn't be saved (" + e.message + "). They'll be gone when you leave this page.";
        return run;
      });
    }).then(function (unsaved) {
      T.running = false; T.abort = null;
      $("run-btn").disabled = false; $("stop-btn").hidden = true;
      return loadRuns(run.id, unsaved);
    });
  }

  // ---------- results ----------

  function suiteByName(name) { return T.suites.filter(function (s) { return s.name === name; })[0] || null; }

  function cell(main, sub, cls) {
    var td = el("td", cls ? { class: cls } : null);
    td.appendChild(el("span", { class: "main" }, main));
    if (sub) td.appendChild(el("span", { class: "sub" }, sub));
    return td;
  }
  function rateText(tokens, ns) { var v = speedOf(tokens, ns); return v > 0 ? Math.round(v).toLocaleString() + " tok/s" : "–"; }

  // Each cell shows the number its scenario was built to measure. key is
  // what "fastest" is judged by.
  function perfCell(sc, r) {
    if (!r) return { main: "–" };
    if (r.skipped) return { main: "Skipped", sub: r.skipped, cls: "skip" };
    if (r.error) return { main: "Error", sub: truncate(r.error, 60), cls: "fail" };
    if (totalOnly(r)) {
      // Cloud models only give the whole time, measured here, network
      // included. It isn't the same measure as a local model's reading or
      // writing time, so it's never marked fastest.
      var what = metricOf(sc) === "reuse" ? pct(reused(r.tokens_cached, r.tokens_in)) + " reused · " :
        metricOf(sc) === "write" || metricOf(sc) === "think" ? compact(r.tokens_out) + " tokens · " : "";
      return { main: dur(r.elapsed || r.total), sub: what + "total time (cloud)" };
    }
    switch (metricOf(sc)) {
      case "load": return { main: dur(r.load), sub: "total " + dur(r.total), key: r.load };
      case "read": return { main: dur(r.read), sub: compact(r.tokens_in) + " tokens · " + rateText(r.tokens_in - r.tokens_cached, r.read), key: r.read };
      case "write": return { main: dur(r.write), sub: compact(r.tokens_out) + " tokens · " + rateText(r.tokens_out, r.write), key: r.write };
      case "reuse": return { main: dur(r.read), sub: pct(reused(r.tokens_cached, r.tokens_in)) + " reused", key: r.read };
      case "think": return { main: dur(r.total), sub: compact(r.tokens_out) + " tokens out", key: r.total };
      case "context":
        return { main: dur(r.read), sub: "kept " + compact(r.tokens_in) + " of ~" + compact(r.sent) + " tokens" +
          (r.tokens_in < r.sent * 0.97 ? ", oldest dropped" : ""), key: r.read };
    }
  }

  // totalOnly is true for results with no reading or writing time: cloud
  // models. Runs saved before cloud results were flagged are spotted by the
  // missing times.
  function totalOnly(r) { return r.cloud || (!r.load && !r.read && !r.write && r.total > 0); }

  // Gives a results table fixed widths for its first columns; the model
  // columns split what's left evenly.
  function fixColumns(table, widths) {
    table.className = "fixed";
    var cg = el("colgroup");
    widths.forEach(function (w) { cg.appendChild(el("col", { style: "width: " + w })); });
    table.insertBefore(cg, table.firstChild);
  }

  // Indexes results by test id and model, so each table cell is a direct
  // lookup instead of a search through every result.
  function indexResults(results) {
    var byKey = {};
    results.forEach(function (r) { byKey[r.id + "\u0000" + r.model] = r; });
    return function (id, model) { return byKey[id + "\u0000" + model]; };
  }

  function drawPerfResults(run, suite, box) {
    var res = run.results.filter(function (r) { return r.kind === "perf"; });
    if (!res.length) { box.appendChild(el("p", { class: "hint" }, "No performance scenarios finished in this run.")); return; }
    var find = indexResults(res);
    var thinkOn = suite.performance.filter(function (sc) { return sc.think === true; })[0];
    var groups = [];
    suite.performance.forEach(function (sc) { if (groups.indexOf(sc.group) === -1) groups.push(sc.group); });
    groups.forEach(function (g) {
      box.appendChild(el("h3", { class: "group-title" }, g));
      var t = el("table");
      fillTable(t, ["Scenario", "Measures"].concat(run.models), suite.performance.filter(function (sc) { return sc.group === g; }).map(function (sc) {
        var tr = el("tr");
        tr.appendChild(el("td", null, sc.title));
        tr.appendChild(el("td", { class: "prompt-cell" }, sc.isolates));
        var cells = run.models.map(function (m) {
          var r = find(sc.id, m), c = perfCell(sc, r);
          // If "thinking off" wrote about as much as "thinking on", the model
          // kept thinking anyway, so say so instead of showing a speedup.
          if (sc.think === false && thinkOn && r && !r.skipped && !r.error) {
            var on = find(thinkOn.id, m);
            if (on && on.tokens_out && r.tokens_out >= on.tokens_out * 0.8) c.sub = "couldn't turn thinking off for this model";
          }
          return c;
        });
        var keys = cells.map(function (c) { return c.key; }).filter(function (k) { return k > 0; });
        var best = keys.length > 1 ? Math.min.apply(null, keys) : null;
        cells.forEach(function (c) { tr.appendChild(cell(c.main, c.sub, (c.cls || "") + (best !== null && c.key === best ? " best" : ""))); });
        return tr;
      }));
      fixColumns(t, ["16%", "24%"]);
      box.appendChild(wrapTable(t));
    });
    box.appendChild(el("p", { class: "hint", style: "margin-top: 12px" }, "Bold marks the fastest model for each scenario. Times are averages when a scenario ran more than once. Cloud models only report their total time, measured from this computer, so they're shown separately and never marked fastest."));
  }

  // Score color bands for the fact check summary.
  var SCORE_GOOD = 0.8, SCORE_MID = 0.5;

  function drawFactResults(run, suite, box) {
    var res = run.results.filter(isFact);
    if (!res.length) { box.appendChild(el("p", { class: "hint" }, "No fact checks finished in this run.")); return; }
    var find = indexResults(res);
    var cats = [], groupOf = {}, categoryOf = {};
    suite.fact_checks.forEach(function (t) {
      if (cats.indexOf(t.category) === -1) cats.push(t.category);
      groupOf[t.category] = t.group || "";
      categoryOf[t.id] = t.category;
    });
    // Count passes per model and category in one pass over the results.
    var tally = {};
    res.forEach(function (r) {
      if (r.error || !categoryOf[r.id]) return;
      [categoryOf[r.id], ""].forEach(function (cat) {
        var k = r.model + "\u0000" + cat;
        var t = tally[k] || (tally[k] = { passed: 0, total: 0 });
        t.total++;
        if (r.pass) t.passed++;
      });
    });

    box.appendChild(el("h3", { class: "group-title" }, "Score by category"));
    var rows = [], lastGroup = null;
    cats.concat([""]).forEach(function (cat) {
      var grp = cat ? groupOf[cat] : "";
      if (grp && grp !== lastGroup) {
        var gh = el("tr");
        gh.appendChild(el("td", { colspan: String(run.models.length + 1), class: "group-title" }, grp));
        rows.push(gh);
        lastGroup = grp;
      }
      var tr = el("tr");
      tr.appendChild(el("td", null, cat || "All fact checks"));
      var scores = run.models.map(function (m) { return tally[m + "\u0000" + cat] || { passed: 0, total: 0 }; });
      var bestRate = Math.max.apply(null, scores.map(function (s) { return s.total ? s.passed / s.total : -1; }));
      scores.forEach(function (s) {
        var r = s.total ? s.passed / s.total : -1;
        var color = r < 0 ? "" : r >= SCORE_GOOD ? " score-good" : r >= SCORE_MID ? " score-mid" : " score-low";
        tr.appendChild(cell(s.total ? s.passed + " of " + s.total : "–", s.total ? Math.round(r * 100) + "%" : "",
          (run.models.length > 1 && r === bestRate && r > 0 ? "best" : "") + color));
      });
      rows.push(tr);
    });
    var t = el("table");
    fillTable(t, ["Category"].concat(run.models), rows);
    fixColumns(t, ["30%"]);
    box.appendChild(wrapTable(t));
    var legend = el("ul", { class: "score-legend" });
    [["--score-good", "80% or more"], ["--score-mid", "50% to 79%"], ["--score-low", "Under 50%"]].forEach(function (x) {
      var li = el("li");
      li.appendChild(el("i", { style: "background: var(" + x[0] + ")" }));
      li.appendChild(document.createTextNode(x[1]));
      legend.appendChild(li);
    });
    box.appendChild(legend);

    box.appendChild(el("h3", { class: "group-title" }, "Every test"));
    var t2 = el("table");
    fillTable(t2, ["Test", "Level"].concat(run.models), suite.fact_checks.filter(function (test) {
      return run.models.some(function (m) { return find(test.id, m); });
    }).map(function (test) {
      var tr = el("tr");
      var pc = el("td", { class: "prompt-cell" });
      pc.appendChild(el("span", { class: "main" }, test.category + (test.group ? " · " + test.group : "")));
      pc.appendChild(el("span", { class: "sub" }, truncate(test.prompt, 120)));
      tr.appendChild(pc);
      tr.appendChild(el("td", null, test.level));
      run.models.forEach(function (m) {
        var r = find(test.id, m), td = el("td");
        if (!r) {
          td.textContent = "–";
        } else if (r.error) {
          td.appendChild(el("span", { class: "fail" }, "Error"));
          td.appendChild(el("span", { class: "sub" }, truncate(r.error, 60)));
        } else {
          // Pass or Fail by default; click to see their answer. Out of
          // room is a fail too, labeled so it isn't mistaken for a wrong
          // answer.
          var d = el("details");
          var room = outOfRoom(r);
          d.appendChild(el("summary", { class: r.pass ? "pass" : "fail" }, r.pass ? "✓ Pass" : room ? "✗ Out of room" : "✗ Fail"));
          if (room) {
            d.appendChild(el("p", null, "Used all " + full(r.tokens_out) + " tokens before giving an answer, usually by thinking too long."));
            if (test.answer) d.appendChild(el("p", null, "Expected: " + Grading.expected(test.answer)));
          } else if (test.answer) {
            d.appendChild(el("p", null, "Answer: " + (r.got != null ? r.got : "–")));
            if (!r.pass) d.appendChild(el("p", null, "Expected: " + Grading.expected(test.answer)));
          } else {
            d.appendChild(el("p", null, r.answer || "(empty answer)"));
          }
          td.appendChild(d);
        }
        tr.appendChild(td);
      });
      return tr;
    }));
    fixColumns(t2, ["30%", "76px"]);
    box.appendChild(wrapTable(t2));
    box.appendChild(el("p", { class: "hint", style: "margin-top: 12px" }, "Click a result to see the model's answer, and what was expected if it's wrong. Out of room means the model used all its tokens before answering; it counts as a fail. Code is checked for recognizable patterns, not run."));
  }

  // Draws one results tab. If the selected run didn't include that section,
  // it shows the most recent run that did, and says so.
  function drawSection(section, run) {
    var box = $(section === "performance" ? "perf-results" : "facts-results");
    var label = section === "performance" ? "performance tests" : "fact checks";
    box.replaceChildren();
    var shown = runHas(run, section) ? run : T.runs.filter(function (r) { return runHas(r, section); })[0];
    if (!shown) {
      box.appendChild(el("p", { class: "hint" }, "No " + label + " yet. Run tests with " + (section === "performance" ? "Performance" : "Fact checks") + " checked."));
      return;
    }
    if (shown !== run) {
      box.appendChild(el("p", { class: "hint" }, "The selected run didn't include " + label + ", so this shows the latest run that did, from " + when(shown.started_at) + "."));
    }
    var suite = suiteByName(shown.suite);
    if (!suite) {
      box.appendChild(el("p", { class: "hint" }, "This run used the \"" + shown.suite + "\" suite, which isn't installed anymore, so its results can't be shown."));
      return;
    }
    (section === "performance" ? drawPerfResults : drawFactResults)(shown, suite, box);
  }

  function showRun(id) {
    var run = T.runs.filter(function (r) { return r.id === id; })[0];
    $("results-card").hidden = !run;
    if (!run) return;
    T.shownRun = run;
    $("run-select").value = run.id;
    var suite = suiteByName(run.suite);
    var meta = $("run-meta");
    meta.replaceChildren(document.createTextNode((suite ? suite.title : run.suite) + " · " + when(run.started_at) + " · " +
      run.models.length + (run.models.length === 1 ? " model" : " models") +
      (run.cancelled ? " · stopped early" : "") + (run.unsaved ? " · not saved" : "") + " · "));
    meta.appendChild(el("a", { href: "/stats#tag=" + encodeURIComponent(run.tag) }, "Speed details on the Usage page"));

    document.querySelectorAll("#results-tabs button").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.tab === T.tab)); });
    $("perf-results").hidden = T.tab !== "perf";
    $("facts-results").hidden = T.tab !== "facts";
    drawSection("performance", run);
    drawSection("fact_checks", run);
  }

  // Loads saved runs and shows one. An unsaved run (when saving failed) is
  // kept at the top of the list so its results stay on screen.
  function loadRuns(selectID, unsaved) {
    return getJSON("/api/stats/suite-runs").then(function (data) { return data.runs || []; }, function () { return T.runs; })
      .then(function (runs) {
        T.runs = runs.filter(function (r) { return !r.unsaved; });
        if (unsaved) T.runs.unshift(unsaved);
        var sel = $("run-select");
        sel.replaceChildren.apply(sel, T.runs.map(function (r) {
          return el("option", { value: r.id }, when(r.started_at) + " · " + r.models.join(", ") + (r.cancelled ? " (stopped)" : "") + (r.unsaved ? " (not saved)" : ""));
        }));
        showRun(selectID || (T.runs[0] && T.runs[0].id));
      });
  }

  // ---------- setup ----------

  function drawSuite() {
    var s = T.suite;
    $("suite-desc").textContent = s.description || s.title;
    $("perf-count").textContent = s.performance.length + " scenarios";
    $("facts-count").textContent = s.fact_checks.length + " tests";
    fillTable($("about-perf"), ["Group", "Scenario", "Measures", "Output cap"], s.performance.map(function (sc) {
      return row([{ text: sc.group }, { text: sc.title + (sc.thinking_only ? " (thinking models)" : "") }, { text: sc.isolates }, { text: sc.output_tokens + " tokens" }]);
    }));
    fillTable($("about-facts"), ["Group", "Category", "Level", "Graded by", "Prompt"], s.fact_checks.map(function (t) {
      var how = t.answer ? "Exact answer (JSON)" : t.check.type === "code" ? "Code patterns" : "Rule: " + t.check.type.replace(/_/g, " ");
      var tr = row([{ text: t.group }, { text: t.category }, { text: t.level }, { text: how }, { text: truncate(t.prompt, 160) }]);
      tr.lastChild.className = "prompt-cell";
      return tr;
    }));
  }

  function drawModels() {
    var saved = [];
    try { saved = JSON.parse(localStorage.getItem("ollama-tests-models") || "[]"); } catch (e) { /* storage blocked */ }
    var box = $("model-list");
    box.replaceChildren();
    if (!T.models.length) { box.appendChild(el("p", { class: "hint" }, "No chat models installed. Download one with: ollama pull llama3.2")); return; }
    T.models.forEach(function (m) {
      var lab = el("label", { class: "check" });
      var input = el("input", { type: "checkbox", value: m.name });
      if (saved.indexOf(m.name) !== -1) input.checked = true;
      lab.appendChild(input);
      lab.appendChild(document.createTextNode(m.name));
      lab.appendChild(el("span", { class: "meta" }, m.cloud ? "cloud" : bytes(m.size)));
      if (m.thinking) lab.appendChild(el("span", { class: "badge" }, "thinking"));
      box.appendChild(lab);
    });
  }

  function initTests() {
    $("usage-view").hidden = true;
    $("tests-view").hidden = false;
    $("nav-tests").setAttribute("aria-current", "page");
    document.title = "Ollama Tests";
    $("run-btn").disabled = true;

    $("run-btn").addEventListener("click", startRun);
    $("stop-btn").addEventListener("click", function () { if (T.abort) T.abort.abort(); });
    $("run-select").addEventListener("change", function (e) { showRun(e.target.value); });
    $("suite-select").addEventListener("change", function (e) { T.suite = suiteByName(e.target.value) || T.suite; drawSuite(); });
    document.querySelectorAll("#results-tabs button").forEach(function (b) {
      b.addEventListener("click", function () { T.tab = b.dataset.tab; if (T.shownRun) showRun(T.shownRun.id); });
    });
    window.addEventListener("beforeunload", function (e) { if (T.running) { e.preventDefault(); e.returnValue = ""; } });

    Promise.all([getJSON("/api/stats/suites"), getJSON("/api/tags")]).then(function (res) {
      T.suites = (res[0].suites || []).map(function (s) {
        s.performance = s.performance || [];
        s.fact_checks = s.fact_checks || [];
        return s;
      });
      T.suite = T.suites[0] || null;
      if (!T.suite) { $("suite-desc").textContent = "No test suites found."; return; }
      var sel = $("suite-select");
      sel.replaceChildren.apply(sel, T.suites.map(function (s) { return el("option", { value: s.name }, s.title + (s.built_in ? " (built in)" : "")); }));
      $("suite-field").hidden = T.suites.length < 2;
      if ((res[0].problems || []).length) {
        $("run-error").textContent = "Some suite files couldn't be used: " +
          res[0].problems.map(function (p) { return p.file + " (" + p.error + ")"; }).join("; ");
      }
      T.models = (res[1].models || []).filter(function (m) {
        return !m.capabilities || m.capabilities.indexOf("completion") !== -1;
      }).map(function (m) {
        return { name: m.name, size: m.size, cloud: !!m.remote_host, thinking: (m.capabilities || []).indexOf("thinking") !== -1 };
      });
      drawSuite();
      drawModels();
      $("run-btn").disabled = false;
      return loadRuns();
    }).catch(function (e) {
      $("suite-desc").textContent = "Can't load the tests. Is Ollama running? Reload the page to try again.";
      $("run-error").textContent = e.message;
    });
  }

  if (/\/stats\/tests\/?$/.test(location.pathname)) initTests();
})();
