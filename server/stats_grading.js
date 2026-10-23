// Grading for the dashboard's Tests page: the JSON shapes that known-answer
// fact checks must follow, how their answers are compared, and the rules for
// free-form checks. It's a separate file, with no page code in it, so it can
// be tested on its own (see TestStatsGradingScript). Loaded by
// stats_dashboard.html as window.OllamaGrading.
(function (root) {
  "use strict";

  // Text answers are capped so the model can't answer with a sentence.
  var TEXT_ANSWER_MAX = 30;
  // Reasoning is capped too, so a model that starts repeating itself is
  // forced to stop and give an answer.
  var REASONING_MAX = 1000;
  // Thinking models think before writing the JSON, so they need room for both.
  var FACT_CHECK_TOKENS = 4000;

  // ---------- free-form replies ----------

  function cleanAnswer(a) {
    return String(a || "").replace(/<think>[\s\S]*?<\/think>/g, "").trim();
  }
  function words(a) { return a.match(/[\p{L}\p{N}'’-]+/gu) || []; }
  function stripDecor(a) { return a.replace(/^[\s*_`"'>#-]+|[\s*_`"'.!]+$/g, ""); }

  // Rules for tests where the format itself is being tested.
  var CHECKS = {
    exact: function (a, c) { return stripDecor(a).toLowerCase() === String(c.value).toLowerCase(); },
    contains: function (a, c) { return a.toLowerCase().indexOf(String(c.value).toLowerCase()) !== -1; },
    word_count: function (a, c) { return words(a).length === Number(c.value); },
    json_list: function (a, c) {
      var m = a.match(/\{[\s\S]*\}/);
      if (!m) return false;
      try { var v = JSON.parse(m[0])[c.key]; return Array.isArray(v) && v.length === c.length; } catch (e) { return false; }
    },
    uppercase: function (a) { return /\p{Lu}/u.test(a) && !/\p{Ll}/u.test(a); },
    excludes_letter: function (a, c) { return a.length > 0 && a.toLowerCase().indexOf(String(c.value).toLowerCase()) === -1; },
    // Code is checked by looking for recognizable patterns, not by running
    // it: running a model's code inside the dashboard wouldn't be safe.
    code: function (a, c) {
      return (c.all || []).every(function (x) { return a.indexOf(x) !== -1; }) &&
        (!(c.any || []).length || c.any.some(function (x) { return a.indexOf(x) !== -1; }));
    }
  };

  function check(test, reply) {
    var rule = CHECKS[test.check.type];
    return !!(rule && rule(reply, test.check));
  }

  // ---------- known answers: structured output ----------

  // The JSON shape a known-answer reply must follow: reasoning first, so the
  // model can work the problem, then a typed answer that's graded exactly.
  function valueSchema(type, options) {
    if (type === "number") return { type: "number" };
    if (type === "choice") return { type: "string", enum: options };
    return { type: "string", maxLength: TEXT_ANSWER_MAX };
  }
  function answerSchema(ans) {
    var answer;
    if (ans.type === "fields") {
      var props = {}, req = [];
      ans.fields.forEach(function (f) { props[f.name] = valueSchema(f.type); req.push(f.name); });
      answer = { type: "object", properties: props, required: req };
    } else {
      answer = valueSchema(ans.type, ans.options);
    }
    return {
      type: "object",
      properties: { reasoning: { type: "string", maxLength: REASONING_MAX }, answer: answer },
      required: ["reasoning", "answer"]
    };
  }

  // Lowercase, without accents or punctuation, so "São Paulo." matches
  // "sao paulo".
  function normText(v) {
    return String(v == null ? "" : v).normalize("NFD").replace(/\p{M}/gu, "").toLowerCase()
      .replace(/[^\p{L}\p{N}: ]/gu, "").replace(/\s+/g, " ").trim();
  }

  // Reads a time like "4:00 PM", "4:00pm", "16:00", or "4:00". period is
  // false when it could be morning or afternoon.
  function parseTime(v) {
    var m = String(v || "").match(/(\d{1,2}):(\d{2})(?:\s*([ap])\.?\s*m\.?)?/i);
    if (!m) return null;
    var h = parseInt(m[1], 10), min = parseInt(m[2], 10);
    if (h > 23 || min > 59) return null;
    var p = m[3] ? m[3].toLowerCase() : "";
    if (p && h > 12) return null;
    if (p === "p" && h < 12) h += 12;
    if (p === "a" && h === 12) h = 0;
    return { minutes: h * 60 + min, period: !!p || h === 0 || h > 12 };
  }
  function sameTime(got, want) {
    var g = parseTime(got), w = parseTime(want);
    if (!g || !w) return false;
    if (g.period && w.period) return g.minutes === w.minutes;
    // One side didn't say AM or PM, so compare on a 12-hour clock.
    return g.minutes % 720 === w.minutes % 720;
  }

  // Words that turn "Mercury" into "not Mercury".
  var NEGATIONS = ["not", "no", "isnt", "never", "except", "neither"];

  function sameValue(type, got, want) {
    if (type === "number") {
      var n = typeof got === "number" ? got : parseFloat(String(got).replace(/[$,]/g, ""));
      return isFinite(n) && Math.abs(n - Number(want)) < 0.01;
    }
    if (type === "time") return sameTime(got, want);
    var g = normText(got), w = normText(want);
    if (g === w) return true;
    if (type !== "text") return false;
    // Text answers are short, so one passes if it contains the expected words
    // and doesn't negate them: "Planet Mercury" counts, "not Mercury" and
    // "Venus" don't.
    var gw = g.split(" ");
    return (" " + g + " ").indexOf(" " + w + " ") !== -1 &&
      !gw.some(function (x) { return NEGATIONS.indexOf(x) !== -1; });
  }

  function gradeAnswer(ans, got) {
    if (ans.type === "fields") {
      if (!got || typeof got !== "object") return false;
      return ans.fields.every(function (f) { return sameValue(f.type, got[f.name], f.value); });
    }
    return sameValue(ans.type, got, ans.value);
  }

  function showValue(v) {
    if (v && typeof v === "object") {
      return Object.keys(v).map(function (k) { return k.replace(/_/g, " ") + ": " + v[k]; }).join(", ");
    }
    return String(v);
  }
  function expected(ans) {
    if (ans.type === "fields") return ans.fields.map(function (f) { return f.name.replace(/_/g, " ") + ": " + f.value; }).join(", ");
    return String(ans.value);
  }

  // Grades a known-answer reply (the JSON text the model wrote). Returns
  // whether it passed and the answer to show.
  function gradeReply(ans, text) {
    var parsed = null;
    try { parsed = JSON.parse(text); } catch (e) { /* not JSON */ }
    if (!parsed || typeof parsed !== "object" || !("answer" in parsed)) return { pass: false, got: null };
    return { pass: gradeAnswer(ans, parsed.answer), got: showValue(parsed.answer) };
  }

  // ---------- tags ----------

  // The same rule as api.ValidateTag on the server.
  var TAG_PATTERN = /^[\p{L}\p{Nd} \-_.:/@]{1,64}$/u;
  function isValidTag(t) { return TAG_PATTERN.test(t); }

  var api = {
    FACT_CHECK_TOKENS: FACT_CHECK_TOKENS,
    cleanAnswer: cleanAnswer,
    check: check,
    answerSchema: answerSchema,
    gradeAnswer: gradeAnswer,
    gradeReply: gradeReply,
    expected: expected,
    isValidTag: isValidTag
  };
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.OllamaGrading = api;
})(this);
