/* The editor page: a view over the server's state; every number is solved
 * server-side. The DOM is rebuilt only when the set of parameters changes, so
 * a broadcast cannot replace the input being edited.
 */
"use strict";

const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};
const css = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();
const seriesColor = (i) => css(`--series-${(i % 8) + 1}`);

let ws = null;
let state = null;
let signature = "";
const inputs = new Map();      // param key → <input|select>
let motorRows = new Map();     // class → {cells}
let chart = null;
let fitData = null;        // the catalogue + curves, sent once on connect
let fitCanvases = [];      // the four panels, rebuilt with the board

/* ── connection ─────────────────────────────────────────────────────────── */

function connect() {
  ws = new WebSocket(`ws://${location.host}/`);
  ws.onopen = () => send({ type: "hello" });
  ws.onmessage = (e) => onState(JSON.parse(e.data));
  ws.onclose = () => {
    setStatus({ text: "Disconnected — is the server still running?", kind: "error" });
    setTimeout(connect, 1500);
  };
}
const send = (msg) => ws && ws.readyState === 1 && ws.send(JSON.stringify(msg));

/* Coalesce drags to one message per frame. */
const pending = new Map();
let flushQueued = false;
function sendCoalesced(id, msg) {
  pending.set(id, msg);
  if (flushQueued) return;
  flushQueued = true;
  requestAnimationFrame(() => {
    flushQueued = false;
    for (const m of pending.values()) send(m);
    pending.clear();
  });
}

/* ── state → screen ─────────────────────────────────────────────────────── */

function onState(s) {
  state = s;
  if (s.fits && !s.fits.error) fitData = s.fits;
  $("#viser").href = s.viser_url;
  const frame = $("#viewer-frame");
  if (frame.dataset.src !== s.viser_url) {
    frame.dataset.src = s.viser_url;
    if (!$("#viewer").hidden) frame.src = s.viser_url;
  } else if (!$("#viewer").hidden && !frame.src) {
    frame.src = s.viser_url;
  }
  renderRobots(s);
  setStatus(s.status);

  $("#unsupported").hidden = !s.unsupported;
  $("#board").hidden = !!s.unsupported;
  if (s.unsupported) {
    $("#unsupported").textContent = s.unsupported;
    return;
  }

  const sig = JSON.stringify([
    s.robot,
    s.groups.map((g) => [g.title, g.params.map((p) => p.key)]),
    (s.joints || []).map((j) => j.name),
    !!s.simulate,
    s.densityGroup.map((p) => p.key),
    (s.motors.classes || []).map((c) => c.cls),
  ]);
  if (sig !== signature) {
    signature = sig;
    buildBoard(s);
  }
  applyValues(s);
  drawFits(s.design_fits || []);
}

function setStatus(st) {
  const n = $("#status");
  n.textContent = st.text;
  n.dataset.kind = /generating|regenerating/i.test(st.text) ? "busy" : st.kind;
}

function renderRobots(s) {
  const nav = $("#robots");
  if (nav.dataset.list !== s.robots.join(",")) {
    nav.dataset.list = s.robots.join(",");
    nav.replaceChildren(...s.robots.map((r) => {
      const b = el("button", null, r.replace(/_/g, " "));
      b.onclick = () => send({ type: "select_robot", robot: r });
      b.dataset.robot = r;
      return b;
    }));
  }
  for (const b of nav.children) b.ariaPressed = String(b.dataset.robot === s.robot);

  const sel = $("#folder");
  const opts = (s.folders || []).join("\u0000");
  if (sel.dataset.opts !== opts) {
    sel.dataset.opts = opts;
    sel.replaceChildren(...(s.folders || []).map((f) => new Option(f, f)));
  }
  if (s.folder && sel.value !== s.folder) sel.value = s.folder;
}

/* ── structure ──────────────────────────────────────────────────────────── */

function card(title, wide) {
  const node = $("#tpl-card").content.firstElementChild.cloneNode(true);
  node.querySelector("h2").textContent = title;
  if (wide) node.classList.add("wide");
  return [node, node.querySelector(".card-body")];
}

function buildBoard(s) {
  inputs.clear();
  motorRows = new Map();
  chart = null;
  const board = $("#board");
  board.replaceChildren();

  board.append(reportCard());
  if (s.joints && s.joints.length) board.append(poseCard(s));
  if (fitData && fitData.points) board.append(fitsCard());
  if (s.motors.classes && s.motors.classes.length) board.append(actuatorCard(s));
  if (s.densityGroup.length || Object.keys(s.densities.derived).length) board.append(densityCard(s));
  for (const g of s.groups) board.append(groupCard(g));
}

function groupCard(g) {
  const [node, body] = card(g.title, g.params.length > 10);
  for (const p of g.params) body.append(paramField(p));
  if (g.title === "Calibrated (datasets/robot_descriptions)") {
    body.append(el("p", "hint",
      "Solved by a datasets/robot_descriptions calibration stage against the measured structural " +
      "laws — re-run that stage rather than editing here."));
  }
  return node;
}

function paramField(p) {
  const wrap = el("label", "field");
  const label = el("span");
  label.append(el("i", "dot"), document.createTextNode(p.key));
  if (p.owner === "calibrated") label.append(el("i", "owner", "calibrated"));
  wrap.append(label);

  let input;
  if (p.kind === "flag") {
    input = el("input"); input.type = "checkbox";
  } else if (p.kind === "motor_class") {
    input = el("select");
    input.replaceChildren(...p.options.map((o) => new Option(o, o)));
  } else if (p.kind === "text") {
    input = el("input"); input.type = "text";
  } else if (p.kind === "color") {
    input = el("input"); input.type = "color";
  } else {
    input = el("input"); input.type = "number"; input.step = "any";
  }
  input.title = p.hint || p.key;
  const emit = () => sendCoalesced(p.key, { type: "patch", key: p.key, value: readInput(input, p.kind) });
  input.addEventListener(p.kind === "text" ? "change" : "input", emit);
  wrap.append(input);
  inputs.set(p.key, { input, kind: p.kind, label });
  return wrap;
}

const readInput = (input, kind) =>
  kind === "flag" ? input.checked
  : kind === "color" ? hexToRgba(input.value)
  : kind === "text" || kind === "motor_class" ? input.value
  : Number(input.value);

/* "r g b a" floats <-> #rrggbb; alpha is preserved, not edited. */
let alphas = new Map();
const rgbaToHex = (s) => {
  const p = String(s).trim().split(/\s+/).map(Number);
  const h = (v) => Math.max(0, Math.min(255, Math.round((v || 0) * 255))).toString(16).padStart(2, "0");
  return `#${h(p[0])}${h(p[1])}${h(p[2])}`;
};
function hexToRgba(hex) {
  const n = parseInt(hex.slice(1), 16);
  const f = (v) => (v / 255).toFixed(3);
  const a = alphas.get(hex) ?? 1.0;
  return `${f((n >> 16) & 255)} ${f((n >> 8) & 255)} ${f(n & 255)} ${a.toFixed(1)}`;
}

/* ── actuator card ──────────────────────────────────────────────────────── */

// [class JSON key, header, parameter field]. `aspect` is never solved by a mode.
const COLS = [
  ["tau", "τ (N·m)", "effort"], ["omega", "ω (rad/s)", "velocity"],
  ["gear", "N", "gear"], ["power", "P (W)", "power"],
  ["aspect", "L/D", "aspect"],
];

function actuatorCard(s) {
  const [node, body] = card("Actuator design", true);
  body.append(el("p", "hint",
    "State two quantities; the fitted laws solve the rest. Greyed cells are " +
    "derived — mass, envelope and armature always are. L/D is never greyed: " +
    "the airgap law fixes r\u00b2\u00b7L, not how it splits between them."));

  const table = el("table");
  const head = el("tr");
  head.append(el("th", null, "Class"), el("th", null, "Designed from"),
    ...COLS.map(([, label]) => el("th", null, label)),
    el("th", null, "Mass"), el("th", null, "⌀ × L"), el("th", null, "Armature"),
    el("th", null, "τ/m"), el("th", null, "P/m"), el("th", null, "Verdict"));
  const thead = el("thead");
  thead.append(head);
  table.append(thead);

  const tbody = el("tbody");
  s.motors.classes.forEach((c, i) => {
    const tr = el("tr");
    const name = el("td");
    const sw = el("i", "swatch");
    sw.style.background = seriesColor(i);
    name.append(sw, document.createTextNode(c.cls));
    tr.append(name);

    const modeCell = el("td");
    const mode = el("select");
    mode.replaceChildren(...s.motors.modes.map((m) => new Option(m.label, m.key)));
    mode.onchange = () => send({ type: "motor_patch", cls: c.cls, field: "mode", value: mode.value });
    modeCell.append(mode);
    tr.append(modeCell);

    const fields = {};
    for (const [key, , field] of COLS) {
      const td = el("td");
      const inp = el("input");
      inp.type = "number"; inp.step = "any";
      inp.addEventListener("input", () => sendCoalesced(`${c.cls}.${field}`,
        { type: "motor_patch", cls: c.cls, field, value: Number(inp.value) }));
      td.append(inp);
      tr.append(td);
      fields[key] = inp;
    }
    const out = {};
    for (const k of ["mass", "env", "armature", "td", "pd", "verdict"]) {
      out[k] = el("td");
      if (k === "verdict") out[k].className = "verdict";
      tr.append(out[k]);
    }
    tbody.append(tr);
    motorRows.set(c.cls, { mode, fields, out });
  });
  table.append(tbody);
  body.append(table);

  body.append(chartBlock());
  return node;
}

/* The paper's four fitted actuator relations (log-log), with this design's
 * actuators on them. */
const FIT_ORDER = ["a", "b", "c", "d"];
const CAT_COLOUR = { QDD: "--series-1", MidGear: "--series-2", Harmonic: "--series-3" };

function fitsCard() {
  const [node, body] = card("Where its actuators land on the fitted trends", true);
  const grid = el("div", "fits-grid");
  fitCanvases = FIT_ORDER.map((k) => {
    const c = document.createElement("canvas");
    c.dataset.panel = k;
    c.setAttribute("aria-label", fitData.panels[k].title);
    grid.append(c);
    return c;
  });
  body.append(grid);
  const legend = el("div", "fits-legend");
  for (const [name, v] of Object.entries(CAT_COLOUR)) {
    const sw = el("span", null, name === "Harmonic" ? "High GR" : name);
    sw.prepend(Object.assign(document.createElement("i"), { style: `--c:var(${v})` }));
    legend.append(sw);
  }
  const mine = el("span", null, "this design");
  mine.prepend(el("i", "this"));
  legend.append(mine);
  body.append(legend);
  body.append(el("p", "hint",
    `Dots are the ${fitData.n} surveyed modules; the line is the trend fitted to them. ` +
    "Panel (c) is drawn at the catalogue's median reduction, so a high-reduction " +
    "actuator sits below it without being wrong."));
  return node;
}

const logPad = (vals, pad = 1.35) => {
  const lo = Math.min(...vals), hi = Math.max(...vals);
  return [Math.log10(lo / pad), Math.log10(hi * pad)];
};

function tickLabel(e) {
  const v = Math.pow(10, e);
  if (e >= 0 && e <= 4) return String(v);
  if (e < 0 && e >= -3) return v.toFixed(-e);
  return `1e${e}`;
}

function drawFits(design) {
  if (!fitData || !fitCanvases.length) return;
  for (const canvas of fitCanvases) drawFitPanel(canvas, canvas.dataset.panel, design);
}

function drawFitPanel(canvas, key, design) {
  const pts = fitData.points[key] || [];
  const law = fitData.laws[key] || [];
  const mine = design.map((d) => [d[key], d.cls]).filter(([p]) => p);
  if (!pts.length) return;

  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const xs = [...pts.map((p) => p[0]), ...mine.map((m) => m[0][0])];
  const ys = [...pts.map((p) => p[1]), ...mine.map((m) => m[0][1])];
  const [x0, x1] = logPad(xs), [y0, y1] = logPad(ys);
  const pad = { l: 44, r: 8, t: 22, b: 28 };
  const X = (v) => pad.l + ((Math.log10(v) - x0) / (x1 - x0)) * (w - pad.l - pad.r);
  const Y = (v) => h - pad.b - ((Math.log10(v) - y0) / (y1 - y0)) * (h - pad.t - pad.b);

  const P = fitData.panels[key];
  ctx.font = "600 11px ui-sans-serif, system-ui, sans-serif";
  ctx.fillStyle = css("--ink");
  ctx.textAlign = "left";
  ctx.fillText(`(${key}) ${P.title}`, pad.l, 12);

  // decade grid
  ctx.font = "9px ui-sans-serif, system-ui, sans-serif";
  ctx.strokeStyle = css("--grid");
  ctx.fillStyle = css("--muted");
  ctx.lineWidth = 1;
  for (let e = Math.ceil(x0); e <= Math.floor(x1); e++) {
    const x = X(Math.pow(10, e));
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
    ctx.textAlign = "center"; ctx.fillText(tickLabel(e), x, h - pad.b + 11);
  }
  for (let e = Math.ceil(y0); e <= Math.floor(y1); e++) {
    const y = Y(Math.pow(10, e));
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(tickLabel(e), pad.l - 4, y + 3);
  }

  // the surveyed catalogue
  for (const [px, py, cat] of pts) {
    ctx.fillStyle = css(CAT_COLOUR[cat] || "--muted");
    ctx.globalAlpha = 0.5;
    ctx.beginPath(); ctx.arc(X(px), Y(py), 2.1, 0, 2 * Math.PI); ctx.fill();
  }
  ctx.globalAlpha = 1;

  // the trend fitted through it
  if (law.length) {
    ctx.strokeStyle = css("--ink");
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    law.forEach(([lx, ly], i) => (i ? ctx.lineTo(X(lx), Y(ly)) : ctx.moveTo(X(lx), Y(ly))));
    ctx.stroke();
  }

  // this design's actuators
  for (const [[px, py], cls] of mine) {
    ctx.fillStyle = css("--accent");
    ctx.strokeStyle = css("--surface");
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(X(px), Y(py), 5, 0, 2 * Math.PI);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = css("--ink");
    ctx.font = "600 10px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(cls, X(px) + 7, Y(py) + 3);
  }

  ctx.fillStyle = css("--muted");
  ctx.font = "9px ui-sans-serif, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(P.x, pad.l + (w - pad.l - pad.r) / 2, h - 2);
}

function chartBlock() {
  const wrap = el("div", "chart-wrap");
  const canvas = el("canvas");
  const tip = el("div", "tip");
  tip.hidden = true;
  const legend = el("div", "legend");
  wrap.append(canvas, tip);
  chart = { canvas, tip, legend, ctx: canvas.getContext("2d"), data: null, box: null };

  canvas.addEventListener("mousemove", (e) => hover(e));
  canvas.addEventListener("mouseleave", () => { chart.tip.hidden = true; draw(); });
  const frag = document.createDocumentFragment();
  frag.append(el("h3", "hint", "Torque–speed envelope"), wrap, legend);
  return frag;
}

/* ── density card ───────────────────────────────────────────────────────── */

function densityCard(s) {
  const [node, body] = card("Density (kg/m³)");
  for (const p of s.densityGroup) body.append(paramField(p));

  const band = el("div");
  band.id = "rho-band";
  body.append(band);

  const derived = Object.entries(s.densities.derived);
  if (derived.length) {
    body.append(el("p", "hint", "Derived by the fits — not editable:"));
    const ul = el("ul", "warnings");
    for (const [k, why] of derived.sort()) ul.append(el("li", null, `${k} — ${why}`));
    body.append(ul);
  } else {
    body.append(el("p", "hint", "No density on this design is law-derived."));
  }
  return node;
}

/* ── report card ────────────────────────────────────────────────────────── */

/* Pose controls: joint sliders, simulate toggle, pose reset. */
function poseCard(s) {
  const [node, body] = card("Pose", true);

  const row = el("div", "pose-actions");
  const sim = el("button", "btn", s.simulate ? "Stop physics" : "Simulate");
  sim.id = "simulate";
  sim.ariaPressed = String(!!s.simulate);
  sim.onclick = () => send({ type: "simulate", on: !state.simulate });
  const rst = el("button", "btn", "Reset pose");
  rst.onclick = () => send({ type: "reset_pose" });
  row.append(sim, rst);
  body.append(row);
  body.append(el("p", "hint", s.simulate
    ? "Physics owns the pose while it runs; the sliders resume when you stop."
    : "Drag a joint to pose the robot. Nothing here changes the design."));

  const grid = el("div", "pose-grid");
  for (const j of s.joints) {
    const wrap = el("label", "pose-joint");
    const head = el("span", "pose-name", j.name);
    const out = el("output", "pose-val", j.value.toFixed(3));
    head.append(out);
    const r = document.createElement("input");
    r.type = "range";
    r.min = String(j.lo);
    r.max = String(j.hi);
    r.step = String(Math.max(1e-4, (j.hi - j.lo) / 200));
    r.value = String(j.value);
    r.disabled = !!s.simulate;
    r.oninput = () => {
      out.textContent = Number(r.value).toFixed(3);
      // One message per frame.
      sendCoalesced(`joint:${j.qadr}`,
        { type: "set_joint", qadr: j.qadr, value: Number(r.value) });
    };
    wrap.append(head, r);
    grid.append(wrap);
  }
  body.append(grid);
  return node;
}

function reportCard() {
  const [node, body] = card("Feasibility report", true);
  body.id = "report-body";
  return node;
}

/* The report: a mass bar, and each feasibility check as a band (the surveyed
 * range with this design's value marked). */
const CHECK_LABEL = {
  actuator_mass_fraction: ["actuator mass fraction", "pct"],
  trunk_mass_fraction: ["trunk mass fraction", "pct"],
  total_mass_vs_size_kg: ["total mass for this size", "kg"],
};

function massBar(r) {
  const box = el("div", "report-block");
  const parts = Object.entries(r.mass_kg).filter(([, v]) => v > 0).sort();
  const total = r.total_mass_kg || parts.reduce((a, [, v]) => a + v, 0);

  const hero = el("div", "hero", `${(r.total_mass_kg || 0).toFixed(2)} kg`);
  hero.append(el("small", null, "total mass"));
  box.append(hero);

  const bar = el("div", "massbar");
  parts.forEach(([, v], i) => {
    const seg = el("span");
    seg.style.flex = String(v);
    seg.style.background = seriesColor(i);
    bar.append(seg);
  });
  box.append(bar);

  const legend = el("div", "massbar-legend");
  parts.forEach(([k, v], i) => {
    const item = el("span", null, null);
    const dot = el("i");
    dot.style.background = seriesColor(i);
    item.append(dot, el("b", null, k), document.createTextNode(
      ` ${v.toFixed(2)} kg · ${Math.round((100 * v) / (total || 1))}%`));
    legend.append(item);
  });
  box.append(legend);
  return box;
}

function checkBand(c) {
  const [label, unit] = CHECK_LABEL[c.quantity]
    || [c.quantity.replace(/_/g, " "), ""];
  const show = (v) => (unit === "pct" ? `${Math.round(v * 100)}%`
    : unit === "kg" ? `${v.toFixed(1)} kg` : v.toFixed(3));

  // p10-p90 for a fraction; the 2-sigma interval for mass-versus-size.
  const lo = c.p10 ?? c.lo_2sigma;
  const hi = c.p90 ?? c.hi_2sigma;
  const ok = c.status === "ok";

  const wrap = el("div", "check");
  const head = el("div", "check-head");
  head.append(el("span", null, label));
  const val = el("b", ok ? "ok" : "warn", show(c.value));
  head.append(val);
  wrap.append(head);

  if (lo != null && hi != null) {
    const spanLo = Math.min(lo, c.value), spanHi = Math.max(hi, c.value);
    const range = spanHi - spanLo || 1;
    const a = spanLo - 0.15 * range, b = spanHi + 0.15 * range;
    const at = (v) => `${(100 * (v - a)) / (b - a)}%`;

    const band = el("div", "band");
    const inner = el("span", "band-in");
    inner.style.left = at(lo);
    inner.style.right = `calc(100% - ${at(hi)})`;
    const mark = el("span", ok ? "band-mark" : "band-mark warn");
    mark.style.left = at(c.value);
    band.append(inner, mark);
    wrap.append(band);

    const foot = el("div", "check-foot");
    foot.append(el("span", null, `surveyed ${show(lo)}–${show(hi)}`));
    if (!ok) foot.append(el("span", "warn", c.status.replace(/_/g, " ")));
    if (c.extrapolated) foot.append(el("span", null, "extrapolated"));
    wrap.append(foot);
  }
  return wrap;
}

function renderReport(r) {
  const body = $("#report-body");
  if (!body) return;
  body.replaceChildren();
  if (!r) {
    body.append(el("p", "hint", "Generate to see the feasibility report."));
    return;
  }
  if (r.refused) {
    body.append(el("p", "refused-head", "Refused by the feasibility gate — nothing was written."),
      el("pre", "refused", r.refused));
    return;
  }

  body.append(massBar(r));

  if (r.checks && r.checks.length) {
    const box = el("div", "report-block");
    box.append(el("h3", "block-title", "Feasibility checks"));
    for (const c of r.checks) box.append(checkBand(c));
    body.append(box);
  }

  const n = r.warnings.length;
  const det = document.createElement("details");
  const sum = document.createElement("summary");
  sum.append(document.createTextNode("Warnings "), el("span", "count", String(n)));
  det.append(sum);
  const ul = el("ul", "warnings");
  for (const w of r.warnings) ul.append(el("li", null, w));
  det.append(ul);
  body.append(det);
}

function applyValues(s) {
  const active = document.activeElement;
  for (const g of [...s.groups.map((x) => x.params), s.densityGroup].flat()) {
    const ref = inputs.get(g.key);
    if (!ref) continue;
    if (ref.input !== active) {
      if (g.kind === "flag") ref.input.checked = !!g.value;
      else if (g.kind === "color") {
        const hex = rgbaToHex(g.value);
        alphas.set(hex, parseFloat(String(g.value).trim().split(/\s+/)[3] ?? 1));
        ref.input.value = hex;
      } else ref.input.value = g.value;
    }
    ref.label.firstChild.style.visibility = g.changed ? "visible" : "hidden";
    ref.label.title = g.changed ? "changed from the robot's shipped value" : "";
  }

  applyMotors(s.motors, active);
  applyBand(s.densities.band);
  renderReport(s.report);
}

function applyMotors(m, active) {
  for (const c of m.classes || []) {
    const row = motorRows.get(c.cls);
    if (!row) continue;
    if (row.mode !== active) row.mode.value = c.mode;
    for (const [key, , field] of COLS) {
      const inp = row.fields[key];
      const derived = !c.active.includes(field);
      inp.disabled = derived;
      inp.title = derived ? "solved by the fitted laws from this mode's two inputs" : "";
      if (inp !== active) inp.value = fmt(c[key]);
    }
    row.out.mass.textContent = c.mass == null ? "—" : `${(c.mass * 1e3).toFixed(0)} g`;
    row.out.env.textContent = `⌀${(c.r * 2e3).toFixed(0)}×${(c.L * 1e3).toFixed(0)} mm`;
    row.out.env.title = c.volume == null ? "" :
      `${(c.volume * 1e6).toFixed(0)} cm³ — set by the mass law; L/D moves r and L inside it`;
    row.out.armature.textContent = c.armature?.toPrecision(3) ?? "—";
    row.out.td.textContent = `${c.torque_density?.toFixed(0)}`;
    row.out.pd.textContent = `${c.power_density?.toFixed(0)}`;
    // Report the worse of the two density verdicts.
    const rank = { ok: 0, above_p90: 1, beyond_frontier: 2 };
    const worst = rank[c.power_status] > rank[c.torque_status] ? c.power_status : c.torque_status;
    row.out.verdict.dataset.s = worst;
    row.out.verdict.textContent =
      `${{ ok: "✓", above_p90: "▲", beyond_frontier: "✕" }[worst]} τ≤${c.tau_ceiling?.toFixed(0)}`;
    row.out.verdict.title =
      `τ/m ${c.torque_density?.toFixed(0)} Nm/kg (${c.torque_status}), ` +
      `P/m ${c.power_density?.toFixed(0)} W/kg (${c.power_status}); ` +
      `ceiling ${c.tau_ceiling?.toFixed(0)} N·m at N=${c.gear?.toFixed(0)}` +
      (m.allow_hypothetical ? " — beyond-frontier is a warning (allow_hypothetical)"
        : " — beyond-frontier refuses generation");
  }
  if (chart) {
    chart.data = m.plot;
    renderLegend(m);
    draw();
  }
}

const fmt = (v) => (v == null ? "" : Math.abs(v) >= 100 ? v.toFixed(1)
  : Math.abs(v) >= 1 ? v.toFixed(3) : v.toPrecision(4));

function applyBand(b) {
  const node = document.getElementById("rho-band");
  if (!node) return;
  node.replaceChildren();
  if (!b) return;
  const lo = b.lo, hi = b.hi, v = b.value;
  const span = Math.max(hi - lo, 1e-9);
  const min = Math.min(lo, v) - span * 0.35, max = Math.max(hi, v) + span * 0.35;
  const pct = (x) => `${((x - min) / (max - min)) * 100}%`;
  const bar = el("div", "band");
  const inside = el("div", "in");
  inside.style.left = pct(lo);
  inside.style.width = `${((hi - lo) / (max - min)) * 100}%`;
  const mark = el("div", "mark");
  mark.style.left = pct(v);
  bar.append(inside, mark);
  const ok = v >= lo && v <= hi;
  node.append(bar, el("p", "hint",
    `${ok ? "✓" : "✕"} link_rho ${v.toFixed(0)} against the measured ${b.population} ` +
    `structural band ${lo.toFixed(0)}–${hi.toFixed(0)} kg/m³ — solved by the ` +
    `datasets/robot_descriptions calibration stage, not picked.`));
}

/* ── chart ──────────────────────────────────────────────────────────────── */

function renderLegend(m) {
  chart.legend.replaceChildren();
  (m.classes || []).forEach((c, i) => {
    const s = el("span");
    const line = el("i");
    line.style.borderTopColor = seriesColor(i);
    s.append(line, document.createTextNode(`class ${c.cls}`));
    chart.legend.append(s);
  });
  const s = el("span");
  const line = el("i");
  line.style.borderTopColor = css("--muted");
  line.style.borderTopStyle = "dashed";
  s.append(line, document.createTextNode("best-in-class frontier"));
  chart.legend.append(s);
}

function draw(hoverX) {
  if (!chart || !chart.data) return;
  const { canvas, ctx, data } = chart;
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = 210;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.height = `${H}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const pad = { l: 46, r: 46, t: 8, b: 26 };
  const xs = data.omegas.filter((v) => v != null);
  const all = data.series.flatMap((s) => [...s.envelope, ...s.frontier]).filter((v) => v != null);
  if (!xs.length || !all.length) return;
  const x0 = 0, x1 = Math.max(...xs);
  const y0 = 0, y1 = Math.max(...all) * 1.06;
  const X = (v) => pad.l + ((v - x0) / (x1 - x0 || 1)) * (W - pad.l - pad.r);
  const Y = (v) => H - pad.b - ((v - y0) / (y1 - y0 || 1)) * (H - pad.t - pad.b);
  chart.box = { X, Y, x0, x1, y0, y1, pad, W, H };

  // Recessive grid and axis; ink stays on text tokens, never on a series colour.
  ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
  ctx.fillStyle = css("--muted");
  ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const v = y0 + ((y1 - y0) * i) / 4, y = Math.round(Y(v)) + 0.5;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(v.toFixed(v < 10 ? 1 : 0), pad.l - 6, y);
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let i = 0; i <= 4; i++) {
    const v = x0 + ((x1 - x0) * i) / 4;
    ctx.fillText(v.toFixed(0), X(v), H - pad.b + 6);
  }
  ctx.strokeStyle = css("--axis");
  ctx.beginPath();
  ctx.moveTo(pad.l, H - pad.b + 0.5); ctx.lineTo(W - pad.r, H - pad.b + 0.5); ctx.stroke();
  ctx.textAlign = "left";
  ctx.fillText("ω (rad/s)", pad.l, H - 11);
  ctx.save(); ctx.translate(11, pad.t + 4); ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "right"; ctx.fillText("τ (N·m)", 0, 0); ctx.restore();

  const line = (ys, color, dash) => {
    ctx.beginPath();
    ctx.setLineDash(dash ? [5, 4] : []);
    ctx.strokeStyle = color;
    ctx.lineWidth = dash ? 1.5 : 2;
    ctx.globalAlpha = dash ? 0.6 : 1;
    let started = false;
    ys.forEach((v, i) => {
      if (v == null || data.omegas[i] == null) { started = false; return; }
      const px = X(data.omegas[i]), py = Y(v);
      started ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      started = true;
    });
    ctx.stroke();
    ctx.globalAlpha = 1; ctx.setLineDash([]);
  };

  data.series.forEach((s, i) => line(s.frontier, css("--muted"), true));
  data.series.forEach((s, i) => line(s.envelope, seriesColor(i), false));

  // Direct labels at each envelope's last drawn point (≤ 4 series here).
  data.series.forEach((s, i) => {
    let last = -1;
    s.envelope.forEach((v, j) => { if (v != null) last = j; });
    if (last < 0) return;
    ctx.fillStyle = css("--ink-2");
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText(s.cls, X(data.omegas[last]) + 5, Y(s.envelope[last]));
  });

  if (hoverX != null) {
    ctx.strokeStyle = css("--axis");
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(hoverX, pad.t); ctx.lineTo(hoverX, H - pad.b); ctx.stroke();
    ctx.setLineDash([]);
  }
}

function hover(e) {
  if (!chart?.box || !chart.data) return;
  const rect = chart.canvas.getBoundingClientRect();
  const px = e.clientX - rect.left;
  const { X, pad, W } = chart.box;
  if (px < pad.l || px > W - pad.r) { chart.tip.hidden = true; draw(); return; }
  let best = 0, bestD = Infinity;
  chart.data.omegas.forEach((o, i) => {
    if (o == null) return;
    const d = Math.abs(X(o) - px);
    if (d < bestD) { bestD = d; best = i; }
  });
  const omega = chart.data.omegas[best];
  const rows = chart.data.series
    .map((s) => s.envelope[best] == null ? null : `${s.cls} ${s.envelope[best].toFixed(1)} N·m`)
    .filter(Boolean);
  chart.tip.textContent = `ω ${omega.toFixed(0)} rad/s — ${rows.join(" · ")}`;
  chart.tip.hidden = false;
  chart.tip.style.left = `${Math.min(px + 10, rect.width - chart.tip.offsetWidth - 4)}px`;
  chart.tip.style.top = "6px";
  draw(X(omega));
}

window.addEventListener("resize", () => draw());
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => draw());

/* ── controls ───────────────────────────────────────────────────────────── */

$("#folder").onchange = (e) => send({ type: "select_folder", folder: e.target.value });
$("#refresh").onclick = () => send({ type: "refresh_folders" });
$("#generate").onclick = () => send({ type: "generate" });
$("#new").onclick = () => send({ type: "new_from_defaults" });
$("#reset").onclick = () => send({ type: "reset_defaults" });
/* Save builds this design into its own folder; Generate reuses the live one. */
$("#save").onclick = () => {
  const name = prompt("Save this design as:", suggestName());
  if (name) send({ type: "save_as", name });
};

function suggestName() {
  const n = (state && state.robot) || "design";
  return `${n}-${new Date().toISOString().slice(0, 10)}`;
}
/* The 3D pane: on by default, remembered per browser. */
function setViewer(show) {
  const box = $("#viewer");
  box.hidden = !show;
  $("#shell").classList.toggle("no-viewer", !show);
  $("#embed").ariaPressed = String(show);
  const frame = $("#viewer-frame");
  if (show && !frame.src) frame.src = frame.dataset.src || "";
  try { localStorage.setItem("draft.viewer", show ? "1" : "0"); } catch { /* private mode */ }
  draw();
}
$("#embed").onclick = () => setViewer($("#viewer").hidden);

/* The bar wraps, so measure its height for the split pane. */
function measureBar() {
  document.documentElement.style.setProperty(
    "--bar-h", `${document.querySelector(".bar").offsetHeight}px`);
}
window.addEventListener("resize", measureBar);
measureBar();

let showViewer = true;
try { showViewer = localStorage.getItem("draft.viewer") !== "0"; } catch { /* private mode */ }
setViewer(showViewer);

connect();
