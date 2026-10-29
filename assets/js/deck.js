/* Segmentation deck: scene engine (from the presentation-neuroimaging deck) + per-scene animations.
   Every number and image comes from Segmentation_UNet.ipynb (assets/generated/, window.SEG in stats.js). */
(function () {
  'use strict';
  var D = window.SEG || {};
  var NS = 'http://www.w3.org/2000/svg';
  var G = 'assets/generated/';
  var SH = {}, TICK = {};
  var CL = D.classes || [];
  var VOID = 11;
  var C = { conv: '#3A86FF', pool: '#FF4D6D', up: '#2EC4B6', out: '#FFD23F', bar: '#C9D6F2', copy: '#AEB4C8', ink: '#0F1220', mute: '#6A7192' };

  function $(s, r) { return (r || document).querySelector(s); }
  function $$(s, r) { return [].slice.call((r || document).querySelectorAll(s)); }
  function fmt(n, d) { return Number(n).toLocaleString('en-US', { minimumFractionDigits: d || 0, maximumFractionDigits: d || 0 }); }
  function set(id, txt) { var e = document.getElementById(id); if (e) e.textContent = txt; }
  function svg(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function col(c) { return c === VOID || !CL[c] ? '#000000' : CL[c].color; }
  function cname(c) { return CL[c] ? CL[c].name : ''; }
  function rgb(h) { h = h.replace('#', ''); return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]; }
  function mixc(a, b, u) { a = rgb(a); b = rgb(b); return 'rgb(' + [0, 1, 2].map(function (k) { return Math.round(a[k] + (b[k] - a[k]) * u); }).join(',') + ')'; }
  function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
  function ez(u) { u = clamp(u, 0, 1); return u < .5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2; }
  function img(src) { var i = new Image(); i.src = src; return i; }
  function arrowDefs(s, id) {
    var defs = svg('defs', {}, s);
    Object.keys(C).forEach(function (k) {
      var m = svg('marker', { id: id + '-' + k, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: 'auto-start-reverse' }, defs);
      svg('path', { d: 'M0 0L10 5L0 10z', fill: C[k] }, m);
    });
  }
  var HERO = img(G + 'hero.jpg'), HERO_GT = img(G + 'hero_gt.png');
  var GRID = D.cover_grid || { h: 24, w: 32, rgb: [], cls: [] };
  var L = D.levels || [], INP = D.input || [192, 256];

  var CELL = 1920 / GRID.w, OFFY = (1080 - CELL * GRID.h) / 2;
  function centroid(c) {
    var xs = [], ys = [];
    GRID.cls.forEach(function (k, i) { if (k === c) { xs.push(i % GRID.w); ys.push(Math.floor(i / GRID.w)); } });
    if (!xs.length) return null;
    xs.sort(function (a, b) { return a - b; }); ys.sort(function (a, b) { return a - b; });
    return [(xs[xs.length >> 1] + .5) * CELL, (ys[ys.length >> 1] + .5) * CELL + OFFY, xs.length];
  }

  /* ================= the U-Net drawing, built from the real layer sizes =================
     Every element carries data-t (what kind of operation) and data-st (which step of the walk-through). */
  function buildUNet(s, opt) {
    opt = opt || {};
    var fs = opt.fs || 1, id = s.id || ('u' + Math.random().toString(36).slice(2, 7)), lab = opt.labels !== false;
    arrowDefs(s, id);
    var rowY = [118, 280, 410, 516, 606], barH = [206, 150, 106, 72, 50], bw = [10, 14, 18, 22, 28], GAP = 26;
    var ch = L.map(function (l) { return l.c; }), res = L.map(function (l) { return l.h + '×' + l.w; });
    var g = svg('g', {}, s);
    function tag(e, t, st) { e.setAttribute('data-t', t); e.setAttribute('data-st', st); e.classList.add('u-el'); return e; }
    function bar(x, i, fill, st, dashed) {
      var a = { x: x, y: rowY[i] - barH[i] / 2, width: bw[i], height: barH[i], rx: 2, fill: dashed ? 'none' : fill };
      if (dashed) { a.stroke = C.copy; a['stroke-width'] = 2.5; a['stroke-dasharray'] = '6 4'; }
      return tag(svg('rect', a, g), dashed ? 'skip' : 'bar', st);
    }
    function arrow(x1, y1, x2, y2, k, st) {
      return tag(svg('path', { d: 'M' + x1 + ' ' + y1 + ' L' + x2 + ' ' + y2, fill: 'none', stroke: C[k], 'stroke-width': 4, 'stroke-linecap': 'round', 'marker-end': 'url(#' + id + '-' + k + ')' }, g), k, st);
    }
    function text(x, y, txt, anchor, size, color, st, t) {
      if (!lab) return;
      var e = svg('text', { x: x, y: y, 'text-anchor': anchor || 'middle', 'font-size': (size || 17) * fs, fill: color || '#AEB4C8' }, g);
      e.textContent = txt; tag(e, t || 'lbl', st);
    }
    tag(svg('image', { href: G + 'hero.jpg', x: 0, y: rowY[0] - 52, width: 140, height: 105, preserveAspectRatio: 'none' }, g), 'io', 'in');
    text(70, rowY[0] + 78, INP[0] + '×' + INP[1] + '×3', 'middle', 17, null, 'in');
    var encEnd = [], decStart = [];
    for (var i = 0; i < 4; i++) {
      var xs = 160 + i * 140, w = bw[i], st = 'L' + i;
      if (i === 0) arrow(142, rowY[0], xs - 4, rowY[0], 'conv', st);
      bar(xs, i, C.bar, st);
      arrow(xs + w + 3, rowY[i], xs + w + GAP - 4, rowY[i], 'conv', st);
      bar(xs + w + GAP, i, C.bar, st);
      text(xs + w + GAP / 2, rowY[i] - barH[i] / 2 - 12, ch[i], 'middle', 18, '#fff', st);
      text(xs - 8, rowY[i] + barH[i] / 2 + 22, res[i], 'start', 16, C.mute, st);
      encEnd.push(xs + 2 * w + GAP);
      var nx = i === 3 ? 832 - bw[4] - GAP / 2 : 160 + (i + 1) * 140, nb = bw[i + 1];
      arrow(xs + 2 * w + GAP - w / 2, rowY[i] + barH[i] / 2 + 4, nx + nb / 2, rowY[i + 1] - barH[i + 1] / 2 - 6, 'pool', 'P' + i);
    }
    var bx = 832 - bw[4] - GAP / 2;
    bar(bx, 4, C.bar, 'B'); arrow(bx + bw[4] + 3, rowY[4], bx + bw[4] + GAP - 4, rowY[4], 'conv', 'B'); bar(bx + bw[4] + GAP, 4, C.bar, 'B');
    text(832, rowY[4] - barH[4] / 2 - 12, ch[4], 'middle', 18, '#fff', 'B');
    text(832, rowY[4] + barH[4] / 2 + 26, res[4] + ' · cuello de botella', 'middle', 16, C.mute, 'B');
    var prevEnd = bx + 2 * bw[4] + GAP, prevI = 4;
    for (i = 3; i >= 0; i--) {
      var w2 = bw[i], end = 1664 - 160 - i * 140, s0 = end - (4 * w2 + 2 * GAP);
      arrow(prevEnd - bw[prevI] / 2, rowY[prevI] - barH[prevI] / 2 - 4, s0 + w2 / 2, rowY[i] + barH[i] / 2 + 6, 'up', 'U' + i);
      bar(s0, i, C.up, 'U' + i);
      bar(s0 + w2, i, null, 'K' + i, true);
      arrow(s0 + 2 * w2 + 3, rowY[i], s0 + 2 * w2 + GAP - 4, rowY[i], 'conv', 'D' + i);
      bar(s0 + 2 * w2 + GAP, i, C.bar, 'D' + i);
      arrow(s0 + 3 * w2 + GAP + 3, rowY[i], s0 + 3 * w2 + 2 * GAP - 4, rowY[i], 'conv', 'D' + i);
      bar(s0 + 3 * w2 + 2 * GAP, i, C.bar, 'D' + i);
      text(s0 + w2, rowY[i] - barH[i] / 2 - 12, ch[i] + '+' + ch[i], 'middle', 18, '#fff', 'K' + i);
      text(end + 8, rowY[i] + barH[i] / 2 + 22, res[i], 'end', 16, C.mute, 'D' + i);
      decStart.push(s0 + w2);
      prevEnd = end; prevI = i;
    }
    decStart.reverse().forEach(function (xd, k) {
      var y = rowY[k];
      tag(svg('path', { 'class': 'skip', d: 'M' + (encEnd[k] + 6) + ' ' + y + ' L' + (xd - 8) + ' ' + y, fill: 'none', stroke: C.copy, 'stroke-width': 4, 'stroke-dasharray': '14 10' }, g), 'skip', 'K' + k);
      text((encEnd[k] + xd) / 2, y - 12, 'copiar ' + res[k], 'middle', 16, C.copy, 'K' + k, 'skip');
    });
    arrow(1504 + 3, rowY[0], 1520, rowY[0], 'out', 'O');
    tag(svg('image', { href: G + 'hero.jpg', x: 1524, y: rowY[0] - 52, width: 140, height: 105, preserveAspectRatio: 'none' }, g), 'out', 'O');
    tag(svg('image', { href: G + 'hero_unet.png', x: 1524, y: rowY[0] - 52, width: 140, height: 105, preserveAspectRatio: 'none', opacity: .85 }, g), 'out', 'O');
    text(1594, rowY[0] + 78, INP[0] + '×' + INP[1] + '×11', 'middle', 17, null, 'O', 'out');
    var mark = null;
    if (opt.mini) {
      svg('text', { x: 330, y: 700, 'text-anchor': 'middle', 'font-size': 64, 'font-weight': 700, fill: '#8A91B4', 'font-family': 'JetBrains Mono' }, g).textContent = 'encoder ↓';
      svg('text', { x: 1334, y: 700, 'text-anchor': 'middle', 'font-size': 64, 'font-weight': 700, fill: '#8A91B4', 'font-family': 'JetBrains Mono' }, g).textContent = 'decoder ↑';
      mark = svg('g', { 'class': 'here' }, s);
      svg('ellipse', { 'class': 'ring', fill: 'none', stroke: '#FFD23F', 'stroke-width': 8 }, mark);
      var chip = svg('g', {}, mark);
      svg('rect', { 'class': 'chipbg', rx: 22, height: 86, width: 230, fill: '#FFD23F' }, chip);
      svg('text', { 'class': 'chiptx', 'font-size': 60, 'font-weight': 700, fill: '#0F1220', 'font-family': 'JetBrains Mono' }, chip).textContent = 'aquí';
    }
    function place() {
      if (!mark) return;
      var on = els.filter(function (e) { return e.classList.contains('hl') && e.getAttribute('data-t') !== 'lbl'; });
      if (!on.length) { mark.style.display = 'none'; return; }
      var x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
      on.forEach(function (e) { var b = e.getBBox(); x0 = Math.min(x0, b.x); y0 = Math.min(y0, b.y); x1 = Math.max(x1, b.x + b.width); y1 = Math.max(y1, b.y + b.height); });
      var cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, rx = (x1 - x0) / 2 + 34, ry = (y1 - y0) / 2 + 30;
      mark.style.display = '';
      var r = mark.querySelector('.ring'); r.setAttribute('cx', cx); r.setAttribute('cy', cy); r.setAttribute('rx', rx); r.setAttribute('ry', ry);
      var tx = clamp(cx - 115, 10, 1664 - 240), ty = y0 - 34 - 86 < 10 ? Math.min(y1 + 34, 720 - 96) : y0 - 34 - 86;
      mark.querySelector('.chipbg').setAttribute('x', tx); mark.querySelector('.chipbg').setAttribute('y', ty);
      var t = mark.querySelector('.chiptx'); t.setAttribute('x', tx + 46); t.setAttribute('y', ty + 62);
    }
    var els = $$('.u-el', s);
    return {
      stages: function (list) { els.forEach(function (e) { var on = !list || list.indexOf(e.getAttribute('data-st')) >= 0; e.classList.toggle('dim', !on); e.classList.toggle('hl', !!list && on); }); place(); },
      types: function (list) { els.forEach(function (e) { var on = !list || list.indexOf(e.getAttribute('data-t')) >= 0; e.classList.toggle('dim', !on); e.classList.toggle('hl', !!list && on); }); if (mark) mark.style.display = 'none'; }
    };
  }

  /* ---------------- 01 portada: photo -> real pixels -> one class per pixel -> mask ---------------- */
  (function () {
    var sc = $('#cover'), cv = $('#coverCv'), g = cv.getContext('2d'), t0 = 0, playing = false, k0 = 0;
    var cells = GRID.cls.map(function (c, i) {
      var r = Math.floor(i / GRID.w), cc = i % GRID.w;
      return { x: cc * CELL, y: r * CELL + OFFY, c: c, rgb: GRID.rgb[i] || [0, 0, 0], cr: rgb(col(c)),
        d: Math.random() * .5, w: cc / GRID.w * .9 + Math.random() * .15 };
    });
    var chipWrap = $('#cvChips');
    [0, 1, 5, 3, 8, 9, 10].forEach(function (c, j) {
      var ct = centroid(c);
      if (!ct || ct[2] < 3) return;
      var s = document.createElement('span');
      s.textContent = cname(c);
      s.style.cssText = 'left:' + clamp(ct[0], 120, 1800) + 'px;top:' + clamp(ct[1], 200, 900) + 'px;background:' + col(c) + ';--d:' + (j * .12) + 's';
      chipWrap.appendChild(s);
    });
    function photo(alpha, zoom) {
      if (!HERO.complete || !HERO.naturalWidth) return;
      var w = 1920 * zoom, h = 1440 * zoom;
      g.globalAlpha = alpha; g.drawImage(HERO, (1920 - w) / 2, (1080 - h) / 2, w, h); g.globalAlpha = 1;
    }
    function mask(alpha) { if (!HERO_GT.complete || !HERO_GT.naturalWidth) return; g.globalAlpha = alpha; g.drawImage(HERO_GT, 0, OFFY, 1920, 1440); g.globalAlpha = 1; }
    function rr(x, y, s, r) { g.beginPath(); if (g.roundRect) g.roundRect(x, y, s, s, r); else g.rect(x, y, s, s); g.fill(); }
    SH.cover = function (s, step) {
      if (step === 1 && !playing) { playing = true; t0 = 0; }
      if (step === 0) { playing = false; sc.classList.remove('chips-on'); }
    };
    SH.cover.reset = function () { playing = false; sc.classList.remove('chips-on'); };
    TICK.cover = function (t) {
      g.clearRect(0, 0, 1920, 1080);
      if (!k0) k0 = t;
      if (!playing) { photo(1, 1 + .04 * Math.sin((t - k0) * .15)); return; }
      if (!t0) t0 = t;
      var e = t - t0;
      if (e < 3.4) photo(1 - ez(e / 1.1), 1);
      cells.forEach(function (o) {
        var a = ez((e - o.d) / .7);
        if (a <= 0 || e > 4.8) return;
        var u = ez((e - 1.3 - o.w * 1.4) / .5), fade = 1 - ez((e - 3.6) / 1);
        var gap = 2 + 6 * a * (1 - ez((e - 3.2) / .5)), s = CELL - gap * 2, c0 = o.rgb, c1 = o.cr;
        g.fillStyle = 'rgba(' + Math.round(c0[0] + (c1[0] - c0[0]) * u) + ',' + Math.round(c0[1] + (c1[1] - c0[1]) * u) + ',' +
          Math.round(c0[2] + (c1[2] - c0[2]) * u) + ',' + (a * fade * (o.c === VOID ? 1 - u * .8 : 1)).toFixed(3) + ')';
        var ss = s * (1 + .25 * Math.sin(Math.PI * clamp((e - 1.3 - o.w * 1.4) / .5, 0, 1)));
        rr(o.x + (CELL - ss) / 2, o.y + (CELL - ss) / 2, ss, 6);
      });
      if (e > 3.4) { var b = ez((e - 3.4) / 1.2); photo(b, 1); mask(b * .62); }
      sc.classList.toggle('chips-on', e > 4.4);
    };
  })();

  /* ---------------- 02 tareas: tags -> boxes -> semantic mask -> instances ---------------- */
  (function () {
    var B = D.boxes || [], s = $('#tkBoxes'), tags = $('#tkTags');
    B.forEach(function (b, i) {
      var x = b.x * 480, y = b.y * 360, w = b.w * 480, h = b.h * 360, c = col(b.cls), d = (i * .12) + 's';
      svg('rect', { x: x, y: y, width: w, height: h, rx: 2, stroke: c, pathLength: 1, style: '--d:' + d }, s);
      var name = cname(b.cls), lw = name.length * 7 + 8;
      svg('rect', { 'class': 'lb', x: x - 1.2, y: Math.max(0, y - 15), width: lw, height: 14, rx: 2, fill: c, style: '--d:' + d }, s);
      var tx = svg('text', { x: x + 3, y: Math.max(0, y - 15) + 11, fill: '#0F1220', style: '--d:' + d }, s);
      tx.textContent = name;
    });
    set('tkNb', B.length + ' cajas');
    var present = [8, 9, 10, 5, 1].filter(function (c) { return GRID.cls.indexOf(c) >= 0; });
    present.forEach(function (c, i) {
      tags.insertAdjacentHTML('beforeend', '<span style="--d:' + (.5 + i * .12) + 's"><i style="background:' + col(c) + '"></i>' + cname(c) + '<em>sí</em></span>');
    });
    tags.insertAdjacentHTML('beforeend', '<span style="--d:' + (.5 + present.length * .12) + 's"><i style="background:#555"></i>Perro<em style="color:#FF4D6D">no</em></span>');
    var li = $$('#tareas .tasks li');
    SH.tareas = function (sc, step) { li.forEach(function (l, i) { l.classList.toggle('on', i === step); l.classList.toggle('done', i < step); }); };
  })();

  /* ---------------- 03 la arquitectura + definiciones ---------------- */
  (function () {
    var U = buildUNet($('#arqSvg'), { fs: 1.35 }), defs = $$('#arqDefs .df');
    var types = [null, ['bar', 'skip'], ['conv'], ['conv'], ['pool'], ['up'], ['skip'], ['out']];
    SH.arq = function (sc, step) {
      U.types(types[step] || null);
      defs.forEach(function (d) { var k = +d.dataset.k; d.classList.toggle('on', k === step); d.classList.toggle('seen', k < step); });
      $('#arqCap').style.opacity = step === 0 ? 1 : 0;
    };
  })();

  /* ---------------- shared: the mini U-Net in each step-by-step slide ---------------- */
  var MINI = {}, MAPS = {
    p01: ['in'], p02: ['in', 'L0'], p03: ['L0'], p04: ['L0'], p05: ['L0'], p06: ['P0'],
    p07: ['L1', 'P1', 'L2', 'P2', 'L3', 'P3'], p08: ['B'], p09: ['U3'], p10: ['K3'], p11: ['D3'],
    p12: ['U2', 'K2', 'D2', 'U1', 'K1', 'D1', 'U0', 'K0', 'D0'], p13: ['O'], p14: ['O'], p15: ['O'],
    skips: ['K0', 'K1', 'K2', 'K3'], salida: ['O'], resultados: ['O'], entrena: null
  };
  $$('svg.mini').forEach(function (s) { var sc = s.closest('.scene'); s.id = 'mini-' + sc.id; MINI[sc.id] = buildUNet(s, { labels: false, mini: true }); });
  function mini(id) { if (!MINI[id]) return; if (id === 'p16') MINI[id].types(['conv', 'up', 'out']); else MINI[id].stages(MAPS[id]); }
  var CD = D.conv_demo;

  /* ---------------- P1 la imagen: 3 matrices de números ---------------- */
  (function () {
    if (!CD) return;
    var box = $('#inBox');
    box.style.left = ((CD.pos[1] - 2) / INP[1] * 100) + '%'; box.style.top = ((CD.pos[0] - 2) / INP[0] * 100) + '%';
    box.style.width = (5 / INP[1] * 100) + '%'; box.style.height = (5 / INP[0] * 100) + '%';
    var tint = ['#E5484D', '#2F9E44', '#3A86FF'];
    $('#inNums').innerHTML = [0, 1, 2].map(function (c) {
      return '<div class="g5n"><b style="color:' + tint[c] + '">' + 'RGB'[c] + '</b><div>' + CD.patch.map(function (row) {
        return row.map(function (p) { var v = p[c] / 255; return '<i style="background:' + mixc('#0F1220', tint[c], v) + '">' + v.toFixed(2) + '</i>'; }).join('');
      }).join('') + '</div></div>';
    }).join('');
  })();

  /* ---------------- P2 la convolución: la ventanita, con la cuenta real ---------------- */
  (function () {
    if (!CD) return;
    var patchEl = $('#cvPatch'), oEl = $('#cvOut'), pc = [], oc = [], kc = [];
    CD.patch.forEach(function (row) { row.forEach(function (p) {
      var e = document.createElement('i'); e.style.background = 'rgb(' + p.join(',') + ')';
      e.innerHTML = '<span>' + p.map(function (v) { return (v / 255).toFixed(2); }).join('</span><span>') + '</span>';
      patchEl.appendChild(e); pc.push(e);
    }); });
    var km = 0; CD.kernel.forEach(function (r) { r.forEach(function (w) { w.forEach(function (v) { km = Math.max(km, Math.abs(v)); }); }); });
    $$('#p02 .g3k').forEach(function (g, c) {
      CD.kernel.forEach(function (r) { r.forEach(function (w) {
        var v = w[c], e = document.createElement('i');
        e.style.background = v < 0 ? mixc('#20253A', '#3A86FF', Math.abs(v) / km) : mixc('#20253A', '#FF4D6D', v / km);
        e.textContent = (v < 0 ? '−' : '') + Math.abs(v).toFixed(2); g.appendChild(e); kc.push(e);
      }); });
    });
    CD.out.forEach(function (r) { r.forEach(function (v) { var e = document.createElement('i'); e.textContent = v.toFixed(2); oEl.appendChild(e); oc.push(e); }); });
    var dot = $('#cvDot'); dot.style.left = (CD.pos[1] / INP[1] * 100) + '%'; dot.style.top = (CD.pos[0] / INP[0] * 100) + '%';
    function sgn(v) { return v < 0 ? '(−' + Math.abs(v).toFixed(2) + ')' : v.toFixed(2); }
    function calc(a, b) {
      var parts = [0, 1, 2].map(function (c) {
        var s = 0, terms = [];
        for (var i = 0; i < 3; i++) for (var j = 0; j < 3; j++) {
          var x = CD.patch[a + i][b + j][c] / 255, w = CD.kernel[i][j][c];
          s += x * w; if (terms.length < 3) terms.push(x.toFixed(2) + '×' + sgn(w));
        }
        return { s: s, t: terms };
      });
      var tot = parts[0].s + parts[1].s + parts[2].s;
      return '<div><b class="r">R</b> ' + parts[0].t.join(' + ') + ' + … = <em>' + parts[0].s.toFixed(3) + '</em></div>' +
        '<div><b class="g">G</b> ' + parts[1].t.join(' + ') + ' + … = <em>' + parts[1].s.toFixed(3) + '</em></div>' +
        '<div><b class="b">B</b> ' + parts[2].t.join(' + ') + ' + … = <em>' + parts[2].s.toFixed(3) + '</em></div>' +
        '<div class="sum">R + G + B = <em>' + tot.toFixed(3) + '</em> → <span>' + CD.out[a][b].toFixed(2) + '</span></div>';
    }
    var t0 = 0, last = -1;
    SH.p02 = function () { t0 = 0; last = -1; };
    TICK.p02 = function (t) {
      if (!t0) t0 = t;
      var k = Math.floor((t - t0) / 1.6) % 10;
      if (k === last) return; last = k;
      var a = Math.floor(Math.min(k, 8) / 3), b = Math.min(k, 8) % 3;
      pc.forEach(function (c, q) { var r = Math.floor(q / 5), cc = q % 5; c.classList.toggle('win', k < 9 && r >= a && r < a + 3 && cc >= b && cc < b + 3); });
      kc.forEach(function (c) { c.classList.toggle('win', k < 9); });
      oc.forEach(function (c, q) { c.classList.toggle('in', q <= k); c.classList.toggle('now', q === k); });
      $('#cvCalc').innerHTML = k < 9 ? calc(a, b) : '<div class="sum">9 posiciones → 9 números. En la foto completa: 192 × 256 números.</div>';
    };
  })();

  /* ---------------- P3 los 16 filtros ---------------- */
  (function () {
    var K = D.conv1_kernels, g = $('#k16');
    if (!K) { $('#p03 .kg').style.visibility = 'hidden'; return; }
    g.innerHTML = K.map(function (k, n) {
      return '<div class="kc">' + k.map(function (row) { return row.map(function (p) { return '<i style="background:rgb(' + p.join(',') + ')"></i>'; }).join(''); }).join('') + '<span>' + (n + 1) + '</span></div>';
    }).join('');
  })();
  if (D.relu_zero != null) set('reluZ', fmt(D.relu_zero * 100, 0) + '%');

  /* ---------------- P5 la segunda convolución: 3 x 3 x 16 ---------------- */
  (function () {
    var s = $('#c2Svg'), wins = [];
    for (var k = 15; k >= 0; k--) {
      var x = 30 + k * 11, y = 30 + k * 11;
      svg('rect', { x: x, y: y, width: 240, height: 180, rx: 4, fill: mixc('#2A1B3D', '#F7A35C', .25 + .6 * Math.abs(Math.sin(k * 1.3))), stroke: '#F1F3F8', 'stroke-width': 1.5 }, s);
      wins.push({ el: svg('rect', { width: 36, height: 36, fill: 'none', stroke: '#FFD23F', 'stroke-width': 3 }, s), x: x, y: y });
    }
    svg('text', { x: 30, y: 22, 'font-size': 18, fill: '#4B5163', 'font-family': 'JetBrains Mono' }, s).textContent = '16 mapas de entrada';
    var out = svg('rect', { x: 400, y: 260, width: 40, height: 40, rx: 4, fill: '#FFD23F' }, s);
    var ar = svg('line', { x1: 0, y1: 0, x2: 392, y2: 280, stroke: '#0F1220', 'stroke-width': 3, 'stroke-dasharray': '6 5' }, s);
    svg('text', { x: 420, y: 330, 'text-anchor': 'middle', 'font-size': 16, fill: '#4B5163', 'font-family': 'JetBrains Mono' }, s).textContent = '1 número';
    svg('text', { x: 30, y: 430, 'font-size': 18, fill: '#0F1220', 'font-family': 'JetBrains Mono', 'font-weight': 700 }, s).textContent = 'ventana 3 × 3 × 16 = 144 pesos';
    var t0 = 0;
    SH.p05 = function () { t0 = 0; };
    TICK.p05 = function (t) {
      if (!t0) t0 = t;
      var e = (t - t0) * .35, px = 20 + 180 * (.5 + .5 * Math.sin(e)), py = 20 + 120 * (.5 + .5 * Math.sin(e * 1.7));
      wins.forEach(function (w) { w.el.setAttribute('x', w.x + px); w.el.setAttribute('y', w.y + py); });
      ar.setAttribute('x1', wins[wins.length - 1].x + px + 36); ar.setAttribute('y1', wins[wins.length - 1].y + py + 18);
    };
  })();

  /* ---------------- P6 max pooling on a real 8 x 8 patch ---------------- */
  (function () {
    var PD = D.pool_demo, ic = [], qc = [];
    if (!PD) return;
    var mx = 0; PD.patch.forEach(function (r) { r.forEach(function (v) { mx = Math.max(mx, v); }); }); mx = mx || 1;
    function cellv(v) { var e = document.createElement('i'); e.style.background = mixc('#1B1F33', '#FFD23F', v / mx); e.style.color = v / mx > .5 ? '#0F1220' : '#fff'; e.textContent = v.toFixed(1); return e; }
    PD.patch.forEach(function (r) { r.forEach(function (v) { var e = cellv(v); $('#mpIn').appendChild(e); ic.push(e); }); });
    PD.out.forEach(function (r) { r.forEach(function (v) { var e = cellv(v); $('#mpOut').appendChild(e); qc.push(e); }); });
    var t0 = 0;
    SH.p06 = function () { t0 = 0; };
    TICK.p06 = function (t) {
      if (!t0) t0 = t;
      var m = Math.floor((t - t0) / .6) % 19, pa = Math.floor(m / 4), pb = m % 4;
      var best = m < 16 ? Math.max(PD.patch[pa * 2][pb * 2], PD.patch[pa * 2][pb * 2 + 1], PD.patch[pa * 2 + 1][pb * 2], PD.patch[pa * 2 + 1][pb * 2 + 1]) : null;
      ic.forEach(function (c, q) {
        var r = Math.floor(q / 8), cc = q % 8, inw = m < 16 && Math.floor(r / 2) === pa && Math.floor(cc / 2) === pb;
        c.classList.toggle('win', inw); c.classList.toggle('max', inw && PD.patch[r][cc] === best);
      });
      qc.forEach(function (c, q) { c.classList.toggle('in', q <= m); c.classList.toggle('now', q === m); });
    };
  })();

  /* ---------------- P7 the four levels of the encoder ---------------- */
  (function () {
    var rep = $('#encRep'), x = 0, cin = [3].concat(L.map(function (l) { return l.c; }));
    L.forEach(function (l, i) {
      var vh = [170, 120, 86, 60, 42][i], vw = vh * 4 / 3, n = 2 + i, off = 8, html = '';
      for (var k = n - 1; k >= 0; k--) html += '<img src="' + G + 'fmap_' + i + '_' + (k % 4) + '.png" alt="" style="width:' + vw + 'px;height:' + vh + 'px;left:' + (k * off) + 'px;bottom:' + (k * off) + 'px">';
      var tw = vw + (n - 1) * off, el = document.createElement('div');
      el.className = 'slab'; el.style.left = x + 'px'; el.style.setProperty('--d', (i * .25) + 's');
      el.innerHTML = '<div class="stk" style="width:' + tw + 'px;height:' + (vh + (n - 1) * off) + 'px">' + html + '</div>';
      rep.appendChild(el); x += tw + 26;
    });
    $('#lvTab').innerHTML = '<tr><th></th>' + L.map(function (l, i) { return '<th>' + (i < 4 ? 'nivel ' + (i + 1) : 'fondo') + '</th>'; }).join('') + '</tr>' +
      '<tr><td>tamaño</td>' + L.map(function (l) { return '<td>' + l.h + ' × ' + l.w + '</td>'; }).join('') + '</tr>' +
      '<tr><td>filtros</td>' + L.map(function (l) { return '<td>' + l.c + '</td>'; }).join('') + '</tr>' +
      '<tr><td>pesos por filtro</td>' + L.map(function (l, i) { return '<td>3×3×' + cin[i] + ' = ' + fmt(9 * cin[i]) + '</td>'; }).join('') + '</tr>';
  })();

  /* ---------------- P8 bottleneck: one position summarises 16 x 16 pixels ---------------- */
  (function () {
    var last = L[L.length - 1], hl = $('#btHl');
    if (last) set('btCell', (INP[0] / last.h) + ' × ' + (INP[1] / last.w));
    var cells = [8, 9, 5, 0, 3].map(function (c) {
      var idx = []; GRID.cls.forEach(function (k, i) { if (k === c) idx.push(i); });
      if (!idx.length) return null;
      var i = idx[idx.length >> 1]; return [Math.floor(Math.floor(i / GRID.w) / 2), Math.floor((i % GRID.w) / 2)];
    }).filter(Boolean);
    var t0 = 0;
    SH.p08 = function () { t0 = 0; };
    TICK.p08 = function (t) {
      if (!t0) t0 = t;
      var c = cells[Math.floor((t - t0) / 1.8) % cells.length];
      if (c) { hl.style.left = (c[1] / 16 * 100) + '%'; hl.style.top = (c[0] / 12 * 100) + '%'; }
    };
  })();

  /* ---------------- P9 transposed convolution: toy 2 x 2 -> 4 x 4 ---------------- */
  (function () {
    var IN = [[1, 3], [2, 4]], K = [[1, .5], [.5, .25]], wrap = $('#p09 .toy');
    function cell(v, mx) { return '<i style="background:' + mixc('#D9DEEA', '#2EC4B6', clamp(v / mx, 0, 1)) + '">' + (Math.round(v * 100) / 100) + '</i>'; }
    $('.gin', wrap).innerHTML = IN.map(function (r) { return r.map(function (v) { return cell(v, 4); }).join(''); }).join('');
    $('.gk', wrap).innerHTML = K.map(function (r) { return r.map(function (v) { return cell(v, 1); }).join(''); }).join('');
    var out = ''; for (var i = 0; i < 4; i++) for (var j = 0; j < 4; j++) out += cell(IN[i >> 1][j >> 1] * K[i % 2][j % 2], 4);
    $('.gout', wrap).innerHTML = out;
    var ins = $$('.gin i', wrap), outs = $$('.gout i', wrap), t0 = 0;
    SH.p09 = function () { t0 = 0; };
    TICK.p09 = function (t) {
      if (!t0) t0 = t;
      var per = .9, cyc = (t - t0) % (4 * per + 2), k = Math.floor(cyc / per);
      ins.forEach(function (c, q) { c.classList.toggle('hot', q === k); });
      outs.forEach(function (c, q) { var src = (Math.floor(q / 4) >> 1) * 2 + ((q % 4) >> 1); c.classList.toggle('in', src < k || (src === k && cyc - k * per > .25)); });
      set('tcEq', k < 4 ? IN[k >> 1][k % 2] + ' × kernel' : '');
    };
  })();

  /* ---------------- P12 up the remaining levels ---------------- */
  (function () {
    var lv = L.slice(0, 4).reverse();
    $('#decRep').innerHTML = lv.map(function (l, k) {
      var src = k === 0 ? 'step_dec4.png' : k === 3 ? 'step_dec1.png' : null, vh = [80, 120, 170, 230][k], vw = vh * 4 / 3;
      return (k ? '<b class="op">→</b>' : '') + '<figure style="width:' + vw + 'px">' + (src ? '<img class="pix" src="' + G + src + '" alt="" style="height:' + vh + 'px">' :
        '<div class="ghost" style="height:' + vh + 'px"></div>') + '<figcaption>' + l.h + ' × ' + l.w + ' × ' + l.c + '</figcaption></figure>';
    }).join('');
  })();

  /* ---------------- P13 1x1 conv: 16 channels of one pixel -> 11 class scores ---------------- */
  (function () {
    var s = $('#c11Svg'), ys = [], yo = [], lines = [];
    for (var i = 0; i < 16; i++) {
      var y = 20 + i * 31; ys.push(y + 12);
      svg('rect', { x: 60, y: y, width: 24, height: 24, rx: 5, fill: mixc('#1B1F33', '#FFD23F', .2 + .8 * Math.abs(Math.sin(i * 1.7))) }, s);
    }
    for (i = 0; i < 11; i++) yo.push(30 + i * 44 + 12);
    ys.forEach(function (a) { yo.forEach(function (b, k) { lines.push(svg('line', { x1: 88, y1: a, x2: 700, y2: b, stroke: col(k), 'stroke-opacity': .12, 'stroke-width': 1.4 }, s)); }); });
    yo.forEach(function (y, k) {
      svg('rect', { x: 704, y: y - 12, width: 24, height: 24, rx: 5, fill: col(k) }, s);
      svg('text', { x: 742, y: y + 8, 'font-size': 22, fill: '#F1F3F8' }, s).textContent = cname(k);
    });
    svg('text', { x: 72, y: 548, 'text-anchor': 'middle', 'font-size': 20, fill: '#AEB4C8' }, s).textContent = '16 canales de 1 píxel';
    svg('text', { x: 400, y: 548, 'text-anchor': 'middle', 'font-size': 20, fill: '#AEB4C8' }, s).textContent = '16 × 11 pesos + 11 sesgos (se entrenan)';
    var t0 = 0;
    SH.p13 = function () { t0 = 0; };
    TICK.p13 = function (t) {
      if (!t0) t0 = t;
      var e = t - t0;
      lines.forEach(function (l, q) { l.setAttribute('stroke-opacity', (.06 + .3 * Math.max(0, Math.sin(e * 2 - (q % 11) * .5))).toFixed(2)); });
    };
  })();

  /* ---------------- P14 softmax: per-pixel probabilities (hover) + probability maps ---------------- */
  (function () {
    var P = D.probs, pr = $('#smProbe'), cur = $('#smCur'), bars = $('#smBars'), manual = false;
    var cw = 620 / (P ? P.w : 32), chh = 465 / (P ? P.h : 24);
    cur.style.width = cw + 'px'; cur.style.height = chh + 'px';
    function show(r, c) {
      if (!P) return;
      r = clamp(r, 0, P.h - 1); c = clamp(c, 0, P.w - 1);
      cur.style.left = (c * cw) + 'px'; cur.style.top = (r * chh) + 'px';
      bars.innerHTML = P.top[r * P.w + c].map(function (p) {
        return '<div>' + cname(p[0]) + '<em>' + fmt(p[1] * 100, 0) + '%</em><i style="background:' + col(p[0]) + ';transform:scaleX(' + Math.max(.02, p[1]).toFixed(3) + ')"></i></div>';
      }).join('');
    }
    pr.addEventListener('pointermove', function (e) {
      manual = true; var b = pr.getBoundingClientRect();
      show(Math.floor((e.clientY - b.top) / b.height * P.h), Math.floor((e.clientX - b.left) / b.width * P.w));
    });
    pr.addEventListener('pointerleave', function () { manual = false; });
    $('#smMaps').innerHTML = (D.prob_maps || []).map(function (m) {
      return '<figure><img src="' + G + 'hero.jpg" alt=""><img class="ov" src="' + G + 'step_prob_' + m.cls + '.png" alt=""><figcaption><i style="background:' + col(m.cls) + '"></i>p(' + cname(m.cls) + ')</figcaption></figure>';
    }).join('');
    var stops = [8, 9, 3, 2, 5, 0, 10, 1].map(function (c) { var idx = []; GRID.cls.forEach(function (k, i) { if (k === c) idx.push(i); }); return idx.length ? idx[idx.length >> 1] : -1; }).filter(function (i) { return i >= 0; });
    var t0 = 0, last = -1;
    SH.p14 = function () { t0 = 0; last = -1; };
    TICK.p14 = function (t) {
      if (!t0) t0 = t;
      if (manual || !stops.length || !P) return;
      var k = Math.floor((t - t0) / 1.6) % stops.length;
      if (k !== last) { last = k; var i = stops[k]; show(Math.floor(i / GRID.w) * (P.h / GRID.h), (i % GRID.w) * (P.w / GRID.w)); }
    };
  })();

  /* ---------------- P16 what is trained and what is not (real parameter counts) ---------------- */
  (function () {
    var Pd = D.params_detail || {};
    function g(k) { return Pd[k] || 0; }
    var yes = [
      ['Conv 3 × 3', g('conv3x3.kernel.train'), '#3A86FF'],
      ['Conv transpuesta', g('Conv2DTranspose.kernel.train') + g('Conv2DTranspose.bias.train'), '#2EC4B6'],
      ['BatchNorm γ, β', g('BatchNormalization.gamma.train') + g('BatchNormalization.beta.train'), '#8C7AA9'],
      ['Conv 1 × 1', g('conv1x1.kernel.train') + g('conv1x1.bias.train'), '#FFD23F']];
    var tot = yes.reduce(function (s, r) { return s + r[1]; }, 0);
    $('#trYes').innerHTML = yes.map(function (r) {
      return '<div class="pr2"><span>' + r[0] + '</span><b>' + (tot ? fmt(r[1]) : '—') + '</b><i style="background:' + r[2] + ';--v:' + (tot ? Math.max(.004, r[1] / tot) : 0).toFixed(4) + '"></i></div>';
    }).join('');
    set('trYesT', tot ? 'Total: ' + fmt(tot) : '');
    var no = [
      ['BatchNorm: media y varianza', fmt(g('BatchNormalization.moving_mean.fixed') + g('BatchNormalization.moving_variance.fixed'))],
      ['Max pooling', '0'], ['ReLU', '0'], ['Concatenar', '0'], ['Softmax y argmax', '0']];
    $('#trNo').innerHTML = no.map(function (r) { return '<div class="pr3"><span>' + r[0] + '</span><b>' + r[1] + '</b></div>'; }).join('');
    if (D.epochs) set('hpEp', D.epochs);
  })();

  /* ---------------- extra motion for the step slides ---------------- */
  (function () {
    /* P1: the 75 numbers light up one by one */
    var nums = $$('#inNums i'), t1 = 0;
    SH.p01 = function () { t1 = 0; };
    TICK.p01 = function (t) { if (!t1) t1 = t; var k = Math.floor(((t - t1) % 9) / .08); nums.forEach(function (c, q) { var i = q % 25; c.classList.toggle('lit', i * 3 + Math.floor(q / 25) <= k); }); };
    /* P3: each filter lights up with its own map */
    var kc = $$('#k16 .kc'), hl = $('#k16hl'), t3 = 0;
    SH.p03 = function () { t3 = 0; };
    TICK.p03 = function (t) {
      if (!t3) t3 = t; var k = Math.floor((t - t3) / .9) % 16;
      kc.forEach(function (c, q) { c.classList.toggle('on', q === k); });
      if (hl) { hl.style.left = (k % 4) * 25 + '%'; hl.style.top = Math.floor(k / 4) * 25 + '%'; }
    };
    /* P4: ReLU sweeps across the map; a dot rides the ReLU curve */
    var sw = $('#p04 .sw'), dot = $('#reluDot'), t4 = 0;
    SH.p04 = function () { t4 = 0; };
    TICK.p04 = function (t) {
      if (!t4) t4 = t; var u = .5 + .5 * Math.sin((t - t4) * .8 - 1.57);
      sw.style.setProperty('--x', (u * 100).toFixed(1) + '%');
      var x = 10 + 210 * (.5 + .5 * Math.sin((t - t4) * 1.6)), y = x < 120 ? 170 : 170 - (x - 120) * 1.2;
      dot.setAttribute('cx', x.toFixed(1)); dot.setAttribute('cy', y.toFixed(1)); dot.setAttribute('fill', x < 120 ? '#3A86FF' : '#FF4D6D');
    };
    /* P15: wipe between prediction and real mask */
    var sw2 = $('#p15 .sw2'), t15 = 0;
    SH.p15 = function () { t15 = 0; };
    TICK.p15 = function (t) { if (!t15) t15 = t; sw2.style.setProperty('--x', ((.5 + .45 * Math.sin((t - t15) * .7)) * 100).toFixed(1) + '%'); };
  })();

  /* ---------------- 07 skips: before/after slider + test mIoU ---------------- */
  (function () {
    var cmp = $('#skCmp'), manual = false, t0 = 0, T = D.test || {};
    function setX(p) { cmp.style.setProperty('--x', (p * 100).toFixed(2) + '%'); }
    function fromEvent(e) { var r = cmp.getBoundingClientRect(); return clamp((e.clientX - r.left) / r.width, 0, 1); }
    cmp.addEventListener('pointerdown', function (e) { manual = true; cmp.setPointerCapture(e.pointerId); setX(fromEvent(e)); });
    cmp.addEventListener('pointermove', function (e) { if (e.buttons) setX(fromEvent(e)); });
    SH.skips = function (sc, step) { if (step === 0) { manual = false; t0 = 0; countTo($('#skA'), T.miou_plain || 0, 2); countTo($('#skB'), T.miou_unet || 0, 2); } };
    TICK.skips = function (t) { if (!t0) t0 = t; if (!manual) setX(.5 + .38 * Math.sin((t - t0) * .9)); };
  })();

  /* ---------------- 08 cómo se entrena: IoU animation ---------------- */
  (function () {
    var s = $('#sdIou');
    svg('rect', { x: 60, y: 40, width: 140, height: 120, rx: 10, fill: '#3A86FF55', stroke: '#3A86FF', 'stroke-width': 4 }, s);
    var B = svg('rect', { y: 50, width: 140, height: 120, rx: 10, fill: '#FF4D6D44', stroke: '#FF4D6D', 'stroke-width': 4 }, s);
    var I = svg('rect', { y: 50, height: 110, fill: '#FFD23F' }, s);
    var ta = svg('text', { x: 70, y: 30, 'font-size': 22, 'font-weight': 700, fill: '#3A86FF' }, s); ta.textContent = 'A';
    var tb = svg('text', { y: 196, 'font-size': 22, 'font-weight': 700, fill: '#FF4D6D' }, s); tb.textContent = 'B';
    TICK.salida = function (t) {
      var bx = 60 + 110 * (.5 + .5 * Math.sin(t * .8)), ix0 = Math.max(60, bx), ix1 = Math.min(200, bx + 140), iw = Math.max(0, ix1 - ix0);
      B.setAttribute('x', bx); tb.setAttribute('x', bx + 10); I.setAttribute('x', ix0); I.setAttribute('width', iw);
      var inter = iw * 110; set('sdIouV', (inter / (140 * 120 * 2 - inter)).toFixed(2));
    };
  })();

  /* ---------------- 09 entrenamiento: prediction after every epoch + validation mIoU ---------------- */
  (function () {
    var SA = D.sprite_unet, SP = D.sprite_plain, H = D.history || {}, a = $('#trA'), b = $('#trB'), cv = $('#trCurve');
    function init(el, m) { if (!m) return; el.style.backgroundImage = 'url(' + G + m.file + ')'; el.style.backgroundSize = '100% ' + m.frames * 100 + '%'; }
    init(a, SP); init(b, SA);
    var n = SA ? SA.frames : 1, Wd = 1664, Hh = 250, x0 = 90, x1 = Wd - 20, y0 = Hh - 46, y1 = 10;
    var mu = (H.unet && H.unet.miou) || [], mp = (H.plain && H.plain.miou) || [];
    var mx = Math.max.apply(null, mu.concat(mp, [.1]));
    var X = function (e) { return x0 + e / Math.max(1, n - 1) * (x1 - x0); }, Y = function (v) { return y0 - v / mx * (y0 - y1); };
    svg('line', { x1: x0, x2: x1, y1: y0, y2: y0, stroke: '#B9C1D3', 'stroke-width': 2 }, cv);
    [0, .25, .5, .75, 1].forEach(function (f) {
      var v = mx * f, yy = Y(v);
      svg('line', { x1: x0, x2: x1, y1: yy, y2: yy, stroke: '#DDE2EC', 'stroke-width': 1 }, cv);
      var tt = svg('text', { x: x0 - 12, y: yy + 6, 'text-anchor': 'end' }, cv); tt.textContent = v.toFixed(2);
    });
    var tl = svg('text', { x: x0, y: Hh - 8 }, cv); tl.textContent = 'época 0';
    var tr = svg('text', { x: x1, y: Hh - 8, 'text-anchor': 'end' }, cv); tr.textContent = 'época ' + (n - 1);
    var ty = svg('text', { x: x0 + 14, y: 26 }, cv); ty.textContent = 'mIoU en validación';
    [[mp, '#FF4D6D'], [mu, '#3A86FF']].forEach(function (p) {
      var d = ''; p[0].forEach(function (v, i) { d += (i ? 'L' : 'M') + X(i + 1).toFixed(1) + ' ' + Y(v).toFixed(1); });
      svg('path', { d: d, fill: 'none', stroke: p[1], 'stroke-width': 5, 'stroke-linejoin': 'round' }, cv);
    });
    var cl = svg('line', { y1: y1, y2: y0, stroke: '#0F1220', 'stroke-width': 3 }, cv);
    var da = svg('circle', { r: 9, fill: '#FF4D6D' }, cv), db = svg('circle', { r: 9, fill: '#3A86FF' }, cv);
    function frame(f) {
      f = clamp(Math.round(f), 0, n - 1);
      [a, b].forEach(function (el) { el.style.backgroundPosition = '0 ' + (n > 1 ? f / (n - 1) * 100 : 0) + '%'; });
      set('trEp', f);
      cl.setAttribute('x1', X(f)); cl.setAttribute('x2', X(f));
      da.setAttribute('cx', X(f)); da.setAttribute('cy', Y(f ? mp[f - 1] || 0 : 0)); db.setAttribute('cx', X(f)); db.setAttribute('cy', Y(f ? mu[f - 1] || 0 : 0));
    }
    var manual = null, t0 = 0;
    function pick(e) { var r = cv.getBoundingClientRect(), xx = (e.clientX - r.left) / r.width * Wd; manual = (xx - x0) / (x1 - x0) * (n - 1); frame(manual); }
    cv.addEventListener('pointerdown', function (e) { cv.setPointerCapture(e.pointerId); pick(e); });
    cv.addEventListener('pointermove', function (e) { if (e.buttons) pick(e); });
    SH.entrena = function () { manual = null; t0 = 0; };
    TICK.entrena = function (t) { if (manual !== null) return; if (!t0) t0 = t; frame(Math.min(n - 1, ((t - t0) * 5) % (n + 10))); };
    frame(0);
  })();

  /* ---------------- 10 resultados: IoU per class on test ---------------- */
  (function () {
    var T = D.test || {}, F = D.pixel_freq || [], wrap = $('#rsBars'), rare = [2, 6, 9, 10];
    F.map(function (f, i) { return i; }).sort(function (a, b) { return F[b] - F[a]; }).forEach(function (c, j) {
      var r = document.createElement('div'), vp = (T.iou_plain || [])[c] || 0, vu = (T.iou_unet || [])[c] || 0;
      r.className = 'rs-row' + (rare.indexOf(c) >= 0 ? ' rare' : '');
      r.innerHTML = '<div class="nm"><i style="background:' + col(c) + '"></i>' + cname(c) + '<small>' + fmt(F[c] * 100, 1) + '%</small></div>' +
        '<div class="tk"><i style="background:#FF4D6D;--v:' + vp.toFixed(3) + ';--d:' + (j * .06) + 's"></i><i style="background:#3A86FF;--v:' + vu.toFixed(3) + ';--d:' + (j * .06 + .1) + 's"></i></div>' +
        '<b>' + vu.toFixed(2) + '</b>';
      wrap.appendChild(r);
    });
  })();

  /* ---------------- 11 variantes: architecture diagrams + mini demos ---------------- */
  (function () {
    function kit(s) {
      arrowDefs(s, s.id);
      return {
        box: function (x, y, w, h, fill, label, sub, dark) {
          svg('rect', { x: x, y: y, width: w, height: h, rx: 8, fill: fill, stroke: dark ? 'none' : '#B9C1D3', 'stroke-width': 1.5 }, s);
          if (label) { var t = svg('text', { x: x + w / 2, y: y + h / 2 + (sub ? -4 : 7), 'text-anchor': 'middle', 'font-size': 19, 'font-weight': 700, fill: dark ? '#fff' : '#0F1220' }, s); t.textContent = label; }
          if (sub) { var u = svg('text', { x: x + w / 2, y: y + h / 2 + 20, 'text-anchor': 'middle', 'font-size': 15, fill: dark ? '#DDE2EC' : '#4B5163' }, s); u.textContent = sub; }
        },
        arrow: function (x1, y1, x2, y2, k, dash, curve) {
          var d = curve ? 'M' + x1 + ' ' + y1 + ' Q' + curve[0] + ' ' + curve[1] + ' ' + x2 + ' ' + y2 : 'M' + x1 + ' ' + y1 + ' L' + x2 + ' ' + y2;
          svg('path', { d: d, fill: 'none', stroke: C[k], 'stroke-width': 4, 'stroke-linecap': 'round', 'stroke-dasharray': dash ? '10 8' : 'none', 'marker-end': 'url(#' + s.id + '-' + k + ')' }, s);
        },
        text: function (x, y, txt, size, color, anchor) { var t = svg('text', { x: x, y: y, 'text-anchor': anchor || 'middle', 'font-size': size || 16, fill: color || '#4B5163' }, s); t.textContent = txt; },
        img: function (x, y, w, h, src, op) { svg('image', { href: G + src, x: x, y: y, width: w, height: h, preserveAspectRatio: 'none', opacity: op || 1 }, s); },
        circ: function (x, y, txt) { svg('circle', { cx: x, cy: y, r: 18, fill: '#FFD23F' }, s); var t = svg('text', { x: x, y: y + 7, 'text-anchor': 'middle', 'font-size': 22, 'font-weight': 700, fill: '#0F1220' }, s); t.textContent = txt; }
      };
    }
    /* FCN-8s */
    (function () {
      var k = kit($('#adFcn')), hs = [200, 160, 124, 96, 74], xs = [170, 320, 470, 620, 770];
      k.img(0, 110, 130, 98, 'hero.jpg'); k.arrow(132, 160, 166, 160, 'conv');
      xs.forEach(function (x, i) {
        k.box(x, 160 - hs[i] / 2, 110, hs[i], '#C9D6F2', 'conv + pool', '1/' + Math.pow(2, i + 1));
        if (i < 4) k.arrow(x + 112, 160, xs[i + 1] - 4, 160, 'pool');
      });
      k.arrow(882, 160, 936, 160, 'out'); k.box(940, 120, 120, 80, '#FFD23F', 'conv 1×1', 'puntajes 1/32');
      k.arrow(1062, 160, 1110, 160, 'up'); k.text(1086, 146, '× 2', 15, '#0F1220');
      k.circ(1130, 160, '+'); k.arrow(675, 210, 1128, 180, 'copy', true, [900, 300]); k.text(905, 292, 'puntajes de pool4 (1/16)', 15);
      k.arrow(1150, 160, 1200, 160, 'up'); k.text(1175, 146, '× 2', 15, '#0F1220');
      k.circ(1220, 160, '+'); k.arrow(525, 224, 1218, 180, 'copy', true, [870, 340]); k.text(860, 326, 'puntajes de pool3 (1/8)', 15);
      k.arrow(1240, 160, 1384, 160, 'up'); k.text(1312, 146, '× 8', 15, '#0F1220');
      k.img(1390, 110, 130, 98, 'hero.jpg'); k.img(1390, 110, 130, 98, 'hero_coarse.png', .8);
      k.text(1455, 232, 'salida', 15);
    })();
    /* SegNet */
    (function () {
      var k = kit($('#adSeg')), hs = [200, 156, 118, 88, 66];
      k.img(0, 116, 120, 90, 'hero.jpg'); k.arrow(122, 161, 146, 161, 'conv');
      hs.forEach(function (h, i) { k.box(150 + i * 128, 161 - h / 2, 96, h, '#C9D6F2', 'conv', '+ pool'); if (i < 4) k.arrow(248 + i * 128, 161, 274 + i * 128, 161, 'pool'); });
      k.arrow(760, 161, 800, 161, 'up');
      hs.slice().reverse().forEach(function (h, i) {
        var x = 804 + i * 128; k.box(x, 161 - h / 2, 96, h, i === 0 ? '#2EC4B6' : '#B9EEE8', 'unpool', '+ conv');
        if (i < 4) k.arrow(x + 98, 161, x + 124, 161, 'up');
        var ei = 4 - i, ex = 150 + ei * 128 + 48;
        k.arrow(ex, 161 - hs[ei] / 2 - 4, x + 48, 161 - h / 2 - 6, 'copy', true, [(ex + x + 48) / 2, 161 - hs[ei] / 2 - 70 - ei * 12]);
      });
      k.text(832, 30, 'índices de max pooling (dónde estaba cada máximo)', 17, '#0F1220');
      k.arrow(1414, 161, 1444, 161, 'out'); k.img(1450, 116, 120, 90, 'hero.jpg'); k.img(1450, 116, 120, 90, 'hero_plain.png', .8);
      k.text(1510, 232, 'softmax', 15);
    })();
    /* DeepLab v3+ */
    (function () {
      var k = kit($('#adDl'));
      k.img(0, 110, 120, 90, 'hero.jpg'); k.arrow(122, 155, 146, 155, 'conv');
      [[150, 180, '1/2'], [270, 150, '1/4'], [390, 120, '1/8'], [510, 96, '1/16'], [630, 96, '1/16']].forEach(function (b, i) {
        k.box(b[0], 155 - b[1] / 2, 100, b[1], i === 4 ? '#8C7AA9' : '#C9D6F2', i === 4 ? 'dilatada' : 'conv', b[2], i === 4);
        if (i < 4) k.arrow(b[0] + 102, 155, b[0] + 116, 155, i < 3 ? 'pool' : 'conv');
      });
      ['1×1', 'rate 6', 'rate 12', 'rate 18', 'promedio'].forEach(function (t, i) { var y = 18 + i * 56; k.arrow(732, 155, 776, y + 22, 'conv'); k.box(780, y, 120, 44, '#E3E8F2', t); k.arrow(902, y + 22, 946, 155, 'conv'); });
      k.text(840, 312, 'ASPP', 18, '#0F1220');
      k.box(950, 125, 110, 60, '#FFD23F', 'concat', '+ 1×1');
      k.arrow(1062, 155, 1112, 155, 'up'); k.text(1087, 141, '× 4', 15, '#0F1220');
      k.box(1116, 120, 100, 70, '#B9EEE8', 'concat');
      k.arrow(320, 232, 1162, 194, 'copy', true, [740, 330]); k.text(560, 318, 'rasgos de 1/4', 15);
      k.arrow(1218, 155, 1246, 155, 'conv'); k.box(1250, 120, 90, 70, '#C9D6F2', 'conv 3×3');
      k.arrow(1342, 155, 1396, 155, 'up'); k.text(1369, 141, '× 4', 15, '#0F1220');
      k.img(1400, 110, 120, 90, 'hero.jpg'); k.img(1400, 110, 120, 90, 'hero_unet.png', .8);
    })();
    /* Mask R-CNN */
    (function () {
      var k = kit($('#adMr'));
      k.img(0, 110, 120, 90, 'hero.jpg'); k.arrow(122, 155, 146, 155, 'conv');
      k.box(150, 60, 150, 190, '#C9D6F2', 'ResNet', 'backbone');
      [[330, 50, 90], [330, 150, 70], [330, 230, 50]].forEach(function (b) { k.box(b[0], b[1], 110, b[2] * .8, '#E3E8F2', 'FPN'); });
      k.arrow(302, 155, 326, 155, 'conv');
      k.arrow(442, 155, 486, 155, 'conv'); k.box(490, 105, 150, 100, '#FF4D6D', 'RPN', 'propone cajas', true);
      k.arrow(642, 155, 686, 155, 'conv'); k.box(690, 105, 150, 100, '#8C7AA9', 'RoIAlign', 'recorta 14 × 14', true);
      [['clase', 'auto, peatón…', 40], ['caja', 'ajusta x, y, w, h', 130], ['máscara', 'mini FCN 28 × 28', 220]].forEach(function (h) {
        k.arrow(842, 155, 896, h[2] + 32, 'conv'); k.box(900, h[2], 190, 64, h[0] === 'máscara' ? '#2EC4B6' : '#E3E8F2', h[0], h[1]);
      });
      k.arrow(1092, 252, 1196, 170, 'out');
      k.img(1200, 110, 120, 90, 'hero.jpg'); k.img(1200, 110, 120, 90, 'hero_instances.png', .9);
      k.text(1260, 228, 'una máscara por objeto', 15);
    })();
    /* SAM */
    (function () {
      var s = $('#adSam'), k = kit(s);
      k.img(0, 60, 140, 105, 'hero.jpg'); k.arrow(142, 112, 186, 112, 'conv');
      k.box(190, 40, 260, 144, '#3A86FF', 'encoder de imagen', 'ViT · una vez por foto', true);
      k.arrow(452, 112, 496, 112, 'conv'); k.box(500, 62, 140, 100, '#C9D6F2', 'embedding', '64 × 64');
      svg('circle', { cx: 70, cy: 258, r: 14, fill: '#FFD23F', stroke: '#0F1220', 'stroke-width': 3 }, s);
      k.text(70, 305, 'un clic', 15); k.arrow(90, 258, 186, 258, 'conv');
      k.box(190, 222, 260, 72, '#E3E8F2', 'encoder de la pista', 'punto, caja o texto');
      k.arrow(642, 112, 726, 160, 'conv'); k.arrow(452, 258, 726, 200, 'conv');
      k.box(730, 120, 220, 120, '#2EC4B6', 'decoder de máscara', 'transformer ligero');
      [0, 1, 2].forEach(function (i) { var y = 30 + i * 100; k.arrow(952, 180, 1046, y + 40, 'out'); k.img(1050, y, 104, 78, 'hero.jpg'); k.img(1050, y, 104, 78, 'hero_things.png', .85); k.text(1170, y + 46, 'máscara ' + (i + 1) + ' + calificación', 15, '#4B5163', 'start'); });
    })();
    /* mini demos */
    var dil = $('#msDil'), cells = [];
    for (var i = 0; i < 81; i++) { var e = document.createElement('i'); dil.appendChild(e); cells.push(e); }
    function rate(r) {
      cells.forEach(function (c, i) {
        var y = Math.floor(i / 9) - 4, x = i % 9 - 4, tap = (y % r === 0 && x % r === 0 && Math.abs(y) <= r && Math.abs(x) <= r);
        c.className = (x === 0 && y === 0) ? 'c' : tap ? 'k' : (Math.abs(x) <= r && Math.abs(y) <= r ? 'rf' : '');
      });
      set('msRate', 'rate = ' + r + ' · ve ' + (2 * r + 1) + ' × ' + (2 * r + 1) + ' con 9 pesos');
    }
    rate(1);
    var bx = $('#msBx'), B = D.boxes || [];
    B.forEach(function (b) { svg('rect', { x: b.x * 480, y: b.y * 360, width: b.w * 480, height: b.h * 360, rx: 3, stroke: col(b.cls) }, bx); });
    var mr = $('#var .mr'), sam = $('#var .sam'), one = $('#msOne'), clk = $('#msClk');
    var car = B.filter(function (b) { return b.cls === 8; }).sort(function (p, q) { return q.w * q.h - p.w * p.h; })[0] || B[0];
    var px = car ? car.x + car.w / 2 : .5, py = car ? car.y + car.h * .6 : .5;
    clk.style.left = (px * 100) + '%'; clk.style.top = (py * 100) + '%';
    var ins = img(G + 'hero_instances.png');
    ins.onload = function () {
      try {
        var c = document.createElement('canvas'); c.width = ins.naturalWidth; c.height = ins.naturalHeight;
        var g = c.getContext('2d'); g.drawImage(ins, 0, 0);
        var d = g.getImageData(0, 0, c.width, c.height), q = (Math.floor(py * c.height) * c.width + Math.floor(px * c.width)) * 4;
        var r0 = d.data[q], g0 = d.data[q + 1], b0 = d.data[q + 2];
        for (var k = 0; k < d.data.length; k += 4) {
          if (d.data[k] !== r0 || d.data[k + 1] !== g0 || d.data[k + 2] !== b0 || !d.data[k + 3]) d.data[k + 3] = 0;
          else { d.data[k] = 255; d.data[k + 1] = 210; d.data[k + 2] = 63; }
        }
        if (d.data[q + 3]) { g.putImageData(d, 0, 0); one.src = c.toDataURL(); }
      } catch (err) { /* file:// pages cannot read pixels; keep every object */ }
    };
    var sc = $('#var'), tabs = $$('.tab', sc), vps = $$('.vp', sc), cur = 0, t0 = 0;
    tabs.forEach(function (b) { b.addEventListener('click', function () { if (window.DECK) DECK.setStep(+b.dataset.i); }); });
    SH.var = function (s, step) {
      cur = step; t0 = 0;
      tabs.forEach(function (b, i) { b.classList.toggle('on', i === step); });
      vps.forEach(function (p, i) { p.classList.toggle('on', i === step); });
    };
    TICK.var = function (t) {
      if (!t0) t0 = t;
      var e = t - t0;
      if (cur === 2) rate([1, 2, 3][Math.floor(e / 1.8) % 3]);
      if (cur === 3) mr.classList.toggle('masks', (e % 4) > 2);
      if (cur === 4) {
        var s2 = e % 5;
        sam.classList.toggle('pt', s2 > .6); sam.classList.toggle('go', s2 > 1.2);
        one.style.clipPath = 'circle(' + (s2 > 1.2 ? 75 : 0) + '% at ' + px * 100 + '% ' + py * 100 + '%)';
      }
    };
  })();

  /* ---------------- 12 fin: the mask as a slowly breathing grid ---------------- */
  (function () {
    var cv = $('#finCv'), g = cv.getContext('2d');
    TICK.fin = function (t) {
      g.clearRect(0, 0, 1920, 1080);
      GRID.cls.forEach(function (c, i) {
        if (c === VOID) return;
        var r = Math.floor(i / GRID.w), cc = i % GRID.w, ph = Math.sin(t * 1.2 - cc * .25 - r * .15);
        var s = (CELL - 8) * (.55 + .35 * (ph * .5 + .5));
        g.fillStyle = col(c); g.globalAlpha = .35 + .4 * (ph * .5 + .5);
        g.fillRect(cc * CELL + (CELL - s) / 2, r * CELL + OFFY + (CELL - s) / 2, s, s);
      });
      g.globalAlpha = 1;
    };
  })();

  /* KaTeX formulas (a plain-text version shows if the CDN is unreachable) */
  $$('[data-tex]').forEach(function (el) {
    if (!window.katex) return;
    try { katex.render(el.getAttribute('data-tex'), el, { displayMode: true, throwOnError: false }); el.classList.add('tex-ok'); } catch (e) { }
  });

  /* ================= engine ================= */
  var stage = $('#stage');
  var scenes = $$('.scene', stage);
  var cur = 0, step = 0, reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;

  function countTo(el, to, dec) {
    if (to == null || !el) return;
    function put(v) { el.textContent = fmt(v, dec); }
    if (reduce) { put(to); return; }
    var t0 = performance.now();
    (function f(t) { var p = Math.min(1, (t - t0) / 1300), e = 1 - Math.pow(1 - p, 3); put(to * e); if (p < 1) requestAnimationFrame(f); })(t0);
  }
  function maxStep(sc) { var m = 0; $$('[data-s]', sc).forEach(function (e) { m = Math.max(m, +e.getAttribute('data-s')); }); return m; }
  function clearScene(sc) {
    sc.removeAttribute('data-step');
    for (var q = 1; q <= 8; q++) sc.classList.remove('st' + q);
    if (SH[sc.id] && SH[sc.id].reset) SH[sc.id].reset(sc);
    $$('[data-s].in', sc).forEach(function (e) { e.classList.remove('in'); });
  }
  function render() {
    scenes.forEach(function (sc, i) { sc.classList.toggle('on', i === cur); });
    var sc = scenes[cur];
    sc.setAttribute('data-step', step);
    for (var q = 1; q <= 8; q++) sc.classList.toggle('st' + q, step >= q);
    if (SH[sc.id]) { try { SH[sc.id](sc, step); } catch (err) { console.error(err); } }
    try { mini(sc.id); } catch (err) { console.error(err); }
    $$('[data-s]', sc).forEach(function (e) {
      var on = +e.getAttribute('data-s') <= step;
      if (on && !e.classList.contains('in')) e.classList.add('in');
      else if (!on && e.classList.contains('in')) e.classList.remove('in');
    });
    stage.classList.toggle('isdark', sc.classList.contains('dark'));
    set('pg', (cur + 1 < 10 ? '0' : '') + (cur + 1) + ' / ' + scenes.length);
    var total = 0, done = 0;
    scenes.forEach(function (s, i) { var m = maxStep(s) + 1; total += m; if (i < cur) done += m; else if (i === cur) done += step + 1; });
    $('#prog').style.width = (done / total * 100) + '%';
    $('#hint').style.opacity = (cur === 0 && step === 0) ? .9 : 0;
    try { history.replaceState(null, '', '#' + (cur + 1)); } catch (e) { }
  }
  function go(n, atEnd) {
    if (n < 0 || n >= scenes.length) return;
    var from = cur; cur = n;
    if (from !== n) {
      var old = scenes[from];
      old.classList.add('leaving');
      setTimeout(function () { old.classList.remove('leaving'); if (scenes[cur] !== old) clearScene(old); }, 1500);
    }
    clearScene(scenes[cur]);
    step = atEnd ? maxStep(scenes[cur]) : 0;
    setTimeout(render, 34);
  }
  function next() { var m = maxStep(scenes[cur]); if (step < m) { step++; render(); } else go(cur + 1, false); }
  function prev() { if (step > 0) { step--; render(); } else if (cur > 0) go(cur - 1, true); }

  document.addEventListener('keydown', function (e) {
    if (['ArrowRight', ' ', 'Enter', 'PageDown', 'ArrowDown'].indexOf(e.key) >= 0) { e.preventDefault(); next(); }
    else if (['ArrowLeft', 'PageUp', 'ArrowUp'].indexOf(e.key) >= 0) { e.preventDefault(); prev(); }
    else if (e.key === 'Home') go(0, false);
    else if (e.key === 'End') go(scenes.length - 1, true);
    else if (e.key === 'f') toggleFs();
  });
  document.addEventListener('click', function (e) {
    if (e.target.closest('button, .noadv')) return;
    if (e.clientX < window.innerWidth * .28) prev(); else next();
  });
  var tx = null;
  document.addEventListener('touchstart', function (e) { tx = e.touches[0].clientX; }, { passive: true });
  document.addEventListener('touchend', function (e) {
    if (tx === null) return;
    var dx = e.changedTouches[0].clientX - tx; tx = null;
    if (Math.abs(dx) > 50) { if (dx < 0) next(); else prev(); }
  }, { passive: true });
  function toggleFs() {
    try { if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen().catch(function () { }); } catch (x) { }
  }
  $('#fs').addEventListener('click', function (e) { e.stopPropagation(); toggleFs(); });
  function fit() { var s = Math.min(window.innerWidth / 1920, window.innerHeight / 1080); stage.style.transform = 'translate(-50%,-50%) scale(' + s + ')'; }
  window.addEventListener('resize', fit); fit();

  (function loop(now) {
    var sc = scenes[cur];
    if (!reduce && TICK[sc.id]) { try { TICK[sc.id](now / 1000, sc); } catch (err) { console.error(err); } }
    requestAnimationFrame(loop);
  })(performance.now());

  var h = parseInt((location.hash || '').slice(1), 10);
  if (h >= 1 && h <= scenes.length) cur = h - 1;
  setTimeout(render, 50);
  window.addEventListener('hashchange', function () {
    var n = parseInt((location.hash || '').slice(1), 10) - 1;
    if (n >= 0 && n < scenes.length && n !== cur) go(n, false);
  });
  window.DECK = { next: next, prev: prev, show: function (n) { go(n - 1, true); },
    setStep: function (n) { step = Math.max(0, Math.min(maxStep(scenes[cur]), n)); render(); } };
})();
