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
  function rgb(h) { h = h.replace('#', ''); return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]; }
  function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
  function ez(u) { u = clamp(u, 0, 1); return u < .5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2; }
  function img(src) { var i = new Image(); i.src = src; return i; }
  var HERO = img(G + 'hero.jpg'), HERO_GT = img(G + 'hero_gt.png');
  var GRID = D.cover_grid || { h: 24, w: 32, rgb: [], cls: [] };

  /* cells of the 24 x 32 grid drawn full-bleed on the 1920 x 1080 stage (photo is 4:3, so it is cropped vertically) */
  var CELL = 1920 / GRID.w, OFFY = (1080 - CELL * GRID.h) / 2;
  function centroid(c) {
    var xs = [], ys = [];
    GRID.cls.forEach(function (k, i) { if (k === c) { xs.push(i % GRID.w); ys.push(Math.floor(i / GRID.w)); } });
    if (!xs.length) return null;
    xs.sort(function (a, b) { return a - b; }); ys.sort(function (a, b) { return a - b; });
    return [(xs[xs.length >> 1] + .5) * CELL, (ys[ys.length >> 1] + .5) * CELL + OFFY, xs.length];
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
    [[0, 'Cielo'], [1, 'Edificio'], [5, 'Árbol'], [3, 'Calle'], [8, 'Auto'], [9, 'Peatón'], [10, 'Ciclista']].forEach(function (p, j) {
      var ct = centroid(p[0]);
      if (!ct || ct[2] < 3) return;
      var s = document.createElement('span');
      s.textContent = CL[p[0]] ? CL[p[0]].name : p[1];
      s.style.cssText = 'left:' + clamp(ct[0], 120, 1800) + 'px;top:' + clamp(ct[1], 200, 900) + 'px;background:' + col(p[0]) + ';--d:' + (j * .12) + 's';
      chipWrap.appendChild(s);
    });
    function photo(alpha, zoom) {
      if (!HERO.complete || !HERO.naturalWidth) return;
      var w = 1920 * zoom, h = 1440 * zoom;
      g.globalAlpha = alpha;
      g.drawImage(HERO, (1920 - w) / 2, (1080 - h) / 2, w, h);
      g.globalAlpha = 1;
    }
    function mask(alpha) {
      if (!HERO_GT.complete || !HERO_GT.naturalWidth) return;
      g.globalAlpha = alpha; g.drawImage(HERO_GT, 0, OFFY, 1920, 1440); g.globalAlpha = 1;
    }
    function rr(x, y, s, r) { g.beginPath(); g.roundRect ? g.roundRect(x, y, s, s, r) : g.rect(x, y, s, s); g.fill(); }
    SH.cover = function (s, step) {
      if (step === 1 && !playing) { playing = true; t0 = 0; }
      if (step === 0) { playing = false; sc.classList.remove('chips-on'); }
    };
    SH.cover.reset = function () { playing = false; sc.classList.remove('chips-on'); };
    TICK.cover = function (t) {
      g.clearRect(0, 0, 1920, 1080);
      if (!k0) k0 = t;
      var zoom = 1 + .04 * Math.sin((t - k0) * .15);
      if (!playing) { photo(1, zoom); return; }
      if (!t0) t0 = t;
      var e = t - t0;
      /* 0-1.3 s: the photo breaks into its real pixels; 1.3-3 s: each pixel takes its class colour; 3.4-4.6 s: back to the photo with its mask */
      if (e < 3.4) photo(1 - ez(e / 1.1), 1);
      cells.forEach(function (o) {
        var a = ez((e - o.d) / .7);
        if (a <= 0 || e > 4.8) return;
        var u = ez((e - 1.3 - o.w * 1.4) / .5), fade = 1 - ez((e - 3.6) / 1);
        var gap = 2 + 6 * a * (1 - ez((e - 3.2) / .5)), s = CELL - gap * 2;
        var c0 = o.rgb, c1 = o.cr;
        g.fillStyle = 'rgba(' + Math.round(c0[0] + (c1[0] - c0[0]) * u) + ',' + Math.round(c0[1] + (c1[1] - c0[1]) * u) + ',' +
          Math.round(c0[2] + (c1[2] - c0[2]) * u) + ',' + (a * fade * (o.c === VOID ? 1 - u * .8 : 1)).toFixed(3) + ')';
        var pop = 1 + .25 * Math.sin(Math.PI * clamp((e - 1.3 - o.w * 1.4) / .5, 0, 1));
        var ss = s * pop;
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
      var name = CL[b.cls] ? CL[b.cls].name : '', lw = name.length * 7 + 8;
      svg('rect', { 'class': 'lb', x: x - 1.2, y: Math.max(0, y - 15), width: lw, height: 14, rx: 2, fill: c, style: '--d:' + d }, s);
      var tx = svg('text', { x: x + 3, y: Math.max(0, y - 15) + 11, fill: '#0F1220', style: '--d:' + d }, s);
      tx.textContent = name;
    });
    set('tkNb', B.length + ' cajas');
    var F = D.pixel_freq || [];
    var present = [8, 9, 10, 5, 1].filter(function (c) {
      return GRID.cls.indexOf(c) >= 0;
    });
    present.forEach(function (c, i) {
      tags.insertAdjacentHTML('beforeend', '<span style="--d:' + (.5 + i * .12) + 's"><i style="background:' + col(c) + '"></i>' + CL[c].name + '<em>sí</em></span>');
    });
    tags.insertAdjacentHTML('beforeend', '<span style="--d:' + (.5 + present.length * .12) + 's"><i style="background:#555"></i>Perro<em style="color:#FF4D6D">no</em></span>');
    var li = $$('#tareas .tasks li');
    SH.tareas = function (sc, step) {
      li.forEach(function (l, i) { l.classList.toggle('on', i === step); l.classList.toggle('done', i < step); });
    };
  })();

  /* ---------------- 03 problema: feature maps shrinking, then "there is a car" ---------------- */
  (function () {
    var L = D.levels || [], pipe = $('#pbPipe'), vis = [300, 210, 148, 104, 72], x = 0;
    var stages = [{ src: G + 'hero.jpg', h: 192, w: 256, c: 3, vh: 190, n: 1 }];
    L.forEach(function (l, i) { stages.push({ src: G + 'fmap_' + i + '_', h: l.h, w: l.w, c: l.c, vh: [190, 135, 95, 66, 46][i], n: 2 + i }); });
    stages.forEach(function (st, i) {
      var vw = st.vh * 4 / 3, el = document.createElement('div'), off = 9, html = '';
      el.className = 'slab';
      el.style.left = x + 'px'; el.style.setProperty('--d', (i * .25) + 's');
      for (var k = st.n - 1; k >= 0; k--) {
        var src = i === 0 ? st.src : st.src + (k % 4) + '.png';
        html += '<img src="' + src + '" alt="" style="width:' + vw + 'px;height:' + st.vh + 'px;left:' + (k * off) + 'px;bottom:' + (k * off) + 'px;opacity:' + (k ? .85 : 1) + '">';
      }
      var tw = vw + (st.n - 1) * off;
      el.innerHTML = '<div class="stk" style="width:' + tw + 'px;height:' + (st.vh + (st.n - 1) * off) + 'px">' + html + '</div>' +
        '<span><b>' + st.h + ' × ' + st.w + '</b>' + st.c + (i === 0 ? ' canales (RGB)' : ' canales') + '</span>';
      pipe.appendChild(el);
      x += tw + (i === 0 ? 36 : 26);
    });
    var dots = $('#pbVec .dots');
    for (var i = 0; i < 26; i++) dots.insertAdjacentHTML('beforeend', '<i style="opacity:' + (.35 + Math.random() * .65).toFixed(2) + '"></i>');
    $('#pbVec .out').innerHTML = [[8, 1], [9, 1], [10, 1], [-1, .06]].map(function (p) {
      var nm = p[0] < 0 ? 'Perro' : CL[p[0]].name, c = p[0] < 0 ? '#555' : col(p[0]);
      return '<div>' + nm + '<em' + (p[1] < .5 ? ' style="color:#FF4D6D"' : '') + '>' + (p[1] > .5 ? 'sí' : 'no') + '</em><i style="background:' + c + ';--v:' + p[1] + '"></i></div>';
    }).join('');
    var inp = D.input || [192, 256], last = L[L.length - 1];
    if (last) set('pbCell', (inp[0] / last.h) + ' × ' + (inp[1] / last.w));
  })();

  /* ---------------- 04 fcn: 12 x 16 grid, 1x1 conv scores in one cell, stretched back ---------------- */
  (function () {
    var fr = $('#fcFrame'), box = document.createElement('div'), colEl = $('#fcCol'), P = D.probs;
    box.className = 'cellbox'; fr.appendChild(box);
    var gh = 12, gw = 16, cw = 920 / gw, chh = 690 / gh;
    var r = 6, c = 7, ci = GRID.cls.indexOf(8);
    if (ci >= 0) {      /* the cell over the car */
      var cars = []; GRID.cls.forEach(function (k, i) { if (k === 8) cars.push(i); });
      ci = cars[cars.length >> 1]; r = Math.floor(Math.floor(ci / GRID.w) / 2); c = Math.floor((ci % GRID.w) / 2);
    }
    box.style.cssText = 'left:' + (c * cw) + 'px;top:' + (r * chh) + 'px;width:' + cw + 'px;height:' + chh + 'px';
    var cx = (c + 1) * cw + 16, cy = Math.max(10, r * chh - 60);
    if (cx + 250 > 900) cx = c * cw - 266;
    colEl.style.left = cx + 'px'; colEl.style.top = cy + 'px';
    var top = P && P.top ? P.top[(r * 2) * P.w + c * 2] : [];
    colEl.innerHTML = '<div class="cap" style="display:block">11 puntajes → softmax</div>' + top.map(function (p) {
      return '<div><span>' + CL[p[0]].name + '</span><i style="background:' + col(p[0]) + ';transform:scaleX(' + Math.max(.04, p[1]).toFixed(3) + ')"></i></div>';
    }).join('') + '<div style="color:#AEB4C8"><span>… 8 más</span><i></i></div>';
    var li = $$('#fcn .steps3 li');
    SH.fcn = function (sc, step) { li.forEach(function (l, i) { l.classList.toggle('on', i === step); l.classList.toggle('done', i < step); }); };
  })();

  /* ---------------- 05 upsampling: nearest, bilinear, transposed convolution ---------------- */
  (function () {
    var IN = [[1, 3], [2, 4]], K = [[1, .5], [.5, .25]];
    function shade(v, mx) {
      var u = clamp(v / mx, 0, 1), a = rgb('#2E3450'), b = rgb('#FFD23F');
      return 'rgb(' + [0, 1, 2].map(function (k) { return Math.round(a[k] + (b[k] - a[k]) * u); }).join(',') + ')';
    }
    function cell(v, mx) { return '<i style="background:' + shade(v, mx) + ';color:' + (v / mx > .45 ? '#0F1220' : '#fff') + '">' + (Math.round(v * 100) / 100) + '</i>'; }
    function fill(el, M, mx) { el.innerHTML = M.map(function (row) { return row.map(function (v) { return cell(v, mx); }).join(''); }).join(''); }
    var near = [], bil = [], tr = [];
    for (var i = 0; i < 4; i++) {
      near.push([]); bil.push([]); tr.push([]);
      for (var j = 0; j < 4; j++) {
        near[i].push(IN[i >> 1][j >> 1]);
        var y = clamp((i + .5) / 2 - .5, 0, 1), x = clamp((j + .5) / 2 - .5, 0, 1);
        bil[i].push(IN[0][0] * (1 - y) * (1 - x) + IN[0][1] * (1 - y) * x + IN[1][0] * y * (1 - x) + IN[1][1] * y * x);
        tr[i].push(IN[i >> 1][j >> 1] * K[i % 2][j % 2]);
      }
    }
    var ups = $$('#upsample .up');
    [near, bil, tr].forEach(function (M, k) { fill($('.gin', ups[k]), IN, 4); fill($('.gout', ups[k]), M, 4); });
    fill($('#upsample .gk'), K, 1);
    var outs = ups.map(function (u) { return $$('.gout i', u); }), ins = $$('.gin i', ups[2]), t0 = 0, cur = -1;
    SH.upsample = function (sc, step) {
      ups.forEach(function (u, i) { u.classList.toggle('on', i === step); u.classList.toggle('done', i < step); });
      if (step !== cur) { cur = step; t0 = 0; }
      outs.forEach(function (o, k) { o.forEach(function (c, q) { c.classList.toggle('in', k < step); c.style.transitionDelay = '0s'; }); });
    };
    SH.upsample.reset = function () { cur = -1; };
    TICK.upsample = function (t) {
      if (cur < 0) return;
      if (!t0) t0 = t;
      var e = t - t0;
      if (cur < 2) {
        outs[cur].forEach(function (c, q) { c.classList.toggle('in', e > .2 + q * .09); });
        return;
      }
      var per = .9, cyc = e % (4 * per + 2), k = Math.floor(cyc / per);
      ins.forEach(function (c, q) { c.classList.toggle('hot', q === k); });
      outs[2].forEach(function (c, q) {
        var i = Math.floor(q / 4), j = q % 4, src = (i >> 1) * 2 + (j >> 1);
        c.classList.toggle('in', src < k || (src === k && cyc - k * per > .25));
      });
      if (k < 4) { var v = IN[k >> 1][k % 2]; set('upEq', v + ' × kernel'); } else set('upEq', '');
    };
  })();

  /* ---------------- 06 encoder-decoder ---------------- */
  (function () {
    var row = $('#edRow'), L = D.levels || [], x = 0;
    var enc = [{ src: G + 'hero.jpg', vh: 190, n: 1, lab: '192 × 256' }];
    L.forEach(function (l, i) { enc.push({ src: G + 'fmap_' + i + '_', vh: [190, 135, 95, 66, 46][i], n: 2 + i, lab: l.h + ' × ' + l.w, k: i }); });
    var dec = enc.slice(1, 5).reverse().map(function (e, i) { return { vh: e.vh, n: e.n, lab: e.lab, grad: true }; });
    dec.push({ src: 'out', vh: 190, n: 1, lab: 'salida' });
    var all = enc.map(function (e) { e.e = true; return e; }).concat(dec);
    all.forEach(function (st, i) {
      var vw = st.vh * 4 / 3, off = 7, el = document.createElement('div'), html = '';
      el.className = 'blk ' + (st.e ? 'e' : 'd');
      el.style.setProperty('--d', ((st.e ? i : i - enc.length) * .18) + 's');
      for (var k = st.n - 1; k >= 0; k--) {
        var s2 = 'width:' + vw + 'px;height:' + st.vh + 'px;left:' + (k * off) + 'px;top:' + ((st.n - 1 - k) * off) + 'px';
        if (st.grad) html += '<i style="' + s2 + ';opacity:' + (1 - k * .12) + '"></i>';
        else if (st.src === 'out') html += '<img src="' + G + 'hero_plain.png" alt="" style="' + s2 + '">';
        else html += '<img src="' + (st.k === undefined ? st.src : st.src + (k % 4) + '.png') + '" alt="" style="' + s2 + '">';
      }
      var tw = vw + (st.n - 1) * off;
      el.style.left = x + 'px';
      el.innerHTML = '<div class="stk" style="width:' + tw + 'px;height:' + (st.vh + (st.n - 1) * off) + 'px">' + html + '</div><span>' + st.lab + '</span>';
      row.appendChild(el);
      x += tw + 24;
    });
    /* fit the row in the 1664 px content width and centre it */
    var tot = x - 24, k = Math.min(1.35, 1664 / tot), firstDec = $('.blk.d', row);
    row.style.transformOrigin = '0 50%'; row.style.transform = 'scale(' + k.toFixed(3) + ')';
    row.style.left = (128 + (1664 - tot * k) / 2) + 'px'; row.style.right = 'auto';
    $('#encdec .ed-lbl.l1').style.left = (128 + (1664 - tot * k) / 2) + 'px';
    if (firstDec) $('#encdec .ed-lbl.l2').style.left = (128 + (1664 - tot * k) / 2 + parseFloat(firstDec.style.left) * k) + 'px';
  })();

  /* ---------------- 07 u-net: the architecture, drawn from the real layer sizes ---------------- */
  (function () {
    var s = $('#unSvg'), L = D.levels || [], inp = D.input || [192, 256];
    var C = { conv: '#3A86FF', pool: '#FF4D6D', up: '#2EC4B6', out: '#FFD23F', bar: '#C9D6F2', upb: '#2EC4B6', copy: '#6A7192' };
    var defs = svg('defs', {}, s);
    ['conv', 'pool', 'up', 'out'].forEach(function (k) {
      var m = svg('marker', { id: 'ah-' + k, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: 'auto-start-reverse' }, defs);
      svg('path', { d: 'M0 0L10 5L0 10z', fill: C[k] }, m);
    });
    var gEnc = svg('g', { 'class': 'g-enc' }, s), gDec = svg('g', { 'class': 'g-dec' }, s), gSk = svg('g', {}, s), gP = svg('g', { 'class': 'pulse' }, s);
    var rowY = [118, 280, 410, 516, 606], barH = [206, 150, 106, 72, 50], bw = [10, 14, 18, 22, 28], GAP = 26, n = 0;
    function D_(v) { return (v * .07).toFixed(2) + 's'; }
    function bar(g, x, i, w, fill, extra) {
      svg('rect', { 'class': 'bar', x: x, y: rowY[i] - barH[i] / 2, width: w, height: barH[i], rx: 2, fill: fill, style: '--d:' + D_(n++) + (extra || '') }, g);
    }
    function arrow(g, x1, y1, x2, y2, k, curve) {
      var d = curve ? 'M' + x1 + ' ' + y1 + ' Q' + curve[0] + ' ' + curve[1] + ' ' + x2 + ' ' + y2 : 'M' + x1 + ' ' + y1 + ' L' + x2 + ' ' + y2;
      svg('path', { 'class': 'ar', d: d, fill: 'none', stroke: C[k], 'stroke-width': 4, 'stroke-linecap': 'round', pathLength: 1, 'marker-end': 'url(#ah-' + k + ')', style: '--d:' + D_(n++) }, g);
    }
    function label(g, x, y, txt, anchor, size, color) {
      var t = svg('text', { 'class': 'lbl', x: x, y: y, 'text-anchor': anchor || 'middle', 'font-size': size || 17, fill: color || '#AEB4C8', style: '--d:' + D_(n) }, g);
      t.textContent = txt;
    }
    var ch = L.map(function (l) { return l.c; }), res = L.map(function (l) { return l.h + '×' + l.w; });
    /* input thumbnail */
    var io1 = svg('image', { 'class': 'io in', href: G + 'hero.jpg', x: 0, y: rowY[0] - 52, width: 140, height: 105, preserveAspectRatio: 'none' }, gEnc);
    label(gEnc, 70, rowY[0] + 78, inp[0] + '×' + inp[1] + '×3');
    var encEnd = [], decStart = [], path = [{ x: 70, y: rowY[0], t: inp[0] + ' × ' + inp[1] + ' × 3', s: 'entrada' }];
    for (var i = 0; i < 4; i++) {
      var xs = 160 + i * 140, w = bw[i];
      if (i === 0) arrow(gEnc, 142, rowY[0], xs - 4, rowY[0], 'conv');
      bar(gEnc, xs, i, w, C.bar);
      arrow(gEnc, xs + w + 3, rowY[i], xs + w + GAP - 4, rowY[i], 'conv');
      bar(gEnc, xs + w + GAP, i, w, C.bar);
      label(gEnc, xs + w + GAP / 2, rowY[i] - barH[i] / 2 - 12, ch[i]);
      label(gEnc, xs - 8, rowY[i] + barH[i] / 2 + 22, res[i], 'start', 16, '#6A7192');
      encEnd.push(xs + 2 * w + GAP);
      path.push({ x: xs + w + GAP / 2, y: rowY[i], t: L[i].h + ' × ' + L[i].w + ' × ' + ch[i], s: 'encoder · nivel ' + (i + 1) });
      var nx = 160 + (i + 1) * 140, nb = bw[i + 1];
      if (i === 3) nx = 832 - nb - GAP / 2;
      arrow(gEnc, xs + 2 * w + GAP - w / 2, rowY[i] + barH[i] / 2 + 4, nx + nb / 2, rowY[i + 1] - barH[i + 1] / 2 - 6, 'pool');
    }
    /* bottleneck */
    var bx = 832 - bw[4] - GAP / 2;
    bar(gEnc, bx, 4, bw[4], C.bar);
    arrow(gEnc, bx + bw[4] + 3, rowY[4], bx + bw[4] + GAP - 4, rowY[4], 'conv');
    bar(gEnc, bx + bw[4] + GAP, 4, bw[4], C.bar);
    label(gEnc, 832, rowY[4] - barH[4] / 2 - 12, ch[4]);
    label(gEnc, 832, rowY[4] + barH[4] / 2 + 24, res[4] + ' · cuello de botella', 'middle', 16, '#6A7192');
    path.push({ x: 832, y: rowY[4], t: L[4].h + ' × ' + L[4].w + ' × ' + ch[4], s: 'cuello de botella' });
    var prevEnd = bx + 2 * bw[4] + GAP, prevI = 4;
    n = 0;
    for (i = 3; i >= 0; i--) {
      var w2 = bw[i], end = 1664 - 160 - i * 140, st = end - (4 * w2 + 2 * GAP);
      arrow(gDec, prevEnd - bw[prevI] / 2, rowY[prevI] - barH[prevI] / 2 - 4, st + w2 / 2, rowY[i] + barH[i] / 2 + 6, 'up');
      bar(gDec, st, i, w2, C.upb);
      svg('rect', { 'class': 'bar', x: st + w2, y: rowY[i] - barH[i] / 2, width: w2, height: barH[i], rx: 2, fill: 'none', stroke: '#AEB4C8', 'stroke-width': 2.5, 'stroke-dasharray': '6 4', style: '--d:' + D_(n++) }, gDec);
      arrow(gDec, st + 2 * w2 + 3, rowY[i], st + 2 * w2 + GAP - 4, rowY[i], 'conv');
      bar(gDec, st + 2 * w2 + GAP, i, w2, C.bar);
      arrow(gDec, st + 3 * w2 + GAP + 3, rowY[i], st + 3 * w2 + 2 * GAP - 4, rowY[i], 'conv');
      bar(gDec, st + 3 * w2 + 2 * GAP, i, w2, C.bar);
      label(gDec, st + w2, rowY[i] - barH[i] / 2 - 12, ch[i] + '+' + ch[i]);
      label(gDec, end + 8, rowY[i] + barH[i] / 2 + 22, res[i], 'end', 16, '#6A7192');
      decStart.push(st + w2);
      path.push({ x: st + w2, y: rowY[i], t: L[i].h + ' × ' + L[i].w + ' × ' + (2 * ch[i]), s: 'decoder · concatena nivel ' + (i + 1) });
      prevEnd = end; prevI = i;
    }
    arrow(gDec, 1504 + 3, rowY[0], 1520, rowY[0], 'out');
    svg('image', { 'class': 'io out', href: G + 'hero.jpg', x: 1524, y: rowY[0] - 52, width: 140, height: 105, preserveAspectRatio: 'none' }, s);
    svg('image', { 'class': 'io out', href: G + 'hero_unet.png', x: 1524, y: rowY[0] - 52, width: 140, height: 105, preserveAspectRatio: 'none', opacity: .85 }, s);
    var lo = svg('text', { 'class': 'io out', x: 1594, y: rowY[0] + 78, 'text-anchor': 'middle', 'font-size': 17, fill: '#AEB4C8' }, s);
    lo.textContent = inp[0] + '×' + inp[1] + '×11';
    path.push({ x: 1594, y: rowY[0], t: inp[0] + ' × ' + inp[1] + ' × 11', s: 'salida · 11 probabilidades por píxel' });
    /* skip connections */
    decStart.reverse().forEach(function (xd, i) {
      var y = rowY[i], d = (i * .25) + 's';
      svg('path', { 'class': 'skip', d: 'M' + (encEnd[i] + 6) + ' ' + y + ' L' + (xd - 8) + ' ' + y, style: '--d:' + d }, gSk);
      var hl = svg('text', { 'class': 'skip-h', x: (encEnd[i] + xd) / 2, y: y - 12, 'text-anchor': 'middle', 'font-size': 17, fill: '#AEB4C8', style: '--d:' + d }, gSk);
      hl.textContent = 'copiar ' + res[i];
    });
    /* pulse travelling through the U */
    var dot = svg('circle', { r: 13, fill: '#FFD23F' }, gP), halo = svg('circle', { r: 26, fill: '#FFD23F', opacity: .25 }, gP);
    var segs = [], tot = 0;
    for (i = 1; i < path.length; i++) { var dl = Math.hypot(path[i].x - path[i - 1].x, path[i].y - path[i - 1].y); segs.push(dl); tot += dl; }
    var t0 = 0, on = false, lastK = -1;
    SH.unet = function (sc, step) { on = step >= 3; t0 = 0; lastK = -1; };
    TICK.unet = function (t) {
      if (!on) return;
      if (!t0) t0 = t;
      var e = ((t - t0) % 9) / 7.5, dd = clamp(e, 0, 1) * tot, k = 0;
      while (k < segs.length - 1 && dd > segs[k]) { dd -= segs[k]; k++; }
      var u = clamp(dd / segs[k], 0, 1), a = path[k], b = path[k + 1];
      var px = a.x + (b.x - a.x) * u, py = a.y + (b.y - a.y) * u;
      dot.setAttribute('cx', px); dot.setAttribute('cy', py); halo.setAttribute('cx', px); halo.setAttribute('cy', py);
      var kk = u > .5 ? k + 1 : k;
      if (kk !== lastK) { lastK = kk; $('#unRead').innerHTML = '<b>' + path[kk].t + '</b><span>' + path[kk].s + '</span>'; }
    };
  })();

  /* ---------------- 08 skips: before/after slider + test mIoU ---------------- */
  (function () {
    var cmp = $('#skCmp'), manual = false, t0 = 0, T = D.test || {};
    function setX(p) { cmp.style.setProperty('--x', (p * 100).toFixed(2) + '%'); }
    function fromEvent(e) { var r = cmp.getBoundingClientRect(); return clamp((e.clientX - r.left) / r.width, 0, 1); }
    cmp.addEventListener('pointerdown', function (e) { manual = true; cmp.setPointerCapture(e.pointerId); setX(fromEvent(e)); });
    cmp.addEventListener('pointermove', function (e) { if (e.buttons) setX(fromEvent(e)); });
    SH.skips = function (sc, step) {
      if (step === 0) { manual = false; t0 = 0; countTo($('#skA'), T.miou_plain || 0, 2); countTo($('#skB'), T.miou_unet || 0, 2); }
    };
    TICK.skips = function (t) { if (!t0) t0 = t; if (!manual) setX(.5 + .38 * Math.sin((t - t0) * .9)); };
  })();

  /* ---------------- 09 salida: per-pixel probabilities, loss, IoU ---------------- */
  (function () {
    var P = D.probs, pr = $('#sdProbe'), cur = $('#sdCur'), bars = $('#sdBars'), manual = false;
    var cw = 800 / (P ? P.w : 32), chh = 600 / (P ? P.h : 24);
    cur.style.width = cw + 'px'; cur.style.height = chh + 'px';
    var intro = document.createElement('p');
    intro.className = 'sd-intro';
    var inp = D.input || [192, 256];
    intro.innerHTML = 'La red no responde una clase: responde <b>11 probabilidades por píxel</b>.<br><br>Salida: <b>' + inp[0] + ' × ' + inp[1] + ' × 11</b>.';
    $('#salida').appendChild(intro);
    function show(r, c) {
      if (!P) return;
      r = clamp(r, 0, P.h - 1); c = clamp(c, 0, P.w - 1);
      cur.style.left = (c * cw) + 'px'; cur.style.top = (r * chh) + 'px';
      var top = P.top[r * P.w + c];
      bars.innerHTML = top.map(function (p) {
        return '<div>' + CL[p[0]].name + '<em>' + fmt(p[1] * 100, 0) + '%</em><i style="background:' + col(p[0]) + ';transform:scaleX(' + Math.max(.02, p[1]).toFixed(3) + ')"></i></div>';
      }).join('');
    }
    pr.addEventListener('pointermove', function (e) {
      manual = true;
      var b = pr.getBoundingClientRect();
      show(Math.floor((e.clientY - b.top) / b.height * P.h), Math.floor((e.clientX - b.left) / b.width * P.w));
    });
    pr.addEventListener('pointerleave', function () { manual = false; });
    /* tour: one stop per class that appears in the photo */
    var stops = [8, 9, 3, 2, 5, 0, 10, 1].map(function (c) {
      var idx = []; GRID.cls.forEach(function (k, i) { if (k === c) idx.push(i); });
      return idx.length ? idx[idx.length >> 1] : -1;
    }).filter(function (i) { return i >= 0; });
    var t0 = 0, last = -1, eqs = $$('#salida .eq');
    SH.salida = function (sc, step) {
      eqs.forEach(function (q, i) { q.classList.toggle('on', i === step - 1); });
      intro.style.opacity = step === 0 ? 1 : 0;
    };
    TICK.salida = function (t) {
      if (!t0) t0 = t;
      if (!manual && stops.length) {
        var k = Math.floor((t - t0) / 1.6) % stops.length;
        if (k !== last) { last = k; var i = stops[k]; show(Math.floor(i / GRID.w) * (P.h / GRID.h), (i % GRID.w) * (P.w / GRID.w)); }
      }
      iouTick(t);
    };
    /* IoU: real (blue) vs predicted (coral) boxes sliding past each other */
    var s = $('#sdIou');
    var A = svg('rect', { x: 60, y: 40, width: 140, height: 120, rx: 10, fill: '#3A86FF55', stroke: '#3A86FF', 'stroke-width': 4 }, s);
    var B = svg('rect', { y: 50, width: 140, height: 120, rx: 10, fill: '#FF4D6D44', stroke: '#FF4D6D', 'stroke-width': 4 }, s);
    var I = svg('rect', { y: 50, height: 110, fill: '#FFD23F' }, s);
    var ta = svg('text', { x: 70, y: 30, 'font-size': 20, fill: '#3A86FF', 'font-family': 'JetBrains Mono' }, s); ta.textContent = 'A = real';
    var tb = svg('text', { y: 195, 'font-size': 20, fill: '#FF4D6D', 'font-family': 'JetBrains Mono' }, s); tb.textContent = 'B = predicción';
    function iouTick(t) {
      var bx = 60 + 110 * (.5 + .5 * Math.sin(t * .8)), ix0 = Math.max(60, bx), ix1 = Math.min(200, bx + 140), iw = Math.max(0, ix1 - ix0);
      B.setAttribute('x', bx); tb.setAttribute('x', bx);
      I.setAttribute('x', ix0); I.setAttribute('width', iw);
      var inter = iw * 110, uni = 140 * 120 * 2 - inter;
      set('sdIouV', (inter / uni).toFixed(2));
    }
  })();

  /* ---------------- 10 entrenamiento: prediction after every epoch + validation mIoU ---------------- */
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
      var d = '';
      p[0].forEach(function (v, i) { d += (i ? 'L' : 'M') + X(i + 1).toFixed(1) + ' ' + Y(v).toFixed(1); });
      svg('path', { d: d, fill: 'none', stroke: p[1], 'stroke-width': 5, 'stroke-linejoin': 'round' }, cv);
    });
    var cl = svg('line', { y1: y1, y2: y0, stroke: '#0F1220', 'stroke-width': 3 }, cv);
    var da = svg('circle', { r: 9, fill: '#FF4D6D' }, cv), db = svg('circle', { r: 9, fill: '#3A86FF' }, cv);
    function frame(f) {
      f = clamp(Math.round(f), 0, n - 1);
      [a, b].forEach(function (el) { el.style.backgroundPosition = '0 ' + (n > 1 ? f / (n - 1) * 100 : 0) + '%'; });
      set('trEp', f);
      cl.setAttribute('x1', X(f)); cl.setAttribute('x2', X(f));
      var vp = f ? mp[f - 1] : 0, vu = f ? mu[f - 1] : 0;
      da.setAttribute('cx', X(f)); da.setAttribute('cy', Y(vp || 0)); db.setAttribute('cx', X(f)); db.setAttribute('cy', Y(vu || 0));
    }
    var manual = null, t0 = 0;
    function pick(e) { var r = cv.getBoundingClientRect(), xx = (e.clientX - r.left) / r.width * Wd; manual = (xx - x0) / (x1 - x0) * (n - 1); frame(manual); }
    cv.addEventListener('pointerdown', function (e) { cv.setPointerCapture(e.pointerId); pick(e); });
    cv.addEventListener('pointermove', function (e) { if (e.buttons) pick(e); });
    SH.entrena = function () { manual = null; t0 = 0; };
    TICK.entrena = function (t) {
      if (manual !== null) return;
      if (!t0) t0 = t;
      var e = (t - t0) * 5, L = n + 10;
      frame(Math.min(n - 1, e % L));
    };
    frame(0);
  })();

  /* ---------------- 11 resultados: IoU per class on test ---------------- */
  (function () {
    var T = D.test || {}, F = D.pixel_freq || [], wrap = $('#rsBars'), rare = [2, 6, 9, 10];
    var order = F.map(function (f, i) { return i; }).sort(function (a, b) { return F[b] - F[a]; });
    order.forEach(function (c, j) {
      var r = document.createElement('div');
      r.className = 'rs-row' + (rare.indexOf(c) >= 0 ? ' rare' : '');
      var vp = (T.iou_plain || [])[c] || 0, vu = (T.iou_unet || [])[c] || 0;
      r.innerHTML = '<div class="nm"><i style="background:' + col(c) + '"></i>' + CL[c].name + '<small>' + fmt(F[c] * 100, 1) + '%</small></div>' +
        '<div class="tk"><i style="background:#FF4D6D;--v:' + vp.toFixed(3) + ';--d:' + (j * .06) + 's"></i><i style="background:#3A86FF;--v:' + vu.toFixed(3) + ';--d:' + (j * .06 + .1) + 's"></i></div>' +
        '<b>' + vu.toFixed(2) + '</b>';
      wrap.appendChild(r);
    });
  })();

  /* ---------------- 12 más allá: dilated conv, Mask R-CNN, SAM ---------------- */
  (function () {
    var dil = $('#msDil'), cells = [];
    for (var i = 0; i < 81; i++) { var e = document.createElement('i'); dil.appendChild(e); cells.push(e); }
    function rate(r) {
      cells.forEach(function (c, i) {
        var y = Math.floor(i / 9) - 4, x = i % 9 - 4, tap = (y % r === 0 && x % r === 0 && Math.abs(y) <= r && Math.abs(x) <= r);
        c.className = (x === 0 && y === 0) ? 'c' : tap ? 'k' : (Math.abs(x) <= r && Math.abs(y) <= r ? 'rf' : '');
      });
      set('msRate', 'rate = ' + r + ' · ve ' + (2 * r + 1) + ' × ' + (2 * r + 1));
    }
    rate(1);
    var bx = $('#msBx'), B = D.boxes || [];
    B.forEach(function (b) { svg('rect', { x: b.x * 480, y: b.y * 360, width: b.w * 480, height: b.h * 360, rx: 3, stroke: col(b.cls) }, bx); });
    var mr = $('#mas .mr'), sam = $('#mas .sam'), one = $('#msOne'), clk = $('#msClk');
    /* SAM: the instance under the click, cut out of hero_instances.png */
    var car = B.filter(function (b) { return b.cls === 8; }).sort(function (p, q) { return q.w * q.h - p.w * p.h; })[0] || B[0];
    var px = car ? car.x + car.w / 2 : .5, py = car ? car.y + car.h * .6 : .5;
    clk.style.left = (px * 100) + '%'; clk.style.top = (py * 100) + '%';
    one.style.clipPath = 'circle(0% at ' + px * 100 + '% ' + py * 100 + '%)';
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
    var t0 = 0;
    SH.mas = function () { t0 = 0; };
    TICK.mas = function (t) {
      if (!t0) t0 = t;
      var e = t - t0;
      rate([1, 2, 3][Math.floor(e / 1.8) % 3]);
      mr.classList.toggle('masks', (e % 4) > 2);
      var s2 = e % 5;
      sam.classList.toggle('pt', s2 > .6);
      sam.classList.toggle('go', s2 > 1.2);
      one.style.clipPath = 'circle(' + (s2 > 1.2 ? 75 : 0) + '% at ' + px * 100 + '% ' + py * 100 + '%)';
    };
  })();

  /* ---------------- 13 fin: the mask as a slowly breathing grid ---------------- */
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

  /* KaTeX formulas */
  function renderTex() {
    $$('[data-tex]').forEach(function (el) { if (!window.katex) return; try { katex.render(el.getAttribute('data-tex'), el, { displayMode: true, throwOnError: false }); el.classList.add('tex'); } catch (e) { } });
  }
  renderTex();

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
