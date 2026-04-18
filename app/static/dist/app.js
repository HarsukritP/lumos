// Lumos PWA — vanilla ES modules, hash routing. No build step.
// The goal is to feel as native as a good React app without paying the
// memory price of building on a Pi Zero 2 W.

const API = '';

const el = (tag, props = {}, ...children) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (k === 'class') n.className = v;
    else if (k === 'html') n.innerHTML = v;
    else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return n;
};

// --- icons (inline SVG, lucide-style) ----------------------------------
const icon = (name) => {
  const paths = {
    library: '<rect x="3" y="3" width="5" height="18" rx="1"/><rect x="10" y="3" width="5" height="18" rx="1"/><path d="m17 3 4 18"/>',
    book: '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
    question: '<circle cx="12" cy="12" r="10"/><path d="M9.1 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
    vocab: '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
    arrow: '<path d="m15 18-6-6 6-6"/>',
    dot: '<circle cx="12" cy="12" r="4"/>',
  };
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '2');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  svg.innerHTML = paths[name] || '';
  return svg;
};

// --- API helpers -------------------------------------------------------
async function api(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(`${path} ${r.status}`);
  return r.json();
}

// --- router ------------------------------------------------------------
const routes = [
  { match: /^#?\/?$/,                   render: renderLibrary, tab: 'library' },
  { match: /^#?\/library$/,             render: renderLibrary, tab: 'library' },
  { match: /^#?\/books\/(\d+)$/,        render: (m) => renderBook(parseInt(m[1])), tab: 'library' },
  { match: /^#?\/questions$/,           render: renderQuestions, tab: 'question' },
  { match: /^#?\/questions\/(\d+)$/,    render: (m) => renderQuestion(parseInt(m[1])), tab: 'question' },
  { match: /^#?\/vocab$/,               render: renderVocab, tab: 'vocab' },
  { match: /^#?\/debug$/,               render: renderDebug, tab: null },
];

// Intervals registered by a view; cleared on navigation.
const _intervals = new Set();
function onView(fn, ms) {
  const id = setInterval(fn, ms);
  _intervals.add(id);
  return id;
}
function clearViewIntervals() {
  for (const id of _intervals) clearInterval(id);
  _intervals.clear();
}

function router() {
  clearViewIntervals();
  const h = location.hash || '#/library';
  const root = document.getElementById('root');
  root.innerHTML = '';
  for (const r of routes) {
    const m = h.match(r.match);
    if (m) {
      Promise.resolve(r.render(m)).then((node) => {
        root.innerHTML = '';
        root.appendChild(node);
        if (r.tab) root.appendChild(renderNav(r.tab));
      }).catch((err) => {
        root.innerHTML = '';
        root.appendChild(errorView(err));
        if (r.tab) root.appendChild(renderNav(r.tab));
      });
      return;
    }
  }
  root.appendChild(errorView(new Error('not found')));
}

window.addEventListener('hashchange', router);
window.addEventListener('DOMContentLoaded', router);

// --- shared bits -------------------------------------------------------
function renderNav(active) {
  const mk = (href, name, label) => {
    const a = el('a', { href: '#' + href, class: active === name ? 'active' : '' });
    a.appendChild(icon(name));
    a.appendChild(el('span', {}, label));
    return a;
  };
  return el('nav', { class: 'tabs' },
    mk('/library', 'library', 'books'),
    mk('/questions', 'question', 'questions'),
    mk('/vocab', 'vocab', 'vocab'),
  );
}

function skeletons(n = 3) {
  const list = el('div');
  for (let i = 0; i < n; i++) list.appendChild(el('div', { class: 'skeleton' }));
  return list;
}

function errorView(err) {
  return el('div', { class: 'app' },
    el('div', { class: 'hero' },
      el('h1', {}, 'Lumos'),
      el('div', { class: 'sub' }, 'something went wrong'),
    ),
    el('div', { class: 'empty' }, String(err?.message || err)),
  );
}

function fmtDate(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  const now = Date.now();
  const diff = (now - d.getTime()) / 1000;
  if (diff < 60) return 'just now';
  if (diff < 3600) return Math.floor(diff/60) + 'm ago';
  if (diff < 86400) return Math.floor(diff/3600) + 'h ago';
  return d.toLocaleDateString();
}

// --- LIBRARY -----------------------------------------------------------
async function renderLibrary() {
  const app = el('div', { class: 'app' });
  const hero = el('div', { class: 'hero scanlines' },
    el('h1', {}, el('span', { class: 'dot' }), 'Lumos'),
    el('div', { class: 'sub' }, 'Reading light, with intelligence.'),
  );
  const status = el('div', { class: 'status-bar' }, 'loading status...');
  hero.appendChild(status);
  app.appendChild(hero);

  app.appendChild(el('h2', { class: 'page-title' }, 'Library'));

  const list = el('div'); list.appendChild(skeletons(2));
  app.appendChild(list);

  let lastCount = -1, lastTop = null, lastPhase = null;

  const paintStatus = (st) => {
    status.innerHTML = '';
    if (!st) {
      status.append(el('span', { class: 'key' }, 'OFFLINE'), el('span', {}, 'device unreachable'));
      return;
    }
    const phase = st.phase || 'connect';
    if (phase !== lastPhase) lastPhase = phase;
    if (phase === 'connect') {
      status.append(el('span', { class: 'key' }, 'CONNECT'), el('span', {}, 'handshake received'));
    } else if (phase === 'hunting') {
      status.append(el('span', { class: 'key' }, 'HUNT'), el('span', {}, 'looking for a book\u2026'));
    } else if (phase === 'reading' && st.book_title && st.book_title !== 'Unknown') {
      status.append(
        el('span', { class: 'key' }, 'NOW'),
        el('span', {}, st.book_title + ' \u00b7 p. ' + (st.current_page || '?')),
      );
    } else {
      status.append(el('span', { class: 'key' }, 'READY'), el('span', {}, 'open a book under the lamp'));
    }
  };

  const paintBooks = (books) => {
    const topId = books[0]?.id ?? null;
    if (books.length === lastCount && topId === lastTop) return;
    lastCount = books.length; lastTop = topId;
    list.innerHTML = '';
    if (!books.length) {
      list.appendChild(el('div', { class: 'empty' }, 'No books yet. Point Lumos at a book to begin.'));
    } else {
      for (const b of books) list.appendChild(bookCard(b));
    }
  };

  const refreshStatus = async () => {
    try { paintStatus(await api('/api/status')); } catch { paintStatus(null); }
  };
  const refreshBooks = async () => {
    try { paintBooks(await api('/api/books')); }
    catch {
      if (lastCount <= 0) {
        list.innerHTML = '';
        list.appendChild(el('div', { class: 'empty' }, 'Device offline. Try again in a moment.'));
      }
    }
  };

  await Promise.all([refreshStatus(), refreshBooks()]);
  onView(refreshStatus, 2000);
  onView(refreshBooks, 4000);
  return app;
}

function bookCard(b) {
  const c = el('a', { class: 'card', href: `#/books/${b.id}` });
  c.appendChild(el('h3', {}, b.title || 'Unknown'));
  if (b.author && b.author !== 'Unknown') c.appendChild(el('div', { class: 'author' }, b.author));
  const meta = el('div', { class: 'card-meta' });
  if (b.current_page) meta.appendChild(el('span', {}, el('strong', {}, 'p.' + b.current_page)));
  meta.appendChild(el('span', {}, (b.page_count || 0) + ' pages seen'));
  meta.appendChild(el('span', {}, (b.question_count || 0) + ' questions'));
  c.appendChild(meta);
  return c;
}

// --- BOOK --------------------------------------------------------------
async function renderBook(id) {
  const app = el('div', { class: 'app' });
  const back = el('a', { href: '#/library', class: 'back' }); back.appendChild(icon('arrow')); back.appendChild(el('span', {}, 'library'));
  app.appendChild(back);

  const head = el('div', { class: 'book-head' }); app.appendChild(head);
  const content = el('div'); app.appendChild(content);
  content.appendChild(skeletons(3));

  try {
    const b = await api(`/api/books/${id}`);
    head.innerHTML = '';
    head.appendChild(el('div', { class: 'tag' }, b.is_textbook ? 'textbook' : 'fiction'));
    head.appendChild(el('h1', {}, b.title || 'Unknown'));
    if (b.author && b.author !== 'Unknown') head.appendChild(el('div', { class: 'author' }, b.author));
    const prog = el('div', { class: 'progress' });
    if (b.current_page) prog.appendChild(el('span', {}, 'on ', el('strong', {}, 'p.' + b.current_page)));
    prog.appendChild(el('span', {}, (b.pages?.length || 0) + ' pages seen'));
    prog.appendChild(el('span', {}, (b.questions?.length || 0) + ' questions'));
    head.appendChild(prog);

    content.innerHTML = '';

    // Recent pages
    if (b.pages && b.pages.length) {
      content.appendChild(sectionHead('Reading log', b.pages.length));
      const log = el('div', { class: 'card' });
      const sorted = [...b.pages].sort((a, z) => (z.page_number - a.page_number));
      for (const p of sorted) {
        log.appendChild(el('div', { class: 'page-entry' },
          el('div', { class: 'p' }, 'PAGE ' + p.page_number),
          el('p', {}, p.summary),
        ));
      }
      content.appendChild(log);
    }

    // Vocab on this book
    const vocab = (b.pages || []).flatMap(p => (p.vocabulary || []).map(v => ({ ...v, page: p.page_number })));
    if (vocab.length) {
      content.appendChild(sectionHead('Vocabulary learned', vocab.length));
      for (const v of vocab) {
        content.appendChild(el('div', { class: 'card vocab-card' },
          el('h3', {}, v.word || ''),
          el('div', { class: 'def' }, v.definition || ''),
          el('div', { class: 'from' }, 'on p. ' + (v.page ?? '?')),
        ));
      }
    }

    // Questions
    if (b.questions && b.questions.length) {
      content.appendChild(sectionHead('Questions asked', b.questions.length));
      for (const q of b.questions) content.appendChild(questionCard(q));
    }

    if (!b.pages?.length && !b.questions?.length) {
      content.appendChild(el('div', { class: 'empty' }, 'Nothing yet. Open the book under the lamp.'));
    }
  } catch (e) {
    content.innerHTML = '';
    content.appendChild(el('div', { class: 'empty' }, 'Book not found.'));
  }
  return app;
}

function sectionHead(title, count) {
  return el('div', { class: 'section-head' },
    el('h2', {}, title),
    el('div', { class: 'count' }, String(count)),
  );
}

function questionCard(q) {
  const cls = 'card q-entry' + (q.is_spoiler_refusal ? ' refused' : '');
  const card = el('a', { class: cls, href: '#/questions/' + q.id });
  if (q.is_spoiler_refusal) card.appendChild(el('div', { class: 'badge' }, 'spoiler-safe'));
  card.appendChild(el('div', { class: 'q' }, q.question));
  card.appendChild(el('div', { class: 'a' }, q.answer));
  const meta = el('div', { class: 'card-meta' });
  if (q.page_number) meta.appendChild(el('span', {}, el('strong', {}, 'p.' + q.page_number)));
  meta.appendChild(el('span', {}, fmtDate(q.created_at)));
  card.appendChild(meta);
  return card;
}

// --- QUESTIONS (all) ---------------------------------------------------
async function renderQuestions() {
  const app = el('div', { class: 'app' });
  app.appendChild(el('div', { class: 'hero' },
    el('h1', {}, 'Questions'),
    el('div', { class: 'sub' }, 'Everything you\u2019ve asked Lumos.'),
  ));
  const list = el('div'); list.appendChild(skeletons(3)); app.appendChild(list);
  try {
    const books = await api('/api/books');
    const allQs = [];
    for (const b of books) {
      try {
        const qs = await api('/api/books/' + b.id + '/questions');
        for (const q of qs) allQs.push({ ...q, book_title: b.title });
      } catch {}
    }
    list.innerHTML = '';
    if (!allQs.length) {
      list.appendChild(el('div', { class: 'empty' }, 'No questions yet. Press the button on Lumos and ask one.'));
    } else {
      allQs.sort((a, z) => z.created_at - a.created_at);
      for (const q of allQs) {
        const card = questionCard(q);
        const meta = card.querySelector('.card-meta');
        if (meta) meta.insertBefore(el('span', {}, q.book_title || ''), meta.firstChild);
        list.appendChild(card);
      }
    }
  } catch (e) {
    list.innerHTML = '';
    list.appendChild(el('div', { class: 'empty' }, 'Device offline.'));
  }
  return app;
}

async function renderQuestion(id) {
  const app = el('div', { class: 'app' });
  const back = el('a', { href: '#/questions', class: 'back' }); back.appendChild(icon('arrow')); back.appendChild(el('span', {}, 'questions'));
  app.appendChild(back);
  const content = el('div'); content.appendChild(skeletons(1)); app.appendChild(content);
  try {
    const q = await api('/api/questions/' + id);
    content.innerHTML = '';
    if (q.book_title) {
      content.appendChild(el('div', { class: 'book-head' },
        el('div', { class: 'tag' }, 'question'),
        el('h1', {}, q.book_title),
        q.book_author ? el('div', { class: 'author' }, q.book_author) : null,
        el('div', { class: 'progress' },
          q.page_number ? el('span', {}, 'asked on ', el('strong', {}, 'p.' + q.page_number)) : null,
          el('span', {}, fmtDate(q.created_at)),
        ),
      ));
    }
    const card = el('div', { class: 'card q-entry' + (q.is_spoiler_refusal ? ' refused' : '') });
    if (q.is_spoiler_refusal) card.appendChild(el('div', { class: 'badge' }, 'spoiler-safe'));
    card.appendChild(el('div', { class: 'q display-font' }, q.question));
    card.appendChild(el('div', { class: 'a' }, q.answer));
    content.appendChild(card);
  } catch (e) {
    content.innerHTML = '';
    content.appendChild(el('div', { class: 'empty' }, 'Not found.'));
  }
  return app;
}

// --- VOCAB -------------------------------------------------------------
async function renderVocab() {
  const app = el('div', { class: 'app' });
  app.appendChild(el('div', { class: 'hero' },
    el('h1', {}, 'Vocab'),
    el('div', { class: 'sub' }, 'Every word Lumos has surfaced.'),
  ));
  const list = el('div'); list.appendChild(skeletons(3)); app.appendChild(list);
  try {
    const vs = await api('/api/vocab');
    list.innerHTML = '';
    if (!vs.length) {
      list.appendChild(el('div', { class: 'empty' }, 'No vocab yet.'));
      return app;
    }
    const byBook = new Map();
    for (const v of vs) {
      if (!byBook.has(v.book_id)) byBook.set(v.book_id, { title: v.book_title, items: [] });
      byBook.get(v.book_id).items.push(v);
    }
    for (const [bid, g] of byBook) {
      list.appendChild(sectionHead(g.title || 'Unknown', g.items.length));
      for (const v of g.items) {
        list.appendChild(el('div', { class: 'card vocab-card' },
          el('h3', {}, v.word),
          el('div', { class: 'def' }, v.definition),
          el('div', { class: 'from' }, `p. ${v.page_number} · ${fmtDate(v.created_at)}`),
        ));
      }
    }
  } catch (e) {
    list.innerHTML = '';
    list.appendChild(el('div', { class: 'empty' }, 'Device offline.'));
  }
  return app;
}

// --- DEBUG -------------------------------------------------------------
async function renderDebug() {
  const app = el('div', { class: 'app debug-view' });

  const back = el('a', { href: '#/library', class: 'back' });
  back.appendChild(icon('arrow'));
  back.appendChild(el('span', {}, 'library'));
  app.appendChild(back);

  app.appendChild(el('div', { class: 'hero' },
    el('h1', {}, 'Debug'),
    el('div', { class: 'sub' }, 'Live camera + detector score. Aim the lamp.'),
  ));

  const frame = el('img', { class: 'debug-frame', alt: 'latest frame' });
  frame.src = '/api/debug/frame?t=' + Date.now();
  frame.onerror = () => { frame.style.display = 'none'; };
  app.appendChild(frame);

  const grid = el('div', { class: 'debug-grid' });
  app.appendChild(grid);

  const mk = (label) => {
    const row = el('div', { class: 'debug-row' });
    const v = el('span', { class: 'debug-value mono' }, '\u2026');
    row.append(el('span', { class: 'debug-label' }, label), v);
    grid.appendChild(row);
    return v;
  };
  const vPhase = mk('phase');
  const vBook = mk('book');
  const vPage = mk('page');
  const vOk = mk('likely-page');
  const vReason = mk('reason');
  const vVar = mk('variance');
  const vMean = mk('brightness');
  const vCaptured = mk('captured');
  const vCommitted = mk('last commit');

  const audioWrap = el('div', { class: 'debug-audio' });
  audioWrap.appendChild(el('h3', {}, 'last question audio'));
  const aud = el('audio', { controls: 'controls', preload: 'none' });
  aud.src = '/api/debug/audio?t=' + Date.now();
  audioWrap.appendChild(aud);
  app.appendChild(audioWrap);

  const fmtTs = (ts) => {
    if (!ts) return '\u2014';
    const d = (Date.now() / 1000) - ts;
    if (d < 2) return 'just now';
    if (d < 60) return Math.floor(d) + 's ago';
    if (d < 3600) return Math.floor(d / 60) + 'm ago';
    return Math.floor(d / 3600) + 'h ago';
  };

  const refresh = async () => {
    try {
      frame.style.display = '';
      frame.src = '/api/debug/frame?t=' + Date.now();
      const s = await api('/api/debug/state');
      vPhase.textContent = s.phase || '\u2014';
      vBook.textContent = (s.book_title && s.book_title !== 'Unknown') ? s.book_title : '\u2014';
      vPage.textContent = s.current_page ? 'p. ' + s.current_page : '\u2014';
      vOk.textContent = s.last_capture_ok ? 'yes' : 'no';
      vOk.className = 'debug-value mono ' + (s.last_capture_ok ? 'ok' : 'bad');
      vReason.textContent = s.last_capture_reason || '\u2014';
      vVar.textContent = (s.last_capture_var ?? 0).toFixed(1);
      vMean.textContent = (s.last_capture_mean ?? 0).toFixed(1);
      vCaptured.textContent = fmtTs(s.last_capture_at);
      vCommitted.textContent = fmtTs(s.last_commit_at);
    } catch (e) {
      vPhase.textContent = 'offline';
    }
  };

  await refresh();
  onView(refresh, 1500);
  return app;
}
