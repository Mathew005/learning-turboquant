/* ── Config ─────────────────────────────────────────────── */
const BASE_PATH = 'docs/';
const HOME = 'article.md';

const PAGES = {
  'article.md':           'Understanding TurboQuant',
  'theory_outliers.md':    'Deep Dive: Activation Outliers',
  'theory_incoherence.md': 'Deep Dive: Random Rotation',
  'theory_qjl.md':         'Deep Dive: QJL Correction',
  'benchmarks.md':         'Benchmarking pyturboquant',
};

// Display order in the sidebar "In This Series" section
const PAGE_ORDER = [
  'article.md',
  'theory_outliers.md',
  'theory_incoherence.md',
  'theory_qjl.md',
  'benchmarks.md',
];

/* ── Marked options ─────────────────────────────────────── */
marked.setOptions({ gfm: true, breaks: false });

/* ── Mermaid init ───────────────────────────────────────── */
mermaid.initialize({ startOnLoad: false, theme: 'default' });

/* ── Post-render: promote mermaid code blocks to diagrams ── */
async function renderMermaidBlocks() {
  const content = document.getElementById('content');
  const blocks = content.querySelectorAll('pre > code.language-mermaid');
  for (const code of blocks) {
    const pre = code.parentElement;
    const definition = code.textContent;
    const id = 'mermaid-' + Math.random().toString(36).slice(2);
    const div = document.createElement('div');
    div.className = 'mermaid-wrapper';
    try {
      const { svg } = await mermaid.render(id, definition);
      div.innerHTML = svg;
    } catch (e) {
      div.textContent = 'Diagram error: ' + e.message;
      div.style.color = 'red';
    }
    pre.replaceWith(div);
  }
}

/* ── KaTeX deferred render ──────────────────────────────── */
function renderMathOnPage() {
  renderMathInElement(document.body, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '$',  right: '$',  display: false },
      { left: '\\(', right: '\\)', display: false },
      { left: '\\[', right: '\\]', display: true },
    ],
    throwOnError: false,
  });
}

/* ── Routing helpers ────────────────────────────────────── */
function hashToFile() {
  const raw = decodeURIComponent(location.hash.slice(1));
  return raw || HOME;
}

function navigateTo(file) {
  location.hash = encodeURIComponent(file);
}

/* ── TOC ─────────────────────────────────────────────────── */
function generateTOC(currentFile) {
  const content = document.getElementById('content');
  const tocList = document.getElementById('toc-list');
  tocList.innerHTML = '';

  // ── "In This Series" ──────────────────────────────────────
  const seriesHeader = document.createElement('li');
  seriesHeader.className = 'toc-section-header';
  seriesHeader.textContent = 'In This Series';
  tocList.appendChild(seriesHeader);

  PAGE_ORDER.forEach(file => {
    const li = document.createElement('li');
    const a  = document.createElement('a');
    a.href = '#' + encodeURIComponent(file);
    a.textContent = PAGES[file] || file;
    a.className = 'toc-page-link' + (file === currentFile ? ' toc-page-active' : '');
    a.addEventListener('click', e => {
      e.preventDefault();
      closeSidebar();
      location.hash = encodeURIComponent(file);
    });
    li.appendChild(a);
    tocList.appendChild(li);
  });

  // ── "On This Page" (H2 only to keep it compact) ───────────
  const headings = Array.from(content.querySelectorAll('h2'));
  if (headings.length > 0) {
    const divider = document.createElement('li');
    divider.className = 'toc-divider';
    tocList.appendChild(divider);

    const onPageHeader = document.createElement('li');
    onPageHeader.className = 'toc-section-header';
    onPageHeader.textContent = 'On This Page';
    tocList.appendChild(onPageHeader);

    headings.forEach((heading, i) => {
      if (!heading.id) {
        heading.id = heading.textContent
          .toLowerCase()
          .replace(/[^\w]+/g, '-')
          .replace(/^-+|-+$/g, '') || ('h-' + i);
      }

      const li = document.createElement('li');
      const a  = document.createElement('a');
      a.href = '#' + heading.id;
      // Strip leading section numbers for cleaner labels
      a.textContent = heading.textContent.replace(/^\d+[a-z]?\.\s*/i, '');
      a.className = 'toc-h2';

      a.addEventListener('click', e => {
        e.preventDefault();
        closeSidebar();
        const y = heading.getBoundingClientRect().top + window.scrollY - 72;
        window.scrollTo({ top: y, behavior: 'smooth' });
      });

      li.appendChild(a);
      tocList.appendChild(li);
    });
  }
}

function openSidebar() {
  document.getElementById('toc-sidebar').classList.add('visible');
  document.getElementById('toc-backdrop').classList.add('visible');
}

function closeSidebar() {
  document.getElementById('toc-sidebar').classList.remove('visible');
  document.getElementById('toc-backdrop').classList.remove('visible');
}

function toggleTOC() {
  const sidebar = document.getElementById('toc-sidebar');
  if (sidebar.classList.contains('visible')) {
    closeSidebar();
  } else {
    openSidebar();
  }
}

/* ── Page loader ────────────────────────────────────────── */
async function loadPage(file) {
  const content = document.getElementById('content');
  const titleEl = document.getElementById('page-title');

  // Start fade out by removing the enter class
  content.classList.remove('page-enter');

  try {
    const res = await fetch(BASE_PATH + file);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    let md = await res.text();

    // Protect math blocks from Marked.js mangling (e.g., underscores turning into italics)
    const mathBlocks = [];
    md = md.replace(/\$\$([\s\S]*?)\$\$/g, (match) => {
      mathBlocks.push(match);
      return `@@MATHBLOCK${mathBlocks.length - 1}@@`;
    });
    md = md.replace(/\$([^\n$]+)\$/g, (match) => {
      mathBlocks.push(match);
      return `@@MATHBLOCK${mathBlocks.length - 1}@@`;
    });

    // Parse markdown
    let html = marked.parse(md);

    // Restore math blocks and rewrite links
    html = html.replace(/@@MATHBLOCK(\d+)@@/g, (_, i) => mathBlocks[i]);
    html = html.replace(/href="([^"]+\.md)"/g, (_, href) => `href="#${href}"`);

    // Give the fade-out a tiny moment to happen if the network is extremely fast
    await new Promise(r => setTimeout(r, 80));

    content.innerHTML = html;

    // Update chrome
    document.title  = PAGES[file] || file;
    titleEl.textContent = PAGES[file] || file;
    document.body.setAttribute('data-page', file);

    // Rebuild TOC for this page
    generateTOC(file);

    window.scrollTo({ top: 0, behavior: 'instant' });

    await renderMermaidBlocks();
    if (window.renderMathInElement) renderMathOnPage();

  } catch (err) {
    if (content) {
      content.innerHTML = `<h2>Page not found</h2>
        <p>Could not load <code>${file}</code>.</p><p>${err}</p>`;
    }
  } finally {
    // Trigger fade in cleanly
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (content) content.classList.add('page-enter');
      });
    });
  }
}

/* ── Intercept rendered .md links ───────────────────────── */
document.getElementById('content').addEventListener('click', e => {
  const a = e.target.closest('a');
  if (!a) return;
  const href = a.getAttribute('href');
  if (href && href.startsWith('#') && href.endsWith('.md')) {
    e.preventDefault();
    location.hash = href.slice(1);
  }
});

/* ── Close sidebar on Escape ────────────────────────────── */
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeSidebar();
});

/* ── Hash-change router ─────────────────────────────────── */
window.addEventListener('hashchange', () => loadPage(hashToFile()));

/* ── Initial load ───────────────────────────────────────── */
loadPage(hashToFile());
