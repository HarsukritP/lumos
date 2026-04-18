// Very small service worker: cache the shell, pass API calls through.
const VERSION = 'lumos-v1';
const SHELL = ['/', '/index.html', '/style.css', '/app.js', '/manifest.webmanifest'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (url.pathname.startsWith('/api/')) return; // always hit network for API
  e.respondWith(
    caches.match(e.request).then((hit) => hit || fetch(e.request).then((resp) => {
      if (resp.ok && e.request.method === 'GET') {
        const copy = resp.clone();
        caches.open(VERSION).then((c) => c.put(e.request, copy)).catch(() => {});
      }
      return resp;
    }).catch(() => caches.match('/index.html')))
  );
});
