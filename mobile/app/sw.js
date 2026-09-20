// Caches the ~20MB of tracker/model assets so only the first visit pays for
// them — everything after loads from the Cache API instead of the network.
// Two strategies, chosen by what's actually safe to cache aggressively:
//
//  - CDN assets (tfjs, MediaPipe wasm + hand model) and our own model/pamphlet
//    files: cache-first. These are either version-pinned in their URL or, for
//    our own model files, only change when we ship a new CACHE_VERSION — so a
//    cache hit is always correct, not just fast.
//  - The app shell (html/js/css): network-first, falling back to cache when
//    offline. These change during development; always preferring the network
//    means a real update is never stuck behind a stale cache, while offline
//    use still works from whatever was cached last.
//
// Bump CACHE_VERSION when the model files change — see mobile/convert_model.py.
const CACHE_VERSION = 'v1';
const CACHE_NAME = `isl-fingerspell-${CACHE_VERSION}`;

const CDN_HOSTS = new Set([
  'cdn.jsdelivr.net',
  'storage.googleapis.com',
  'fonts.googleapis.com',
  'fonts.gstatic.com',
]);

function isCacheFirst(url) {
  if (CDN_HOSTS.has(url.hostname)) return true;
  if (url.origin === self.location.origin) {
    return url.pathname.includes('/model/') || url.pathname.includes('/pamphlet/');
  }
  return false;
}

self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))),
    ),
  );
  self.clients.claim();
});

async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (response.ok) {
    const cache = await caches.open(CACHE_NAME);
    cache.put(request, response.clone());
  }
  return response;
}

async function networkFirst(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    const cached = await caches.match(request);
    if (cached) return cached;
    throw error;
  }
}

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  const url = new URL(event.request.url);
  event.respondWith(isCacheFirst(url) ? cacheFirst(event.request) : networkFirst(event.request));
});
