/*
  Shared client-side auth helpers for Gatekeeper's static UI pages.

  There is no session/cookie layer here -- Gatekeeper's only credential is
  an API key (see core/auth.py), so "being logged in" just means: a key is
  held in sessionStorage (cleared when the tab closes, never persisted to
  disk) and it has been checked against the real KeyStore via
  GET /api/v1/whoami, not merely present.
*/
const GK_STORAGE_KEY = "gatekeeper_api_key";
const GK_STORAGE_IDENTITY = "gatekeeper_identity";

function gkApiBase() {
  return window.location.origin;
}

function gkGetKey() {
  return sessionStorage.getItem(GK_STORAGE_KEY) || "";
}

function gkSetSession(key, identity) {
  sessionStorage.setItem(GK_STORAGE_KEY, key);
  sessionStorage.setItem(GK_STORAGE_IDENTITY, JSON.stringify(identity));
}

function gkGetIdentity() {
  try {
    return JSON.parse(sessionStorage.getItem(GK_STORAGE_IDENTITY) || "null");
  } catch (e) {
    return null;
  }
}

function gkClearSession() {
  sessionStorage.removeItem(GK_STORAGE_KEY);
  sessionStorage.removeItem(GK_STORAGE_IDENTITY);
}

function gkAuthHeaders() {
  const key = gkGetKey();
  return key ? { "Authorization": "Bearer " + key } : {};
}

function gkLoginUrl(next) {
  const target = next || (window.location.pathname + window.location.search);
  return "/ui/login/index.html?next=" + encodeURIComponent(target);
}

/*
  Verifies the held key against GET /api/v1/whoami. Returns the identity
  object on success. On any failure, clears the session and redirects to
  the login page -- called at the top of every protected page so a
  revoked/expired key never sits in the UI showing stale data.
*/
async function gkRequireAuth() {
  const key = gkGetKey();
  if (!key) {
    window.location.replace(gkLoginUrl());
    return null;
  }
  try {
    const res = await fetch(gkApiBase() + "/api/v1/whoami", { headers: gkAuthHeaders() });
    if (!res.ok) {
      gkClearSession();
      window.location.replace(gkLoginUrl());
      return null;
    }
    const identity = await res.json();
    gkSetSession(key, identity);
    return identity;
  } catch (e) {
    // Network failure: keep the session rather than bouncing to login on a
    // transient outage. The page's own API calls will surface the error.
    return gkGetIdentity();
  }
}

function gkSignOut() {
  gkClearSession();
  window.location.replace(gkLoginUrl());
}

/*
  Cross-page nav links shown in every protected page's header. Review
  Queue only appears for INTERNAL callers -- matching the endpoint's own
  gate (GET /api/v1/review requires INTERNAL), so the link itself is
  never shown to a caller who would just get a 403 for clicking it.
*/
function gkNavLinks(identity, activePage) {
  const links = [
    { href: "/ui/activity/index.html", label: "Activity", page: "activity" },
    { href: "/ui/trace/index.html", label: "Trace", page: "trace" },
    { href: "/ui/settings/index.html", label: "Settings", page: "settings" },
  ];
  if (identity.capability === "INTERNAL") {
    links.push(
      { href: "/ui/review/index.html", label: "Review Queue", page: "review" },
      { href: "/ui/gateways/index.html", label: "Gateways", page: "gateways" },
      { href: "/ui/logs/index.html", label: "Logs", page: "logs" },
      { href: "/ui/benchmarks/index.html", label: "Benchmarks", page: "benchmarks" },
      { href: "/ui/policy/index.html", label: "Policy", page: "policy" },
    );
  }
  return links.map(l =>
    `<a href="${l.href}" class="gk-nav-link${l.page === activePage ? ' active' : ''}">${l.label}</a>`
  ).join("");
}
