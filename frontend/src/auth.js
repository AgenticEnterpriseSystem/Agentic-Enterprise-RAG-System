// Session and CSRF values exist only in this controller's memory. Cookies stay
// HttpOnly; broadcasts carry no identity, token, question or evidence payload.
export function createAuthLifecycle({ onChange, onInvalidate, fetchImpl = (...args) => fetch(...args),
  channelFactory = () => typeof BroadcastChannel === "undefined" ? null : new BroadcastChannel("redwood-auth"),
} = {}) {
  let alive = true, generation = 0, enabled = null, controller = null, expiry = null;
  let pendingLogout = null, channel = null;
  let state = Object.freeze({ status: "checking", enabled, user: null, csrf_token: null, expires_at: null, error: "" });
  const isCurrent = value => alive && value === generation;
  const notify = (status, values = {}, epoch = generation) => {
    if (!isCurrent(epoch)) return;
    state = Object.freeze({ status, enabled, user: null, csrf_token: null, expires_at: null, error: "", ...values });
    onChange?.(state);
  };
  const invalidate = status => {
    generation += 1;
    controller?.abort();
    controller = null;
    clearTimeout(expiry);
    expiry = null;
    onInvalidate?.();
    notify(status);
    return generation;
  };
  const broadcast = () => { try { channel?.postMessage(null); } catch { /* optional notification */ } };
  const request = async (path, options, epoch) => {
    if (!isCurrent(epoch)) return null;
    const response = await fetchImpl(path, { credentials: "same-origin", cache: "no-store", signal: controller.signal, ...options });
    let body = null;
    if (response.status !== 204) { try { body = await response.json(); } catch { /* invalid body fails closed below */ } }
    return isCurrent(epoch) ? { response, body } : null;
  };
  const message = (body, fallback) => typeof body?.detail?.message === "string" ? body.detail.message : fallback;
  const acceptSession = (body, epoch) => {
    if (!body?.user || typeof body.user.user_id !== "string" || !body.user.user_id
      || typeof body.csrf_token !== "string" || !body.csrf_token || !Number.isFinite(Date.parse(body.expires_at))) {
      throw new Error("Invalid session response");
    }
    if (pendingLogout) {
      notify("logout_pending", { error: "Sign-out is not confirmed. Retry sign out or sign in again." }, epoch);
      return;
    }
    const remaining = Date.parse(body.expires_at) - Date.now();
    if (remaining <= 0) {
      notify("anonymous", { error: "Your session expired. Sign in again." }, epoch);
      return;
    }
    notify("authenticated", { user: Object.freeze({ ...body.user }), csrf_token: body.csrf_token,
      expires_at: body.expires_at }, epoch);
    expiry = setTimeout(() => { if (isCurrent(epoch)) revalidate(); }, Math.min(remaining, 2 ** 31 - 1));
  };
  const loadMe = async epoch => {
    const value = await request("/api/auth/me", {}, epoch);
    if (!value) return;
    if (value.response.status === 401) {
      pendingLogout = null;
      notify("anonymous", {}, epoch);
    } else if (value.response.ok) acceptSession(value.body, epoch);
    else notify(pendingLogout ? "logout_pending" : "unavailable",
      { error: "Sign-in status is unavailable. Search remains closed; retry shortly." }, epoch);
  };
  const start = async () => {
    const epoch = invalidate("checking");
    controller = new AbortController();
    try {
      const value = await request("/api/auth/profile", {}, epoch);
      if (!value) return;
      if (!value.response.ok || typeof value.body?.enabled !== "boolean") throw new Error("Unknown authentication profile");
      enabled = value.body.enabled;
      if (!enabled) { pendingLogout = null; notify("benchmark", {}, epoch); }
      else await loadMe(epoch);
    } catch {
      if (isCurrent(epoch)) notify(pendingLogout ? "logout_pending" : "unavailable",
        { error: "Cannot confirm sign-in status. Retry the connection." }, epoch);
    }
  };
  const revalidate = async () => {
    if (!alive) return;
    if (enabled === null) return start();
    if (!enabled) {
      if (state.status !== "benchmark") notify("benchmark");
      return; // Ordinary benchmark focus does not invalidate a search.
    }
    const epoch = invalidate("checking");
    controller = new AbortController();
    try { await loadMe(epoch); }
    catch {
      if (isCurrent(epoch)) notify(pendingLogout ? "logout_pending" : "unavailable",
        { error: "Cannot confirm sign-in status. Search remains closed." }, epoch);
    }
  };
  const login = async (loginValue, password) => {
    if (enabled !== true || !alive) return;
    const epoch = invalidate("signing_in");
    controller = new AbortController();
    try {
      const bootstrap = await request("/api/auth/csrf", {}, epoch);
      if (!bootstrap) return;
      if (!bootstrap.response.ok || typeof bootstrap.body?.csrf_token !== "string") throw new Error("CSRF bootstrap unavailable");
      const value = await request("/api/auth/login", { method: "POST", headers: {
        "Content-Type": "application/json", "X-CSRF-Token": bootstrap.body.csrf_token },
        body: JSON.stringify({ login: loginValue, password }) }, epoch);
      if (!value) return;
      if (!value.response.ok) {
        notify(pendingLogout ? "logout_pending" : "anonymous",
          { error: message(value.body, "Sign-in failed. Check your login and try again.") }, epoch);
        return;
      }
      pendingLogout = null;
      acceptSession(value.body, epoch);
      broadcast();
    } catch {
      if (isCurrent(epoch)) notify(pendingLogout ? "logout_pending" : "anonymous",
        { error: "Sign-in could not finish. Retry shortly." }, epoch);
    }
  };
  const logout = async () => {
    if (enabled !== true || !alive) return;
    const csrf = state.csrf_token || pendingLogout?.csrf;
    pendingLogout = { csrf };
    const epoch = invalidate("signing_out");
    controller = new AbortController();
    broadcast();
    try {
      if (!csrf) throw new Error("Missing session verification");
      const value = await request("/api/auth/logout", { method: "POST", headers: { "X-CSRF-Token": csrf } }, epoch);
      if (!value) return;
      if (value.response.ok || value.response.status === 401) {
        pendingLogout = null;
        notify("anonymous", {}, epoch);
        broadcast();
      } else notify("logout_pending", { error: "Sign-out is not confirmed. Search stays closed; retry sign out." }, epoch);
    } catch {
      if (isCurrent(epoch)) notify("logout_pending",
        { error: "Sign-out is not confirmed. Search stays closed; retry sign out." }, epoch);
    }
  };
  const focus = () => { if (typeof document === "undefined" || document.visibilityState === "visible") revalidate(); };
  const pageShow = event => { if (event.persisted) revalidate(); };
  const pageHide = () => { if (alive) invalidate("checking"); };
  try { channel = channelFactory(); } catch { /* focus revalidation remains available */ }
  if (channel) channel.onmessage = () => revalidate();
  if (typeof window !== "undefined") {
    window.addEventListener("focus", focus);
    window.addEventListener("pageshow", pageShow);
    window.addEventListener("pagehide", pageHide);
    document.addEventListener("visibilitychange", focus);
  }
  return {
    start, revalidate, login, logout, isCurrent,
    get generation() { return generation; },
    get snapshot() { return state; },
    dispose() {
      alive = false; generation += 1; controller?.abort(); clearTimeout(expiry);
      channel?.close();
      if (typeof window !== "undefined") {
        window.removeEventListener("focus", focus);
        window.removeEventListener("pageshow", pageShow);
        window.removeEventListener("pagehide", pageHide);
        document.removeEventListener("visibilitychange", focus);
      }
      pendingLogout = null;
      state = null;
    },
  };
}
