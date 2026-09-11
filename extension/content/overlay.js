/* AetherCheck HUD - renders risk_update payloads from the stream core.
 *
 * Runs both as a content script (messages relayed from background.js) and as a
 * plain page script in demo/hud_preview.html, where it subscribes to
 * ws://localhost:8000/ws/hud directly. Keeping one renderer for both means the
 * thing we demo is the thing that ships.
 */

(function () {
  if (window.__aethercheckHud) return;

  const BANDS = { normal: "Monitoring", elevated: "Elevated", critical: "Scam Likely" };

  function build() {
    const el = document.createElement("div");
    el.id = "aethercheck-hud";
    el.dataset.band = "normal";
    el.innerHTML = `
      <div class="ac-head">
        <span class="ac-dot"></span>
        <span class="ac-title">AetherCheck</span>
        <span class="ac-status" data-ref="status">Connecting</span>
      </div>
      <div class="ac-dial">
        <span class="ac-risk" data-ref="risk">0</span>
        <span class="ac-risk-unit">%</span>
        <span class="ac-risk-label" data-ref="band">Monitoring</span>
      </div>
      <div class="ac-bar"><div class="ac-bar-fill" data-ref="fill"></div></div>
      <div class="ac-meters">
        <div class="ac-meter" data-ref="physicsBox">
          <div class="ac-meter-label">Turn Physics</div>
          <div class="ac-meter-value" data-ref="physics">0</div>
          <div class="ac-meter-sub" data-ref="latency">no turns yet</div>
        </div>
        <div class="ac-meter" data-ref="intentBox">
          <div class="ac-meter-label">Scam Intent</div>
          <div class="ac-meter-value" data-ref="intent">0</div>
          <div class="ac-meter-sub" data-ref="intentSub">listening</div>
        </div>
      </div>
      <div class="ac-triggers" data-ref="triggers">
        <div class="ac-empty">No anomalies detected</div>
      </div>
      <div class="ac-transcript" data-ref="transcript" hidden></div>`;
    document.documentElement.appendChild(el);

    const refs = {};
    el.querySelectorAll("[data-ref]").forEach((n) => (refs[n.dataset.ref] = n));
    return { el, refs };
  }

  const { el, refs } = build();
  let lastAlertAt = 0;
  let lastTriggerSignature = null;

  function render(msg) {
    if (msg.type === "session_start") {
      refs.status.textContent = "Live";
      const intentOff = msg.engines && !msg.engines.intent.available;
      refs.intentBox.classList.toggle("offline", !!intentOff);
      if (intentOff) refs.intentSub.textContent = "engine offline";
      return;
    }
    if (msg.type !== "risk_update") return;

    el.dataset.band = msg.band;
    refs.risk.textContent = Math.round(msg.risk);
    refs.band.textContent = BANDS[msg.band] || msg.band;
    refs.fill.style.width = Math.min(100, msg.risk) + "%";

    refs.physics.textContent = Math.round(msg.physics.score);
    refs.latency.textContent =
      msg.physics.last_latency_ms != null
        ? `${Math.round(msg.physics.last_latency_ms)}ms last turn`
        : "no turns yet";

    refs.intent.textContent = Math.round(msg.intent.score);
    refs.intentBox.classList.toggle("offline", !msg.intent.available);
    if (msg.intent.available) {
      refs.intentSub.textContent = msg.intent.transcript ? "transcribing" : "listening";
    }

    // Scores update ~4x/second. Rebuilding the badges every time restarts their
    // entry animation on each frame, which makes them flicker and read as
    // invisible. Only touch the DOM when the set of badges actually changes.
    const signature = msg.triggers.map((t) => t.id + "|" + t.detail).join("~");
    if (signature !== lastTriggerSignature) {
      lastTriggerSignature = signature;
      refs.triggers.innerHTML = msg.triggers.length
        ? msg.triggers
            .map(
              (t) => `<div class="ac-trigger ${t.source}">
                    <div class="ac-trigger-label">${esc(t.label)}</div>
                    <div class="ac-trigger-detail">${esc(t.detail)}</div>
                  </div>`
            )
            .join("")
        : `<div class="ac-empty">No anomalies detected</div>`;
    }

    if (msg.intent.transcript) {
      refs.transcript.hidden = false;
      refs.transcript.textContent = "“" + msg.intent.transcript.slice(-160) + "”";
    }

    // The server sets alert exactly once per crossing of the threshold, so this
    // does not need its own debounce - but guard anyway in case of a reconnect.
    if (msg.alert && Date.now() - lastAlertAt > 5000) {
      lastAlertAt = Date.now();
      notify(msg);
    }
  }

  function notify(msg) {
    const labels = msg.triggers.map((t) => t.label).join(" + ") || "Anomalous call dynamics";
    if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.sendMessage) {
      chrome.runtime.sendMessage({ type: "AETHERCHECK_ALERT", risk: msg.risk, detail: labels });
    } else if (typeof Notification !== "undefined" && Notification.permission === "granted") {
      new Notification(`AetherCheck: ${Math.round(msg.risk)}% scam risk`, { body: labels });
    }
  }

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])
    );
  }

  // In the extension, background.js relays frames. Standalone, subscribe direct.
  if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.onMessage) {
    chrome.runtime.onMessage.addListener((m) => {
      if (m && (m.type === "risk_update" || m.type === "session_start")) render(m);
    });
  } else {
    const connect = () => {
      const ws = new WebSocket("ws://localhost:8000/ws/hud");
      ws.onopen = () => (refs.status.textContent = "Live");
      ws.onmessage = (e) => render(JSON.parse(e.data));
      ws.onclose = () => {
        refs.status.textContent = "Reconnecting";
        setTimeout(connect, 1500);
      };
    };
    connect();
  }

  window.__aethercheckHud = { render };
})();
