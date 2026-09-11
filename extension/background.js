let isRunning = false;
let activeTabId = null;

chrome.action.onClicked.addListener(async (tab) => {
  if (isRunning) {
    console.log("[AetherCheck] Stopping capture session...");
    await chrome.offscreen.closeDocument();
    isRunning = false;
    chrome.action.setBadgeText({ text: "" });
    return;
  }

  // 1. Get a stream ID for the active tab audio
  const streamId = await chrome.tabCapture.getMediaStreamId({
    targetTabId: tab.id
  });

  // 2. Ensure the offscreen document exists
  const existingContexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT']
  });

  if (existingContexts.length === 0) {
    await chrome.offscreen.createDocument({
      url: 'offscreen.html',
      reasons: ['USER_MEDIA', 'AUDIO_PLAYBACK'],
      justification: 'Real-time dual-channel audio stream capture and analysis'
    });
  }

  // 3. Hand off the tab stream ID to the offscreen worker
  chrome.runtime.sendMessage({
    type: 'START_RECORDING',
    targetTabId: tab.id,
    data: streamId
  });

  activeTabId = tab.id;
  isRunning = true;
  chrome.action.setBadgeText({ text: "LIVE" });
  chrome.action.setBadgeBackgroundColor({ color: "#22c55e" });
});
// Relay scores from the offscreen socket to the HUD in the call tab, and raise
// a desktop notification when the score crosses the alert threshold.
chrome.runtime.onMessage.addListener((message) => {
  if (!message) return;

  if (message.type === "risk_update" || message.type === "session_start") {
    if (activeTabId != null) {
      chrome.tabs.sendMessage(activeTabId, message).catch(() => {});
    }
    if (message.type === "risk_update") {
      chrome.action.setBadgeText({ text: String(Math.round(message.risk)) });
      chrome.action.setBadgeBackgroundColor({
        color: message.band === "critical" ? "#ef4444"
             : message.band === "elevated" ? "#f59e0b" : "#22c55e"
      });
    }
    return;
  }

  if (message.type === "AETHERCHECK_ALERT") {
    chrome.notifications.create({
      type: "basic",
      iconUrl: "icon128.png",
      title: `AetherCheck: ${Math.round(message.risk)}% scam risk`,
      message: message.detail,
      priority: 2
    });
  }
});
