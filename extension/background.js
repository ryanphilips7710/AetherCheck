let isRunning = false;

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

  isRunning = true;
  chrome.action.setBadgeText({ text: "LIVE" });
  chrome.action.setBadgeBackgroundColor({ color: "#22c55e" });
});