let socket = null;
let tabAudioCtx = null;
let micAudioCtx = null;

chrome.runtime.onMessage.addListener(async (message) => {
  if (message.type === 'START_RECORDING') {
    initWebSocket();
    await startTabCapture(message.data);
    await startMicCapture();
  }
});

function initWebSocket() {
  socket = new WebSocket("ws://localhost:8000/ws/stream");

  socket.onopen = () => {
    console.log("[AetherCheck] WebSocket connected to backend engine.");
  };

  socket.onerror = (err) => {
    console.error("[AetherCheck] WebSocket Error:", err);
  };

  socket.onclose = () => {
    console.warn("[AetherCheck] WebSocket connection closed.");
  };
}

async function startTabCapture(streamId) {
  try {
    const tabStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        mandatory: {
          chromeMediaSource: 'tab',
          chromeMediaSourceId: streamId
        }
      },
      video: false
    });

    tabAudioCtx = new AudioContext({ sampleRate: 16000 });
    const source = tabAudioCtx.createMediaStreamSource(tabStream);

    // Maintain tab audio output so you can hear the caller
    source.connect(tabAudioCtx.destination);

    setupProcessor(tabAudioCtx, source, "remote");
    console.log("[AetherCheck] Tab (remote) audio capture started.");
  } catch (err) {
    console.error("[AetherCheck] Failed to capture tab audio:", err);
  }
}

async function startMicCapture() {
  try {
    const micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      },
      video: false
    });

    micAudioCtx = new AudioContext({ sampleRate: 16000 });
    const source = micAudioCtx.createMediaStreamSource(micStream);

    setupProcessor(micAudioCtx, source, "local");
    console.log("[AetherCheck] Mic (local) audio capture started.");
  } catch (err) {
    console.error("[AetherCheck] Failed to capture mic audio:", err);
  }
}

function setupProcessor(audioCtx, sourceNode, sourceLabel) {
  // Use ScriptProcessor (bufferSize = 4096 gives ~256ms chunk @ 16kHz)
  const processor = audioCtx.createScriptProcessor(4096, 1, 1);

  processor.onaudioprocess = (e) => {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;

    const inputData = e.inputBuffer.getChannelData(0);
    const pcm16Buffer = convertFloat32ToInt16(inputData);

    const base64Data = arrayBufferToBase64(pcm16Buffer);

    socket.send(JSON.stringify({
      source: sourceLabel, // "remote" or "local"
      timestamp_ms: Date.now(),
      audio_data: base64Data
    }));
  };

  sourceNode.connect(processor);
  processor.connect(audioCtx.destination);
}

function convertFloat32ToInt16(buffer) {
  let l = buffer.length;
  let buf = new Int16Array(l);
  while (l--) {
    let s = Math.max(-1, Math.min(1, buffer[l]));
    buf[l] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return buf.buffer;
}

function arrayBufferToBase64(buffer) {
  let binary = '';
  let bytes = new Uint8Array(buffer);
  let len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}