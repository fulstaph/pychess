import { Chessground } from 'https://cdn.jsdelivr.net/npm/@lichess-org/chessground/+esm';

// ---------------------------------------------------------------------------
// Module state: the Chessground board instance, the last state payload,
// the active session id (persisted in localStorage so a reload returns to
// the same game), a deferred promotion awaiting a piece choice, and the
// current orientation.
// ---------------------------------------------------------------------------
let ground = null;
let currentState = null;
let currentSessionId = null;
let pendingPromotion = null;
let boardOrientation = 'white';

// Web Audio Sound synthesis
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
function playBeep(freq, type = 'sine', duration = 0.08) {
  if (audioCtx.state === 'suspended') audioCtx.resume();
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
  gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + duration);
  osc.connect(gain);
  gain.connect(audioCtx.destination);
  osc.start();
  osc.stop(audioCtx.currentTime + duration);
}

const GLYPHS = { p: '♟', r: '♜', n: '♞', b: '♝', q: '♛', k: '♚' };

// Remember the active session id across reloads.
function saveSessionId() {
  try {
    if (currentSessionId !== null) localStorage.setItem('pychess-session', String(currentSessionId));
  } catch {
    // Storage unavailable (private mode); the id simply won't persist.
  }
}

function loadStoredSessionId() {
  try {
    const raw = localStorage.getItem('pychess-session');
    return raw ? Number(raw) : null;
  } catch {
    return null;
  }
}

// State polling: fetchState pulls the active session's state and pushes it
// through updateUI.
export async function fetchState() {
  if (currentSessionId === null) return;
  try {
    const res = await fetch(`/api/v1/sessions/${currentSessionId}`);
    if (!res.ok) return;
    updateUI(await res.json());
  } catch {
    // Server unreachable; keep the last known board state.
  }
}

// Send JSON and return the parsed body (null for 204); throws Error(server
// detail) on !ok.
export async function sendJSON(path, body, method = 'POST') {
  const res = await fetch(path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: method === 'GET' ? undefined : JSON.stringify(body ?? {}),
  });
  const data = res.status === 204 ? null : await res.json();
  if (!res.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : res.statusText;
    throw new Error(detail);
  }
  return data;
}

// Board init + full UI refresh: creates the Chessground board on first call,
// then re-syncs FEN/orientation/movable state afterwards.
export function updateUI(data) {
  currentState = data;
  const gameOver = Boolean(data.is_checkmate || data.is_draw);
  // Legal-move highlighting: server dests maps each origin square to
  // its list of legal destinations, consumed by Chessground's movable.
  const destsMap = new Map();
  if (!gameOver && data.dests) {
    for (const [from, toList] of Object.entries(data.dests)) {
      destsMap.set(from, toList);
    }
  }

  const turnColor = data.turn === 'w' ? 'white' : 'black';
  const aiMode = document.getElementById('ai-mode').value;
  const isHumanTurn =
    !gameOver &&
    (aiMode === 'human' ||
      data.turn === (boardOrientation === 'white' ? 'w' : 'b'));

  // The human may drag only on their own turn (pass & play, or the
  // board oriented toward the side to move); otherwise moves are locked.
  if (!ground) {
    const wrap = document.getElementById('cg-board');
    ground = Chessground(wrap, {
      fen: data.fen,
      orientation: boardOrientation,
      turnColor: turnColor,
      movable: {
        color: isHumanTurn ? turnColor : null,
        free: false,
        dests: destsMap,
        events: { after: onMovePiece },
      },
    });
  } else {
    ground.set({
      fen: data.fen,
      orientation: boardOrientation,
      turnColor: turnColor,
      movable: {
        color: isHumanTurn ? turnColor : null,
        dests: destsMap,
      },
    });
  }

  // Update text indicators
  const turnTxt = data.turn === 'w' ? 'White to move' : 'Black to move';
  document.getElementById('turn-text').textContent = turnTxt;
  const stateCard = document.getElementById('status-card');
  stateCard.className = 'status-banner';
  if (data.is_checkmate) {
    stateCard.classList.add('status-mate');
    document.getElementById('state-text').textContent = 'CHECKMATE';
  } else if (data.is_draw) {
    stateCard.classList.add('status-draw');
    document.getElementById('state-text').textContent = 'DRAW';
  } else if (data.is_check) {
    stateCard.classList.add('status-check');
    document.getElementById('state-text').textContent = 'CHECK';
  } else {
    document.getElementById('state-text').textContent = 'ACTIVE';
  }

  // Captures
  document.getElementById('cap-w').textContent = (data.captured_w || [])
    .map((p) => GLYPHS[p] || p)
    .join(' ');
  document.getElementById('cap-b').textContent = (data.captured_b || [])
    .map((p) => GLYPHS[p] || p)
    .join(' ');

  // Eval bar (bounded -1000 to +1000)
  const ev = Math.max(-1000, Math.min(1000, data.eval || 0));
  const pct = Math.round(((ev + 1000) / 2000) * 100);
  document.getElementById('eval-fill').style.width = pct + '%';

  // Moves history
  const histBox = document.getElementById('moves-history');
  histBox.innerHTML = '';
  const hist = data.history || [];
  for (let i = 0; i < hist.length; i += 2) {
    const row = document.createElement('div');
    row.className = 'move-row';
    const num = i / 2 + 1;
    const wMove = hist[i];
    const bMove = hist[i + 1] || '';
    row.textContent = `${num}. ${wMove}  ${bMove}`;
    histBox.appendChild(row);
  }
  histBox.scrollTop = histBox.scrollHeight;

  // FEN input placeholder
  document.getElementById('fen-input').value = data.fen;
}

// Refresh the Sessions dropdown from GET /api/v1/sessions; preserves the
// current selection when its id is still present.
export async function refreshSessions() {
  const select = document.getElementById('sessions-select');
  if (!select) return;
  let sessions = [];
  try {
    const res = await fetch('/api/v1/sessions?page_size=100');
    if (res.ok) sessions = (await res.json()).items;
  } catch {
    return;
  }
  const prev = currentSessionId !== null ? String(currentSessionId) : '';
  select.innerHTML = '';
  for (const s of sessions) {
    const opt = document.createElement('option');
    opt.value = String(s.id);
    opt.textContent = `#${s.id} ${s.name || ''} (${s.move_count} plies)`;
    select.appendChild(opt);
  }
  if (prev && sessions.some((s) => String(s.id) === prev)) select.value = prev;
}

// Switch to another session from the dropdown: remember it and fetch state.
document.getElementById('sessions-select').onchange = async (e) => {
  const id = e.target.value;
  if (!id) return;
  currentSessionId = Number(id);
  saveSessionId();
  try {
    const data = await sendJSON(`/api/v1/sessions/${currentSessionId}`, {}, 'GET');
    updateUI(data);
  } catch (err) {
    alert(err.message);
    fetchState();
  }
};

// New session: POST /api/v1/sessions (standard start), then switch to it.
document.getElementById('btn-new').onclick = async () => {
  try {
    const data = await sendJSON('/api/v1/sessions', {});
    currentSessionId = data.id;
    saveSessionId();
    updateUI(data);
    refreshSessions();
  } catch (err) {
    alert(err.message);
    fetchState();
  }
};

// Delete the active session; switches to another one when the server
// still has any.
document.getElementById('btn-session-delete').onclick = async () => {
  if (currentSessionId === null) return;
  if (!confirm(`Delete session #${currentSessionId}?`)) return;
  try {
    await sendJSON(`/api/v1/sessions/${currentSessionId}`, {}, 'DELETE');
  } catch (err) {
    alert(err.message);
    refreshSessions();
    return;
  }
  currentSessionId = null;
  saveSessionId();
  refreshSessions().then(() => {
    const select = document.getElementById('sessions-select');
    if (select.value) {
      currentSessionId = Number(select.value);
      saveSessionId();
      fetchState();
    }
  });
};

// Promotion detection via a UCI probe: if "<orig><dest>q" appears in
// legal_moves the drag was a pawn promotion, so open the choice dialog
// instead of submitting the move directly.
export async function onMovePiece(orig, dest) {
  // Check for pawn promotion by verifying if promoting with 'q' is a legal move
  const isPromotion =
    currentState &&
    currentState.legal_moves &&
    currentState.legal_moves.includes(orig + dest + 'q');
  if (isPromotion) {
    pendingPromotion = { orig, dest };
    const isWhite = currentState.turn === 'w';
    document.querySelector('.promo-btn[data-piece="q"]').textContent = isWhite
      ? '♕'
      : '♛';
    document.querySelector('.promo-btn[data-piece="r"]').textContent = isWhite
      ? '♖'
      : '♜';
    document.querySelector('.promo-btn[data-piece="b"]').textContent = isWhite
      ? '♗'
      : '♝';
    document.querySelector('.promo-btn[data-piece="n"]').textContent = isWhite
      ? '♘'
      : '♞';
    document.getElementById('promo-modal').style.display = 'flex';
    return;
  }

  await submitMove(orig + dest);
}

// Move submission: POST the move to the active session with the mode's AI
// reply, depth, and Stockfish flag; on server error, alert the detail and
// re-sync state.
export async function submitMove(moveStr) {
  const mode = document.getElementById('ai-mode').value;
  playBeep(220, 'triangle', 0.05);
  try {
    const data = await sendJSON(`/api/v1/sessions/${currentSessionId}/move`, {
      move: moveStr,
      ai_reply: mode !== 'human',
      depth: mode === 'minimax-1' ? 1 : 2,
      stockfish: mode === 'stockfish',
    });
    if (data.is_check) playBeep(520, 'sine', 0.12);
    updateUI(data);
  } catch (err) {
    alert(err.message);
    fetchState();
  }
}

// Event listeners: promotion dialog, and the undo/flip/ai/FEN controls.
document.querySelectorAll('.promo-btn').forEach((btn) => {
  btn.addEventListener('click', async () => {
    document.getElementById('promo-modal').style.display = 'none';
    if (pendingPromotion) {
      const promoPiece = btn.dataset.piece;
      const moveStr =
        pendingPromotion.orig + pendingPromotion.dest + promoPiece;
      pendingPromotion = null;
      await submitMove(moveStr);
    }
  });
});

document.getElementById('promo-modal').addEventListener('click', (e) => {
  if (e.target.id === 'promo-modal') {
    document.getElementById('promo-modal').style.display = 'none';
    pendingPromotion = null;
    fetchState();
  }
});

document.getElementById('btn-undo').onclick = async () => {
  const mode = document.getElementById('ai-mode').value;
  try {
    const data = await sendJSON(`/api/v1/sessions/${currentSessionId}/undo`, {
      steps: mode === 'human' ? 1 : 2,
    });
    updateUI(data);
  } catch (err) {
    alert(err.message);
    fetchState();
  }
};

document.getElementById('btn-flip').onclick = () => {
  boardOrientation = boardOrientation === 'white' ? 'black' : 'white';
  if (ground) ground.set({ orientation: boardOrientation });
};

document.getElementById('btn-ai').onclick = async () => {
  const mode = document.getElementById('ai-mode').value;
  const depth = mode === 'minimax-1' ? 1 : 2;
  const useSf = mode === 'stockfish';
  try {
    const data = await sendJSON(`/api/v1/sessions/${currentSessionId}/ai-move`, {
      depth,
      stockfish: useSf,
    });
    updateUI(data);
  } catch (err) {
    alert(err.message);
    fetchState();
  }
};

document.getElementById('fen-input').onchange = async (e) => {
  const fen = e.target.value.trim();
  if (!fen) return;
  try {
    const data = await sendJSON(`/api/v1/sessions/${currentSessionId}/reset`, { fen });
    updateUI(data);
  } catch (err) {
    alert(err.message);
    fetchState();
  }
};

// Initial load: list sessions, restore the stored id when it still exists,
// otherwise create a fresh session; then fetch the active state.
(async () => {
  await refreshSessions();
  const select = document.getElementById('sessions-select');
  const stored = loadStoredSessionId();
  if (stored !== null && sessionsHaveId(stored)) {
    currentSessionId = stored;
  } else if (select.value) {
    currentSessionId = Number(select.value);
  } else {
    try {
      const data = await sendJSON('/api/v1/sessions', {});
      currentSessionId = data.id;
      await refreshSessions();
    } catch (err) {
      console.error('Failed to create a session', err);
      return;
    }
  }
  saveSessionId();
  select.value = String(currentSessionId);
  fetchState();

  async function sessionsHaveId(id) {
    try {
      const res = await fetch(`/api/v1/sessions/${id}`);
      return res.ok;
    } catch {
      return false;
    }
  }
})();
