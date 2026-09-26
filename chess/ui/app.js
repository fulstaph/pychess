import { Chessground } from 'https://cdn.jsdelivr.net/npm/@lichess-org/chessground/+esm';

// ---------------------------------------------------------------------------
// Module state: the Chessground board instance, the last /api/state payload,
// a deferred promotion awaiting a piece choice, and current orientation.
// ---------------------------------------------------------------------------
let ground = null;
let currentState = null;
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

// State polling: fetchState pulls /api/state and pushes it through updateUI.
export async function fetchState() {
  const res = await fetch('/api/state');
  const data = await res.json();
  updateUI(data);
}

// Board init + full UI refresh: creates the Chessground board on first call,
// then re-syncs FEN/orientation/movable state afterwards.
export function updateUI(data) {
  currentState = data;
  // Legal-move highlighting: server dests maps each origin square to
  // its list of legal destinations, consumed by Chessground's movable.
  const destsMap = new Map();
  if (data.dests) {
    for (const [from, toList] of Object.entries(data.dests)) {
      destsMap.set(from, toList);
    }
  }

  const turnColor = data.turn === 'w' ? 'white' : 'black';
  const aiMode = document.getElementById('ai-mode').value;
  const isHumanTurn =
    aiMode === 'human' ||
    data.turn === (boardOrientation === 'white' ? 'w' : 'b');

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
  } else if (data.is_check) {
    stateCard.classList.add('status-check');
    document.getElementById('state-text').textContent = 'CHECK';
  } else if (data.is_draw) {
    document.getElementById('state-text').textContent = 'DRAW';
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

// Move submission: POST /api/move with the mode's AI reply, depth, and
// Stockfish flag; refreshes state (or re-syncs after a server error).
export async function submitMove(moveStr) {
  const mode = document.getElementById('ai-mode').value;
  const aiReply = mode !== 'human';
  const depth = mode === 'minimax-1' ? 1 : 2;
  const useSf = mode === 'stockfish';

  playBeep(220, 'triangle', 0.05);

  const res = await fetch('/api/move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      move: moveStr,
      ai_reply: aiReply,
      depth: depth,
      stockfish: useSf,
    }),
  });
  const data = await res.json();
  if (data.error) {
    alert(data.error);
    fetchState();
  } else {
    if (data.is_check) playBeep(520, 'sine', 0.12);
    updateUI(data);
  }
}

// Event listeners: promotion dialog, and the undo/flip/ai/new/FEN controls.
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
  const res = await fetch('/api/undo', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ steps: mode === 'human' ? 1 : 2 }),
  });
  updateUI(await res.json());
};

document.getElementById('btn-flip').onclick = () => {
  boardOrientation = boardOrientation === 'white' ? 'black' : 'white';
  if (ground) ground.set({ orientation: boardOrientation });
};

document.getElementById('btn-new').onclick = async () => {
  const res = await fetch('/api/reset', { method: 'POST' });
  updateUI(await res.json());
};

document.getElementById('btn-ai').onclick = async () => {
  const mode = document.getElementById('ai-mode').value;
  const depth = mode === 'minimax-1' ? 1 : 2;
  const useSf = mode === 'stockfish';
  const res = await fetch('/api/ai_move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ depth, stockfish: useSf }),
  });
  updateUI(await res.json());
};

document.getElementById('fen-input').onchange = async (e) => {
  const fen = e.target.value.trim();
  if (!fen) return;
  const res = await fetch('/api/reset', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fen }),
  });
  const data = await res.json();
  if (data.error) alert(data.error);
  else updateUI(data);
};

// Initial load
fetchState();
