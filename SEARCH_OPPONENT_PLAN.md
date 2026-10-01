# Search engine architecture

## Public configuration

`chess.search.EngineConfig` controls the built-in engine. Its defaults are
depth 3, quiescence enabled with a three-ply tactical horizon, a 65,536-entry
transposition table, and the `basic` evaluator. Search depth is bounded to
1–8. `chess.engine.choose_move` accepts a per-call depth/config override;
`SearchEngine` retains its table and move-ordering heuristics between calls and
supports `clear()` at match boundaries. The `positional` profile is opt-in
pending stronger evidence of playing-strength improvement.

The terminal, web, Stockfish-match, and UCI entry points expose depth settings.
Web move bodies may override the app's default depth. The UCI server advertises
depth and quiescence options and returns the move from the last completed
iterative-deepening iteration when its clock expires.

## Search behavior

- White-positive alpha-beta minimax with iterative deepening.
- Transposition hash move first, then captures/promotions, then killer and
  history moves; bounded storage uses exact/lower/upper score bounds.
- Mate scores include distance and are normalized when stored in the table.
- Quiescence is enabled by default. It searches captures and promotions, and
  searches all legal evasions while in check.
- Quiescence move generation is tactical-only: in a non-check node it asks
  `ChessGame._has_legal_move` (short-circuits on the first legal move) for the
  stalemate test, then generates only legal captures and promotions, so quiet
  candidates are never executed for king-safety filtering. Check nodes still
  generate every legal evasion.
- The table key includes board, turn, castling rights, en-passant state,
  halfmove clock, full repetition context, quiescence settings, and evaluator
  profile. Full Python key equality resolves hash collisions; the table does
  not depend on Zobrist hashes.
- Search-generated transitions defer status work until legal moves or a
  terminal score need it. `Board._updated` copies only affected rows and
  refreshes king caches only when a king moves. `Move.execute` and castling
  generation read squares with `Board._get_fast` and apply changes with
  `Board._updated_validated`, skipping coordinate validation for squares that
  the generator itself produced.

## Evaluation profiles

`basic` preserves material plus the original sparse piece-square tables.
`positional` adds phase-tapered king activity, bishop-pair value, and
isolated/doubled/passed-pawn terms. Match evidence remains exploratory:
the positional profile leads in one balanced sample against weakened Stockfish,
but the paired uncertainty includes no difference; `basic` remains the default.

## Measurement and verification

- Start-position perft: 20 / 400 / 8,902 nodes at depths 1 / 2 / 3.
- Kiwipete perft: 48 / 2,039 nodes at depths 1 / 2.
- `uv run --locked python -m chess.bench --depth 3 --repeats 3` measures cold
  searches over fixed start, tactical, middlegame, and endgame positions. It
  reports node counts, TT hits, elapsed time, and NPS without timing thresholds.
- Three serial depth-3 benchmark runs with quiescence enabled measured mean
  `basic` / `positional` elapsed time (ms): start 104.7 / 111.6, tactical
  493.0 / 550.1, middlegame 751.5 / 964.5, endgame 11.6 / 13.4. The positional
  profile took 7–28% longer in these samples. Start/tactical/middlegame choices
  matched; the endgame choice differed (`g2g1` / `g2f3`). Timing is observational.
- One Python 3.14.7 cProfile run of the default `basic` depth-3 search recorded
  885,786 calls; legal generation accumulated 0.182 s across 827 calls,
  `Move.execute` 0.098 s across 21,583 calls, and `_updated_validated` 0.043 s.
  This is one instrumented run, not a timing gate.
- Three 20,000-update samples measured `_updated_validated` at 16.4–16.6 ms;
  applying the same row copies followed by `Board(...)` validation took
  59.4–59.7 ms. This isolates transition validation, not end-to-end search.
- Earlier same-engine self-play was inconclusive: four depth-2 games capped at
  20 full moves and twelve depth-3 games capped at 60 plies all drew.
- Extended match set: 64 games against Stockfish 19 Skill Level 0 (one thread,
  16 MB hash, 40 ms/move), with pychess at fixed depth 3 plus quiescence. Four
  opening lines, four repeats, and both colors were balanced per profile;
  Full opening histories were replayed from start: open game `e4 e5 Nf3 Nc6`;
  Queen's Gambit `d4 d5 c4 e6 Nc3 Nf6`; Sicilian
  `e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6`; English
  `c4 e5 Nc3 Nf6 g3 d5 cxd5 Nxd5`.
  Pychess had no wall-clock limit (mean 409 ms/move for `basic`, 503 ms/move for
  `positional`); this is a fixed-depth comparison, not an equal-time strength test.
  Results: `basic` 19-6-7, 22/32 points (68.8%); `positional` 23-6-3, 26/32
  (81.2%).
  By color, basic scored 11/16 with either side; positional scored 13/16
  with either side.
- In 32 opening/color/repeat pairs, positional scored better in 11, basic in 7,
  and 14 tied. The rough paired 95% t interval for positional-minus-basic score
  is -0.08 to +0.33 points per pair; it includes zero. By opening, basic /
  positional points out of 8 were English 6.5 / 7, open game 4 / 5.5, Queen's
  Gambit 5 / 6.5, and Sicilian 6.5 / 7. Outcomes: 52 checkmates and 12
  experimental 100-ply caps. This single weakened-opponent setting does not
  establish general strength; keep `basic` as default pending replication.

- Equal-requested-time replication: `chess.match_bench` ran 128 games against
  Stockfish 19 (one thread, 16 MB hash), skills 0 and 5, across four openings,
  four repeats, both colors, and both profiles. Seed 211; 100 ms/move requested
  for each engine; maximum 100 full moves. No forfeits; 117 checkmates, 10
  move-limit draws, and one insufficient-material draw. Results (W-D-L):
  `basic` 16-8-40, 20/64 points (31.2%); `positional` 16-3-45, 17.5/64
  (27.3%). By skill, basic / positional scored 18.5/32 / 16.5/32 against skill
  0, and 1.5/32 / 1/32 against skill 5.
  Across 64 matched opening/repeat/color/skill cells, positional scored better
  in 11, basic in 13, and 40 tied; the rough paired 95% t interval for
  positional-minus-basic score is -0.164 to +0.085 points and includes zero.
  Points by color (White / Black): basic 7/32 / 13/32, positional 7.5/32 /
  10/32. Points by opening (basic / positional, out of 16): English 5.5 / 6,
  open game 6 / 5, Queen's Gambit 5 / 3.5, Sicilian 3.5 / 3.
  Pychess averaged 99.7 ms/move (`basic`) and 99.4 ms (`positional`); Stockfish
  call wall time averaged 95.7 ms. Pychess's median was about 100.1 ms, with
  rare outliers up to 170.7 ms, so its budget is cooperative rather than a hard
  wall-clock cap. Mean completed depth was 2.18 (`basic`) and 1.93
  (`positional`); the positional evaluator reached fewer deeper iterations in
  this equal-time sample. These results still do not establish a general
  strength difference. Raw UCI games and per-ply records:
  `/tmp/pychess-balanced-100ms-s211-2026-10-14.json`.
- One positional depth-3 cProfile run over the four fixed benchmark positions
  recorded 13.5M calls in 3.20 s under instrumentation. Legal move generation
  accumulated 2.49 s across 9,370 calls; `Move.execute` accumulated 1.42 s;
  `evaluate` accumulated 0.41 s across 9,558 calls. This sample points to move
  generation/transitions, not evaluator terms, as the larger optimization
  target; it is not an uninstrumented timing comparison.
- Tactical-only quiescence generation (see Search behavior) returned identical
  scores, node counts, and best moves on the four fixed positions for both
  profiles, and cut depth-3 time by 1.7–3.8x (`basic` ms, before / after: start
  97 / 28, tactical 461 / 157, middlegame 689 / 208, endgame 11.0 / 5.9).
  `tests/test_rules.py` compares it with filtered full generation over a
  tree walk that includes promotions, pins, en passant, mate, and stalemate.
- Same seed and schedule as the 128-game equal-time run, after that change:
  mean completed depth rose from 2.18 to 2.88 (`basic`) and 1.92 to 2.74
  (`positional`). Points out of 64 (baseline → now): `basic` 20.0 → 15.0,
  `positional` 17.5 → 21.5. Pooled change per matched game is -0.008 (95% CI
  -0.098 to +0.082): no detectable strength change from the extra depth at this
  sample size. Positional minus basic is now +0.102 per matched cell (95% CI
  -0.013 to +0.217; positional better in 15, basic in 9, 40 tied), consistent
  with the profile's speed penalty shrinking once depth is similar, but still
  inconclusive. Pychess overshot the 100 ms budget at p99 142 ms, max 226 ms.
  Raw data: `/tmp/pychess-fastq-100ms-s211.json`.
- Pruning experiment, removed: null-move (R = 2), late-move reductions, and
  ±50 cp aspiration windows were implemented and measured as Stockfish
  depth-12 centipawn loss of pychess's chosen move on 291 positions sampled
  from recorded games (100 ms/move; 197 positions at 400 ms). Repeating an
  identical configuration changed mean loss by 1–16 cp, more than any gap
  between configurations (100 ms after the quiescence change: no pruning
  105.3 / 104.0, all three 106.8 / 103.8; 400 ms: 81.3 / 97.0 vs 83.9 / 90.6).
  Completed depth rose only 3.20 → 3.49 at 400 ms. With no measurable gain, the
  code and its zugzwang/mate-score risks were not kept; reconsider when
  searches routinely complete depth 5+. The same centipawn-loss harness
  showed the quiescence change itself lowering mean loss from about 129 to
  105 cp at 100 ms (depth 1.85 → 2.42), though that comparison was not
  repeated across runs.
- Pin-aware legality and cheaper transitions: outside check, `_legal_candidates`
  simulates only king moves, pinned-piece moves, and en passant (every move
  while in check); castling is accepted as generated because `_castling_moves`
  already proves its path safe. Tactical generation now skips constructing
  quiet moves, `Board._moved` handles two-square moves and updates the king
  cache without a board rescan, `_basic_evaluate` reads precomputed per-piece
  square scores, and `_is_insufficient_material` / `_next_rights` exit early.
  Ordered legal and tactical move lists were byte-identical to the previous
  generator over a 181,921-node tree walk of eight perft/en-passant/pin
  positions, and node/qnode counts and best moves were unchanged on the four
  benchmark positions. Depth-3 `basic` cold search, ms (before / after): start
  28 / 17.5, tactical 148 / 89, middlegame 187 / 120, endgame 5.6 / 4.0 (NPS
  28k / 46k on start and middlegame). `tests/test_rules.py` compares legal
  generation with brute-force execute-and-reject, including pinned pieces and
  an en passant that would expose the king along a rank.

## Deferred optimization

The engine keeps immutable state transitions as the correctness baseline.
Incremental hashes, make/unmake, cached evaluation accumulators, and more
aggressive pruning remain deferred until a representative profile shows
transition cost still dominates and rule-state restoration can be tested
independently.
