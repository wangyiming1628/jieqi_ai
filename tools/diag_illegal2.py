import sys, os, random, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import referee as R
import jieqi_engine as E

def engine_idx(row, col):
    return (row + 3) * 16 + (col + 3)

def check_move_in_gen(view, side, uci, check_state):
    """Return (ok_engine, ok_referee, detail)."""
    st = E.State.from_string(E.board_to_engine_string(view, side))
    gen = set(st.gen_moves())
    src_v, dst_v = R.uci_to_view_rc(uci)
    mv = (engine_idx(*src_v), engine_idx(*dst_v))
    ok_engine = mv in gen
    ok_ref = R.is_legal(view, side, src_v, dst_v)
    return ok_engine, ok_ref, (src_v, dst_v, mv in gen)

def random_pos(seed):
    rng = random.Random(seed)
    board, secret = R.new_game(rng.randint(0, 10**9))
    # random walk a few plies to get a midgame position
    trackers = {s: R.PerpCheckTracker() for s in "rb"}
    eng = E.JieQiEngine()
    for ply in range(rng.randint(5, 25)):
        side = "r" if ply % 2 == 0 else "b"
        view = board if side == "r" else R.flip_board(board)
        if not R.legal_engine_moves(view, side):
            break
        cs = trackers[side].state_for()
        uci, _, _ = eng.get_best_move(view, side, think_time=0.02, check_state=cs)
        if uci is None:
            break
        src_v, dst_v = R.uci_to_view_rc(uci)
        if not R.is_legal(view, side, src_v, dst_v):
            break
        src = src_v if side == "r" else R.rc_flip(src_v)
        dst = dst_v if side == "r" else R.rc_flip(dst_v)
        cap, rev = R.apply_move(board, secret, side, src, dst)
        ns = "b" if side == "r" else "r"
        for s in "rb":
            trackers[s].on_any_move(src, dst, cap != ".")
        if R.in_check(board, ns):
            trackers[side].deliver_check(dst)
        else:
            trackers[side].reset()
    return board, secret, trackers

def run(mode, n=400, think=0.05):
    eng = E.JieQiEngine()
    cnt_bad = 0
    examples = []
    for i in range(n):
        board, secret, trackers = random_pos(i * 7 + 1)
        side = "r" if i % 2 == 0 else "b"
        view = board if side == "r" else R.flip_board(board)
        if not R.legal_engine_moves(view, side):
            continue
        cs = trackers[side].state_for() if mode == "ref" else None
        uci, score, depth = eng.get_best_move(view, side, think_time=think, check_state=cs)
        if uci is None:
            continue
        ok_e, ok_r, det = check_move_in_gen(view, side, uci, cs)
        if not ok_e or not ok_r:
            cnt_bad += 1
            if len(examples) < 5:
                examples.append({
                    "mode": mode, "i": i, "side": side, "uci": uci,
                    "ok_engine": ok_e, "ok_referee": ok_r, "detail": det,
                    "view": [row[:] for row in view],
                })
    return cnt_bad, examples

if __name__ == "__main__":
    for mode in ("live", "ref"):
        bad, ex = run(mode, n=400, think=0.05)
        print(f"mode={mode} illegal_returned={bad}")
        for e in ex:
            print("  EX", json.dumps(e, ensure_ascii=False))
