import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import referee as R
import jieqi_engine as E

def engine_idx(row, col):
    return (row + 3) * 16 + (col + 3)

def play_and_catch(seed, think=0.3, max_ply=200):
    board, secret = R.new_game(seed)
    trackers = {s: R.PerpCheckTracker() for s in "rb"}
    eng = E.JieQiEngine()
    for ply in range(max_ply):
        side = "r" if ply % 2 == 0 else "b"
        view = board if side == "r" else R.flip_board(board)
        moves = R.legal_engine_moves(view, side)
        if not moves:
            return ("no_legal", ply, side, None, None, None)
        cs = trackers[side].state_for()
        uci, score, depth = eng.get_best_move(view, side, think_time=think, check_state=cs)
        if uci is None:
            return ("no_move", ply, side, None, None, None)
        src_v, dst_v = R.uci_to_view_rc(uci)
        legal = R.is_legal(view, side, src_v, dst_v)
        if not legal:
            # dump
            st = E.State.from_string(E.board_to_engine_string(view, side))
            src_i = engine_idx(*src_v)
            dst_i = engine_idx(*dst_v)
            piece_at_src = st.board[src_i]
            # 裁判真身
            tru = secret.get(src_v, None)
            return ("ILLEGAL", ply, side, uci, (src_v, dst_v), {
                "engine_piece_at_src": piece_at_src,
                "referee_true_type_at_src": tru,
                "src_engine_idx": src_i, "dst_engine_idx": dst_i,
                "stm": st.stm,
                "board_str_view": [row[:] for row in view],
            })
        src = src_v if side == "r" else R.rc_flip(src_v)
        dst = dst_v if side == "r" else R.rc_flip(dst_v)
        captured, reveal = R.apply_move(board, secret, side, src, dst)
        next_side = "b" if side == "r" else "r"
        for s in "rb":
            trackers[s].on_any_move(src, dst, captured != ".")
        if R.in_check(board, next_side):
            trackers[side].deliver_check(dst)
        else:
            trackers[side].reset()
        if captured in R.KING.values():
            return ("king_captured", ply, side, uci, (src_v, dst_v), None)
    return ("max_ply", None, None, None, None, None)

if __name__ == "__main__":
    seeds = [330013, 330026, 330065, 330104, 330117, 330130, 330052, 330200, 330333, 330001, 330002, 330050]
    out = []
    for s in seeds:
        for trial in range(3):
            res = play_and_catch(s, think=0.3)
            if res[0] == "ILLEGAL":
                out.append((s, trial, res))
                print(f"[ILLEGAL] seed={s} trial={trial} ply={res[2]} side={res[3]} uci={res[4]} diag={json.dumps(res[5], ensure_ascii=False)}")
            elif res[0] in ("no_legal", "no_move"):
                print(f"[{res[0]}] seed={s} trial={trial} ply={res[1]} side={res[2]}")
    print("done, illegal count=", len(out))
