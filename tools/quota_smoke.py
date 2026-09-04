import importlib
import jieqi_engine as new_mod
import jieqi_engine_v514 as base_mod
from jieqi_engine import _PerpQuota, _engine_idx_to_row_col, _row_col_to_engine_idx

# 1) tracker unit: mirrors referee.PerpCheckTracker (exceeded = count > quota)
q = _PerpQuota(count=3, sq={(5, 4): True}, retired=0)   # quota = min(3*1,9)=3
assert q.quota == 3
q2 = q.copy(); q2.on_any_move((5, 4), (5, 5), False); q2.deliver_check((5, 5))
assert q2.exceeded(), "4th consecutive check must exceed quota"
q3 = q.copy(); q3.on_any_move((5, 4), (5, 5), False); q3.reset()
assert q3.count == 0 and not q3.exceeded()
# multi-piece: 2 pieces -> quota 6; count==quota is allowed (exceeded is strict >)
q4 = _PerpQuota(count=5, sq={(5, 4): True, (6, 1): True}, retired=0)
assert q4.quota == 6 and not q4.exceeded()
q5 = q4.copy(); q5.deliver_check((6, 1)); assert not q5.exceeded()   # 6 == quota -> allowed
q5b = q4.copy(); q5b.deliver_check((6, 2)); assert not q5b.exceeded()  # 6 vs quota 9 -> allowed
# strict-exceed at the boundary: count 6 (==quota) then one more check -> 7 > 6
q6 = _PerpQuota(count=6, sq={(5, 4): True}, retired=0)
q6b = q6.copy(); q6b.deliver_check((5, 5)); assert q6b.exceeded()     # 7 > 6 -> banned
print("[1] _PerpQuota unit OK")

# 2) opening smoke: both engines run, identical move (pruning inert at count=0)
new_e = new_mod.JieQiEngine()
base_e = base_mod.JieQiEngine()

def start_board():
    b = [["."] * 9 for _ in range(10)]
    for c in range(9):
        b[9][c] = "r帥" if c == 4 else "r?"
        b[0][c] = "b將" if c == 4 else "b?"
    for c in (0, 2, 4, 6, 8):
        b[6][c] = "r?"; b[3][c] = "b?"
    return b

b = start_board()
n_uci, n_sc, n_dp = new_e.get_best_move(b, "r", think_time=0.3)
o_uci, o_sc, o_dp = base_e.get_best_move(b, "r", think_time=0.3)
print(f"[2] opening  new={n_uci}({n_sc},{n_dp})  base={o_uci}({o_sc},{o_dp})")
assert n_uci == o_uci, "normal position must be byte-identical (pruning inert)"
print("[2] opening byte-identical OK")

# 3) referee-mode check_state: count=3 single piece -> any further check banned.
#    engine must return a legal (non-banned) move from search, depth != -2.
cs = {"count": 3, "squares": [[5, 4]], "retired": 0}
u, s, d = new_e.get_best_move(b, "r", think_time=0.3, check_state=cs)
print(f"[3] check_state count=3 -> {u} ({s},{d})")
assert u is not None and d != -2, "must come from search, not 1-ply fallback"
print("[3] ran without crash OK")

# 4) functional: a banned check must be pruned, not played.
def uci_to_move(uci):
    fcol = ord(uci[0]) - ord('a') + 3; frow = 12 - int(uci[1])
    tcol = ord(uci[2]) - ord('a') + 3; trow = 12 - int(uci[3])
    return (_row_col_to_engine_idx(frow, fcol), _row_col_to_engine_idx(trow, tcol))

b2 = start_board()
b2[0][4] = "b將"          # 黑王
b2[3][4] = "r车"          # 红车(已翻) 可沿第4列照将
cs = {"count": 3, "squares": [[3, 4]], "retired": 0}   # 单车配额=3, 再照即第4将→判负
u, s, d = new_e.get_best_move(b2, "r", think_time=0.3, check_state=cs)
mv = uci_to_move(u)
st2 = new_mod.State.from_string(new_mod.board_to_engine_string(b2, "r"))
assert not st2.gives_check(mv), f"禁忌将军着 {u} 竟被走出 (count=3 应被剪枝)!"
print(f"[4] functional: 禁忌将军着已被剪枝, 改走 {u} (depth={d})")

print("\nALL SMOKE TESTS PASSED")
