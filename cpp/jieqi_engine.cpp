// 揭棋引擎 C++ 移植 (v5.17) —— 与 jieqi_engine.py 逐行同构
// 移植范围: State(增量评估/make/unmake/gen_moves)、Searcher(PVS/LMR/nullmove/TT/
//           qsearch/双时限)、DarkPoolTracker、长将配额安检(裁判+实战模式)、
//           JSON stdio 服务端 (协议与 engine_server.py 完全一致)。
// 等价性: 固定深度下与 Python 版节点数/根值/着法逐字节相同 (见 tools 测试)。
// 编译: g++ -O2 -std=c++17 -ffp-contract=off -o jieqi_engine jieqi_engine.cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <tuple>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include "pst_tables.h"
#include "piece_map.h"

using std::string;
using std::vector;
using std::pair;
using std::make_pair;

// ---------------- 常量 ----------------
static const int PIECE_VALUE_P = 44, PIECE_VALUE_N = 108, PIECE_VALUE_B = 23,
                 PIECE_VALUE_R = 233, PIECE_VALUE_A = 23, PIECE_VALUE_C = 101,
                 PIECE_VALUE_K = 2500;
static const int MATE_LOWER = PIECE_VALUE_K - (2*PIECE_VALUE_R + 2*PIECE_VALUE_N + 2*PIECE_VALUE_B + 2*PIECE_VALUE_A + 2*PIECE_VALUE_C + 5*PIECE_VALUE_P);
static const int MATE_UPPER = PIECE_VALUE_K + (2*PIECE_VALUE_R + 2*PIECE_VALUE_N + 2*PIECE_VALUE_B + 2*PIECE_VALUE_A + 2*PIECE_VALUE_C + 5*PIECE_VALUE_P);
static const double TABLE_SIZE = 1e7;

// MVV-LVA 排序基值 (非评估值)
static int MVV_BASE(int ch) {
    switch (ch) {
        case 'P': case 'I': return 100;
        case 'N': case 'B': case 'A': case 'C':
        case 'E': case 'F': case 'G': case 'H': return 300;
        case 'R': case 'D': return 500;
        case 'K': return 10000;
        case 'U': return 200;
        default: return 200;
    }
}

// 方向
static const int DN = -16, DE = 1, DS = 16, DW = -1;

// 大写方 (红/己方) 走法方向表
static const int DIR_UP(char p) {
    switch (p) {
        case 'P': return 0;  // (N, W, E) 特判
        default: return 0;
    }
}
// 用查表: 每个字母一组方向
struct DirTable { const int* d; int n; };
static const int DIR_P_UP[]   = {DN, DW, DE};
static const int DIR_I_UP[]   = {DN};
static const int DIR_N_UP[]   = {DN+DN+DE, DE+DN+DE, DE+DS+DE, DS+DS+DE, DS+DS+DW, DW+DS+DW, DW+DN+DW, DN+DN+DW};
static const int DIR_E_UP[]   = {DN+DN+DE, DE+DN+DE, DW+DN+DW, DN+DN+DW};
static const int DIR_B_UP[]   = {2*DN+2*DE, 2*DS+2*DE, 2*DS+2*DW, 2*DN+2*DW};
static const int DIR_F_UP[]   = {2*DN+2*DE, 2*DN+2*DW};
static const int DIR_R_UP[]   = {DN, DE, DS, DW};
static const int DIR_D_UP[]   = {DN, DE, DW};
static const int DIR_C_UP[]   = {DN, DE, DS, DW};
static const int DIR_H_UP[]   = {DN, DE, DS, DW};
static const int DIR_A_UP[]   = {DN+DE, DS+DE, DS+DW, DN+DW};
static const int DIR_G_UP[]   = {DN+DE, DN+DW};
static const int DIR_K_UP[]   = {DN, DE, DS, DW};
static const int DIR_p_LW[]   = {DS, DE, DW};
static const int DIR_i_LW[]   = {DS};
static const int DIR_n_LW[]   = {DS+DS+DW, DW+DS+DW, DW+DN+DW, DN+DN+DW, DN+DN+DE, DE+DN+DE, DE+DS+DE, DS+DS+DE};
static const int DIR_e_LW[]   = {DS+DS+DW, DW+DS+DW, DE+DS+DE, DS+DS+DE};
static const int DIR_b_LW[]   = {2*DS+2*DW, 2*DN+2*DW, 2*DN+2*DE, 2*DS+2*DE};
static const int DIR_f_LW[]   = {2*DS+2*DW, 2*DS+2*DE};
static const int DIR_r_LW[]   = {DS, DW, DN, DE};
static const int DIR_d_LW[]   = {DS, DW, DE};
static const int DIR_c_LW[]   = {DS, DW, DN, DE};
static const int DIR_h_LW[]   = {DS, DW, DN, DE};
static const int DIR_a_LW[]   = {DS+DW, DN+DW, DN+DE, DS+DE};
static const int DIR_g_LW[]   = {DS+DW, DS+DE};
static const int DIR_k_LW[]   = {DS, DW, DN, DE};

static DirTable DIRS_UP(char p) {
    switch (p) {
        case 'P': return {DIR_P_UP, 3};
        case 'I': return {DIR_I_UP, 1};
        case 'N': return {DIR_N_UP, 8};
        case 'E': return {DIR_E_UP, 4};
        case 'B': return {DIR_B_UP, 4};
        case 'F': return {DIR_F_UP, 2};
        case 'R': return {DIR_R_UP, 4};
        case 'D': return {DIR_D_UP, 3};
        case 'C': return {DIR_C_UP, 4};
        case 'H': return {DIR_H_UP, 4};
        case 'A': return {DIR_A_UP, 4};
        case 'G': return {DIR_G_UP, 2};
        case 'K': return {DIR_K_UP, 4};
    }
    return {nullptr, 0};
}
static DirTable DIRS_LW(char p) {
    switch (p) {
        case 'p': return {DIR_p_LW, 3};
        case 'i': return {DIR_i_LW, 1};
        case 'n': return {DIR_n_LW, 8};
        case 'e': return {DIR_e_LW, 4};
        case 'b': return {DIR_b_LW, 4};
        case 'f': return {DIR_f_LW, 2};
        case 'r': return {DIR_r_LW, 4};
        case 'd': return {DIR_d_LW, 3};
        case 'c': return {DIR_c_LW, 4};
        case 'h': return {DIR_h_LW, 4};
        case 'a': return {DIR_a_LW, 4};
        case 'g': return {DIR_g_LW, 2};
        case 'k': return {DIR_k_LW, 4};
    }
    return {nullptr, 0};
}

static const int A0 = 12*16+3, I0 = 12*16+11, A9 = 3*16+3, I9 = 3*16+11;
static inline bool is_upper_ch(char c) { return c >= 'A' && c <= 'Z'; }
static inline bool is_lower_ch(char c) { return c >= 'a' && c <= 'z'; }
static inline bool is_space_ch(char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }
static inline char to_up(char c) { return is_lower_ch(c) ? char(c - 32) : c; }
static inline char to_lo(char c) { return is_upper_ch(c) ? char(c + 32) : c; }
static inline char swapcase_ch(char c) {
    if (is_upper_ch(c)) return (char)(c + 32);
    if (is_lower_ch(c)) return (char)(c - 32);
    return c;
}

static const char LIGHT_UP[] = "RNBAKCP";
static inline bool in_LIGHT_UP(char c) { return strchr(LIGHT_UP, c) != nullptr && c != 0; }
static const char BACK_SET_U[] = "DEFGRNC";
static const char BACK_SET_L[] = "defgrnc";
static const char DARK_UPPER[] = "DEFGHI";
static const char DARK_LOWER[] = "defghi";
static const char LIGHT_UPPER_POOL[] = "RNBACP";   // 入池明子 (不含 K)
static const char LIGHT_LOWER_POOL[] = "rnbacp";
static inline bool in_DARK_UP(char c) { return c >= 'D' && c <= 'I'; }
static inline bool in_DARK_LW(char c) { return c >= 'd' && c <= 'i'; }
static const char U_AFTER_MOVE[] = "DEFGHIUdefghiu";
static const char REVEAL_TYPES[] = "RNBACP";
static const int POOL_INIT_IDX[6] = {2, 2, 2, 2, 2, 5};   // R N B A C P
static const char POOL_KEYS[6] = {'R', 'N', 'B', 'A', 'C', 'P'};
static const double DISCOUNT = 1.5;

// ---------------- Zobrist ----------------
// Python 版用 random.Random(20260824) 的 MT19937; 这里用固定种子 splitmix64。
// TT 键只要求进程内自洽, 不同键序列仅影响 (概率~0 的) 哈希碰撞, 不影响搜索树。
static uint64_t sm_state = 0x20260824ull;
static uint64_t sm_next() {
    uint64_t z = (sm_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static uint64_t ZOB[256][256];      // [棋子char][格idx]
static uint64_t Z_STM, Z_TURN;
static void zobrist_init() {
    static const char* pieces = "RNBAKCPDEFGHIU" "rnbakcpdefghiu";
    for (const char* pc = pieces; *pc; ++pc)
        for (int i = 0; i < 256; i++) ZOB[(unsigned char)*pc][i] = sm_next();
    Z_STM = sm_next();
    Z_TURN = sm_next();
}

// ---------------- 全局池/期望表 (对应 Python 的 r/b/di/sumall/average) ----------------
static double g_di[2][6];           // [turn(0=upper方池,1=lower方池)][R N B A C P]
static double g_sumall[2];
static int    g_avg_cov[2];         // average[version][turn][False]
static int    g_avg_u[2][256];      // average[version][turn][True][i]
static int    g_type_idx(char t) {
    switch (to_up(t)) {
        case 'R': return 0; case 'N': return 1; case 'B': return 2;
        case 'A': return 3; case 'C': return 4; case 'P': return 5;
    }
    return -1;
}
static inline int pst1(double key_idx) { return 0; }  // 占位(未用)

// pst["1"][type] —— 直接用生成的常量
static double PST1_OF(char t) {
    switch (t) {
        case 'R': return PST1_R; case 'N': return PST1_N; case 'B': return PST1_B;
        case 'A': return PST1_A; case 'C': return PST1_C; case 'P': return PST1_P;
    }
    return 0;
}
static const int* PST_OF(char t) {
    switch (t) {
        case 'R': return PST_R; case 'N': return PST_N; case 'B': return PST_B;
        case 'A': return PST_A; case 'C': return PST_C; case 'P': return PST_P;
        case 'K': return PST_K;
    }
    return nullptr;
}
// Python round(): 银行家舍入 (half-to-even) —— lrint 在默认 FP 舍入模式下等价
static inline int py_round(double v) { return (int)std::lrint(v); }

// ---------------- State ----------------
struct Undo { int i, j; char src, dst; double score; uint64_t hkey; };

struct State {
    char board[256];
    bool stm, turn;
    int version;
    double score;
    uint64_t hkey;
    vector<Undo> undo;
    long long rough;
    int che_u, che_l, zu_u, zu_l, cov_u, cov_l, back_u, back_l;
    int rnci_u[4], rnci_l[4];   // R N C I
    int kt_u, kt_l, kts_u, kts_l;

    State(bool stm_ = true, bool turn_ = true, int version_ = 0)
        : stm(stm_), turn(turn_), version(version_), score(0), hkey(0),
          rough(0), che_u(0), che_l(0), zu_u(0), zu_l(0),
          cov_u(0), cov_l(0), back_u(0), back_l(0),
          kt_u(0), kt_l(0), kts_u(0), kts_l(0) {
        memset(board, ' ', 256);
        for (int i = 15; i < 256; i += 16) board[i] = '\n';
        rnci_u[0] = rnci_u[1] = rnci_u[2] = rnci_u[3] = 0;
        rnci_l[0] = rnci_l[1] = rnci_l[2] = rnci_l[3] = 0;
    }

    static State from_string(const string& s, int version_ = 0) {
        State st(true, true, version_);
        for (int i = 0; i < 256 && i < (int)s.size(); i++) st.board[i] = s[i];
        uint64_t h = 0;
        for (int i = 0; i < 256; i++) {
            char ch = st.board[i];
            if (!(ch == '.' || is_space_ch(ch))) {
                st._pstats(ch, i, +1);
                h ^= ZOB[(unsigned char)ch][i];
            }
        }
        st.hkey = h;
        st._kongtou();
        return st;
    }

    string board_str() const { return string(board, board + 256); }

    // ---- 增量统计 ----
    void _pstats(char ch, int i, int s) {
        if (ch == '.' || ch == '\n' || ch == ' ') return;
        if (is_upper_ch(ch)) {
            if (ch == 'U') {
                rough += (long long)s * g_avg_u[1][i];      // average[v][True][True][i]
                cov_u += s;
            } else if (in_LIGHT_UP(ch)) {
                rough += (long long)s * PST_OF(ch)[i];
                if (ch == 'R') { che_u += s; rnci_u[0] += s; }
                else if (ch == 'P') zu_u += s;
                else if (ch == 'N') rnci_u[1] += s;
                else if (ch == 'C') rnci_u[2] += s;
                // K: 无计数
            } else {                      // 暗子 DEFGHI
                cov_u += s;
                if (ch == 'I') rnci_u[3] += s;
            }
            if ((i >> 4) == 12 && strchr(BACK_SET_U, ch) && ch != 0)
                back_u += s;
        } else {
            if (ch == 'u') {
                rough -= (long long)s * g_avg_u[0][254 - i]; // average[v][False][True][254-i]
                cov_l += s;
            } else if (strchr("rnbakcp", ch) && ch != 0) {  // 仅明子, 不含暗子 defghi
                rough -= (long long)s * PST_OF(to_up(ch))[254 - i];
                if (ch == 'r') { che_l += s; rnci_l[0] += s; }
                else if (ch == 'p') zu_l += s;
                else if (ch == 'n') rnci_l[1] += s;
                else if (ch == 'c') rnci_l[2] += s;
            } else {                       // 暗子 defghi
                cov_l += s;
                if (ch == 'i') rnci_l[3] += s;
            }
            if ((i >> 4) == 3 && strchr(BACK_SET_L, ch) && ch != 0)
                back_l += s;
        }
    }

    // ---- 空头炮 ----
    void _kongtou() {
        int ku = 0, kl = 0;
        bool done_u = false, done_l = false;
        for (int i = 51; i < 204; i++) {
            if ((i & 15) != 7) continue;
            char ch = board[i];
            if (ch == 'C' && !done_u) {
                done_u = true;
                for (int sp = i - 16; sp > 51; sp -= 16) {
                    char q = board[sp];
                    if (is_space_ch(q)) break;
                    if (q == 'C') continue;
                    else if (q != '.') { if (q == 'k') ku++; break; }
                    else ku++;
                }
            } else if (ch == 'c' && !done_l) {
                done_l = true;
                for (int sp = i + 16; sp < 204; sp += 16) {
                    char q = board[sp];
                    if (is_space_ch(q)) break;
                    if (q == 'c') continue;
                    else if (q != '.') { if (q == 'K') kl++; break; }
                    else kl++;
                }
            }
        }
        kt_u = ku; kt_l = kl;
        int kts_u_ = 0, kts_l_ = 0;
        if ((ku > 0 && kl <= 0) || (ku > kl && kl > 0))
            kts_u_ = ((che_u >= che_l && che_u > 0) || ku >= 3) ? 100 : 70;
        else if ((kl > 0 && ku <= 0) || (kl > ku && ku > 0))
            kts_l_ = ((che_l >= che_u && che_l > 0) || kl >= 3) ? 100 : 70;
        kts_u = kts_u_; kts_l = kts_l_;
    }

    int back_m() const { return stm ? back_l : back_u; }   // endline_m

    double static_eval() {
        // 必须返回浮点: Python 的 static() 返回 score(可能是 .5 小数) + 空头炮分差,
        // 截断成 int 会让 stand-pat 判定与 Python 不同 (实测导致搜索树分歧)
        if (stm) return score + kts_u - kts_l;
        return score + kts_l - kts_u;
    }

    bool has_rnci() const {
        const int* t = stm ? rnci_u : rnci_l;
        return t[0] || t[1] || t[2] || t[3];
    }

    // ---- make / unmake ----
    void make(int i, int j) {
        char src_ch = board[i], dst_ch = board[j];
        double mv = value(i, j);
        double ns = (mv < MATE_UPPER) ? score + mv : (double)MATE_UPPER;
        undo.push_back({i, j, src_ch, dst_ch, score, hkey});
        _pstats(src_ch, i, -1);
        _pstats(dst_ch, j, -1);
        char new_ch;
        if (in_LIGHT_UP(to_up(src_ch))) new_ch = src_ch;
        else new_ch = stm ? 'U' : 'u';
        board[i] = '.';
        board[j] = new_ch;
        _pstats(new_ch, j, +1);
        score = -ns;
        hkey ^= ZOB[(unsigned char)src_ch][i] ^ ZOB[(unsigned char)dst_ch][j]
              ^ ZOB[(unsigned char)new_ch][j] ^ Z_STM ^ Z_TURN;
        stm = !stm;
        turn = !turn;
        _kongtou();
    }

    void unmake() {
        Undo u = undo.back(); undo.pop_back();
        char new_ch = board[u.j];
        score = u.score;
        hkey = u.hkey;
        stm = !stm;
        turn = !turn;
        _pstats(new_ch, u.j, -1);
        _pstats(u.src, u.i, +1);
        _pstats(u.dst, u.j, +1);
        board[u.i] = u.src;
        board[u.j] = u.dst;
        _kongtou();
    }

    void flip_stm() {
        stm = !stm;
        turn = !turn;
        score = -score;
        hkey ^= Z_STM ^ Z_TURN;
    }

    // ---- 走法生成 ----
    void gen_moves(bool upper, vector<pair<int,int>>& out) const {
        const char* bd = board;
        if (upper) {
            for (int i = 51; i < 204; i++) {
                char p = bd[i];
                if (!is_upper_ch(p) || p == 'U') continue;
                if (p == 'K') {
                    for (int sp = i - 16; sp > A9; sp -= 16) {
                        if (bd[sp] == 'k') out.push_back({i, sp});
                        else if (bd[sp] != '.') break;
                    }
                }
                if (p == 'C' || p == 'H') {
                    DirTable dt = DIRS_UP(p);
                    for (int di_ = 0; di_ < dt.n; di_++) {
                        int d = dt.d[di_];
                        int cfoot = 0;
                        for (int j = i + d; ; j += d) {
                            char q = bd[j];
                            if (is_space_ch(q)) break;
                            if (cfoot == 0 && q == '.') out.push_back({i, j});
                            else if (cfoot == 0 && q != '.') cfoot++;
                            else if (cfoot == 1 && is_lower_ch(q)) { out.push_back({i, j}); break; }
                            else if (cfoot == 1 && is_upper_ch(q)) break;
                        }
                    }
                    continue;
                }
                DirTable dt = DIRS_UP(p);
                for (int di_ = 0; di_ < dt.n; di_++) {
                    int d = dt.d[di_];
                    for (int j = i + d; ; j += d) {
                        char q = bd[j];
                        if (is_space_ch(q) || is_upper_ch(q)) break;
                        if (p == 'P' && (d == DE || d == DW) && i > 128) break;
                        else if (p == 'K' && (j < 160 || (j & 15) > 8 || (j & 15) < 6)) break;
                        else if (p == 'G' && j != 183) break;
                        else if ((p == 'N' || p == 'E')) {
                            int n_diff_x = (j - i) & 15;
                            if (n_diff_x == 14 || n_diff_x == 2) {
                                if (bd[i + (n_diff_x == 2 ? 1 : -1)] != '.') break;
                            } else {
                                if (j > i && bd[i + 16] != '.') break;
                                else if (j < i && bd[i - 16] != '.') break;
                            }
                        } else if ((p == 'B' || p == 'F') && bd[i + d / 2] != '.') break;
                        out.push_back({i, j});
                        if (strchr("PNBAKIEFG", p) && p != 0) break;
                        if (is_lower_ch(q)) break;
                    }
                }
            }
        } else {
            for (int i = 203; i > 50; i--) {
                char p = bd[i];
                if (!is_lower_ch(p) || p == 'u') continue;
                if (p == 'k') {
                    for (int sp = i + 16; sp < I0; sp += 16) {
                        if (bd[sp] == 'K') out.push_back({i, sp});
                        else if (bd[sp] != '.') break;
                    }
                }
                if (p == 'c' || p == 'h') {
                    DirTable dt = DIRS_LW(p);
                    for (int di_ = 0; di_ < dt.n; di_++) {
                        int d = dt.d[di_];
                        int cfoot = 0;
                        for (int j = i + d; ; j += d) {
                            char q = bd[j];
                            if (is_space_ch(q)) break;
                            if (cfoot == 0 && q == '.') out.push_back({i, j});
                            else if (cfoot == 0 && q != '.') cfoot++;
                            else if (cfoot == 1 && is_upper_ch(q)) { out.push_back({i, j}); break; }
                            else if (cfoot == 1 && is_lower_ch(q)) break;
                        }
                    }
                    continue;
                }
                DirTable dt = DIRS_LW(p);
                for (int di_ = 0; di_ < dt.n; di_++) {
                    int d = dt.d[di_];
                    for (int j = i + d; ; j += d) {
                        char q = bd[j];
                        if (is_space_ch(q) || is_lower_ch(q)) break;
                        if (p == 'p' && (d == DE || d == DW) && i < 126) break;
                        else if (p == 'k' && (j > 94 || (j & 15) > 8 || (j & 15) < 6)) break;
                        else if (p == 'g' && j != 71) break;
                        else if ((p == 'n' || p == 'e')) {
                            int n_diff_x = (j - i) & 15;
                            if (n_diff_x == 14 || n_diff_x == 2) {
                                if (bd[i + (n_diff_x == 2 ? 1 : -1)] != '.') break;
                            } else {
                                if (j > i && bd[i + 16] != '.') break;
                                else if (j < i && bd[i - 16] != '.') break;
                            }
                        } else if ((p == 'b' || p == 'f') && bd[i + d / 2] != '.') break;
                        out.push_back({i, j});
                        if (strchr("pnbakiefg", p) && p != 0) break;
                        if (is_upper_ch(q)) break;
                    }
                }
            }
        }
    }

    vector<pair<int,int>> gen_moves() const {   // 当前轮走方
        vector<pair<int,int>> v;
        gen_moves(stm, v);
        return v;
    }

    bool can_capture_king(bool side) const {
        char target = side ? 'k' : 'K';
        vector<pair<int,int>> ms;
        gen_moves(side, ms);
        for (auto& m : ms)
            if (board[m.second] == target) return true;
        return false;
    }

    // ---- 单步增量评估 ----
    double value(int i, int j) {
        int vi, vj;
        const int* covered_p; const int* covered_o_p;
        int che, che_o, endline;
        double score_rough;
        int t = turn ? 1 : 0;
        if (stm) {
            vi = i; vj = j;
            covered_p = &cov_u; covered_o_p = &cov_l;
            che = che_u; che_o = che_l;
            endline = back_l;
            score_rough = (double)rough;
        } else {
            vi = 254 - i; vj = 254 - j;
            covered_p = &cov_l; covered_o_p = &cov_u;
            che = che_l; che_o = che_u;
            endline = back_u;
            score_rough = -(double)rough;
        }
        int covered = *covered_p, covered_o = *covered_o_p;
        // R(x): stm=True 直接取, 否则镜像+swapcase (与 Python bd[254-x].swapcase() 一致)
        #define RAt(x) (stm ? board[x] : swapcase_ch(board[254 - (x)]))
        double sc;
        char p = RAt(vi);
        char q_raw = board[j];
        char q = to_up(q_raw);
        double possible_che = (g_sumall[t] == 0) ? 0
            : (double)covered * g_di[t][0] / g_sumall[t];              // di[v][t]["R"/"r"]
        double possible_che_opponent = (g_sumall[1 - t] == 0) ? 0
            : (double)covered_o * g_di[1 - t][0] / g_sumall[1 - t];    // di[v][not t]["r"/"R"]
        if (q == 'K') return (double)MATE_UPPER;
        if (is_upper_ch(p) && strchr("RNBAKCP", p) && p != 0) {
            sc = (double)(PST_OF(p)[vj] - PST_OF(p)[vi]);
            if (p == 'C') {
                if ((vi >> 4) != 3 && (vj >> 4) == 3 && endline <= 2) {
                    if ((vj == 51 || vj == 52) && RAt(53) == 'f' && RAt(54) == 'g') { }
                    else if ((vj == 59 || vj == 58) && RAt(57) == 'f' && RAt(56) == 'g') { }
                    else sc -= (endline == 0) ? 55 : 30;
                }
                if ((vi >> 4) == 3 && (vj >> 4) != 3 && endline <= 2) {
                    if ((vi == 51 || vi == 52) && RAt(53) == 'f' && RAt(54) == 'g') { }
                    else if ((vi == 59 || vi == 58) && RAt(57) == 'f' && RAt(56) == 'g') { }
                    else sc += (endline == 0) ? 55 : 30;
                }
            } else if (p == 'R') {
                if (strchr("dr", RAt(51)) == nullptr && RAt(51) != 0 && RAt(54) != 'a'
                        && RAt(71) != 'a' && (RAt(71) == 'p' || RAt(87) != 'n')) {
                    if ((vj & 15) == 6 && (vi & 15) != 6) sc += 30;
                    if ((vj & 15) != 6 && (vi & 15) == 6) sc -= 30;
                }
                if (strchr("dr", RAt(59)) == nullptr && RAt(59) != 0 && RAt(56) != 'a'
                        && RAt(71) != 'a' && (RAt(71) == 'p' || RAt(87) != 'n')) {
                    if ((vj & 15) == 8 && (vi & 15) != 8) sc += 30;
                    if ((vj & 15) != 8 && (vi & 15) == 8) sc -= 30;
                }
                if ((vi >> 4) == 3 && (vj >> 4) != 3 && (endline <= 1 || score_rough < -150))
                    sc += (score_rough < -150) ? 40 : 30;
                if ((vi >> 4) != 3 && (vj >> 4) == 3 && (endline <= 1 || score_rough < -150))
                    sc -= (score_rough < -150) ? 40 : 30;
            }
        } else {
            sc = (double)(g_avg_u[t][vj] - g_avg_cov[t]) + 20;
            if (p == 'D') {
                double minus = 30 * (possible_che_opponent / 2 + che_o);
                sc -= minus;
                if (score_rough < -150) sc -= std::floor(minus / 2);
            } else if (p == 'I') {
                if (strchr("rp", RAt(vi - 32)) && RAt(vi - 32) != 0)
                    sc -= g_avg_cov[t] / 2;
                else if (strchr("nc", RAt(vi - 32)) && RAt(vi - 32) != 0)
                    sc += 30;
                else if (RAt(vi - 48) == 'i')
                    sc += 30;
                else
                    sc += 20;
            }
        }
        if (is_upper_ch(q_raw) || is_lower_ch(q_raw)) {   // q = R(vj).upper(): 任意棋子字母都为真
            int k = 254 - vj;
            if (strchr("RNBAKCP", q) && q != 0) {
                sc += PST_OF(q)[k];
                if (q == 'P' && RAt(vj + 32) == 'I') sc += 30;
            } else {
                if (q != 'U') {
                    sc += g_avg_cov[1 - t];
                    if (q == 'I') sc += 10;
                } else {
                    sc += g_avg_u[1 - t][k];
                    if ((vj >> 4) == 7 && (vj & 1) == 1) sc += 10;
                }
                if (q == 'D') {
                    double addition = 30 * (possible_che / 2 + che);
                    sc += addition;
                    if (score_rough > 150) sc += std::floor(addition / 2);
                }
            }
        }
        #undef RAt
        return sc;
    }
};

// ---------------- TT ----------------
struct TTKey {
    uint64_t h; double score; int depth; bool root;
    bool operator==(const TTKey& o) const {
        return h == o.h && score == o.score && depth == o.depth && root == o.root;
    }
};
struct TTKeyHash {
    uint64_t mix(uint64_t x) const {
        x += 0x9E3779B97F4A7C15ull;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        return x ^ (x >> 31);
    }
    size_t operator()(const TTKey& k) const {
        uint64_t s;
        memcpy(&s, &k.score, 8);
        return (size_t)mix(k.h ^ mix(s) ^ mix((uint64_t)k.depth * 131 + (k.root ? 7 : 3)));
    }
};
struct MVKey {
    uint64_t h; double score;
    bool operator==(const MVKey& o) const { return h == o.h && score == o.score; }
};
struct MVKeyHash {
    size_t operator()(const MVKey& k) const {
        uint64_t s; memcpy(&s, &k.score, 8);
        return (size_t)(k.h ^ (s * 0x9E3779B97F4A7C15ull));
    }
};
struct Entry { double lower, upper; };

static inline int mvv_lva(const char* bd, const pair<int,int>& m, bool upper_moves) {
    char attacker = bd[m.first], victim = bd[m.second];
    if (victim == '.' || is_space_ch(victim)) return 0;
    if (upper_moves) { if (is_upper_ch(victim)) return 0; }
    else { if (is_lower_ch(victim)) return 0; }
    int v_val = MVV_BASE(to_up(victim));
    int a_val = MVV_BASE(to_up(attacker));
    return v_val * 1024 - a_val;
}

struct SearchTimeoutThrow {};

struct Searcher {
    std::unordered_map<TTKey, Entry, TTKeyHash> tp_score;
    std::unordered_map<MVKey, pair<int,int>, MVKeyHash> tp_move;
    std::unordered_map<uint32_t, long long> history_heur;
    long long nodes = 0;
    int lmr_full_moves = 2, lmr_min_depth = 2, lmr_base_reduction = 1;
    double deadline = 0.0;
    int check_interval = 2048;
    int qs_depth = 8;
    vector<pair<int,int>> forbid;    // 空 = 无禁着
    bool has_forbid = false;
    bool no_tt = false;              // 调试: 禁用置换表

    TTKey tt_key(const State& st, int depth, bool root) const {
        return {st.hkey, st.score, depth, root};
    }
    MVKey mv_key(const State& st) const { return {st.hkey, st.score}; }

    bool in_forbid(const pair<int,int>& m) const {
        for (auto& f : forbid) if (f == m) return true;
        return false;
    }

    vector<pair<int,int>> order_moves(const State& st,
                                      const vector<pair<int,int>>& moves,
                                      const pair<int,int>* tt_move) {
        vector<pair<int,int>> out;
        if (tt_move) out.push_back(*tt_move);
        vector<pair<int,int>> caps, quiets;
        for (auto& m : moves) {
            if (tt_move && m == *tt_move) continue;
            if (mvv_lva(st.board, m, st.stm) > 0) caps.push_back(m);
            else quiets.push_back(m);
        }
        std::stable_sort(caps.begin(), caps.end(), [&](const auto& a, const auto& b) {
            return mvv_lva(st.board, a, st.stm) > mvv_lva(st.board, b, st.stm);
        });
        std::stable_sort(quiets.begin(), quiets.end(), [&](const auto& a, const auto& b) {
            auto ia = history_heur.find((uint32_t)(a.first * 256 + a.second));
            auto ib = history_heur.find((uint32_t)(b.first * 256 + b.second));
            long long va = ia == history_heur.end() ? 0 : ia->second;
            long long vb = ib == history_heur.end() ? 0 : ib->second;
            return va > vb;
        });
        for (auto& m : caps) out.push_back(m);
        for (auto& m : quiets) out.push_back(m);
        return out;
    }

    static bool is_capture(const State& st, const pair<int,int>& move) {
        char q = st.board[move.second];
        return q != '.' && !is_space_ch(q) && (st.stm ? is_lower_ch(q) : is_upper_ch(q));
    }

    double qsearch(State& st, double alpha, double beta, int qdepth) {
        nodes++;
        if (deadline > 0 && nodes % check_interval == 0 && now_sec() > deadline)
            throw SearchTimeoutThrow();
        if (st.score <= -MATE_LOWER) return -MATE_UPPER;
        double stand = st.static_eval();
        if (qdepth <= 0) return stand;
        if (stand >= beta) return stand;
        if (stand > alpha) alpha = stand;
        char king_ch = st.stm ? 'k' : 'K';
        vector<pair<int,int>> caps;
        for (auto& m : st.gen_moves()) {
            char q = st.board[m.second];
            if (q == king_ch) return (double)MATE_UPPER;
            if (q != '.' && !is_space_ch(q)) caps.push_back(m);
        }
        std::stable_sort(caps.begin(), caps.end(), [&](const auto& a, const auto& b) {
            return mvv_lva(st.board, a, st.stm) > mvv_lva(st.board, b, st.stm);
        });
        double best = stand;
        for (auto& m : caps) {
            st.make(m.first, m.second);
            double val = -qsearch(st, -beta, -alpha, qdepth - 1);
            st.unmake();
            if (val > best) {
                best = val;
                if (val > alpha) alpha = val;
                if (val >= beta) break;
            }
        }
        return best;
    }

    double alphabeta(State& st, double alpha, double beta, int depth,
                     bool root = true, bool nullmove = false, bool nullmove_now = false) {
        nodes++;
        if (deadline > 0 && nodes % check_interval == 0 && now_sec() > deadline)
            throw SearchTimeoutThrow();
        if (depth < 0) depth = 0;
        if (st.score <= -MATE_LOWER) return -MATE_UPPER;
        char king_ch = st.stm ? 'k' : 'K';
        vector<pair<int,int>> raw_moves = st.gen_moves();
        // killer = tp_move[mv_key]
        MVKey mk = mv_key(st);
        auto kit = tp_move.find(mk);
        const pair<int,int>* killer = kit != tp_move.end() ? &kit->second : nullptr;
        // 杀棋检查
        {
            bool found = false;
            if (killer && st.board[killer->second] == king_ch) found = true;
            if (!found) for (auto& m : raw_moves)
                if (st.board[m.second] == king_ch) { found = true; break; }
            if (found) {
                if (killer && st.board[killer->second] == king_ch)
                    tp_move[mk] = *killer;
                else {
                    // 与 Python 一致: 存的是触发杀棋的那个着 (首个匹配)
                    for (auto& m : raw_moves)
                        if (st.board[m.second] == king_ch) { tp_move[mk] = m; break; }
                }
                return (double)MATE_UPPER;
            }
        }
        vector<pair<int,int>> search_moves = raw_moves;
        if (root && has_forbid) {
            vector<pair<int,int>> filtered;
            for (auto& m : raw_moves) if (!in_forbid(m)) filtered.push_back(m);
            search_moves = filtered;
            if (killer && in_forbid(*killer)) killer = nullptr;
        }
        TTKey key = tt_key(st, depth, root);
        auto eit = no_tt ? tp_score.end() : tp_score.find(key);
        Entry entry = eit != tp_score.end() ? eit->second : Entry{(double)-MATE_UPPER, (double)MATE_UPPER};
        if (!no_tt && entry.lower >= beta && std::fabs(entry.lower) < MATE_LOWER
                && (!root || tp_move.count(mk)))
            return entry.lower;
        if (!no_tt && entry.upper < alpha && std::fabs(entry.upper) < MATE_LOWER)
            return entry.upper;
        if (nullmove_now && depth > 3 && !root && st.has_rnci()) {
            if (!st.can_capture_king(!st.stm)) {
                st.flip_stm();
                double val = -alphabeta(st, -beta, 1 - beta, depth - 3, false, nullmove, false);
                st.flip_stm();
                if (val >= beta && alphabeta(st, alpha, beta, depth - 3, false, nullmove, false) != 0.0)
                    return val;
            }
        }
        nullmove_now = nullmove;
        if (depth == 0) return qsearch(st, alpha, beta, qs_depth);
        vector<pair<int,int>> moves = order_moves(st, search_moves, killer);
        double best = -MATE_UPPER;
        pair<int,int> mvBest{0, 0};
        bool have_best = false;
        int move_idx = 0;
        for (auto& move : moves) {
            bool is_cap = is_capture(st, move);
            bool do_lmr = (!root && depth >= lmr_min_depth && move_idx >= lmr_full_moves
                           && !is_cap && have_best && best > -MATE_UPPER);
            double val;
            if (!have_best) {
                st.make(move.first, move.second);
                val = -alphabeta(st, -beta, -alpha, depth - 1, false, nullmove, nullmove_now);
                st.unmake();
            } else {
                if (do_lmr) {
                    // 对数缩减: 越深/越靠后的着法减得越多 (标准 LMR 公式)
                    int R = (int)(0.5 + std::log((double)depth) * std::log((double)(move_idx + 1)) / 2.0);
                    if (R < lmr_base_reduction) R = lmr_base_reduction;
                    int reduced = depth - 1 - R;
                    if (reduced < 1) reduced = 1;
                    st.make(move.first, move.second);
                    val = -alphabeta(st, -alpha - 1, -alpha, reduced, false, nullmove, nullmove_now);
                    st.unmake();
                    if (val > alpha) {
                        st.make(move.first, move.second);
                        val = -alphabeta(st, -alpha - 1, -alpha, depth - 1, false, nullmove, nullmove_now);
                        st.unmake();
                    }
                } else {
                    st.make(move.first, move.second);
                    val = -alphabeta(st, -alpha - 1, -alpha, depth - 1, false, nullmove, nullmove_now);
                    st.unmake();
                }
                if (val > alpha && val < beta) {
                    st.make(move.first, move.second);
                    val = -alphabeta(st, -beta, -alpha, depth - 1, false, nullmove, nullmove_now);
                    st.unmake();
                }
            }
            if (val >= MATE_UPPER) {
                st.make(move.first, move.second);
                bool ok = st.can_capture_king(st.stm);
                st.unmake();
                if (ok) {
                    mvBest = move; have_best = true; best = val;
                    break;
                }
            }
            if (val > best && val > -MATE_UPPER) {
                best = val;
                mvBest = move;
                have_best = true;
                if (val > beta) {
                    if (!is_cap)
                        history_heur[(uint32_t)(move.first * 256 + move.second)] += (long long)depth * depth;
                    break;
                }
                if (val > alpha) alpha = val;
            }
            move_idx++;
        }
        if (!have_best && !moves.empty()) { mvBest = moves[0]; have_best = true; }
        if (have_best) {
            if ((double)tp_move.size() > TABLE_SIZE) tp_move.clear();
            tp_move[mk] = mvBest;
        }
        if (best < alpha && best < 0 && depth > 0) {
            bool dead_all = true;
            for (auto& m : raw_moves) {
                st.make(m.first, m.second);
                bool dead = st.can_capture_king(st.stm);
                st.unmake();
                if (!dead) { dead_all = false; break; }
            }
            if (dead_all) {
                bool in_check = st.can_capture_king(!st.stm);
                best = in_check ? (double)-MATE_UPPER : 0.0;
            }
        }
        if ((double)tp_score.size() > TABLE_SIZE) tp_score.clear();
        if (!no_tt) {
            if (best >= beta) tp_score[key] = Entry{best, entry.upper};
            if (best < alpha) tp_score[key] = Entry{entry.lower, best};
        }
        return best;
    }

    // 返回 (move, score, depth); move 可为 {-1,-1}
    std::tuple<pair<int,int>, double, int> search(State& st, double max_time) {
        nodes = 0;
        calc_average();
        tp_score.clear();
        tp_move.clear();
        history_heur.clear();
        double start = now_sec();
        deadline = start + max_time;
        pair<int,int> best_move{-1, -1};
        double best_score = 0;
        int best_depth = 0;
        for (int depth = 2; ; depth++) {
            double iter_start = now_sec();
            double root_val;
            try {
                root_val = alphabeta(st, -MATE_UPPER, MATE_UPPER, depth, true, true, true);
            } catch (SearchTimeoutThrow&) {
                break;
            }
            double iter_time = now_sec() - iter_start;
            auto it = tp_move.find(mv_key(st));
            if (it != tp_move.end()) {
                best_move = it->second;
                best_depth = depth;
                best_score = root_val;
            }
            if (now_sec() - start + 2.0 * iter_time > max_time) break;
            if (depth >= 12) break;
        }
        if (best_move.first < 0) {
            deadline = 0.0;
            double root_val = alphabeta(st, -MATE_UPPER, MATE_UPPER, 2, true, true, true);
            auto it = tp_move.find(mv_key(st));
            if (it != tp_move.end()) {
                best_move = it->second;
                best_depth = 2;
                best_score = root_val;
            }
        }
        return {best_move, best_score, best_depth};
    }

    void calc_average() {
        // di[0][True/False] 池, 顺序 R N B A C P (与 Python dict 插入序一致)
        double numr = 0, numb = 0;
        for (int k = 0; k < 6; k++) { numr += g_di[1][k]; numb += g_di[0][k]; }
        int avg_cov_r = 0, avg_cov_b = 0;
        static int averager[256], averageb[256];
        if (numr == 0) {
            avg_cov_r = 0;
            for (int i = 51; i < 204; i++) averager[i] = 0;
        } else {
            double sumr = 0;
            for (int k = 0; k < 6; k++)
                sumr += PST1_OF(POOL_KEYS[k]) * g_di[1][k] / DISCOUNT;
            avg_cov_r = py_round(sumr / numr);
            for (int i = 51; i < 204; i++) {
                double s2 = 0;
                for (int k = 0; k < 6; k++)
                    s2 += (double)PST_OF(POOL_KEYS[k])[i] * g_di[1][k];
                averager[i] = py_round(s2 / numr);
            }
        }
        if (numb == 0) {
            avg_cov_b = 0;
            for (int i = 51; i < 204; i++) averageb[i] = 0;
        } else {
            double sumb = 0;
            for (int k = 0; k < 6; k++)
                sumb += PST1_OF(POOL_KEYS[k]) * g_di[0][k] / DISCOUNT;
            avg_cov_b = py_round(sumb / numb);
            for (int i = 51; i < 204; i++) {
                double s2 = 0;
                for (int k = 0; k < 6; k++)
                    s2 += (double)PST_OF(POOL_KEYS[k])[i] * g_di[0][k];
                averageb[i] = py_round(s2 / numb);
            }
        }
        // average[version][turn]: True=大写方池 → g_avg_u[1], False → g_avg_u[0]
        g_avg_cov[1] = avg_cov_r;
        g_avg_cov[0] = avg_cov_b;
        for (int i = 0; i < 256; i++) { g_avg_u[1][i] = averager[i]; g_avg_u[0][i] = averageb[i]; }
    }

    static double now_sec() {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
};

// ---------------- 暗子池跨回合跟踪 ----------------
struct DarkPoolTracker {
    int rev[2][6];              // [mine][R N B A C P]
    bool have_prev;
    string prev;                // 上次观测局面串
    bool have_my_move;
    pair<int,int> my_move;
    long long stats_own_revealed = 0, stats_own_lost = 0, stats_oppo_revealed = 0,
              stats_chain_breaks = 0, stats_floor_bumps = 0, stats_resets = 0;

    DarkPoolTracker() { reset(); }

    void reset() {
        memset(rev, 0, sizeof(rev));
        have_prev = false; have_my_move = false;
        stats_resets++;
    }

    bool bump(int mine, char t) {
        int idx = g_type_idx(t);
        if (rev[mine][idx] < POOL_INIT_IDX[idx]) { rev[mine][idx]++; return true; }
        return false;
    }

    void floor_(const string& estr) {
        for (int m = 0; m < 2; m++) {
            const char* light = m == 0 ? LIGHT_UPPER_POOL : LIGHT_LOWER_POOL;
            int seen[6] = {0, 0, 0, 0, 0, 0};
            for (int i = 51; i < 204; i++) {
                char ch = estr[i];
                if (strchr(light, ch) && ch != 0) {
                    int idx = g_type_idx(ch);
                    seen[idx]++;
                }
            }
            for (int k = 0; k < 6; k++) {
                if (rev[m][k] < seen[k]) {
                    rev[m][k] = std::min(seen[k], POOL_INIT_IDX[k]);
                    stats_floor_bumps++;
                }
            }
        }
    }

    // 返回 (oppo_src, oppo_dst); -1 表示无
    pair<int,int> locate_oppo_move(const string& prevs, const string& estr, int src, int dst) const {
        if (estr[dst] == '.') return {-1, -1};
        vector<int> rest;
        for (int i = 51; i < 204; i++)
            if (prevs[i] != estr[i] && i != src && i != dst) rest.push_back(i);
        if ((int)rest.size() == 2 && estr[src] == '.'
                && (estr[rest[0]] == '.') != (estr[rest[1]] == '.')) {
            if (estr[rest[0]] != '.') return {rest[1], rest[0]};
            return {rest[0], rest[1]};
        }
        if ((int)rest.size() == 1 && estr[rest[0]] == '.')
            return {rest[0], estr[src] != '.' ? src : dst};
        return {-1, -1};
    }

    void observe(const string& estr) {
        if (have_prev && have_my_move) {
            const string& pv = prev;
            int src = my_move.first, dst = my_move.second;
            if (in_DARK_UP(pv[src])) {
                if (strchr(LIGHT_UPPER_POOL, estr[dst]) && estr[dst] != 0) {
                    bump(1, to_up(estr[dst]));
                    stats_own_revealed++;
                } else stats_own_lost++;
            }
            auto od = locate_oppo_move(pv, estr, src, dst);
            if (od.second < 0) stats_chain_breaks++;
            else if (in_DARK_LW(pv[od.first]) && strchr(LIGHT_LOWER_POOL, estr[od.second]) && estr[od.second] != 0) {
                bump(0, to_up(estr[od.second]));
                stats_oppo_revealed++;
            }
        }
        floor_(estr);
        prev = estr; have_prev = true;
        have_my_move = false;
    }

    void commit(const pair<int,int>& move) { my_move = move; have_my_move = true; }

    // mine 方的池写入 out[6] (R N B A C P), 保证 sum == 盘面朝下子数
    void pool(const string& estr, bool mine, double out[6]) {
        const char* dark = mine ? DARK_UPPER : DARK_LOWER;
        int n_dark = 0;
        for (int i = 51; i < 204; i++)
            if (strchr(dark, estr[i]) && estr[i] != 0) n_dark++;
        int m = mine ? 1 : 0;
        if (n_dark <= 0) {
            for (int k = 0; k < 6; k++) out[k] = 0;
            return;
        }
        double base[6], total = 0;
        for (int k = 0; k < 6; k++) {
            base[k] = POOL_INIT_IDX[k] - rev[m][k];
            if (base[k] < 0) base[k] = 0;
            total += base[k];
        }
        if (total <= 0) {
            for (int k = 0; k < 6; k++) { base[k] = POOL_INIT_IDX[k]; total += base[k]; }
        }
        double scale = (double)n_dark / total;
        for (int k = 0; k < 6; k++) out[k] = base[k] * scale;
    }

    void apply(const string& estr) {
        double mine[6], oppo[6];
        pool(estr, true, mine);
        pool(estr, false, oppo);
        for (int k = 0; k < 6; k++) {
            g_di[1][k] = mine[k];                       // r = mine (大写方池)
            g_di[0][k] = oppo[k];                       // b = oppo (小写字母键, 同索引)
        }
        double sr = 0, sb = 0;
        for (int k = 0; k < 6; k++) { sr += mine[k]; sb += oppo[k]; }
        g_sumall[1] = sr; g_sumall[0] = sb;
    }
};

// ---------------- board 转换 ----------------
static inline int rc_to_idx(int row, int col) { return (row + 3) * 16 + (3 + col); }
static inline pair<int,int> idx_to_rc(int idx) { return {idx / 16 - 3, idx % 16 - 3}; }
static inline string idx_to_uci(int idx) {
    string s;
    s += (char)('a' + idx % 16 - 3);
    s += std::to_string(12 - idx / 16);
    return s;
}

static string board_to_engine_string(const vector<vector<string>>& board, char my_side) {
    string eb(256, ' ');
    for (int i = 15; i < 256; i += 16) eb[i] = '\n';
    for (int row = 0; row < 10; row++) {
        for (int col = 0; col < 9; col++) {
            const string& piece = board[row][col];
            int idx = rc_to_idx(row, col);
            if (piece == "." || piece.empty()) { eb[idx] = '.'; continue; }
            bool is_ours = (piece[0] == my_side);
            if (piece.size() >= 2 && piece.back() == '?') {
                char letter = INIT_DARK_LETTER[row][col];
                if (letter == '.') { eb[idx] = '.'; continue; }
                eb[idx] = is_ours ? to_up(letter) : to_lo(letter);
            } else {
                auto it = TYPE_LETTER_MAP.find(piece.substr(1));
                if (it == TYPE_LETTER_MAP.end()) { eb[idx] = '.'; continue; }
                eb[idx] = is_ours ? it->second : to_lo(it->second);
            }
        }
    }
    return eb;
}

// ---------------- 长将安检 + 对外接口 ----------------
struct CheckState {
    int count = 0;
    vector<pair<int,int>> squares;   // 引擎视角 row-col (转换后)
    int retired = 0;
};

struct JieQiEngine {
    Searcher searcher;
    DarkPoolTracker pool;
    char pool_side = 0;

    // ---- live 模式记忆 (check_state 为空时自维护) ----
    char mem_side = 0;
    vector<string> mem_boards;
    vector<pair<int,int>> mem_moves;
    vector<bool> mem_my_checks;
    vector<bool> mem_oppo_checks;
    vector<int> mem_oppo_dsts;

    void memory_reset() {
        mem_boards.clear(); mem_moves.clear();
        mem_my_checks.clear(); mem_oppo_checks.clear(); mem_oppo_dsts.clear();
    }

    bool entry_in_check(const string& estr, bool side_to_move_upper, bool my_upper) {
        State st = State::from_string(estr);
        if (side_to_move_upper == my_upper) return st.can_capture_king(false);
        return st.can_capture_king(true);
    }

    bool gives_check(State& st, const pair<int,int>& move) {
        char king_ch = st.stm ? 'k' : 'K';
        if (st.board[move.second] == king_ch) return false;
        int src = move.first, dst = move.second;
        char p = st.board[src];
        if (strchr(U_AFTER_MOVE, p) && p != 0) {
            bool upper = is_upper_ch(p);
            for (const char* t = REVEAL_TYPES; *t; t++) {
                st.board[src] = upper ? *t : to_lo(*t);
                st.make(src, dst);
                bool chk = st.can_capture_king(!st.stm);
                st.unmake();
                st.board[src] = p;
                if (chk) return true;
            }
            return false;
        }
        st.make(src, dst);
        bool chk = st.can_capture_king(!st.stm);
        st.unmake();
        return chk;
    }

    bool memory_quota_blocks(const pair<int,int>& move, bool cand_chk) {
        if (!cand_chk) return false;
        int n = (int)mem_moves.size();
        int start = n;
        while (start > 0 && mem_my_checks[start - 1]) start--;
        std::map<int, bool> squares;
        int retired = 0, count = 0;
        for (int k = start; k < n; k++) {
            int src = mem_moves[k].first, dst = mem_moves[k].second;
            squares.erase(src);
            squares[dst] = true;
            count++;
            int od = (k + 1 < (int)mem_oppo_dsts.size()) ? mem_oppo_dsts[k + 1] : -1;
            if (od >= 0 && squares.count(od)) {
                squares.erase(od);
                retired++;
            }
        }
        count++;
        int total = (int)squares.size() + retired;
        if (!squares.count(move.first)) total++;
        return count > std::min(3 * total, 9);
    }

    void memory_step(char side, const string& estr) {
        if (mem_side != 0 && mem_side != side) memory_reset();
        mem_side = side;
        if (mem_boards.empty()) {
            mem_boards.push_back(estr);
            mem_oppo_checks.push_back(false);
            mem_oppo_dsts.push_back(-1);
            return;
        }
        if ((int)mem_boards.size() > (int)mem_moves.size()) return;
        const string& prev_b = mem_boards.back();
        int src = mem_moves.back().first, dst = mem_moves.back().second;
        vector<int> diff, rest;
        for (int i = 0; i < 256; i++)
            if (prev_b[i] != estr[i]) diff.push_back(i);
        for (int i : diff) if (i != src && i != dst) rest.push_back(i);
        int oppo_dst = -1;
        if (estr[dst] != '.') {
            if ((int)rest.size() == 2 && estr[src] == '.'
                    && (estr[rest[0]] == '.') != (estr[rest[1]] == '.')) {
                oppo_dst = estr[rest[0]] != '.' ? rest[0] : rest[1];
            } else if ((int)rest.size() == 1 && estr[rest[0]] == '.') {
                oppo_dst = estr[src] != '.' ? src : dst;
            }
        }
        if (oppo_dst < 0) {
            fprintf(stderr, "[长将] 记忆链断裂重置 (diff格数=%d rest格数=%d)\n",
                    (int)diff.size(), (int)rest.size());
            memory_reset();
            mem_side = side;
            mem_boards.push_back(estr);
            mem_oppo_checks.push_back(false);
            mem_oppo_dsts.push_back(-1);
            return;
        }
        mem_boards.push_back(estr);
        mem_oppo_checks.push_back(entry_in_check(estr, true, true));  // side==my_side 恒真 (引擎串恒己方视角)
        mem_oppo_dsts.push_back(oppo_dst);
    }

    void memory_commit(const pair<int,int>& move, State& st) {
        mem_moves.push_back(move);
        mem_my_checks.push_back(gives_check(st, move));
    }

    bool quota_blocks(State& st, char my_side, const pair<int,int>& move,
                      const CheckState& cs, const std::map<pair<int,int>,bool>& tracked) {
        if (gives_check(st, move)) {
            auto src_rc = idx_to_rc(move.first);
            int total = (int)tracked.size() + cs.retired;
            if (!tracked.count(src_rc)) total++;
            if (cs.count + 1 > std::min(3 * total, 9)) return true;
        }
        return false;
    }

    // 返回 (move{-1,-1} 表示无, score, depth)
    std::tuple<pair<int,int>, double, int> re_search_avoiding(State& st, double think_time,
                                                              const vector<pair<int,int>>& forbid) {
        if (forbid.empty()) return {{-1, -1}, 0.0, 0};
        searcher.forbid = forbid;
        searcher.has_forbid = true;
        auto r = searcher.search(st, think_time);
        searcher.forbid.clear();
        searcher.has_forbid = false;
        if (std::get<0>(r).first >= 0) return r;
        vector<pair<int,int>> safe;
        for (auto& m : st.gen_moves())
            if (std::find(forbid.begin(), forbid.end(), m) == forbid.end()) safe.push_back(m);
        if (!safe.empty()) {
            auto pick = *std::max_element(safe.begin(), safe.end(),
                [&](const auto& a, const auto& b) {
                    return st.value(a.first, a.second) < st.value(b.first, b.second);
                });
            return {pick, st.value(pick.first, pick.second), -2};
        }
        return {{-1, -1}, 0.0, 0};
    }

    // 返回 (uci 或空, score, depth)
    std::tuple<string, double, int> get_best_move(const vector<vector<string>>& board,
                                                  char my_side, double think_time,
                                                  const CheckState* check_state) {
        string estr = board_to_engine_string(board, my_side);
        if (pool_side != 0 && pool_side != my_side) pool.reset();
        pool_side = my_side;
        pool.observe(estr);
        pool.apply(estr);

        State st = State::from_string(estr);
        bool live = (check_state == nullptr);
        if (live) memory_step(my_side, estr);

        double t_search = Searcher::now_sec();
        auto r = searcher.search(st, think_time);
        pair<int,int> move = std::get<0>(r);
        double score = std::get<1>(r);
        int depth = std::get<2>(r);
        double elapsed = Searcher::now_sec() - t_search;

        st = State::from_string(estr);
        {
            auto legal = st.gen_moves();
            bool ok = move.first >= 0 &&
                      std::find(legal.begin(), legal.end(), move) != legal.end();
            if (move.first >= 0 && !ok) {
                if (!legal.empty()) {
                    auto best = *std::max_element(legal.begin(), legal.end(),
                        [&](const auto& a, const auto& b) {
                            return st.value(a.first, a.second) < st.value(b.first, b.second);
                        });
                    move = best;
                    score = st.value(best.first, best.second);
                    depth = -1;
                }
            }
        }
        double rethink = std::max(0.5, think_time - elapsed);
        if (move.first >= 0 && check_state) {
            std::map<pair<int,int>,bool> tracked;
            for (auto& rc : check_state->squares) tracked[rc] = true;
            if (quota_blocks(st, my_side, move, *check_state, tracked)) {
                vector<pair<int,int>> forbid;
                for (auto& m : st.gen_moves())
                    if (quota_blocks(st, my_side, m, *check_state, tracked))
                        forbid.push_back(m);
                auto r2 = re_search_avoiding(st, rethink, forbid);
                if (std::get<0>(r2).first >= 0) { move = std::get<0>(r2); score = std::get<1>(r2); depth = std::get<2>(r2); }
                else { move = {-1, -1}; score = 0; depth = 0; }
            }
        } else if (move.first >= 0 && live && (int)mem_moves.size() < (int)mem_boards.size()) {
            bool cand_chk = gives_check(st, move);
            if (cand_chk && memory_quota_blocks(move, cand_chk)) {
                vector<pair<int,int>> forbid;
                for (auto& m : st.gen_moves())
                    if (gives_check(st, m) && memory_quota_blocks(m, true))
                        forbid.push_back(m);
                auto r2 = re_search_avoiding(st, rethink, forbid);
                if (std::get<0>(r2).first >= 0) { move = std::get<0>(r2); score = std::get<1>(r2); depth = std::get<2>(r2); }
                else { move = {-1, -1}; score = 0; depth = 0; }
            }
        }
        st = State::from_string(estr);
        if (move.first >= 0 && live) memory_commit(move, st);
        if (live) {
            fprintf(stderr, "[长将] my_checks=[");
            for (size_t i = 0; i < mem_my_checks.size(); i++)
                fprintf(stderr, "%s%s", i ? ", " : "", mem_my_checks[i] ? "True" : "False");
            fprintf(stderr, "] depth=%d boards=%d moves=%d\n", depth,
                    (int)mem_boards.size(), (int)mem_moves.size());
        }
        if (move.first >= 0) {
            pool.commit(move);
            return {idx_to_uci(move.first) + idx_to_uci(move.second), score, depth};
        }
        return {"", 0.0, 0};
    }
};

// ---------------- 最小 JSON ----------------
namespace minijson {
struct Val;
using ValP = std::shared_ptr<Val>;
struct Val {
    enum T { NUL, BOO, NUM, STR, ARR, OBJ } t = NUL;
    bool b = false;
    double num = 0;
    string str;
    vector<ValP> arr;
    std::map<string, ValP> obj;
};
struct Parser {
    const char* p;
    explicit Parser(const char* s) : p(s) {}
    void ws() { while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++; }
    ValP parse() {
        ws();
        char c = *p;
        if (c == '{') return obj();
        if (c == '[') return arr();
        if (c == '"') return strv();
        if (c == 't') { p += 4; auto v = std::make_shared<Val>(); v->t = Val::BOO; v->b = true; return v; }
        if (c == 'f') { p += 5; auto v = std::make_shared<Val>(); v->t = Val::BOO; v->b = false; return v; }
        if (c == 'n') { p += 4; return std::make_shared<Val>(); }
        return numv();
    }
    ValP obj() {
        auto v = std::make_shared<Val>(); v->t = Val::OBJ;
        p++; ws();
        if (*p == '}') { p++; return v; }
        while (true) {
            ws();
            auto k = strv();
            ws(); if (*p == ':') p++;
            v->obj[k->str] = parse();
            ws();
            if (*p == ',') { p++; continue; }
            if (*p == '}') { p++; break; }
            break;
        }
        return v;
    }
    ValP arr() {
        auto v = std::make_shared<Val>(); v->t = Val::ARR;
        p++; ws();
        if (*p == ']') { p++; return v; }
        while (true) {
            v->arr.push_back(parse());
            ws();
            if (*p == ',') { p++; continue; }
            if (*p == ']') { p++; break; }
            break;
        }
        return v;
    }
    ValP strv() {
        auto v = std::make_shared<Val>(); v->t = Val::STR;
        p++;  // 开引号
        while (*p && *p != '"') {
            if (*p == '\\') {
                p++;
                switch (*p) {
                    case 'n': v->str += '\n'; break;
                    case 't': v->str += '\t'; break;
                    case 'r': v->str += '\r'; break;
                    case '"': v->str += '"'; break;
                    case '\\': v->str += '\\'; break;
                    case '/': v->str += '/'; break;
                    case 'u': {  // \uXXXX → UTF-8 ( referee 用 ensure_ascii=False, 一般不会出现)
                        unsigned cp = 0;
                        for (int i = 1; i <= 4; i++) {
                            char h = p[i];
                            cp = cp * 16 + (h <= '9' ? h - '0' : (h | 32) - 'a' + 10);
                        }
                        p += 4;
                        if (cp < 0x80) v->str += (char)cp;
                        else if (cp < 0x800) {
                            v->str += (char)(0xC0 | (cp >> 6));
                            v->str += (char)(0x80 | (cp & 63));
                        } else {
                            v->str += (char)(0xE0 | (cp >> 12));
                            v->str += (char)(0x80 | ((cp >> 6) & 63));
                            v->str += (char)(0x80 | (cp & 63));
                        }
                        break;
                    }
                    default: v->str += *p;
                }
                p++;
            } else {
                v->str += *p++;
            }
        }
        if (*p == '"') p++;
        return v;
    }
    ValP numv() {
        auto v = std::make_shared<Val>(); v->t = Val::NUM;
        char* end;
        v->num = strtod(p, &end);
        p = end;
        return v;
    }
};
}  // namespace minijson

// ---------------- 服务端 ----------------
static JieQiEngine g_engine;

// 调试用: 从请求里解析 pool 写入全局池 (gen/val/avg/val1/search_fixed 共用)
static void apply_pool(const std::map<string, minijson::ValP>& obj) {
    auto it = obj.find("pool");
    if (it == obj.end() || it->second->t != minijson::Val::OBJ) return;
    auto& po = it->second->obj;
    for (int side = 0; side < 2; side++) {
        string key = side == 1 ? "mine" : "oppo";
        auto kit = po.find(key);
        if (kit == po.end()) continue;
        double sum = 0;
        for (int k = 0; k < 6; k++) {
            string kk(1, POOL_KEYS[k]);
            double v = (kit->second->t == minijson::Val::OBJ && kit->second->obj.count(kk))
                       ? kit->second->obj[kk]->num : 0;
            g_di[side][k] = v;
            sum += v;
        }
        g_sumall[side] = sum;
    }
    Searcher tmp;
    tmp.calc_average();
}

static void print_json_line(const string& s) {
    fputs(s.c_str(), stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

static string num_str(double v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

int main() {
    zobrist_init();
    // 初始满池 (与 Python di 初值一致: 每方 R2 N2 B2 A2 C2 P5)
    for (int k = 0; k < 6; k++) { g_di[1][k] = POOL_INIT_IDX[k]; g_di[0][k] = POOL_INIT_IDX[k]; }
    g_sumall[1] = g_sumall[0] = 15;
    // 预热
    {
        vector<vector<string>> warm(10, vector<string>(9, "."));
        for (int c = 0; c < 9; c++) {
            warm[9][c] = c == 4 ? "r帥" : "r?";
            warm[0][c] = c == 4 ? "b將" : "b?";
        }
        warm[7][1] = "r?"; warm[7][7] = "r?";
        warm[2][1] = "b?"; warm[2][7] = "b?";
        for (int c = 0; c < 9; c += 2) { warm[6][c] = "r?"; warm[3][c] = "b?"; }
        CheckState cs; cs.count = 0; cs.retired = 0;
        g_engine.get_best_move(warm, 'r', 0.3, &cs);
    }
    print_json_line("{\"ok\": true, \"ready\": true}");
    fprintf(stderr, "[engine_cpp] 就绪\n");

    char line[1 << 20];
    while (fgets(line, sizeof(line), stdin)) {
        size_t len = strlen(line);
        if (len && line[len - 1] == '\n') line[len - 1] = 0;
        if (!line[0]) continue;
        minijson::ValP req;
        try {
            minijson::Parser ps(line);
            req = ps.parse();
        } catch (...) {
            print_json_line("{\"ok\": false, \"error\": \"json parse\"}");
            continue;
        }
        if (!req || req->t != minijson::Val::OBJ) {
            print_json_line("{\"ok\": false, \"error\": \"bad req\"}");
            continue;
        }
        auto& obj = req->obj;
        string cmd = obj.count("cmd") ? obj["cmd"]->str : "";
        if (cmd == "quit") break;
        if (cmd == "ping") { print_json_line("{\"ok\": true, \"pong\": true}"); continue; }
        if (cmd == "gen") {
            // 调试: 走法生成对比 {"estr":..., "make":[i,j] 可选, "pool": 可选}
            string estr = obj.count("estr") ? obj["estr"]->str : "";
            apply_pool(obj);
            State st = State::from_string(estr);
            if (obj.count("make") && obj["make"]->t == minijson::Val::ARR
                    && obj["make"]->arr.size() >= 2)
                st.make((int)obj["make"]->arr[0]->num, (int)obj["make"]->arr[1]->num);
            if (obj.count("makes") && obj["makes"]->t == minijson::Val::ARR)
                for (auto& mv : obj["makes"]->arr)
                    if (mv->t == minijson::Val::ARR && mv->arr.size() >= 2)
                        st.make((int)mv->arr[0]->num, (int)mv->arr[1]->num);
            auto ms = st.gen_moves();
            string out = "{\"ok\": true, \"moves\": [";
            for (size_t i = 0; i < ms.size(); i++) {
                out += "[" + std::to_string(ms[i].first) + "," + std::to_string(ms[i].second) + "]";
                if (i + 1 < ms.size()) out += ",";
            }
            out += "], \"static\": " + num_str(st.static_eval()) +
                   ", \"score\": " + num_str(st.score) +
                   ", \"ktu\": " + std::to_string(st.kt_u) +
                   ", \"ktl\": " + std::to_string(st.kt_l) +
                   ", \"ktsu\": " + std::to_string(st.kts_u) +
                   ", \"ktsl\": " + std::to_string(st.kts_l) + "}";
            print_json_line(out);
            continue;
        }
        if (cmd == "val") {
            // 调试: 每步 value 对比 {"estr":..., "pool": 可选}
            string estr = obj.count("estr") ? obj["estr"]->str : "";
            if (obj.count("pool") && obj["pool"]->t == minijson::Val::OBJ) {
                auto& po = obj["pool"]->obj;
                for (int side = 0; side < 2; side++) {
                    string key = side == 1 ? "mine" : "oppo";
                    if (!po.count(key)) continue;
                    double sum = 0;
                    for (int k = 0; k < 6; k++) {
                        string kk(1, POOL_KEYS[k]);
                        g_di[side][k] = (po[key]->t == minijson::Val::OBJ
                                         && po[key]->obj.count(kk))
                                        ? po[key]->obj[kk]->num : 0;
                        sum += g_di[side][k];
                    }
                    g_sumall[side] = sum;
                }
                Searcher tmp; tmp.calc_average();
            }
            State st = State::from_string(estr);
            auto ms = st.gen_moves();
            string out = "{\"ok\": true, \"vals\": [";
            char buf[64];
            for (size_t i = 0; i < ms.size(); i++) {
                snprintf(buf, sizeof(buf), "[%d,%d,%s]", ms[i].first, ms[i].second,
                         num_str(st.value(ms[i].first, ms[i].second)).c_str());
                out += buf;
                if (i + 1 < ms.size()) out += ",";
            }
            out += "]}";
            print_json_line(out);
            continue;
        }
        if (cmd == "avg") {
            // 调试: 返回 calc_average 的期望表 {"avg_cov":[r,b], "avg_u":[[...256],[...256]]}
            if (obj.count("pool") && obj["pool"]->t == minijson::Val::OBJ) {
                auto& po = obj["pool"]->obj;
                for (int side = 0; side < 2; side++) {
                    string key = side == 1 ? "mine" : "oppo";
                    if (!po.count(key)) continue;
                    double sum = 0;
                    for (int k = 0; k < 6; k++) {
                        string kk(1, POOL_KEYS[k]);
                        double v = (po[key]->t == minijson::Val::OBJ && po[key]->obj.count(kk))
                                   ? po[key]->obj[kk]->num : 0;
                        g_di[side][k] = v;
                        sum += v;
                    }
                    g_sumall[side] = sum;
                }
            }
            Searcher s;
            s.calc_average();
            string out = "{\"ok\": true, \"avg_cov\": [" +
                std::to_string(g_avg_cov[1]) + "," + std::to_string(g_avg_cov[0]) + "], \"avg_u\": [";
            for (int t = 0; t < 2; t++) {
                out += "[";
                for (int i = 51; i < 204; i++) {
                    out += std::to_string(g_avg_u[t][i]);
                    if (i < 203) out += ",";
                }
                out += t == 0 ? "]," : "]";
            }
            out += "]}";
            print_json_line(out);
            continue;
        }
        if (cmd == "val1") {
            // 调试: 单步 value 内部量 {"estr":..., "i":..., "j":...}
            string estr = obj.count("estr") ? obj["estr"]->str : "";
            int i_ = obj.count("i") ? (int)obj["i"]->num : 0;
            int j_ = obj.count("j") ? (int)obj["j"]->num : 0;
            if (obj.count("pool") && obj["pool"]->t == minijson::Val::OBJ) {
                auto& po = obj["pool"]->obj;
                for (int side = 0; side < 2; side++) {
                    string key = side == 1 ? "mine" : "oppo";
                    if (!po.count(key)) continue;
                    double sum = 0;
                    for (int k = 0; k < 6; k++) {
                        string kk(1, POOL_KEYS[k]);
                        g_di[side][k] = (po[key]->t == minijson::Val::OBJ
                                         && po[key]->obj.count(kk))
                                        ? po[key]->obj[kk]->num : 0;
                        sum += g_di[side][k];
                    }
                    g_sumall[side] = sum;
                }
                Searcher tmp; tmp.calc_average();
            }
            State st = State::from_string(estr);
            if (obj.count("makes") && obj["makes"]->t == minijson::Val::ARR)
                for (auto& mv : obj["makes"]->arr)
                    if (mv->t == minijson::Val::ARR && mv->arr.size() >= 2)
                        st.make((int)mv->arr[0]->num, (int)mv->arr[1]->num);
            char buf[512];
            snprintf(buf, sizeof(buf),
                     "{\"ok\": true, \"p\": \"%c\", \"stm\": %d, \"turn\": %d, "
                     "\"cov_u\": %d, \"cov_l\": %d, \"che_l\": %d, \"di0R\": %s, "
                     "\"sum0\": %s, \"avg_cov1\": %d, \"avg_u179\": %d, \"val\": %s}",
                     st.board[i_], st.stm ? 1 : 0, st.turn ? 1 : 0,
                     st.cov_u, st.cov_l, st.che_l, num_str(g_di[0][0]).c_str(),
                     num_str(g_sumall[0]).c_str(), g_avg_cov[1],
                     g_avg_u[1][179], num_str(st.value(i_, j_)).c_str());
            print_json_line(buf);
            continue;
        }
        if (cmd == "search_fixed") {
            // 调试: 固定深度确定性搜索 {"estr":..., "depth":N, "pool": 可选}
            string estr = obj.count("estr") ? obj["estr"]->str : "";
            int depth = obj.count("depth") ? (int)obj["depth"]->num : 5;
            if (obj.count("pool") && obj["pool"]->t == minijson::Val::OBJ) {
                auto& po = obj["pool"]->obj;
                for (int side = 0; side < 2; side++) {
                    string key = side == 1 ? "mine" : "oppo";
                    if (!po.count(key)) continue;
                    double sum = 0;
                    for (int k = 0; k < 6; k++) {
                        string kk(1, POOL_KEYS[k]);
                        double v = (po[key]->t == minijson::Val::OBJ && po[key]->obj.count(kk))
                                   ? po[key]->obj[kk]->num : 0;
                        g_di[side][k] = v;
                        sum += v;
                    }
                    g_sumall[side] = sum;
                }
            }
            State st = State::from_string(estr);
            if (obj.count("make") && obj["make"]->t == minijson::Val::ARR
                    && obj["make"]->arr.size() >= 2) {
                int mi = (int)obj["make"]->arr[0]->num;
                int mj = (int)obj["make"]->arr[1]->num;
                st.make(mi, mj);
            }
            Searcher s;
            s.deadline = 0.0;
            s.no_tt = obj.count("no_tt") ? (bool)obj["no_tt"]->num : false;
            s.calc_average();
            double val;
            try {
                val = s.alphabeta(st, -MATE_UPPER, MATE_UPPER, depth, true, true, true);
            } catch (SearchTimeoutThrow&) { val = 0; }
            auto mv = s.tp_move.find(s.mv_key(st));
            char buf[512];
            if (mv != s.tp_move.end())
                snprintf(buf, sizeof(buf), "{\"ok\": true, \"nodes\": %lld, \"score\": %s, "
                         "\"move\": [%d, %d]}", s.nodes, num_str(val).c_str(),
                         mv->second.first, mv->second.second);
            else
                snprintf(buf, sizeof(buf), "{\"ok\": true, \"nodes\": %lld, \"score\": %s, \"move\": null}",
                         s.nodes, num_str(val).c_str());
            print_json_line(buf);
            continue;
        }
        if (cmd == "go") {
            vector<vector<string>> board;
            if (obj.count("board") && obj["board"]->t == minijson::Val::ARR) {
                for (auto& row : obj["board"]->arr) {
                    vector<string> r;
                    for (auto& c : row->arr) r.push_back(c->str);
                    board.push_back(r);
                }
            }
            char my_side = obj.count("my_side") && !obj["my_side"]->str.empty()
                           ? obj["my_side"]->str[0] : 'r';
            double think_time = obj.count("think_time") ? obj["think_time"]->num : 2.0;
            CheckState cs;
            bool has_cs = false;
            if (obj.count("check_state") && obj["check_state"]->t == minijson::Val::OBJ) {
                has_cs = true;
                auto& cso = obj["check_state"]->obj;
                cs.count = cso.count("count") ? (int)cso["count"]->num : 0;
                cs.retired = cso.count("retired") ? (int)cso["retired"]->num : 0;
                if (cso.count("squares") && cso["squares"]->t == minijson::Val::ARR) {
                    bool flip = (my_side == 'b');
                    for (auto& sq : cso["squares"]->arr) {
                        if (sq->t == minijson::Val::ARR && sq->arr.size() >= 2) {
                            int r_ = (int)sq->arr[0]->num, c_ = (int)sq->arr[1]->num;
                            if (flip) { r_ = 9 - r_; c_ = 8 - c_; }
                            cs.squares.push_back({r_, c_});
                        }
                    }
                }
            }
            try {
                auto r = g_engine.get_best_move(board, my_side, think_time,
                                                has_cs ? &cs : nullptr);
                string uci = std::get<0>(r);
                char buf[512];
                if (!uci.empty())
                    snprintf(buf, sizeof(buf), "{\"ok\": true, \"uci\": \"%s\", \"score\": %s, \"depth\": %d}",
                             uci.c_str(), num_str(std::get<1>(r)).c_str(), std::get<2>(r));
                else
                    snprintf(buf, sizeof(buf), "{\"ok\": false, \"error\": \"no move\"}");
                print_json_line(buf);
            } catch (const std::exception& e) {
                char buf[256];
                snprintf(buf, sizeof(buf), "{\"ok\": false, \"error\": \"%s\"}", e.what());
                print_json_line(buf);
            } catch (...) {
                print_json_line("{\"ok\": false, \"error\": \"unknown\"}");
            }
            continue;
        }
        print_json_line("{\"ok\": false, \"error\": \"unknown cmd\"}");
    }
    return 0;
}
