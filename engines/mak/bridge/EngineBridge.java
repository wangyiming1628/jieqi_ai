package edu.bupt.jieqi.bridge;

import edu.bupt.jieqi.ai.Agent;
import edu.bupt.jieqi.ai.ExpectiminimaxAgent;
import edu.bupt.jieqi.ai.MaterialEvaluator;
import edu.bupt.jieqi.ai.SearchBudget;
import edu.bupt.jieqi.model.Board;
import edu.bupt.jieqi.model.Color;
import edu.bupt.jieqi.model.GameState;
import edu.bupt.jieqi.model.GameStatus;
import edu.bupt.jieqi.model.HiddenPiecePool;
import edu.bupt.jieqi.model.Move;
import edu.bupt.jieqi.model.Piece;
import edu.bupt.jieqi.model.PieceType;
import edu.bupt.jieqi.model.Position;
import edu.bupt.jieqi.model.PlayerView;
import edu.bupt.jieqi.rules.GameEngine;
import edu.bupt.jieqi.rules.StandardGameEngine;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Makinuohara/2026-jieqi-AI 引擎的桥接服务端。
 *
 * 常驻进程, 通过 stdin/stdout 逐行 JSON 通信:
 *   请求 <- {"cmd":"go","board":[[...]],"my_side":"r","think_time":2.0}
 *          {"cmd":"ping"} / {"cmd":"quit"}
 *   响应 -> {"ok":true,"uci":"a3a4","score":0,"depth":0}
 *          {"ok":true,"pong":true}
 *          {"ok":true,"ready":true}   (启动就绪)
 *
 * board 输入 (本项目己方视角): board[row][col], row0-4=对手(上), row5-9=己方(下)。
 *   棋子字符串: "r帥/r仕/r相/r馬/r車/r炮/r兵", "b將/.../b卒", "r?"/"b?"(暗子), "."(空)。
 *   己方统一映射为引擎 RED(rank 0-3 底部), 对手映射为 BLACK。
 *   坐标: 引擎 rank = 9 - board_row, file = col。返回 uci 用引擎 rank (= 9 - row),
 *   与本项目 execute_move 的 'r' 分支一致。
 *
 * stdout 只放 JSON; 日志走 stderr。
 */
public final class EngineBridge {

    private final Agent agent = new ExpectiminimaxAgent(new MaterialEvaluator());
    private final GameEngine engine = new StandardGameEngine();
    private final PrintStream out;

    public EngineBridge() {
        this.out = new PrintStream(System.out, true, StandardCharsets.UTF_8);
    }

    // 中文棋子名 -> 引擎 PieceType
    private static PieceType typeFromName(String name) {
        switch (name) {
            case "帥": case "將": return PieceType.KING;
            case "車": return PieceType.ROOK;
            case "馬": return PieceType.KNIGHT;
            case "炮": return PieceType.CANNON;
            case "兵": case "卒": return PieceType.PAWN;
            case "仕": case "士": return PieceType.GUARD;
            case "相": case "象": return PieceType.BISHOP;
            default: return null;
        }
    }

    // 暗子初始摆位类型 (标准开局, 上下半场镜像)。row 为本项目 board 行(0-9)。
    private static PieceType initialVirtualType(int row, int col) {
        PieceType[] back = {
            PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.GUARD,
            PieceType.KING, PieceType.GUARD, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK
        };
        if (row == 0 || row == 9) return back[col];
        if (row == 2 || row == 7) return (col == 1 || col == 7) ? PieceType.CANNON : null;
        if (row == 3 || row == 6) return (col % 2 == 0) ? PieceType.PAWN : null;
        return null;
    }

    /**
     * 把本项目 board 转成引擎 Board。
     * 己方(my_side 对应色) -> RED, 对手 -> BLACK。引擎 rank = 9 - row, file = col。
     * 归属必须按棋子颜色前缀 'r'/'b' 判断, 不能按行号——揭棋中双方棋子会过河穿插到对方半场。
     */
    private Board toEngineBoard(String[][] board, char mySide) {
        Map<Position, Piece> pieces = new LinkedHashMap<>();
        for (int row = 0; row < 10; row++) {
            for (int col = 0; col < 9; col++) {
                String cell = board[row][col];
                if (cell == null || cell.isEmpty() || cell.equals(".")) continue;
                char c = cell.charAt(0);              // 'r' 或 'b' (棋子颜色)
                String rest = cell.substring(1);
                // 己方(=mySide)映射为 RED, 对手映射为 BLACK
                Color owner = (c == mySide) ? Color.RED : Color.BLACK;
                int rank = 9 - row;
                Position pos = new Position(col, rank);
                if (rest.equals("?")) {
                    // 暗子第一次移动才翻开, 仍是暗子者必在初始位置, 按物理格子查初始布局类型
                    PieceType vt = initialVirtualType(row, col);
                    if (vt == null) vt = PieceType.PAWN;
                    pieces.put(pos, Piece.hidden(owner, vt));
                } else {
                    PieceType t = typeFromName(rest);
                    if (t == null) continue;
                    pieces.put(pos, Piece.visible(owner, t));
                }
            }
        }
        return new Board(pieces);
    }

    private static String moveToUci(Move m) {
        if (m == null) return null;
        Position s = m.source();
        Position d = m.destination();
        // 引擎 rank 直接作为 uci 的 rank (= 9 - our_row), 与 execute_move 'r' 分支一致
        return "" + (char) ('a' + s.file()) + s.rank()
                  + (char) ('a' + d.file()) + d.rank();
    }

    private String handleGo(String[][] board, char mySide, double thinkSeconds) {
        Board engineBoard = toEngineBoard(board, mySide);
        // 己方恒映射为 RED 走子
        GameState state = new GameState(
                engineBoard, Color.RED, 0,
                0, 0, null, null,
                0, 0, null, null,
                System.currentTimeMillis(), GameStatus.PLAYING,
                HiddenPiecePool.standard(), HiddenPiecePool.standard());
        List<Move> legal = engine.legalMoves(state);
        if (legal.isEmpty()) {
            return "{\"ok\":true,\"uci\":null,\"score\":0,\"depth\":0}";
        }
        PlayerView view = PlayerView.from(state, Color.RED, legal);
        long millis = Math.max(50L, (long) (thinkSeconds * 1000));
        SearchBudget budget = new SearchBudget(Duration.ofMillis(millis), 64);
        Move best = agent.chooseMove(view, budget);
        String uci = moveToUci(best);
        return "{\"ok\":true,\"uci\":" + (uci == null ? "null" : "\"" + uci + "\"")
                + ",\"score\":0,\"depth\":0}";
    }

    private void log(String msg) {
        System.err.println(msg);
        System.err.flush();
    }

    public void run() throws Exception {
        log("[mak_engine_bridge] 就绪 (Java " + System.getProperty("java.version") + ")");
        out.println("{\"ok\":true,\"ready\":true}");

        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        String line;
        while ((line = in.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;
            try {
                String cmd = extractString(line, "cmd");
                if ("quit".equals(cmd)) break;
                if ("ping".equals(cmd)) { out.println("{\"ok\":true,\"pong\":true}"); continue; }
                // go
                String[][] board = extractBoard(line);
                String mySideStr = extractString(line, "my_side");
                char mySide = (mySideStr != null && !mySideStr.isEmpty()) ? mySideStr.charAt(0) : 'r';
                double think = extractNumber(line, "think_time", 2.0);
                out.println(handleGo(board, mySide, think));
            } catch (Exception e) {
                log(stackTrace(e));
                out.println("{\"ok\":false,\"error\":\"" + escape(String.valueOf(e.getMessage())) + "\"}");
            }
        }
    }

    // ---- 极简 JSON 解析 (只针对本协议固定结构, 避免引入 gson 依赖) ----

    private static String extractString(String json, String key) {
        String pat = "\"" + key + "\"";
        int i = json.indexOf(pat);
        if (i < 0) return null;
        int colon = json.indexOf(':', i + pat.length());
        int q1 = json.indexOf('"', colon + 1);
        if (q1 < 0) return null;
        int q2 = json.indexOf('"', q1 + 1);
        return json.substring(q1 + 1, q2);
    }

    private static double extractNumber(String json, String key, double dflt) {
        String pat = "\"" + key + "\"";
        int i = json.indexOf(pat);
        if (i < 0) return dflt;
        int colon = json.indexOf(':', i + pat.length());
        int j = colon + 1;
        while (j < json.length() && (json.charAt(j) == ' ')) j++;
        int start = j;
        while (j < json.length() && "0123456789.+-eE".indexOf(json.charAt(j)) >= 0) j++;
        try { return Double.parseDouble(json.substring(start, j)); }
        catch (Exception e) { return dflt; }
    }

    /**
     * 解析 "board":[["r車","r馬",...],[...],...] 为 String[10][9]。
     * board 值为字符串二维数组, 元素是短字符串, 无嵌套引号问题。
     */
    private static String[][] extractBoard(String json) {
        String pat = "\"board\"";
        int i = json.indexOf(pat);
        int lb = json.indexOf('[', i);
        String[][] board = new String[10][9];
        for (String[] r : board) java.util.Arrays.fill(r, ".");
        int p = lb + 1;
        int row = 0;
        while (row < 10) {
            // 找该行的 '['
            int rowStart = json.indexOf('[', p);
            if (rowStart < 0) break;
            int rowEnd = json.indexOf(']', rowStart);
            String rowContent = json.substring(rowStart + 1, rowEnd);
            // 拆分逗号分隔的带引号元素
            int col = 0;
            int k = 0;
            while (col < 9 && k < rowContent.length()) {
                int q1 = rowContent.indexOf('"', k);
                if (q1 < 0) break;
                int q2 = rowContent.indexOf('"', q1 + 1);
                String val = rowContent.substring(q1 + 1, q2);
                board[row][col] = val.isEmpty() ? "." : val;
                col++;
                k = q2 + 1;
            }
            p = rowEnd + 1;
            row++;
        }
        return board;
    }

    private static String escape(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", " ");
    }

    private static String stackTrace(Throwable t) {
        StringBuilder sb = new StringBuilder(t.toString());
        for (StackTraceElement e : t.getStackTrace()) sb.append("\n  at ").append(e);
        return sb.toString();
    }

    public static void main(String[] args) throws Exception {
        new EngineBridge().run();
    }
}
