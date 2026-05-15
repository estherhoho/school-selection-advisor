"""
中考志愿填报推荐 v2
====================
基于排名（非分数）+ 自主招生分流 + 志愿优先规则的概率模型。

新特性：
  1. 输入改为排名（含标准差），不再用分数
  2. 自主招生分流（按排名加权）
  3. 严格"志愿优先"录取规则建模
  4. 三维评估：学校声誉 + 交通便利度 + 一中名额
  5. 双引擎：解析公式 + 蒙特卡洛 10万次模拟（可对比）
  6. 两种模式：A 自填志愿看结果 / B 模型推荐冲/稳/保
  7. 全中文界面
"""

import json
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import binom

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent
PROFILES_FILE = DATA_DIR / "student_profiles.json"
TRANSPORT_SCORES = {"高": 1.0, "中": 0.5, "低": 0.0}

# 所有可保存的输入参数（key 名 = session_state key 名）
PROFILE_KEYS = [
    "middle_school", "student_name", "student_rank", "student_std",
    "latest_score",
    "T", "w_rep", "w_trans", "w_quota",
    "zizu_top3", "zizu_top6", "zizu_top9", "zizu_rest",
    "choice_1", "choice_2", "choice_3",
    "n_sim_mc", "seed",
]
DEFAULTS = {
    # 学生信息字段不预填，让用户自己填（值为 None / 空字符串）
    "T": 1.0,
    "w_rep": 0.5,
    "w_trans": 0.25,
    "w_quota": 0.25,
    "zizu_top3": 0.25,
    "zizu_top6": 0.10,
    "zizu_top9": 0.03,
    "zizu_rest": 0.01,
    "choice_1": 0,
    "choice_2": 1,
    "choice_3": 4,
    "n_sim_mc": 100_000,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Profile Persistence (保存/加载学生档案)
# ---------------------------------------------------------------------------
def load_profiles() -> dict:
    """读取所有保存的学生档案。"""
    if PROFILES_FILE.exists():
        try:
            return json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_profiles(profiles: dict) -> None:
    """写入所有学生档案到磁盘。"""
    PROFILES_FILE.write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_profile_from_state() -> dict:
    """从当前 session_state 抽取可保存的字段。"""
    return {k: st.session_state.get(k) for k in PROFILE_KEYS if k in st.session_state}


def apply_profile(profile: dict) -> None:
    """把档案数据写入 session_state，下次渲染时输入框会自动取这些值。"""
    for k, v in profile.items():
        if k in PROFILE_KEYS:
            st.session_state[k] = v


def init_session_defaults(middle_school_default: str) -> None:
    """首次启动时初始化所有默认值到 session_state。"""
    if "middle_school" not in st.session_state:
        st.session_state.middle_school = middle_school_default
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if "_session_id" not in st.session_state:
        import uuid
        st.session_state["_session_id"] = str(uuid.uuid4())[:12]


# ---------------------------------------------------------------------------
# 静默日志：每次点「立即分析」自动写一行到本地 jsonl + Google Sheet
# ---------------------------------------------------------------------------
SILENT_LOG_FILE = DATA_DIR / "silent_log.jsonl"
GOOGLE_SHEET_ID = "1k7KDzAsfVLm90b-_k4t_JaQE7PkWjlokbpc2QNYYYIM"
GOOGLE_SHEET_RANGE = "events!A1"


SCHOOL_NAMES_BY_INDEX = [
    "华附(石牌)", "执信(越秀)", "广雅(荔湾)",
    "省实(荔湾)", "六中(海珠)", "广附(本部)",
]


def _idx_to_school(v) -> str:
    """把 choice index (0-5) 或 label string 转成可读学校名。"""
    if isinstance(v, int) and 0 <= v < len(SCHOOL_NAMES_BY_INDEX):
        return SCHOOL_NAMES_BY_INDEX[v]
    if isinstance(v, str):
        return v
    return ""


def _build_row_from_record(record: dict) -> list:
    inputs = record.get("inputs", {})
    extra = record.get("extra", {})
    event = record["event"]

    # 用户填的 1/2/3 志愿（可读名）
    if event == "custom_run":
        choices_lbl = extra.get("choices", [])
        c1 = choices_lbl[0] if len(choices_lbl) > 0 else ""
        c2 = choices_lbl[1] if len(choices_lbl) > 1 else ""
        c3 = choices_lbl[2] if len(choices_lbl) > 2 else ""
    else:
        c1 = _idx_to_school(inputs.get("choice_1"))
        c2 = _idx_to_school(inputs.get("choice_2"))
        c3 = _idx_to_school(inputs.get("choice_3"))

    # 推荐冲/稳/保（只在 analyze 事件才有）
    top_picks = extra.get("top_picks", {}) if isinstance(extra, dict) else {}
    rec_chong = top_picks.get("冲", {}).get("1志", "") if isinstance(top_picks.get("冲"), dict) else ""
    rec_wen = top_picks.get("稳", {}).get("1志", "") if isinstance(top_picks.get("稳"), dict) else ""
    rec_bao = top_picks.get("保", {}).get("1志", "") if isinstance(top_picks.get("保"), dict) else ""

    # 偏好 (声誉/交通/名额)
    def _to_pct(x):
        try: return f"{float(x)*100:.0f}%"
        except: return ""
    pref = (
        f"{_to_pct(inputs.get('w_rep'))} / "
        f"{_to_pct(inputs.get('w_trans'))} / "
        f"{_to_pct(inputs.get('w_quota'))}"
    )

    # 综合录取率：custom_run 用 p_any；analyze 用保方案的综合
    if event == "custom_run":
        p_any = extra.get("p_any") if isinstance(extra, dict) else None
    else:
        bao = top_picks.get("保", {}) if isinstance(top_picks.get("保"), dict) else {}
        p_any = bao.get("综合录取率")
    p_any_str = f"{float(p_any)*100:.1f}%" if p_any is not None else ""

    return [
        record["timestamp"],
        record.get("session_id", ""),
        event,
        str(inputs.get("student_name", "") or ""),
        str(inputs.get("middle_school", "") or ""),
        inputs.get("student_rank") if inputs.get("student_rank") is not None else "",
        inputs.get("student_std") if inputs.get("student_std") is not None else "",
        inputs.get("latest_score") if inputs.get("latest_score") is not None else "",
        c1, c2, c3,
        rec_chong, rec_wen, rec_bao,
        pref,
        p_any_str,
        json.dumps(
            {"inputs": inputs, "extra": extra},
            ensure_ascii=False,
        ),
    ]


@st.cache_resource(show_spinner=False)
def _get_gsheet_worksheet():
    """初始化 Google Sheet 客户端 — 缓存连接对象。
    优先级：
      1. Streamlit secrets["gcp_service_account"]（云端部署）
      2. 本地 service account JSON 文件（开发）
      3. 返回 None（写日志会 fallback 到 gws CLI）
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = None

        # 优先 Streamlit secrets
        if hasattr(st, "secrets") and "gcp_service_account" in st.secrets:
            info = dict(st.secrets["gcp_service_account"])
            creds = Credentials.from_service_account_info(info, scopes=scopes)
        else:
            # 找本地 JSON 文件（仅开发）
            for fname in [
                "school-advisor-logger-42d23ea1328b.json",
            ]:
                p = DATA_DIR / fname
                if p.exists():
                    creds = Credentials.from_service_account_file(str(p), scopes=scopes)
                    break

        if creds is None:
            return None

        client = gspread.authorize(creds)
        return client.open_by_key(GOOGLE_SHEET_ID).sheet1
    except Exception:
        return None


def _write_to_google_sheet(record: dict) -> None:
    """优先用 gspread (服务账号) 写 Sheet；失败则尝试 gws CLI；都失败静默吞。"""
    row = _build_row_from_record(record)
    # 1) 试 gspread
    try:
        ws = _get_gsheet_worksheet()
        if ws is not None:
            ws.append_row(row, value_input_option="USER_ENTERED")
            return
    except Exception:
        pass
    # 2) Fallback: gws CLI（仅本地开发能用）
    try:
        import subprocess
        params = {
            "spreadsheetId": GOOGLE_SHEET_ID,
            "range": GOOGLE_SHEET_RANGE,
            "valueInputOption": "USER_ENTERED",
            "insertDataOption": "INSERT_ROWS",
        }
        subprocess.run(
            [
                "gws", "sheets", "spreadsheets", "values", "append",
                "--params", json.dumps(params),
                "--json", json.dumps({"values": [row]}, ensure_ascii=False),
            ],
            capture_output=True, timeout=15, check=False,
        )
    except Exception:
        pass


def silent_log(event_type: str, extra: dict | None = None) -> None:
    """静默写日志。失败不影响 UI。同时写本地文件 + Google Sheet。"""
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": st.session_state.get("_session_id", "unknown"),
        "event": event_type,
        "inputs": {
            k: st.session_state.get(k)
            for k in PROFILE_KEYS if k in st.session_state
        },
    }
    if extra:
        record["extra"] = extra

    # 1. 本地 JSONL（备份）
    try:
        with open(SILENT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

    # 2. Google Sheet（在后台线程，避免阻塞 UI）
    try:
        import threading
        threading.Thread(
            target=_write_to_google_sheet, args=(record,), daemon=True
        ).start()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_excel(DATA_DIR / "school_v2.xlsx")


# ---------------------------------------------------------------------------
# Core Model
# ---------------------------------------------------------------------------
class SchoolAdvisorV2:
    """择校推荐引擎 v2：排名 + 自招分流 + 志愿优先。"""

    def __init__(
        self,
        schools_df: pd.DataFrame,
        top_n: int = 15,
        zizu_rates: list[float] | None = None,
    ):
        self.schools = schools_df.reset_index(drop=True)
        self.top_n = top_n
        self.n_schools = len(self.schools)
        self.quotas = self.schools["local_quota"].values.astype(np.int32)

        if zizu_rates is None:
            zizu_rates = [0.25] * 3 + [0.10] * 3 + [0.03] * 3 + [0.01] * max(0, top_n - 9)
        # 确保长度等于 top_n
        zizu_rates = (list(zizu_rates) + [0.01] * top_n)[:top_n]
        self.zizu_rates = np.array(zizu_rates, dtype=float)

    # -----------------------------------------------------------------
    # 学校热度（softmax）
    # -----------------------------------------------------------------
    def school_popularity(self, weights: dict, T: float = 1.0) -> np.ndarray:
        """计算 top 学生把每所学校选为1志的概率（softmax）。"""
        s = self.schools
        # 声誉：全市名额倒数 → 归一化到 [0, 1]（越尖端越高）
        rep = 1.0 / s["city_quota"].values.astype(float)
        rep_norm = (rep - rep.min()) / (rep.max() - rep.min() + 1e-9)
        # 交通：高=1, 中=0.5, 低=0
        trans = s["transport"].map(TRANSPORT_SCORES).values.astype(float)
        # 一中名额归一化（越多越保险，得分越高）
        lq = s["local_quota"].values.astype(float)
        quota_norm = (lq - lq.min()) / (lq.max() - lq.min() + 1e-9)

        attr = (
            weights["reputation"] * rep_norm
            + weights["transport"] * trans
            + weights["quota"] * quota_norm
        )
        scaled = attr / max(T, 1e-6)
        exp_s = np.exp(scaled - scaled.max())
        return exp_s / exp_s.sum()

    # -----------------------------------------------------------------
    # 解析引擎（公式法）
    # -----------------------------------------------------------------
    def analytical(
        self, student_rank: int, weights: dict, T: float = 1.0
    ) -> pd.DataFrame:
        """公式法估算每所学校 1/2/3 志愿的录取率。"""
        n_above = max(0, student_rank - 1)
        # 自招分流期望
        zizu_expected = (
            self.zizu_rates[:n_above].sum() if n_above > 0 else 0.0
        )
        n_eff = max(0.0, n_above - zizu_expected)
        n_int = int(round(n_eff))

        pop = self.school_popularity(weights, T)
        rows = []

        for i in range(self.n_schools):
            row = self.schools.iloc[i]
            quota = int(self.quotas[i])
            p = float(pop[i])

            # P(1志) = P(竞争者中 < quota 人填这所为1志)
            if n_int == 0:
                p_1 = 1.0
            else:
                p_1 = float(binom.cdf(quota - 1, n_int, p))

            # P(2志) 近似：1志没填满 × 没有更高排名的2志填这所
            #   (1-p)^n 表示没人把它放1志的概率
            #   再乘以 (1-p*0.4)^n 表示其他人把它放2志的概率较低
            if n_int == 0:
                p_2 = 1.0
                p_3 = 1.0
            else:
                p_no_1st = (1 - p) ** n_int  # 没人填1志
                p_no_2nd_peer = (1 - p * 0.4) ** n_int  # 没人在2志也填这所
                p_2 = p_no_1st * p_no_2nd_peer * 0.7
                p_3 = p_2 * 0.3

            rows.append(
                {
                    "学校": row["high_school"],
                    "校区": row["campus"],
                    "全市名额": int(row["city_quota"]),
                    "一中名额": quota,
                    "交通": row["transport"],
                    "热度": f"{p * 100:.1f}%",
                    "P(1志)": f"{p_1 * 100:.1f}%",
                    "P(2志)": f"{p_2 * 100:.1f}%",
                    "P(3志)": f"{p_3 * 100:.1f}%",
                    "_p_1st": p_1,
                    "_p_2nd": p_2,
                    "_p_3rd": p_3,
                    "_p_popularity": p,
                }
            )
        return pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # 蒙特卡洛核心：预生成场景 + 计算 "竞争对手每轮后剩余名额"
    # -----------------------------------------------------------------
    def _ensure_top_n(self, needed: int) -> None:
        """如果 top_n 不够覆盖学生排名，自动扩展（防止 clip bug）。"""
        if needed > self.top_n:
            old_top_n = self.top_n
            self.top_n = needed
            # 扩展 zizu_rates（新增的位置用最后一个值兜底）
            tail = self.zizu_rates[-1] if len(self.zizu_rates) else 0.01
            extra = np.full(needed - old_top_n, tail, dtype=float)
            self.zizu_rates = np.concatenate([self.zizu_rates, extra])

    def _simulate_scenarios(
        self,
        student_rank: int,
        student_std: float,
        weights: dict,
        T: float,
        n_sim: int,
        seed: int,
    ):
        # 自动确保 top_n 够用（覆盖 ±3σ）
        std_for_buf = max(int(student_std * 3), 5)
        needed = max(15, int(student_rank) + std_for_buf)
        self._ensure_top_n(needed)
        """
        预先模拟 n_sim 个场景，每个场景返回：
          - 竞争对手处理完每轮志愿后剩余的名额
        关键洞察：希哥是排名最低的，所以竞争对手不受希哥影响。
        我们可以预先算出"竞争对手每轮跑完后每所学校剩多少名额"，
        然后对任意希哥策略 (c1, c2, c3) 都能秒判结果。
        """
        rng = np.random.default_rng(seed)
        pop = self.school_popularity(weights, T)
        top_n = self.top_n

        # 1. 学生排名
        student_ranks = rng.normal(student_rank, max(student_std, 1e-3), size=n_sim)
        student_ranks = np.clip(np.round(student_ranks).astype(int), 1, top_n)

        # 2. 自招（每个 top_n 学生独立伯努利）
        zizu_mask = rng.random((n_sim, top_n)) < self.zizu_rates[None, :]

        # 3. 每个学生（除希哥）抽 3 个不同学校（Gumbel-Max trick）
        gumbels = -np.log(-np.log(rng.random((n_sim, top_n, self.n_schools)) + 1e-30) + 1e-30)
        log_pop = np.log(pop + 1e-30)
        scores = log_pop[None, None, :] + gumbels
        # 按分数降序取 top 3：第0个=1志，第1个=2志，第2个=3志
        sorted_idx = np.argsort(-scores, axis=2)
        top3 = sorted_idx[:, :, :3].astype(np.int8)  # (n_sim, top_n, 3)

        # 4. 逐 sim 计算"竞争对手每轮后剩余名额"
        quotas_after = np.zeros((n_sim, 3, self.n_schools), dtype=np.int32)
        n_schools = self.n_schools
        quotas_init = self.quotas

        for sim in range(n_sim):
            s_rank = student_ranks[sim]
            n_above = s_rank - 1
            current = quotas_init.copy()

            if n_above == 0:
                # 希哥是第1名 → 所有竞争对手不存在
                quotas_after[sim, 0] = current
                quotas_after[sim, 1] = current
                quotas_after[sim, 2] = current
                continue

            # 跟踪每个竞争对手是否已被录取
            admitted = np.full(n_above, -1, dtype=np.int8)

            for round_idx in range(3):
                # 单趟扫描：每个竞争对手按排名顺序处理本轮志愿
                for k in range(n_above):
                    if zizu_mask[sim, k]:
                        continue  # 走自招了，不参与
                    if admitted[k] != -1:
                        continue  # 已被录取
                    school = top3[sim, k, round_idx]
                    if current[school] > 0:
                        admitted[k] = school
                        current[school] -= 1
                quotas_after[sim, round_idx] = current

        return student_ranks, quotas_after

    # -----------------------------------------------------------------
    # 蒙特卡洛（指定 1/2/3 志愿）
    # -----------------------------------------------------------------
    def monte_carlo(
        self,
        student_rank: int,
        student_std: float,
        student_choices: list[int],
        weights: dict,
        T: float = 1.0,
        n_sim: int = 100_000,
        seed: int = 42,
    ) -> tuple[dict, np.ndarray]:
        """指定希哥的 (c1, c2, c3)，返回各学校录取率。"""
        _, quotas_after = self._simulate_scenarios(
            student_rank, student_std, weights, T, n_sim, seed
        )

        c1, c2, c3 = student_choices
        outcomes = np.full(n_sim, -1, dtype=np.int8)
        # 1志
        mask1 = quotas_after[:, 0, c1] > 0
        outcomes[mask1] = c1
        # 2志（未被1志录取的）
        mask2 = (~mask1) & (quotas_after[:, 1, c2] > 0)
        outcomes[mask2] = c2
        # 3志（未被1/2志录取的）
        mask3 = (~mask1) & (~mask2) & (quotas_after[:, 2, c3] > 0)
        outcomes[mask3] = c3

        result = {}
        for i in range(self.n_schools):
            result[i] = float(np.mean(outcomes == i))
        result["未录取"] = float(np.mean(outcomes == -1))
        return result, outcomes

    # -----------------------------------------------------------------
    # 模式 B：枚举所有 120 种组合，分组冲/稳/保
    # -----------------------------------------------------------------
    def find_best_strategies(
        self,
        student_rank: int,
        student_std: float,
        weights: dict,
        T: float = 1.0,
        n_sim: int = 100_000,
        seed: int = 42,
    ) -> pd.DataFrame:
        """枚举所有 1/2/3 志愿组合，输出每个的录取概率。"""
        _, quotas_after = self._simulate_scenarios(
            student_rank, student_std, weights, T, n_sim, seed
        )
        # 学校声誉分数（用于"期望质量"计算）
        prestige = 1000.0 / self.schools["city_quota"].values

        all_combos = []
        for c1, c2, c3 in permutations(range(self.n_schools), 3):
            mask1 = quotas_after[:, 0, c1] > 0
            mask2 = (~mask1) & (quotas_after[:, 1, c2] > 0)
            mask3 = (~mask1) & (~mask2) & (quotas_after[:, 2, c3] > 0)

            p_1st = float(np.mean(mask1))
            p_2nd = float(np.mean(mask2))
            p_3rd = float(np.mean(mask3))
            p_any = p_1st + p_2nd + p_3rd
            p_none = 1.0 - p_any

            # 期望声誉
            exp_prestige = (
                p_1st * prestige[c1] + p_2nd * prestige[c2] + p_3rd * prestige[c3]
            )

            # 策略分类：基于"1志可不可能上"
            # 冲：1志录取率 < 30%
            # 稳：1志录取率 30%-70%
            # 保：1志录取率 > 70%
            if p_1st < 0.30:
                strategy = "冲"
            elif p_1st < 0.70:
                strategy = "稳"
            else:
                strategy = "保"

            all_combos.append(
                {
                    "1志": f"{self.schools.iloc[c1]['high_school']}({self.schools.iloc[c1]['campus']})",
                    "2志": f"{self.schools.iloc[c2]['high_school']}({self.schools.iloc[c2]['campus']})",
                    "3志": f"{self.schools.iloc[c3]['high_school']}({self.schools.iloc[c3]['campus']})",
                    "策略": strategy,
                    "1志录取率": p_1st,
                    "2志录取率": p_2nd,
                    "3志录取率": p_3rd,
                    "综合录取率": p_any,
                    "未录取": p_none,
                    "期望声誉": exp_prestige,
                    "_indices": (c1, c2, c3),
                }
            )
        return pd.DataFrame(all_combos)


# ---------------------------------------------------------------------------
# 可视化图表
# ---------------------------------------------------------------------------
SCHOOL_COLORS = px.colors.qualitative.Set2  # 6 所学校用同一套配色
STRATEGY_COLORS = {"冲": "#EF4444", "稳": "#F59E0B", "保": "#10B981"}


def chart_school_radar(schools_df: pd.DataFrame) -> go.Figure:
    """每所学校在三维上的雷达图（全部叠在一个图）。"""
    rep = 1.0 / schools_df["city_quota"].values.astype(float)
    rep = (rep - rep.min()) / (rep.max() - rep.min() + 1e-9)
    trans = schools_df["transport"].map(TRANSPORT_SCORES).values.astype(float)
    quota = schools_df["local_quota"].values.astype(float)
    quota = (quota - quota.min()) / (quota.max() - quota.min() + 1e-9)
    categories = ["学校声誉", "交通便利", "本校名额", "学校声誉"]

    fig = go.Figure()
    for i, (_, row) in enumerate(schools_df.iterrows()):
        fig.add_trace(
            go.Scatterpolar(
                r=[rep[i], trans[i], quota[i], rep[i]],
                theta=categories,
                fill="toself",
                name=f"{row['high_school']}({row['campus']})",
                line=dict(color=SCHOOL_COLORS[i % len(SCHOOL_COLORS)]),
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    return fig


def chart_popularity_donut(schools_df: pd.DataFrame, pop: np.ndarray) -> go.Figure:
    """学校热度甜甜圈图。"""
    labels = [f"{r['high_school']}({r['campus']})" for _, r in schools_df.iterrows()]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=(pop * 100).round(1),
            hole=0.55,
            marker=dict(colors=SCHOOL_COLORS),
            textinfo="label+percent",
            textposition="outside",
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
        annotations=[
            dict(text="Top 学生<br>1志偏好", x=0.5, y=0.5, font_size=14, showarrow=False)
        ],
    )
    return fig


def chart_admission_heatmap(analytical_df: pd.DataFrame) -> go.Figure:
    """录取概率热力图：行=学校, 列=1/2/3志, 数值=录取率%。"""
    matrix = analytical_df[["_p_1st", "_p_2nd", "_p_3rd"]].values * 100
    labels_y = [
        f"{r['学校']}({r['校区']})" for _, r in analytical_df.iterrows()
    ]
    labels_x = ["填为1志", "填为2志", "填为3志"]
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels_x,
            y=labels_y,
            text=[[f"{v:.1f}%" for v in row] for row in matrix],
            texttemplate="%{text}",
            textfont=dict(size=14),
            colorscale=[
                [0, "#FEE2E2"], [0.3, "#FCD34D"],
                [0.6, "#86EFAC"], [1, "#16A34A"],
            ],
            colorbar=dict(title="录取率(%)"),
            hovertemplate="<b>%{y}</b><br>%{x}<br>录取率: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


def chart_outcome_compare(
    mc_result: dict, schools_df: pd.DataFrame,
    analytical_df: pd.DataFrame, idx_1: int, idx_2: int, idx_3: int,
) -> go.Figure:
    """录取分布对比柱状图：解析法 vs 蒙特卡洛。"""
    n = len(schools_df)
    school_labels = [
        f"{r['high_school']}({r['campus']})" for _, r in schools_df.iterrows()
    ]
    a_probs, mc_probs, roles = [], [], []
    role_map = {idx_1: "🥇1志", idx_2: "🥈2志", idx_3: "🥉3志"}
    for i in range(n):
        if i == idx_1:
            a_probs.append(analytical_df.iloc[i]["_p_1st"] * 100)
        elif i == idx_2:
            a_probs.append(analytical_df.iloc[i]["_p_2nd"] * 100)
        elif i == idx_3:
            a_probs.append(analytical_df.iloc[i]["_p_3rd"] * 100)
        else:
            a_probs.append(0.0)
        mc_probs.append(mc_result[i] * 100)
        roles.append(role_map.get(i, "—"))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="解析法（公式估算）",
        x=school_labels, y=a_probs,
        text=[f"{p:.1f}%" for p in a_probs],
        textposition="outside",
        marker_color="#93C5FD",
    ))
    fig.add_trace(go.Bar(
        name="蒙特卡洛（10万次模拟）",
        x=school_labels, y=mc_probs,
        text=[f"{p:.1f}%" for p in mc_probs],
        textposition="outside",
        marker_color="#1D4ED8",
    ))
    fig.update_layout(
        barmode="group",
        yaxis_title="录取率 (%)",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def chart_outcome_pie(mc_result: dict, schools_df: pd.DataFrame) -> go.Figure:
    """蒙特卡洛录取分布饼图（含未录取）。"""
    labels = [
        f"{r['high_school']}({r['campus']})" for _, r in schools_df.iterrows()
    ]
    values = [mc_result[i] * 100 for i in range(len(schools_df))]
    labels.append("❌ 未录取")
    values.append(mc_result["未录取"] * 100)

    colors = list(SCHOOL_COLORS) + ["#9CA3AF"]
    # 隐藏 0% 的项
    show = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0.05]
    if not show:
        return go.Figure()
    labels2, values2, colors2 = zip(*show)

    fig = go.Figure(
        go.Pie(
            labels=labels2,
            values=values2,
            marker=dict(colors=colors2),
            textinfo="label+percent",
            hole=0.4,
        )
    )
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
    )
    return fig


def chart_strategies_scatter(strategies_df: pd.DataFrame) -> go.Figure:
    """120 组合散点图：X=综合录取率, Y=期望声誉, 颜色=策略。"""
    df = strategies_df.copy()
    df["1志录取率_%"] = df["1志录取率"] * 100
    df["综合录取率_%"] = df["综合录取率"] * 100
    df["方案"] = df["1志"] + " → " + df["2志"] + " → " + df["3志"]

    fig = px.scatter(
        df, x="综合录取率_%", y="期望声誉",
        color="策略",
        color_discrete_map=STRATEGY_COLORS,
        size="1志录取率_%",
        hover_name="方案",
        hover_data={
            "1志录取率_%": ":.1f",
            "综合录取率_%": ":.1f",
            "期望声誉": ":.2f",
            "策略": False,
        },
        labels={"综合录取率_%": "综合录取率 (%)", "期望声誉": "期望学校声誉（越高越尖端）"},
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="white")))
    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def chart_sensitivity_lines(
    advisor: "SchoolAdvisorV2", student_rank: int, student_std: float,
    weights: dict, T: float, schools_df: pd.DataFrame,
) -> go.Figure:
    """敏感性折线图：X=排名变化, Y=各校 P(1志)。"""
    rank_min = max(1, int(student_rank - 2 * student_std))
    rank_max = max(rank_min + 1, int(student_rank + 2 * student_std) + 1)
    rows = []
    for r in range(rank_min, rank_max + 1):
        adf = advisor.analytical(r, weights, T)
        for i, row in adf.iterrows():
            rows.append({
                "排名": r,
                "学校": f"{row['学校']}({row['校区']})",
                "P(1志)": row["_p_1st"] * 100,
            })
    df = pd.DataFrame(rows)
    fig = px.line(
        df, x="排名", y="P(1志)", color="学校",
        markers=True,
        color_discrete_sequence=SCHOOL_COLORS,
    )
    fig.add_vline(x=student_rank, line_dash="dash", line_color="gray",
                   annotation_text="当前排名")
    fig.update_layout(
        yaxis_title="1志录取率 (%)",
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def render_school_options(schools_df: pd.DataFrame) -> list[str]:
    return [
        f"{row['high_school']}({row['campus']})"
        for _, row in schools_df.iterrows()
    ]




# ---------------------------------------------------------------------------
# 自定义 CSS — 提升颜值
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
  /* 主背景：柔和薄荷绿 */
  html, body, [class*="css"], .stApp {
    font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
    background-color: #F0FDF4 !important;
    color: #064E3B !important;
  }
  .main .block-container { color: #064E3B !important; }

  /* 大标题深绿 */
  h1, h2, h3, h4 { color: #064E3B !important; letter-spacing: -0.5px; }
  h1 { font-weight: 800 !important; font-size: 3rem !important; line-height: 1.1; }
  h2 { font-weight: 700 !important; font-size: 1.8rem !important; }
  h3 { font-weight: 600 !important; }

  /* ===== 📱 移动端适配 (iPhone < 768px) ===== */
  @media (max-width: 768px) {
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.3rem !important; }
    h3 { font-size: 1.1rem !important; }
    .card-title { font-size: 18px !important; }
    .card-row { font-size: 14px !important; }
    .card-rate { font-size: 28px !important; }
    .card-rate-label { font-size: 12px !important; }
    .student-card { padding: 16px !important; }
    .student-card div { font-size: 14px !important; }
    .recommend-card { padding: 16px !important; margin: 6px 0 !important; }
    /* 主区域两侧留白小一点 */
    .main .block-container { padding: 1rem 0.5rem !important; }
    /* 让默认 columns 在窄屏自动换行（Streamlit 在小屏会自动堆叠，但加这层保险） */
    [data-testid="column"] { min-width: 100% !important; }
  }
  /* 让所有 plotly 图自适应宽度 */
  .js-plotly-plot, .plotly { width: 100% !important; }

  /* 侧边栏：白底 */
  section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #D1FAE5;
  }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 {
    color: #047857 !important;
  }

  /* 推荐卡片（所有文字强制深色，不受主题影响） */
  .recommend-card {
    border-radius: 16px;
    padding: 24px;
    margin: 8px 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    border: 2px solid;
    color: #1F2937 !important;
  }
  .recommend-card * { color: #1F2937 !important; }
  .card-chong { background: linear-gradient(135deg,#FEF2F2 0%,#FEE2E2 100%); border-color: #EF4444; }
  .card-wen   { background: linear-gradient(135deg,#FFFBEB 0%,#FEF3C7 100%); border-color: #F59E0B; }
  .card-bao   { background: linear-gradient(135deg,#F0FDF4 0%,#DCFCE7 100%); border-color: #10B981; }

  .card-title { font-size: 22px; font-weight: 700; margin-bottom: 12px; color: #111827 !important; }
  .card-row { font-size: 16px; margin: 6px 0; color: #1F2937 !important; }
  .card-rate { font-size: 36px; font-weight: 700; margin: 12px 0 4px; color: #111827 !important; }
  .card-rate-label { font-size: 13px; color: #4B5563 !important; }

  /* 学生卡片 */
  .student-card {
    background: linear-gradient(135deg, #DBEAFE 0%, #EFF6FF 100%);
    border-radius: 16px;
    padding: 24px;
    border-left: 6px solid #3B82F6;
    color: #1E3A8A !important;
  }
  .student-card * { color: #1E3A8A !important; }

  /* 欢迎屏 */
  .welcome-box {
    background: linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%);
    border-radius: 20px;
    padding: 60px 40px;
    text-align: center;
    border: 2px dashed #D1D5DB;
    color: #1F2937 !important;
  }
  .welcome-box * { color: #1F2937 !important; }

  /* 让 Run 按钮更显眼 */
  div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
    color: white;
    border: none;
    padding: 14px 28px;
    font-size: 17px;
    font-weight: 600;
    border-radius: 10px;
    box-shadow: 0 4px 14px rgba(37,99,235,0.35);
  }
  div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1D4ED8 0%, #6D28D9 100%);
    transform: translateY(-1px);
  }

  /* 紧凑 expander */
  .streamlit-expanderHeader { font-weight: 600; font-size: 16px; }
</style>
"""


def render_school_options(schools_df: pd.DataFrame) -> list[str]:
    return [f"{r['high_school']}({r['campus']})" for _, r in schools_df.iterrows()]


def render_recommend_card(strat_name: str, color: str, css_class: str,
                          subtitle: str, top1: dict) -> None:
    """渲染冲/稳/保推荐卡片。突出"第一志愿命中率"为主指标。"""
    st.markdown(
        f"""
<div class="recommend-card {css_class}">
  <div class="card-title">{color} {strat_name} — {subtitle}</div>
  <div class="card-row">🥇 第一志愿：<b>{top1['1志']}</b></div>
  <div class="card-row">🥈 第二志愿：<b>{top1['2志']}</b></div>
  <div class="card-row">🥉 第三志愿：<b>{top1['3志']}</b></div>
  <div class="card-rate">{top1['1志录取率']*100:.0f}%</div>
  <div class="card-rate-label">🎯 直接被第一志愿录取的概率</div>
  <div style="margin-top:14px;font-size:14px;color:#374151;">
    总体不滑档概率：<b>{top1['综合录取率']*100:.0f}%</b>
    <span style="color:#6B7280;font-size:12px;">（任意一所被录取）</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="中考志愿填报推荐 | Esther 制作",
        page_icon="🎓",
        layout="wide",
        menu_items={
            "About": "中考第二批次志愿填报推荐工具 v2 · "
                     "基于排名 + 志愿优先规则的概率模型 · "
                     "制作 by Esther",
        },
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    schools_df = load_data()
    middle_school_options = schools_df["middle_school"].unique().tolist()
    init_session_defaults(middle_school_options[0])

    # =================================================================
    # 顶部标题
    # =================================================================
    st.title("🎓 中考升学规划助手")
    st.markdown(
        '<p style="font-size:18px;color:#047857;margin-top:-12px;">'
        "🌿 科学填报，稳中有进 · 基于排名 + 名额博弈的概率推荐 · 适用广州中考第二批次"
        "</p>",
        unsafe_allow_html=True,
    )

    # =================================================================
    # 侧边栏 — 全部输入
    # =================================================================
    st.sidebar.markdown("## 📝 请填写信息")

    st.sidebar.markdown("### 1️⃣ 孩子的基本情况")
    st.sidebar.selectbox("孩子的初中学校", middle_school_options, key="middle_school")
    st.sidebar.text_input(
        "孩子姓名", key="student_name", placeholder="例：张三",
    )
    st.sidebar.number_input(
        "集团排名（若不确定，预估）", min_value=1, max_value=50, step=1,
        value=None, key="student_rank",
        placeholder="例：10（数字越小越好）",
        help=(
            "孩子在初中所属教育集团内的最新排名。\n"
            "1 = 第一名（最好）。\n"
            "若不确定具体排名，请凭近期模考表现预估。"
        ),
    )
    st.sidebar.number_input(
        "排名波动幅度（约 ±N 名）",
        min_value=0.0, max_value=20.0, step=0.5,
        value=None, key="student_std",
        placeholder="例：3（不确定就填 3）",
        help=(
            "孩子每次考试排名上下浮动的范围。\n"
            "经常在 8-12 名 → 填 2；7-13 名 → 填 3；非常稳定 → 填 1。"
        ),
    )
    st.sidebar.number_input(
        "考生最近一次模拟考成绩（可选）",
        min_value=0.0, max_value=900.0, step=1.0,
        value=None, key="latest_score",
        placeholder="例：720",
        help=(
            "最近一次模拟考的总分。\n"
            "此项**不影响推荐计算**（模型基于排名），\n"
            "但有助于我们后续校准模型 / 你自己留存记录。"
        ),
    )

    # 排名超过 20 时给警示（当前数据库只覆盖 6 所顶尖高中，名额有限）
    _rank_check = st.session_state.get("student_rank")
    if _rank_check is not None and _rank_check > 20:
        st.sidebar.warning(
            f"⚠️ 排名 {_rank_check} 偏后\n\n"
            "本工具目前只收录了 **6 所顶尖高中**（华附/执信/广雅/省实/六中/广附），"
            "总共只有 **15 个名额**给一中。\n\n"
            "排名 20 以后被这 6 所学校录取的概率较低，"
            "结果**仅供参考**，建议同时考虑其他批次的高中。"
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 2️⃣ 您最看重学校的什么")
    st.sidebar.caption("调整三个滑块，决定推荐方案的偏好")
    w_rep = st.sidebar.slider(
        "🏆 学校档次（声誉）", 0.0, 1.0, step=0.05, key="w_rep",
        help="越高 = 越优先推荐顶尖学校（如华附）",
    )
    w_trans = st.sidebar.slider(
        "🚌 上学距离（交通）", 0.0, 1.0, step=0.05, key="w_trans",
        help="越高 = 越优先推荐交通方便的学校",
    )
    w_quota = st.sidebar.slider(
        "🎯 录取保险（名额多）", 0.0, 1.0, step=0.05, key="w_quota",
        help="越高 = 越优先推荐本初中分到名额多的学校",
    )
    total_w = w_rep + w_trans + w_quota
    if total_w > 0:
        w_rep_n, w_trans_n, w_quota_n = w_rep/total_w, w_trans/total_w, w_quota/total_w
    else:
        w_rep_n = w_trans_n = w_quota_n = 1/3
    st.sidebar.caption(
        f"占比：档次 **{w_rep_n:.0%}** / 距离 **{w_trans_n:.0%}** / 保险 **{w_quota_n:.0%}**"
    )

    st.sidebar.markdown("---")

    # 高级设置（默认折叠）
    with st.sidebar.expander("⚙️ 高级设置（一般不用动）", expanded=False):
        st.markdown("**竞争集中度**")
        T = st.slider(
            "扎堆 ◀────▶ 分散",
            0.05, 5.0, step=0.05, key="T",
            help="顶尖学生选学校的扎堆程度。默认 1.0 即可。",
        )
        st.caption(f"当前值：{T:.2f}（默认 1.0）")

        st.markdown("**自主招生分流概率**")
        st.caption("前几名同学走自招提前上岸的概率")
        zizu_top3 = st.slider("第 1-3 名", 0.0, 0.6, step=0.05, key="zizu_top3")
        zizu_top6 = st.slider("第 4-6 名", 0.0, 0.4, step=0.05, key="zizu_top6")
        zizu_top9 = st.slider("第 7-9 名", 0.0, 0.2, step=0.01, key="zizu_top9")
        zizu_rest = st.slider("第 10-15 名", 0.0, 0.1, step=0.005, key="zizu_rest")

        st.markdown("**模拟精度**")
        n_sim_options = [10_000, 50_000, 100_000, 500_000, 1_000_000]
        n_sim_mc = st.select_slider(
            "模拟次数", options=n_sim_options, key="n_sim_mc",
            help="次数越多越准。10万 ≈ 1秒，100万 ≈ 8秒。",
        )
        st.caption(f"精度约 ±{1.0/np.sqrt(n_sim_mc)*100:.2f}%")

        # 随机种子隐藏不显示，固定使用 session_state["seed"]（默认 42）
        seed = st.session_state.get("seed", 42)

    st.sidebar.markdown("---")

    # 大按钮：立即分析
    run_clicked = st.sidebar.button(
        "🚀 立即分析", type="primary", use_container_width=True,
    )

    st.sidebar.caption(
        "👆 填好上面信息后，点这里开始分析\n\n"
        "分析约需 1-3 秒"
    )

    # =================================================================
    # 计算参数
    # =================================================================
    student_name = st.session_state.get("student_name") or ""
    middle_school = st.session_state.get("middle_school")
    student_rank = st.session_state.get("student_rank")
    student_std = st.session_state.get("student_std")

    # 关键：top_n 必须大于学生排名（否则模型把排名 30 的学生当成排名 15）
    # 留 10 名 buffer 给 ±2σ 波动
    rank_for_topn = student_rank if student_rank is not None else 15
    std_for_topn = int(student_std) if student_std is not None else 3
    effective_top_n = max(15, rank_for_topn + max(10, std_for_topn * 3))

    weights = {"reputation": w_rep_n, "transport": w_trans_n, "quota": w_quota_n}
    # 自招概率：前 9 名按用户配置，10 名后用兜底值（默认 0.005，远低于前面）
    zizu_rates = (
        [zizu_top3] * 3 + [zizu_top6] * 3 + [zizu_top9] * 3
        + [zizu_rest] * (effective_top_n - 9)
    )
    advisor = SchoolAdvisorV2(
        schools_df, top_n=effective_top_n, zizu_rates=zizu_rates
    )

    # =================================================================
    # 点击「立即分析」时跑算法 + 静默存档（含输入校验）
    # =================================================================
    missing = []
    if not student_name.strip():
        missing.append("孩子姓名")
    if student_rank is None:
        missing.append("最新班级排名")
    if student_std is None:
        missing.append("排名波动幅度")

    if run_clicked and missing:
        st.sidebar.error(
            f"⚠️ 请先填写：{ '、'.join(missing) }"
        )
    elif run_clicked:
        with st.spinner("⏳ 分析中… 正在跑 10 万次模拟"):
            t0 = time.time()
            analytical_df = advisor.analytical(student_rank, weights, T)
            strategies_df = advisor.find_best_strategies(
                student_rank, student_std, weights, T,
                n_sim=n_sim_mc, seed=int(seed),
            )
            elapsed = time.time() - t0

        st.session_state["_analyzed"] = True
        st.session_state["_analytical_df"] = analytical_df.to_dict(orient="records")
        st.session_state["_strategies_df"] = strategies_df.to_dict(orient="records")
        st.session_state["_analysis_meta"] = {
            "elapsed": elapsed,
            "n_sim": int(n_sim_mc),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 静默保存输入 + 结果摘要
        top_picks_summary = {}
        for strat in ["冲", "稳", "保"]:
            sub = strategies_df[strategies_df["策略"] == strat]
            if len(sub) > 0:
                t = sub.sort_values("期望声誉", ascending=False).iloc[0]
                top_picks_summary[strat] = {
                    "1志": t["1志"], "2志": t["2志"], "3志": t["3志"],
                    "1志录取率": float(t["1志录取率"]),
                    "综合录取率": float(t["综合录取率"]),
                    "期望声誉": float(t["期望声誉"]),
                }
        silent_log("analyze", {
            "elapsed": elapsed,
            "top_picks": top_picks_summary,
        })

    # =================================================================
    # 主区域：未分析 vs 已分析
    # =================================================================
    if not st.session_state.get("_analyzed"):
        st.markdown(
            """
<div class="welcome-box">
  <div style="font-size:48px;margin-bottom:8px;">👈</div>
  <div style="font-size:24px;font-weight:600;margin-bottom:8px;">
    请在左侧填写信息，然后点击「🚀 立即分析」
  </div>
  <div style="font-size:16px;color:#6B7280;">
    我们会用 10 万次模拟，为您算出冲 / 稳 / 保三套填报方案。
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # 介绍：本工具能做什么
        st.markdown("###")
        intro_col1, intro_col2, intro_col3 = st.columns(3)
        with intro_col1:
            st.markdown(
                """
### 🔥 冲
**敢拼一把**

第一志愿填顶尖学校，
万一中了就赚到。
"""
            )
        with intro_col2:
            st.markdown(
                """
### ✅ 稳
**踏实选择**

第一志愿填把握中等的学校，
进可攻退可守。
"""
            )
        with intro_col3:
            st.markdown(
                """
### 🛡️ 保
**安全上岸**

第一志愿填一定能上的学校，
确保不滑档。
"""
            )

        return  # 还没分析，到此为止

    # =================================================================
    # 已分析：渲染所有结果
    # =================================================================
    analytical_df = pd.DataFrame(st.session_state["_analytical_df"])
    strategies_df = pd.DataFrame(st.session_state["_strategies_df"])
    meta = st.session_state["_analysis_meta"]

    # ---- 学生信息卡片 ----
    st.markdown(
        f"""
<div class="student-card">
  <div style="font-size:22px;font-weight:600;margin-bottom:8px;">
    👤 {student_name} · {middle_school}
  </div>
  <div style="display:flex;gap:32px;font-size:16px;color:#1E3A8A;">
    <div>📊 最新排名：<b style="font-size:20px;">第 {student_rank} 名</b></div>
    <div>📈 排名波动：<b style="font-size:20px;">± {student_std:.0f} 名</b></div>
    <div>🎯 偏好：档次 {w_rep_n:.0%} / 距离 {w_trans_n:.0%} / 保险 {w_quota_n:.0%}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.caption(
        f"📅 {meta['timestamp']} · 跑了 {meta['n_sim']:,} 次模拟（耗时 {meta['elapsed']:.1f} 秒）"
    )

    # =================================================================
    # 🏆 HERO：三档推荐方案（先做"低录取率"健康检查）
    # =================================================================

    # 找保方案最优组合，看综合录取率高不高
    bao_df = strategies_df[strategies_df["策略"] == "保"]
    if len(bao_df) > 0:
        best_bao = bao_df.sort_values("综合录取率", ascending=False).iloc[0]
        best_bao_p_any = float(best_bao["综合录取率"])
    else:
        best_bao_p_any = 0.0

    chong_df = strategies_df[strategies_df["策略"] == "冲"]
    if len(chong_df) > 0:
        best_chong_p1 = float(chong_df["1志录取率"].max())
    else:
        best_chong_p1 = 0.0

    # 警示 1：所有方案的综合录取率都不高（< 80%）→ 危险信号
    if best_bao_p_any < 0.80:
        st.error(
            f"### ⚠️ 重要提醒\n\n"
            f"根据您的排名（**第 {student_rank} 名**），即使填**最稳的方案**，"
            f"被这 6 所目标高中录取的最高概率也只有 **{best_bao_p_any * 100:.0f}%**。\n\n"
            f"**这意味着**：很可能 3 个志愿全部滑档（落到第三批次或职高）。\n\n"
            f"**建议**：\n"
            f"- 本工具目前只收录了 **6 所顶尖高中**（华附/执信/广雅/省实/六中/广附）。\n"
            f"- 您的排名可能更适合**第三批次**的中等高中（如 协和、真光、一中等）。\n"
            f"- **请把本工具的结果作为参考**，务必结合班主任建议综合判断。"
        )
    elif best_bao_p_any < 0.95:
        st.warning(
            f"### 💡 温馨提示\n\n"
            f"以您的排名（**第 {student_rank} 名**），保方案的录取率约 "
            f"**{best_bao_p_any * 100:.0f}%**。\n\n"
            f"虽然有较大把握，但**仍有 {(1-best_bao_p_any)*100:.0f}% 滑档风险**。"
            f"建议同时关注**第三批次**的备选学校。"
        )

    st.markdown("##")
    st.markdown("## 🏆 我们为您推荐的 3 套填报方案")
    st.caption("根据您的偏好，从 120 种可能组合中挑出最优的三档")

    top_picks = {}
    card_col1, card_col2, card_col3 = st.columns(3)
    # 每档不同的排序策略，确保 保的综合 ≥ 稳 ≥ 冲 (符合直觉)
    sort_keys = {
        "冲": ("期望声誉", False),     # 冲：追求最尖端学校
        "稳": ("综合录取率", False),   # 稳：在 1志 30-70% 区间里挑综合最高
        "保": ("综合录取率", False),   # 保：综合录取率最高优先
    }
    for col, strat_name, color, css_class, subtitle in [
        (card_col1, "冲", "🔥", "card-chong", "敢拼一把"),
        (card_col2, "稳", "✅", "card-wen",   "中等把握"),
        (card_col3, "保", "🛡️", "card-bao",   "安全上岸"),
    ]:
        with col:
            sub = strategies_df[strategies_df["策略"] == strat_name].copy()
            if len(sub) == 0:
                st.markdown(f"### {color} {strat_name}")
                st.warning("当前条件下无符合方案")
                continue
            sort_col, asc = sort_keys[strat_name]
            top1 = sub.sort_values(sort_col, ascending=asc).iloc[0].to_dict()
            top_picks[strat_name] = top1
            render_recommend_card(strat_name, color, css_class, subtitle, top1)

    # 解读说明
    with st.expander("💡 怎么看这三套方案？"):
        st.markdown(
            """
- **🔥 冲**：第一志愿是您"够一够能上"的好学校，命中率不高但赚到了
- **✅ 稳**：第一志愿是您"努力一下能上"的学校，进退都不亏
- **🛡️ 保**：第一志愿是您"基本能上"的学校，确保孩子有学上

**综合录取率** = 三个志愿中至少被一所学校录取的概率。这个数字越接近 100%，越不会滑档。

**第一志愿命中率** = 直接被第一志愿学校录取的概率。
在「志愿优先」规则下，**第一志愿决定一切**——大部分名额在第一轮就分完了。
"""
        )

    # =================================================================
    # 详细分析 — 折叠
    # =================================================================
    st.markdown("##")
    st.markdown("## 📊 详细分析")

    pop = advisor.school_popularity(weights, T)

    with st.expander("🌐 6 所目标高中三维概览（雷达图）", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**学校三维对比**（档次 / 交通 / 名额）")
            st.plotly_chart(chart_school_radar(schools_df), use_container_width=True)
        with col2:
            st.markdown("**顶尖学生最想去哪所学校**")
            st.plotly_chart(
                chart_popularity_donut(schools_df, pop), use_container_width=True
            )
        with st.expander("📋 学校原始数据"):
            display_df = schools_df[
                ["high_school", "campus", "city_quota", "local_quota", "transport"]
            ].copy()
            display_df.columns = ["学校", "校区", "全市总名额", "本校名额", "交通便利度"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    with st.expander("🌡️ 录取概率热力图（每所学校 × 每个志愿位置）", expanded=False):
        st.plotly_chart(
            chart_admission_heatmap(analytical_df), use_container_width=True
        )
        st.caption(
            "**怎么读**：行 = 学校；列 = 你把它填在第几志愿；颜色越绿 = 录取率越高。"
            "你会发现 2志/3志 普遍是红/黄色 — 这就是「志愿优先」的残酷现实。"
        )

    with st.expander("📍 全部 120 种方案散点图（找你最满意的那一个）", expanded=False):
        st.caption(
            "**X = 综合录取率**（越右越稳）｜**Y = 学校档次**（越上越尖端）｜"
            "**点大小 = 第一志愿命中率**｜**颜色 = 策略类型**"
        )
        st.plotly_chart(
            chart_strategies_scatter(strategies_df), use_container_width=True
        )
        st.caption("👉 鼠标悬停到点上，看具体方案。右上角的点最理想。")

    with st.expander("🔬 排名波动对录取率的影响（如果排名变了会怎样）", expanded=False):
        st.caption(
            f"假设排名在 **{max(1, int(student_rank - 2*student_std))}** 到 "
            f"**{int(student_rank + 2*student_std)}** 之间波动，每所学校 1志录取率会怎么变？"
        )
        st.plotly_chart(
            chart_sensitivity_lines(
                advisor, student_rank, student_std, weights, T, schools_df
            ),
            use_container_width=True,
        )
        st.caption(
            "线在高位平稳 = 这所学校无论排名怎么变都稳上；"
            "线陡峭 = 排名变化对录取率影响大。"
        )

    # =================================================================
    # 自定义方案 — 模式 A
    # =================================================================
    st.markdown("##")
    with st.expander("🎯 我想自己选 3 个志愿，看录取概率", expanded=False):
        st.caption("如果您心里已经有方案，可以在这里填进去看看效果")

        school_options = render_school_options(schools_df)

        def _safe_idx(state_key: str, fallback: int) -> int:
            v = st.session_state.get(state_key, fallback)
            if isinstance(v, str) and v in school_options:
                return school_options.index(v)
            if isinstance(v, int) and 0 <= v < len(school_options):
                return v
            return fallback

        col_1, col_2, col_3 = st.columns(3)
        with col_1:
            c1_label = st.selectbox(
                "🥇 第一志愿", school_options, index=_safe_idx("choice_1", 0),
            )
        with col_2:
            c2_label = st.selectbox(
                "🥈 第二志愿", school_options, index=_safe_idx("choice_2", 1),
            )
        with col_3:
            c3_label = st.selectbox(
                "🥉 第三志愿", school_options, index=_safe_idx("choice_3", 4),
            )

        idx_1 = school_options.index(c1_label)
        idx_2 = school_options.index(c2_label)
        idx_3 = school_options.index(c3_label)
        st.session_state["choice_1"] = idx_1
        st.session_state["choice_2"] = idx_2
        st.session_state["choice_3"] = idx_3

        if len({idx_1, idx_2, idx_3}) < 3:
            st.warning("⚠️ 三个志愿不能重复")
        else:
            choices = [idx_1, idx_2, idx_3]
            if st.button("📊 分析这套方案", type="primary"):
                with st.spinner(f"正在模拟 {n_sim_mc:,} 次..."):
                    t0 = time.time()
                    mc_result, _ = advisor.monte_carlo(
                        student_rank, student_std, choices, weights, T,
                        n_sim=n_sim_mc, seed=int(seed),
                    )
                    elapsed = time.time() - t0
                p_any = sum(mc_result[i] for i in range(advisor.n_schools))
                st.session_state["last_mc_result"] = {
                    "mc_result": {str(k): float(v) for k, v in mc_result.items()},
                    "choices": choices,
                    "c1_label": c1_label, "c2_label": c2_label, "c3_label": c3_label,
                    "p_any": float(p_any),
                    "elapsed": elapsed,
                    "n_sim": int(n_sim_mc),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                silent_log("custom_run", {
                    "choices": [c1_label, c2_label, c3_label],
                    "p_any": float(p_any),
                    "mc_result": {str(k): float(v) for k, v in mc_result.items()},
                })

            last = st.session_state.get("last_mc_result")
            if last and last["choices"] == choices:
                mc_result = {int(k) if k.isdigit() else k: v
                             for k, v in last["mc_result"].items()}
                p_any_mc = 1.0 - mc_result["未录取"]

                st.success(f"✅ 方案：{c1_label} → {c2_label} → {c3_label}")

                # ===== vs 最优方案对照 =====
                # 在已生成的 strategies_df 里找综合录取率最高的方案
                best_overall = strategies_df.sort_values(
                    "综合录取率", ascending=False
                ).iloc[0].to_dict()
                best_p_any = float(best_overall["综合录取率"])
                gap = p_any_mc - best_p_any
                # 计算用户方案在 120 种里的排名
                strategies_df_sorted = strategies_df.sort_values(
                    "综合录取率", ascending=False
                ).reset_index(drop=True)
                user_rank_in_120 = None
                for i, row in strategies_df_sorted.iterrows():
                    if abs(row["综合录取率"] - p_any_mc) < 0.005:
                        user_rank_in_120 = i + 1
                        break
                if user_rank_in_120 is None:
                    # 找最接近的
                    diffs = (strategies_df_sorted["综合录取率"] - p_any_mc).abs()
                    user_rank_in_120 = int(diffs.idxmin()) + 1

                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "✅ 你的综合录取率", f"{p_any_mc*100:.1f}%",
                    delta=f"{gap*100:+.1f}pp vs 最优",
                    delta_color="normal" if gap >= -0.05 else "inverse",
                )
                m2.metric("🎯 第一志愿命中", f"{mc_result[idx_1]*100:.1f}%")
                m3.metric("❌ 滑档风险", f"{mc_result['未录取']*100:.1f}%")

                # 排名 + 最优方案推荐
                if user_rank_in_120 <= 5:
                    st.success(
                        f"🌟 **太棒了！** 你的方案在 120 种可能组合中**排第 {user_rank_in_120} 名**，"
                        f"已经接近最优。"
                    )
                elif user_rank_in_120 <= 30:
                    st.info(
                        f"👍 你的方案在 120 种组合中**排第 {user_rank_in_120} 名**（前 25%）。"
                        f"\n\n💡 模型最优方案是：**{best_overall['1志']} → {best_overall['2志']} → "
                        f"{best_overall['3志']}**（综合 {best_p_any*100:.1f}%）"
                    )
                else:
                    st.warning(
                        f"⚠️ 你的方案在 120 种组合中**排第 {user_rank_in_120} 名**，还有提升空间。"
                        f"\n\n💡 **强烈推荐**改填：**{best_overall['1志']} → {best_overall['2志']} → "
                        f"{best_overall['3志']}**\n\n"
                        f"录取率从 **{p_any_mc*100:.1f}% → {best_p_any*100:.1f}%**（提升 {(best_p_any-p_any_mc)*100:.1f}pp）"
                    )

                st.plotly_chart(
                    chart_outcome_pie(mc_result, schools_df),
                    use_container_width=True,
                )

    # ============================================================
    # 底部署名 + 模型说明
    # ============================================================
    st.markdown("---")
    st.markdown(
        """
<div style="text-align: center; padding: 16px 0; color: #4B5563; font-size: 13px;">
  📌 <b>模型说明</b>：基于广州中考<b>第二批次「志愿优先」</b>规则建模 ·
  考虑前几名同学<b>自招分流</b> ·
  <b>10 万次蒙特卡洛</b>保证结果稳定<br>
  <br>
  ⚠️ 本工具结果<b>仅供参考</b>，不构成最终填报建议。请结合孩子真实意愿、
  家庭情况与<b>班主任专业建议</b>综合判断。<br>
  <br>
  💚 制作 by Esther · 数据来源：广州市招考办 2026 年公开数据 ·
  <a href="https://github.com/estherhoho/school-selection-advisor" target="_blank" style="color:#10B981;">GitHub</a>
</div>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
