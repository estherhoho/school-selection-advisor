"""
School Selection Advisor
========================
A Streamlit web app that recommends optimal 3-school selection strategies
based on student exam performance and historical admission data.

Usage:
    cd "Claude Code Project/School_selection"
    source .venv/bin/activate
    streamlit run tools/school_advisor.py

Algorithm:
    1. Predict cutoffs via weighted trend on 5 years of data
    2. Estimate class score distribution from student's score + rank
    3. Model competition via binomial distribution of top-student choices
    4. Calculate admission probability at each choice level (1st/2nd/3rd)
    5. Search all valid 3-school permutations grouped by strategy type
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from itertools import permutations
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent
SCORE_MAX = 810


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_school_data() -> pd.DataFrame:
    return pd.read_excel(DATA_DIR / "school.xlsx")


@st.cache_data
def load_student_data() -> pd.DataFrame:
    return pd.read_excel(DATA_DIR / "student.xlsx")


# ---------------------------------------------------------------------------
# Core Algorithm
# ---------------------------------------------------------------------------
class SchoolAdvisor:
    """Recommendation engine for school selection strategy."""

    def __init__(self, school_df: pd.DataFrame, total_students: int = 200):
        self.school_df = school_df
        self.total_students = total_students
        self.schools = sorted(school_df["shool_id"].unique())
        self.predictions = self._predict_cutoffs()
        self.popularity = self._compute_popularity()

    def _predict_cutoffs(self) -> dict:
        """Weighted trend analysis on historical cutoff data."""
        predictions = {}
        for school_id in self.schools:
            data = self.school_df[
                self.school_df["shool_id"] == school_id
            ].sort_values("year")
            years = data["year"].values.astype(float)
            scores = data["min_score"].values.astype(float)
            quotas = data["quota"].values.astype(int)
            n = len(years)

            weights = np.array([2.0**i for i in range(n)])
            weights /= weights.sum()
            weighted_avg = float(np.average(scores, weights=weights))

            if n >= 3:
                slope, intercept, *_ = stats.linregress(years, scores)
                next_year = years[-1] + 1
                trend_pred = slope * next_year + intercept
                predicted = 0.6 * weighted_avg + 0.4 * trend_pred
                trend_label = (
                    "Rising" if slope > 2 else ("Falling" if slope < -2 else "Stable")
                )
            else:
                predicted = weighted_avg
                trend_label = "N/A"
                slope = 0.0

            score_std = max(10.0, float(np.std(scores)))
            predictions[school_id] = {
                "predicted_cutoff": round(predicted, 1),
                "cutoff_std": round(score_std, 1),
                "quota": int(quotas[-1]),
                "trend": trend_label,
                "slope": round(slope, 2),
                "hist_years": years.tolist(),
                "hist_scores": scores.tolist(),
            }
        return predictions

    def _compute_popularity(self) -> dict:
        """
        Estimate probability a random top student picks each school as 1st choice.
        Uses softmax over predicted cutoffs with two temperature scenarios.
        """
        cutoffs = np.array(
            [self.predictions[s]["predicted_cutoff"] for s in self.schools]
        )
        c_norm = (cutoffs - cutoffs.min()) / max(1, cutoffs.max() - cutoffs.min())

        popularity = {}
        # Optimistic: choices spread out (high temp) -> less competition at top
        # Pessimistic: choices concentrate at top (low temp) -> more competition
        for scenario, temp in [("optimistic", 3.0), ("pessimistic", 6.0)]:
            logits = c_norm * temp
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            popularity[scenario] = {
                s: float(p) for s, p in zip(self.schools, probs)
            }
        return popularity

    def _class_distribution(self, score: float, rank: int) -> tuple[float, float]:
        """Estimate Normal(mu, sigma) for the class score distribution."""
        T = self.total_students
        percentile = (T - rank + 0.5) / T
        z = stats.norm.ppf(min(0.999, max(0.001, percentile)))
        class_std = 35.0
        class_mean = score - z * class_std
        return float(class_mean), class_std

    def analyze(self, score: float, rank: int, score_std: float = 20.0) -> dict:
        """
        Compute per-school admission probabilities at each choice level.
        """
        class_mean, class_std = self._class_distribution(score, rank)
        results = {}

        for school_id, pred in self.predictions.items():
            cutoff = pred["predicted_cutoff"]
            cutoff_unc = pred["cutoff_std"]
            quota = pred["quota"]

            combined_std = np.sqrt(score_std**2 + cutoff_unc**2)
            p_eligible = float(stats.norm.cdf((score - cutoff) / combined_std))
            margin = round(score - cutoff, 1)

            entry = {
                "predicted_cutoff": cutoff,
                "cutoff_std": cutoff_unc,
                "quota": quota,
                "trend": pred["trend"],
                "score_margin": margin,
                "p_eligible": round(p_eligible, 4),
            }

            for scenario in ("optimistic", "pessimistic"):
                p_school = self.popularity[scenario][school_id]

                # --- 1st CHOICE ---
                # Binomial: among (rank-1) students above us, how many pick
                # this school? Need fewer than quota to pick it.
                if rank <= 1:
                    p_rank_1st = 1.0
                else:
                    p_rank_1st = float(
                        stats.binom.cdf(quota - 1, rank - 1, p_school)
                    )
                p_1st = p_eligible * p_rank_1st

                # --- Expected 1st-round fill ---
                # How many eligible students (from whole class) pick this as 1st?
                p_above = 1 - stats.norm.cdf(cutoff, class_mean, class_std)
                exp_1st_eligible = max(0.3, self.total_students * p_above * p_school)

                # P(seats remaining after 1st round) = P(Poisson(lam) < quota)
                p_seat_after_1st = float(
                    stats.poisson.cdf(quota - 1, exp_1st_eligible)
                )

                # --- 2nd CHOICE ---
                # Need: seat remaining AND outrank other 2nd-choice applicants.
                # 2nd-choice applicants = students who failed 1st and put this as 2nd.
                # ~93% of students fail 1st round (only ~14/200 seats total in round 1).
                # Among those, some fraction pick this school as 2nd choice.
                p_2nd_popularity = p_school * 1.2  # slightly broader for 2nd choice
                exp_2nd_eligible = max(
                    0.2,
                    self.total_students * p_above * p_2nd_popularity * 0.93,
                )
                # How many of those outrank me?
                p_better = (rank - 1) / max(1, self.total_students - 1)
                # Expected 2nd-round competitors who are better than me
                exp_better_2nd = exp_2nd_eligible * p_better
                # P(I get a remaining seat): need remaining > 0 AND I outrank others
                # Approximate: P(seat available) * P(fewer than remaining_seats
                # people ahead of me in 2nd round)
                if quota == 1:
                    # With quota 1, either the seat was taken in round 1 or not
                    p_rank_2nd = float(
                        stats.poisson.cdf(0, exp_better_2nd)
                    )
                else:
                    # With quota 2, maybe 1 seat left
                    p_rank_2nd = float(
                        stats.poisson.cdf(max(0, quota - 1), exp_better_2nd)
                    )

                p_2nd = p_eligible * p_seat_after_1st * p_rank_2nd

                # --- 3rd CHOICE ---
                # After rounds 1+2, estimate remaining seats.
                # Use Poisson model for combined fill from rounds 1 and 2.
                exp_total_12 = exp_1st_eligible + exp_2nd_eligible * p_seat_after_1st
                p_seat_after_12 = float(
                    stats.poisson.cdf(quota - 1, exp_total_12)
                )
                # 3rd-round competition is lighter (most placed or picked other 3rds)
                exp_better_3rd = exp_better_2nd * 0.5
                p_rank_3rd = float(
                    stats.poisson.cdf(max(0, quota - 1), exp_better_3rd)
                )
                p_3rd = p_eligible * p_seat_after_12 * p_rank_3rd

                entry[f"p_1st_{scenario}"] = round(min(0.99, max(0.005, p_1st)), 4)
                entry[f"p_2nd_{scenario}"] = round(min(0.90, max(0.002, p_2nd)), 4)
                entry[f"p_3rd_{scenario}"] = round(min(0.80, max(0.001, p_3rd)), 4)

            results[school_id] = entry
        return results

    def get_recommendations(
        self, analysis: dict, scenario: str
    ) -> dict[str, list]:
        """
        Return diverse strategy recommendations grouped by type.

        Enforces descending quality order (1st >= 2nd >= 3rd by cutoff)
        to reflect sound strategy: aim high first, step down for backup.
        """
        schools = list(analysis.keys())
        all_combos = []

        for perm in permutations(schools, 3):
            s1, s2, s3 = perm
            a1, a2, a3 = analysis[s1], analysis[s2], analysis[s3]

            # Enforce: 1st choice quality >= 2nd >= 3rd
            if a1["predicted_cutoff"] < a2["predicted_cutoff"]:
                continue
            if a2["predicted_cutoff"] < a3["predicted_cutoff"]:
                continue

            p1 = a1[f"p_1st_{scenario}"]
            p2 = a2[f"p_2nd_{scenario}"]
            p3 = a3[f"p_3rd_{scenario}"]

            p_get_1 = p1
            p_get_2 = (1 - p1) * p2
            p_get_3 = (1 - p1) * (1 - p2) * p3
            p_any = p_get_1 + p_get_2 + p_get_3

            q1 = a1["predicted_cutoff"]
            q2 = a2["predicted_cutoff"]
            q3 = a3["predicted_cutoff"]
            # Expected quality (no fallback penalty — let P(any) speak for itself)
            if p_any > 0.001:
                exp_quality = (q1 * p_get_1 + q2 * p_get_2 + q3 * p_get_3) / p_any
            else:
                exp_quality = q3

            # Classify by 1st-choice category
            m1 = a1["score_margin"]
            if m1 < -10:
                strategy = "Aggressive"
            elif m1 <= 10:
                strategy = "Balanced"
            else:
                strategy = "Conservative"

            all_combos.append({
                "1st": s1,
                "2nd": s2,
                "3rd": s3,
                "p_1st": round(p_get_1, 4),
                "p_2nd": round(p_get_2, 4),
                "p_3rd": round(p_get_3, 4),
                "p_any": round(p_any, 4),
                "exp_quality": round(exp_quality, 1),
                "strategy": strategy,
            })

        # Group by strategy and sort each group by expected quality
        grouped = {"Aggressive": [], "Balanced": [], "Conservative": []}
        for c in all_combos:
            grouped[c["strategy"]].append(c)

        for key in grouped:
            grouped[key].sort(key=lambda c: c["exp_quality"], reverse=True)
            # Deduplicate: keep only combos with distinct 1st-choice schools
            seen_1st = set()
            deduped = []
            for c in grouped[key]:
                if c["1st"] not in seen_1st:
                    seen_1st.add(c["1st"])
                    deduped.append(c)
                if len(deduped) >= 5:
                    break
            grouped[key] = deduped

        return grouped


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="School Selection Advisor",
        page_icon=":material/school:",
        layout="wide",
    )

    st.title("School Selection Advisor")
    st.caption(
        "Recommend optimal 3-school strategies based on your exam score, "
        "class rank, and 5 years of historical admission data."
    )

    school_df = load_school_data()
    student_df = load_student_data()

    # =================================================================
    # SIDEBAR
    # =================================================================
    st.sidebar.header("Your Information")

    student_score = st.sidebar.number_input(
        "Latest Exam Score",
        min_value=400.0,
        max_value=810.0,
        value=700.0,
        step=0.5,
        help="Your most recent exam total score",
    )
    student_rank = st.sidebar.number_input(
        "Class Rank (1 = best)",
        min_value=1,
        max_value=500,
        value=10,
        step=1,
        help="Your rank in the class based on latest exam",
    )
    total_students = st.sidebar.number_input(
        "Total Students in Class",
        min_value=50,
        max_value=500,
        value=200,
        step=10,
    )

    with st.sidebar.expander("Advanced Settings"):
        score_variability = st.slider(
            "Your Score Variability (+/-)",
            min_value=5,
            max_value=50,
            value=20,
            help="Standard deviation of your scores across recent exams",
        )

    st.sidebar.markdown("---")
    run = st.sidebar.button(
        "Analyze & Recommend", type="primary", use_container_width=True
    )

    with st.sidebar.expander("Sample Students (for testing)"):
        for sid in student_df["Stu_id"].unique():
            sub = student_df[student_df["Stu_id"] == sid].sort_values("date")
            latest = sub.iloc[-1]
            st.write(
                f"**{sid.title()}** — Score {latest['score']:.1f}, "
                f"Rank {int(latest['rank'])}"
            )
            st.caption(
                f"Mean {sub['score'].mean():.0f} +/- {sub['score'].std():.0f}  |  "
                f"Last 3: {[round(x, 1) for x in sub.tail(3)['score'].tolist()]}"
            )

    # =================================================================
    # LANDING PAGE (before analysis)
    # =================================================================
    if not run:
        st.info(
            "Enter your score and rank in the sidebar, "
            "then click **Analyze & Recommend**."
        )
        st.header("Historical Cutoff Scores")
        pivot = school_df.pivot_table(
            index="shool_id", columns="year", values="min_score"
        )
        pivot = pivot.sort_values(pivot.columns[-1], ascending=False)
        st.dataframe(pivot.style.format("{:.0f}"), use_container_width=True)
        return

    # =================================================================
    # RUN ANALYSIS
    # =================================================================
    advisor = SchoolAdvisor(school_df, total_students)
    analysis = advisor.analyze(student_score, student_rank, score_variability)

    # ------------------------------------------------------------------
    # 1. Per-school table
    # ------------------------------------------------------------------
    st.header("School-by-School Analysis")

    rows = []
    for sid in sorted(
        analysis, key=lambda s: analysis[s]["predicted_cutoff"], reverse=True
    ):
        a = analysis[sid]
        m = a["score_margin"]
        cat = "Reach" if m < -10 else ("Target" if m <= 10 else "Safety")
        rows.append({
            "School": sid,
            "Pred. Cutoff": f"{a['predicted_cutoff']:.0f}",
            "Trend": a["trend"],
            "Quota": a["quota"],
            "Your Margin": f"{m:+.0f}",
            "P(Eligible)": f"{a['p_eligible']:.0%}",
            "1st Opt": f"{a['p_1st_optimistic']:.1%}",
            "1st Pes": f"{a['p_1st_pessimistic']:.1%}",
            "2nd Opt": f"{a['p_2nd_optimistic']:.1%}",
            "2nd Pes": f"{a['p_2nd_pessimistic']:.1%}",
            "Category": cat,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("""
**Reading the table:**
- **Margin** = your score - predicted cutoff (positive is good)
- **1st / 2nd** = probability of admission at that choice level
- **Opt / Pes** = optimistic (competition spread) vs pessimistic (competition concentrated)
- **Category**: *Reach* (margin < -10), *Target* (-10 to +10), *Safety* (> +10)
""")

    # ------------------------------------------------------------------
    # 2. Strategy recommendations
    # ------------------------------------------------------------------
    st.header("Recommended Strategies")

    # Run for both scenarios
    for scenario, scenario_label in [
        ("optimistic", "Optimistic Scenario (competition spreads out)"),
        ("pessimistic", "Pessimistic Scenario (competition concentrates at top)"),
    ]:
        st.subheader(scenario_label)
        recs = advisor.get_recommendations(analysis, scenario)

        col_agg, col_bal, col_con = st.columns(3)

        for col, strat_name, strat_desc in [
            (col_agg, "Aggressive", "1st choice is a Reach school — high risk, high reward"),
            (col_bal, "Balanced", "1st choice is a Target school — realistic stretch"),
            (col_con, "Conservative", "1st choice is a Safety school — maximize placement"),
        ]:
            with col:
                st.markdown(f"**{strat_name}**")
                st.caption(strat_desc)
                combos = recs.get(strat_name, [])
                if not combos:
                    st.write("*No viable options in this category.*")
                    continue

                for i, c in enumerate(combos[:3]):
                    label = f"{c['1st']} -> {c['2nd']} -> {c['3rd']}"
                    with st.expander(label, expanded=(i == 0)):
                        mc1, mc2 = st.columns(2)
                        mc1.write(f"**1st:** {c['1st']} ({c['p_1st']:.1%})")
                        mc1.write(f"**2nd:** {c['2nd']} ({c['p_2nd']:.1%})")
                        mc1.write(f"**3rd:** {c['3rd']} ({c['p_3rd']:.1%})")
                        mc2.metric("P(Any School)", f"{c['p_any']:.1%}")
                        mc2.metric("Avg Quality", f"{c['exp_quality']:.0f}")

        st.markdown("---")

    # ------------------------------------------------------------------
    # 3. Quick Summary
    # ------------------------------------------------------------------
    st.header("Strategic Summary")

    reaches = sorted(
        [s for s, a in analysis.items() if a["score_margin"] < -10],
        key=lambda s: analysis[s]["predicted_cutoff"],
        reverse=True,
    )
    targets = sorted(
        [s for s, a in analysis.items() if -10 <= a["score_margin"] <= 10],
        key=lambda s: analysis[s]["predicted_cutoff"],
        reverse=True,
    )
    safeties = sorted(
        [s for s, a in analysis.items() if a["score_margin"] > 10],
        key=lambda s: analysis[s]["predicted_cutoff"],
        reverse=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Reach Schools", len(reaches))
    c2.metric("Target Schools", len(targets))
    c3.metric("Safety Schools", len(safeties))

    st.markdown(f"""
**Your Profile:** Score **{student_score}**, Rank **{student_rank}** / {total_students}

**Key Principles:**
1. **1st Choice is critical** — with only 1-2 quota per school, most seats fill in round 1.
   Pick the best school where you have a *realistic* chance.
2. **2nd Choice = backup** — seats here are scarce (leftovers from round 1).
   Pick a school likely to have remaining seats.
3. **3rd Choice = safety net** — maximize the chance you land *somewhere*.
""")

    if reaches:
        items = [
            f"{s} (cutoff ~{analysis[s]['predicted_cutoff']:.0f}, margin {analysis[s]['score_margin']:+.0f})"
            for s in reaches
        ]
        st.write(f"**Reach:** {', '.join(items)}")
    if targets:
        items = [
            f"{s} (cutoff ~{analysis[s]['predicted_cutoff']:.0f}, margin {analysis[s]['score_margin']:+.0f})"
            for s in targets
        ]
        st.write(f"**Target:** {', '.join(items)}")
    if safeties:
        items = [
            f"{s} (cutoff ~{analysis[s]['predicted_cutoff']:.0f}, margin {analysis[s]['score_margin']:+.0f})"
            for s in safeties
        ]
        st.write(f"**Safety:** {', '.join(items)}")

    # ------------------------------------------------------------------
    # 4. Historical chart
    # ------------------------------------------------------------------
    st.header("Historical Cutoff Trends")

    chart_data = []
    for sid in analysis:
        pred = advisor.predictions[sid]
        for y, s in zip(pred["hist_years"], pred["hist_scores"]):
            chart_data.append({"School": sid, "Year": int(y), "Min Score": s})

    chart_pivot = pd.DataFrame(chart_data).pivot(
        index="Year", columns="School", values="Min Score"
    )
    st.line_chart(chart_pivot, use_container_width=True)

    st.markdown("---")
    st.caption(
        "Assumptions: class scores ~ Normal; competition via binomial model; "
        "school popularity from cutoff softmax. Probabilities are estimates."
    )


if __name__ == "__main__":
    main()
