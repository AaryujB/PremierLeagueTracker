#!/usr/bin/env python3
"""
Premier League Match Predictor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Prediction model — 4 factors weighted equally at 25% each:

  1. Base win rate       P(win | home/away venue) over last 25 games
  2. Markov chain form   3-state chain: GOOD → NEUTRAL → POOR
                         Uses stationary distribution π to measure
                         long-run momentum, not just raw win count
  3. xG model            Expected goals from attack vs defence matchup
  4. Head-to-head        Direct record between these two clubs

Final probabilities are normalised so P(home) + P(draw) + P(away) = 1
"""

import os
import sys
import math
import httpx

API_BASE    = "https://api.football-data.org/v4"
GAME_WINDOW = 25   # only look at the last N finished games

# ── Terminal colours ──────────────────────────────────────────
R  = "\033[0m";   B  = "\033[1m"
CY = "\033[96m";  GR = "\033[92m"
RD = "\033[91m";  YL = "\033[93m"
PU = "\033[95m";  GY = "\033[90m"
WH = "\033[97m"


# ════════════════════════════════════════════════════════════════
# API
# ════════════════════════════════════════════════════════════════

def get_key():
    key = os.getenv("FOOTBALL_DATA_API_KEY", "")
    if not key:
        key = input("Enter your football-data.org API key: ").strip()
    return key

def fetch(url, key, params=None):
    r = httpx.get(url, headers={"X-Auth-Token": key},
                  params=params, timeout=15)
    if r.status_code == 403:
        sys.exit("Error: Invalid API key.")
    if r.status_code != 200:
        sys.exit(f"API error {r.status_code}: {r.text}")
    return r.json()

def get_teams(key):
    return fetch(f"{API_BASE}/competitions/PL/teams", key).get("teams", [])

def get_matches(team_id, key):
    return fetch(f"{API_BASE}/teams/{team_id}/matches", key,
                 params={"status": "FINISHED", "limit": 40}).get("matches", [])


# ════════════════════════════════════════════════════════════════
# MARKOV CHAIN FORM MODEL
#
# We classify each point in the result sequence into one of 3 states
# based on the two most recent results at that point:
#
#   GOOD    (0) — two consecutive wins
#   NEUTRAL (1) — anything else (win+draw, draw+draw, etc.)
#   POOR    (2) — two consecutive losses
#
# We then count every state→state transition across the last 25
# results to build a 3×3 transition matrix T, apply Laplace
# smoothing (add 1 to every cell) so no row is zero, and
# normalise each row to get probabilities.
#
# Raising T to a high power via repeated squaring gives the
# stationary distribution π — the long-run fraction of time
# the team spends in each state.  This captures momentum:
# a team that consistently transitions back to GOOD has structural
# form, not just a lucky recent streak.
#
# Final form score = (π_good − π_poor + 1) / 2  →  0 to 1
# ════════════════════════════════════════════════════════════════

GOOD, NEUTRAL, POOR = 0, 1, 2
STATE_NAMES = ["GOOD", "NEUTRAL", "POOR"]
STATE_COLS  = [GR, GY, RD]

def classify(results, idx):
    """State at position idx (results[0] = most recent game)."""
    if idx >= len(results):
        return NEUTRAL
    r0 = results[idx]
    r1 = results[idx + 1] if idx + 1 < len(results) else None
    if r0 == "W" and r1 == "W":
        return GOOD
    if r0 == "L" and r1 == "L":
        return POOR
    return NEUTRAL

def build_matrix(results):
    """3×3 transition matrix with Laplace smoothing."""
    counts = [[1.0]*3 for _ in range(3)]
    for i in range(len(results) - 2):
        from_s = classify(results, i + 1)   # older state
        to_s   = classify(results, i)        # state it moved to
        counts[from_s][to_s] += 1.0
    return [[v / sum(row) for v in row] for row in counts]

def mat_mul(A, B):
    n = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(n))
             for j in range(n)] for i in range(n)]

def stationary(T):
    """Raise T to 2^8 = 256 — all rows converge to π."""
    M = [row[:] for row in T]
    for _ in range(8):
        M = mat_mul(M, M)
    return M[0]   # all rows identical at convergence

def markov_form(results):
    """Returns (score 0–1, transition matrix, stationary dist)."""
    if len(results) < 3:
        return 0.5, [[1/3]*3]*3, [1/3, 1/3, 1/3]
    T  = build_matrix(results)
    pi = stationary(T)
    score = max(0.0, min(1.0, (pi[GOOD] - pi[POOR] + 1) / 2))
    return score, T, pi


# ════════════════════════════════════════════════════════════════
# STATS  (capped to last GAME_WINDOW games)
# ════════════════════════════════════════════════════════════════

def compute_stats(matches, team_id):
    results, home_r, away_r, gf_list, ga_list = [], [], [], [], []

    for m in matches:
        ft = m.get("score", {}).get("fullTime", {})
        hg, ag = ft.get("home"), ft.get("away")
        if hg is None or ag is None:
            continue
        if m["homeTeam"]["id"] == team_id:
            gf, ga, home = hg, ag, True
        elif m["awayTeam"]["id"] == team_id:
            gf, ga, home = ag, hg, False
        else:
            continue
        r = "W" if gf > ga else ("D" if gf == ga else "L")
        results.append(r); gf_list.append(gf); ga_list.append(ga)
        (home_r if home else away_r).append(r)

    results = results[:GAME_WINDOW]
    gf_list = gf_list[:GAME_WINDOW]
    ga_list = ga_list[:GAME_WINDOW]
    home_r  = home_r[:GAME_WINDOW]
    away_r  = away_r[:GAME_WINDOW]

    n = len(results)
    if n == 0:
        return None

    def rate(lst, v): return lst.count(v) / len(lst) if lst else 0.0

    mk_score, mk_T, mk_pi = markov_form(results)

    return {
        "total_games":        n,
        "win_rate":           rate(results, "W"),
        "draw_rate":          rate(results, "D"),
        "home_win_rate":      rate(home_r, "W") if home_r else 0.45,
        "away_win_rate":      rate(away_r, "W") if away_r else 0.30,
        "avg_goals_scored":   sum(gf_list) / n,
        "avg_goals_conceded": sum(ga_list) / n,
        "recent_form":        results[:10],
        "form_score":         mk_score,
        "markov_T":           mk_T,
        "markov_pi":          mk_pi,
    }


# ════════════════════════════════════════════════════════════════
# PREDICTION
# ════════════════════════════════════════════════════════════════

def predict(hs, as_, h2h_matches, home_id):
    # ── Factor 1: base win rate ──────────────────────────────
    base_h = hs["home_win_rate"]
    base_a = as_["away_win_rate"]
    base_d = max(0.05, 1 - base_h - base_a)

    # ── Factor 2: Markov form score ──────────────────────────
    tf     = hs["form_score"] + as_["form_score"]
    form_h = hs["form_score"] / tf * 0.75 if tf > 0 else 0.45
    form_a = as_["form_score"] / tf * 0.75 if tf > 0 else 0.30
    form_d = max(0.05, 1 - form_h - form_a)

    # ── Factor 3: expected goals ─────────────────────────────
    xg_h   = (hs["avg_goals_scored"] + as_["avg_goals_conceded"]) / 2
    xg_a   = (as_["avg_goals_scored"] + hs["avg_goals_conceded"]) / 2
    gd     = xg_h - xg_a
    p_xg_h = max(0.05, min(0.80,  0.45 + gd * 0.08))
    p_xg_a = max(0.05, min(0.70,  0.30 - gd * 0.06))
    p_xg_d = max(0.05, 1 - p_xg_h - p_xg_a)

    # ── Factor 4: head to head ───────────────────────────────
    hw = hd = ha = 0
    for m in h2h_matches[:10]:
        ft = m.get("score", {}).get("fullTime", {})
        hg, ag = ft.get("home"), ft.get("away")
        if hg is None or ag is None:
            continue
        if m["homeTeam"]["id"] == home_id:
            if hg > ag: hw += 1
            elif hg == ag: hd += 1
            else: ha += 1
        else:
            if ag > hg: hw += 1
            elif ag == hg: hd += 1
            else: ha += 1
    hn = hw + hd + ha
    h2h_h, h2h_d, h2h_a = (hw/hn, hd/hn, ha/hn) if hn > 0 else (0.45, 0.25, 0.30)

    # ── Combine (25% each) and normalise ────────────────────
    W  = 0.25
    rh = W*base_h + W*form_h + W*p_xg_h + W*h2h_h
    ra = W*base_a + W*form_a + W*p_xg_a + W*h2h_a
    rd = W*base_d + W*form_d + W*p_xg_d + W*h2h_d
    t  = rh + ra + rd
    ph, pa, pd = rh/t, ra/t, rd/t

    if ph >= pa and ph >= pd:   winner, conf = "home", ph
    elif pa >= ph and pa >= pd: winner, conf = "away", pa
    else:                       winner, conf = "draw", pd

    return {
        "winner": winner, "confidence": conf,
        "home_win": ph, "draw": pd, "away_win": pa,
        "xg_h": xg_h, "xg_a": xg_a,
        "h2h": f"{hw}W-{hd}D-{ha}L", "h2h_games": hn,
        "factors": {
            "Base Win Rate": (base_h, base_a),
            "Markov Form":   (form_h, form_a),
            "xG Model":      (p_xg_h, p_xg_a),
            "Head to Head":  (h2h_h,  h2h_a),
        }
    }


# ════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ════════════════════════════════════════════════════════════════

def form_dots(recent):
    m = {"W": f"{GR}●{R}", "D": f"{YL}●{R}", "L": f"{RD}●{R}"}
    return " ".join(m.get(r, "?") for r in recent)

def prob_bar(ph, pd, pa, width=44):
    h = round(ph / 100 * width)
    d = round(pd / 100 * width)
    a = max(0, width - h - d)
    return f"{CY}{'█'*h}{R}{GY}{'█'*d}{R}{RD}{'█'*a}{R}"

def mini_bar(val, width=20):
    filled = round(val * width)
    return f"{'█'*filled}{'░'*(width-filled)}"

def section(title):
    print(f"\n  {B}{WH}{title}{R}")

def divider(w=60):
    print(f"{GY}{'─'*w}{R}")


# ════════════════════════════════════════════════════════════════
# MARKOV DISPLAY
# Shows the transition matrix and stationary distribution for
# both teams so the viewer can understand the momentum model
# ════════════════════════════════════════════════════════════════

def print_markov(name, T, pi, color):
    dominant = STATE_NAMES[pi.index(max(pi))]
    dom_col  = STATE_COLS[pi.index(max(pi))]
    score    = max(0.0, min(1.0, (pi[GOOD] - pi[POOR] + 1) / 2))

    print(f"\n  {color}{B}{name}{R}  "
          f"{GY}dominant state: {dom_col}{dominant}{R}  "
          f"{GY}form score: {GR}{score*100:.1f}%{R}")

    # Stationary distribution bar chart
    for i, (sname, scol) in enumerate(zip(STATE_NAMES, STATE_COLS)):
        bar = mini_bar(pi[i], width=18)
        print(f"    {scol}{sname:<8}{R}  {scol}{bar}{R}  {pi[i]*100:>5.1f}%")

    # Transition matrix
    print(f"\n    {GY}Transition matrix  (row=FROM  col=TO){R}")
    header = f"    {GY}{'':10}  {'GOOD':>8}  {'NEUTRAL':>8}  {'POOR':>8}{R}"
    print(header)
    for ri, (row, sname, scol) in enumerate(zip(T, STATE_NAMES, STATE_COLS)):
        cells = []
        max_v = max(row)
        for ci, v in enumerate(row):
            val_str = f"{v*100:.0f}%"
            if v == max_v:
                cells.append(f"{STATE_COLS[ci]}{B}{val_str:>8}{R}")
            else:
                cells.append(f"{GY}{val_str:>8}{R}")
        print(f"    {scol}{sname:<10}{R}  {'  '.join(cells)}")

    # Plain-english interpretation
    stay_good = T[GOOD][GOOD]
    stay_poor = T[POOR][POOR]
    recover   = T[POOR][GOOD]
    print(f"\n    {GY}When in {GR}GOOD{GY} form, stay there {GR}{stay_good*100:.0f}%{GY} of the time{R}")
    print(f"    {GY}When in {RD}POOR{GY} form, stay there {RD}{stay_poor*100:.0f}%{GY} of the time{R}")
    print(f"    {GY}When in {RD}POOR{GY} form, recover to {GR}GOOD{GY}: {GR}{recover*100:.0f}%{GY} of the time{R}")


# ════════════════════════════════════════════════════════════════
# MAIN RESULT PRINTOUT
# ════════════════════════════════════════════════════════════════

def print_result(home_name, away_name, hs, as_, result):
    ph = result["home_win"] * 100
    pd = result["draw"]     * 100
    pa = result["away_win"] * 100
    W  = 60

    print()
    divider(W)
    print(f"  {CY}{B}{home_name}{R}  vs  {RD}{B}{away_name}{R}")
    divider(W)

    # ── Win probability ──────────────────────────────────────
    section("WIN PROBABILITY")
    print(f"  {prob_bar(ph, pd, pa)}")
    print(f"  {CY}{home_name:<24}{ph:>5.1f}%{R}")
    print(f"  {GY}{'Draw':<24}{pd:>5.1f}%{R}")
    print(f"  {RD}{away_name:<24}{pa:>5.1f}%{R}")

    # ── Prediction ───────────────────────────────────────────
    section("PREDICTION")
    if result["winner"] == "home":
        label = f"{GR}{B}🏆  {home_name}{R}"
    elif result["winner"] == "away":
        label = f"{GR}{B}🏆  {away_name}{R}"
    else:
        label = f"{YL}{B}🤝  Draw{R}"
    print(f"  {label}  {GY}({result['confidence']*100:.1f}% confidence){R}")

    # ── Conditional probability factors ──────────────────────
    section(f"CONDITIONAL PROBABILITY FACTORS  {GY}(25% each){R}")
    cols = [CY, GR, YL, PU]
    for (fname, (fh, fa)), col in zip(result["factors"].items(), cols):
        arrow = f"{CY}←{R}" if fh > fa else (f"{RD}→{R}" if fa > fh else "=")
        hbar = mini_bar(fh, 10)
        abar = mini_bar(fa, 10)
        print(f"  {col}{fname:<16}{R}  "
              f"{CY}{hbar} {fh*100:>4.1f}%{R}  {arrow}  "
              f"{RD}{abar} {fa*100:>4.1f}%{R}")

    # ── Markov chain breakdown ────────────────────────────────
    section("MARKOV CHAIN FORM MODEL")
    print(f"  {GY}States: {GR}GOOD{GY}=2 consecutive wins  "
          f"{GY}NEUTRAL{GY}=mixed  {RD}POOR{GY}=2 consecutive losses{R}")
    print(f"  {GY}Stationary distribution π = long-run % of time in each state{R}")
    print(f"  {GY}Form score = (π_good − π_poor + 1) ÷ 2  →  0 to 1{R}")
    print_markov(home_name, hs["markov_T"], hs["markov_pi"], CY)
    print_markov(away_name, as_["markov_T"], as_["markov_pi"], RD)

    # ── Recent form ──────────────────────────────────────────
    section(f"RECENT FORM  {GY}(last 10 games){R}")
    print(f"  {CY}{home_name[:20]:<20}{R}  {form_dots(hs['recent_form'])}")
    print(f"  {RD}{away_name[:20]:<20}{R}  {form_dots(as_['recent_form'])}")

    # ── Stats ────────────────────────────────────────────────
    section(f"STATS  {GY}(last {hs['total_games']} / {as_['total_games']} games){R}")
    rows = [
        ("Win Rate",      f"{hs['win_rate']*100:.0f}%",     f"{as_['win_rate']*100:.0f}%"),
        ("Goals For",     f"{hs['avg_goals_scored']:.2f}",   f"{as_['avg_goals_scored']:.2f}"),
        ("Goals Against", f"{hs['avg_goals_conceded']:.2f}", f"{as_['avg_goals_conceded']:.2f}"),
        ("xG",            f"{result['xg_h']:.2f}",           f"{result['xg_a']:.2f}"),
        ("H2H Record",    result["h2h"],                     f"({result['h2h_games']} games)"),
    ]
    for lbl, hv, av in rows:
        print(f"  {GY}{lbl:<16}{R}  {CY}{hv:>8}{R}    {RD}{av:>8}{R}")

    print()
    divider(W)
    print()


# ════════════════════════════════════════════════════════════════
# TEAM SEARCH
# ════════════════════════════════════════════════════════════════

def pick_team(teams, prompt):
    while True:
        query = input(f"\n{B}{prompt}{R} ").strip().lower()
        if not query:
            continue
        found = [t for t in teams
                 if query in t["name"].lower()
                 or query in t.get("shortName", "").lower()]
        if not found:
            print(f"  {RD}No team found. Try again.{R}")
            continue
        if len(found) == 1:
            print(f"  {GR}✓ {found[0]['name']}{R}")
            return found[0]
        for i, t in enumerate(found, 1):
            print(f"  {i}. {t['name']}")
        try:
            n = int(input("  Pick a number: ")) - 1
            if 0 <= n < len(found):
                return found[n]
        except ValueError:
            pass
        print(f"  {RD}Invalid. Try again.{R}")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print(f"\n{CY}{B}  Premier League Match Predictor{R}")
    print(f"{GY}  Conditional probability + Markov chain form · last {GAME_WINDOW} games{R}\n")

    key = get_key()

    print(f"{GY}  Loading teams...{R}", end="", flush=True)
    teams = get_teams(key)
    print(f"\r  {GR}✓ {len(teams)} teams loaded{R}          ")

    while True:
        home = pick_team(teams, "Home team:")
        away = pick_team(teams, "Away team:")

        if home["id"] == away["id"]:
            print(f"  {RD}Pick two different teams.{R}")
            continue

        print(f"\n{GY}  Fetching match data...{R}", end="", flush=True)
        home_matches = get_matches(home["id"], key)
        away_matches = get_matches(away["id"], key)
        print(f"\r  {GR}✓ Done{R}                    ")

        hs  = compute_stats(home_matches, home["id"])
        as_ = compute_stats(away_matches, away["id"])

        if not hs or not as_:
            print(f"  {RD}Not enough match history. Try different teams.{R}")
            continue

        h2h = [m for m in away_matches
               if m["id"] in {x["id"] for x in home_matches}]
        result = predict(hs, as_, h2h, home["id"])
        print_result(home["name"], away["name"], hs, as_, result)

        if input("  Predict another? (y/n): ").strip().lower() != "y":
            break

    print(f"\n{GY}  Goodbye.{R}\n")


if __name__ == "__main__":
    main()