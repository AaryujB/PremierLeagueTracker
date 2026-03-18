#!/usr/bin/env python3
import os, math, httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio

app = FastAPI()

API_BASE    = "https://api.football-data.org/v4"
GAME_WINDOW = 25

# ── Markov chain ──────────────────────────────────────────────
GOOD, NEUTRAL, POOR = 0, 1, 2
STATE_NAMES = ["GOOD", "NEUTRAL", "POOR"]

def classify(results, idx):
    if idx >= len(results): return NEUTRAL
    r0 = results[idx]
    r1 = results[idx + 1] if idx + 1 < len(results) else None
    if r0 == "W" and r1 == "W": return GOOD
    if r0 == "L" and r1 == "L": return POOR
    return NEUTRAL

def build_matrix(results):
    counts = [[1.0]*3 for _ in range(3)]
    for i in range(len(results) - 2):
        from_s = classify(results, i + 1)
        to_s   = classify(results, i)
        counts[from_s][to_s] += 1.0
    return [[v / sum(row) for v in row] for row in counts]

def mat_mul(A, B):
    n = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

def stationary(T):
    M = [row[:] for row in T]
    for _ in range(8): M = mat_mul(M, M)
    return M[0]

def markov_form(results):
    if len(results) < 3: return 0.5, [[1/3]*3]*3, [1/3, 1/3, 1/3]
    T  = build_matrix(results)
    pi = stationary(T)
    score = max(0.0, min(1.0, (pi[GOOD] - pi[POOR] + 1) / 2))
    return score, T, pi

# ── Bayes' Theorem ─────────────────────────────────────────────
# We compute P(win | form_state) for each Markov state.
#
# Bayes' Theorem:
#   P(win | state) = P(state | win) × P(win) / P(state)
#
# Where:
#   P(win)         = base win rate from historical results
#   P(state | win) = fraction of wins that were preceded by 'state'
#   P(state)       = π_state from the stationary distribution
#
# This lets us use form as *evidence* to update our win probability,
# rather than just averaging it in as a flat weight.

def compute_bayes(results, pi):
    """
    Given the ordered list of results (most-recent first) and the
    stationary distribution pi, return:
      - p_win_given_state: dict mapping state index -> P(win | state)
      - p_state_given_win: dict mapping state index -> P(state | win)  [the likelihood]
      - p_win:             base win rate (prior)
      - bayes_details:     dict of all components for display
    """
    n = len(results)
    if n < 3:
        uniform = {GOOD: 1/3, NEUTRAL: 1/3, POOR: 1/3}
        p_win = 0.5
        return (
            {GOOD: 0.5, NEUTRAL: 0.5, POOR: 0.5},
            uniform, p_win,
            {"note": "insufficient data"}
        )

    # P(win) — the prior
    p_win = results.count("W") / n

    # Count how often each Markov state preceded a win / any result
    state_before_win   = [0, 0, 0]   # numerator counts
    state_before_any   = [0, 0, 0]   # denominator counts

    # Walk oldest → newest so idx+1 is the *previous* game
    # results[0] = most recent, results[-1] = oldest
    # We look at results[i] (the outcome) and the Markov state
    # formed by results[i] and results[i+1] (the two games before it)
    for i in range(n - 2):
        outcome = results[i]               # the game we're predicting into
        state   = classify(results, i + 1) # state formed by the game before + the one before that
        state_before_any[state] += 1
        if outcome == "W":
            state_before_win[state] += 1

    # P(state | win)  — likelihood
    total_wins_with_context = sum(state_before_win)
    if total_wins_with_context == 0:
        p_state_given_win = {GOOD: 1/3, NEUTRAL: 1/3, POOR: 1/3}
    else:
        p_state_given_win = {
            s: state_before_win[s] / total_wins_with_context
            for s in (GOOD, NEUTRAL, POOR)
        }

    # P(state) — the marginal, taken from the stationary distribution
    # (this is the long-run probability of being in each state)
    p_state = {GOOD: pi[GOOD], NEUTRAL: pi[NEUTRAL], POOR: pi[POOR]}

    # Bayes update: P(win | state) = P(state | win) × P(win) / P(state)
    # Clamp to [0,1] for robustness with small samples
    p_win_given_state = {}
    for s in (GOOD, NEUTRAL, POOR):
        denom = p_state[s]
        if denom < 1e-9:
            p_win_given_state[s] = p_win
        else:
            raw = (p_state_given_win[s] * p_win) / denom
            p_win_given_state[s] = max(0.0, min(1.0, raw))

    bayes_details = {
        "p_win":             p_win,
        "p_state_given_win": p_state_given_win,
        "p_state":           p_state,
        "p_win_given_state": p_win_given_state,
        "state_counts_win":  state_before_win,
        "state_counts_any":  state_before_any,
    }
    return p_win_given_state, p_state_given_win, p_win, bayes_details

# ── Stats ─────────────────────────────────────────────────────
def compute_stats(matches, team_id):
    results, home_r, away_r, gf_list, ga_list = [], [], [], [], []
    for m in matches:
        ft = m.get("score", {}).get("fullTime", {})
        hg, ag = ft.get("home"), ft.get("away")
        if hg is None or ag is None: continue
        if m["homeTeam"]["id"] == team_id:
            gf, ga, home = hg, ag, True
        elif m["awayTeam"]["id"] == team_id:
            gf, ga, home = ag, hg, False
        else: continue
        r = "W" if gf > ga else ("D" if gf == ga else "L")
        results.append(r); gf_list.append(gf); ga_list.append(ga)
        (home_r if home else away_r).append(r)
    results = results[:GAME_WINDOW]; gf_list = gf_list[:GAME_WINDOW]
    ga_list = ga_list[:GAME_WINDOW]; home_r = home_r[:GAME_WINDOW]; away_r = away_r[:GAME_WINDOW]
    n = len(results)
    if n == 0: return None
    def rate(lst, v): return lst.count(v) / len(lst) if lst else 0.0
    mk_score, mk_T, mk_pi = markov_form(results)

    # Compute Bayes posteriors from this team's history
    p_win_given_state, p_state_given_win, p_win_prior, bayes_details = compute_bayes(results, mk_pi)

    # Determine the team's *current* Markov state (from the 2 most recent results)
    current_state = classify(results, 0) if len(results) >= 2 else NEUTRAL

    return {
        "total_games": n, "win_rate": rate(results, "W"), "draw_rate": rate(results, "D"),
        "home_win_rate": rate(home_r, "W") if home_r else 0.45,
        "away_win_rate": rate(away_r, "W") if away_r else 0.30,
        "avg_goals_scored": sum(gf_list) / n, "avg_goals_conceded": sum(ga_list) / n,
        "recent_form": results[:10], "form_score": mk_score, "markov_T": mk_T, "markov_pi": mk_pi,
        # Bayes additions
        "current_state":      current_state,
        "p_win_given_state":  p_win_given_state,
        "p_state_given_win":  p_state_given_win,
        "p_win_prior":        p_win_prior,
        "bayes_details":      bayes_details,
    }

# ── Prediction ────────────────────────────────────────────────
def predict(hs, as_, h2h_matches, home_id):
    base_h = hs["home_win_rate"]; base_a = as_["away_win_rate"]
    base_d = max(0.05, 1 - base_h - base_a)

    # ── Markov form factor (unchanged) ────────────────────────
    tf = hs["form_score"] + as_["form_score"]
    form_h = hs["form_score"] / tf * 0.75 if tf > 0 else 0.45
    form_a = as_["form_score"] / tf * 0.75 if tf > 0 else 0.30
    form_d = max(0.05, 1 - form_h - form_a)

    # ── Bayes factor ─────────────────────────────────────────
    # Replace raw form_score with P(win | current Markov state)
    # This is the Bayesian-updated win probability given observed form state.
    bayes_h = hs["p_win_given_state"][hs["current_state"]]
    bayes_a = as_["p_win_given_state"][as_["current_state"]]
    # Normalise so they compete as home-win / away-win probabilities
    bayes_sum = bayes_h + bayes_a
    bayes_ph  = bayes_h / bayes_sum * 0.75 if bayes_sum > 0 else 0.45
    bayes_pa  = bayes_a / bayes_sum * 0.75 if bayes_sum > 0 else 0.30
    bayes_pd  = max(0.05, 1 - bayes_ph - bayes_pa)

    # ── xG factor ────────────────────────────────────────────
    xg_h = (hs["avg_goals_scored"] + as_["avg_goals_conceded"]) / 2
    xg_a = (as_["avg_goals_scored"] + hs["avg_goals_conceded"]) / 2
    gd = xg_h - xg_a
    p_xg_h = max(0.05, min(0.80, 0.45 + gd * 0.08))
    p_xg_a = max(0.05, min(0.70, 0.30 - gd * 0.06))
    p_xg_d = max(0.05, 1 - p_xg_h - p_xg_a)

    # ── H2H factor ───────────────────────────────────────────
    hw = hd = ha = 0
    for m in h2h_matches[:10]:
        ft = m.get("score", {}).get("fullTime", {})
        hg, ag = ft.get("home"), ft.get("away")
        if hg is None or ag is None: continue
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

    # ── Blend: Base 25%, Markov Form 20%, Bayes 20%, xG 20%, H2H 15% ──
    # (Bayes replaces one quarter of what was "form weight" and gets its own slot)
    W_base  = 0.25
    W_form  = 0.20
    W_bayes = 0.20
    W_xg    = 0.20
    W_h2h   = 0.15

    rh = W_base*base_h + W_form*form_h + W_bayes*bayes_ph + W_xg*p_xg_h + W_h2h*h2h_h
    ra = W_base*base_a + W_form*form_a + W_bayes*bayes_pa + W_xg*p_xg_a + W_h2h*h2h_a
    rd = W_base*base_d + W_form*form_d + W_bayes*bayes_pd + W_xg*p_xg_d + W_h2h*h2h_d
    t = rh + ra + rd
    ph, pa, pd = rh/t, ra/t, rd/t

    if ph >= pa and ph >= pd: winner, conf = "home", ph
    elif pa >= ph and pa >= pd: winner, conf = "away", pa
    else: winner, conf = "draw", pd

    return {
        "winner": winner, "confidence": conf,
        "home_win": ph, "draw": pd, "away_win": pa,
        "xg_h": xg_h, "xg_a": xg_a,
        "h2h": f"{hw}W-{hd}D-{ha}L", "h2h_games": hn,
        "factors": {
            "Base Win Rate":  (base_h,   base_a),
            "Markov Form":    (form_h,   form_a),
            "Bayes P(W|state)":(bayes_ph, bayes_pa),
            "xG Model":       (p_xg_h,   p_xg_a),
            "Head to Head":   (h2h_h,    h2h_a),
        },
        # Raw Bayes numbers for the educational display
        "bayes_h": {
            "p_win":             hs["p_win_prior"],
            "current_state":     hs["current_state"],
            "p_state_given_win": hs["p_state_given_win"][hs["current_state"]],
            "p_state":           hs["markov_pi"][hs["current_state"]],
            "p_win_given_state": hs["p_win_given_state"][hs["current_state"]],
            "all_posteriors":    hs["p_win_given_state"],
        },
        "bayes_a": {
            "p_win":             as_["p_win_prior"],
            "current_state":     as_["current_state"],
            "p_state_given_win": as_["p_state_given_win"][as_["current_state"]],
            "p_state":           as_["markov_pi"][as_["current_state"]],
            "p_win_given_state": as_["p_win_given_state"][as_["current_state"]],
            "all_posteriors":    as_["p_win_given_state"],
        },
    }

# ── ANSI helpers ──────────────────────────────────────────────
def ansi(code): return f"\033[{code}m"
RESET=ansi(0); BOLD=ansi(1); CY=ansi(96); GR=ansi(92); RD=ansi(91)
YL=ansi(93); PU=ansi(95); GY=ansi(90); WH=ansi(97)

def c(col, text, bold=False): return f"{BOLD if bold else ''}{col}{text}{RESET}"
def form_char(r):
    if r=="W": return f"{GR}●{RESET}"
    if r=="D": return f"{YL}●{RESET}"
    return f"{RD}●{RESET}"

def prob_bar(ph, pd, pa, width=44):
    h = round(ph/100*width); d = round(pd/100*width); a = max(0,width-h-d)
    return f"{CY}{'█'*h}{RESET}{GY}{'█'*d}{RESET}{RD}{'█'*a}{RESET}"

def mini_bar(val, width=20):
    filled = round(val*width)
    return f"{'█'*filled}{'░'*(width-filled)}"

def divider(w=60): return f"{GY}{'─'*w}{RESET}"
def thin(w=50):    return f"{GY}{'·'*w}{RESET}"

async def send(ws, text): await ws.send_text(text + "\r\n")
async def send_raw(ws, text): await ws.send_text(text)

# ── Terminal session ──────────────────────────────────────────
class Session:
    def __init__(self, ws):
        self.ws = ws
        self.key = os.getenv("FOOTBALL_DATA_API_KEY", "")
        self.teams = []
        self.buf = ""

    async def readline(self, prompt=""):
        if prompt: await send_raw(self.ws, prompt)
        self.buf = ""
        while True:
            data = await self.ws.receive_text()
            if data == "\r" or data == "\n":
                await send(self.ws, "")
                return self.buf.strip()
            elif data in ("\x7f", "\b"):
                if self.buf:
                    self.buf = self.buf[:-1]
                    await send_raw(self.ws, "\b \b")
            elif data == "\x03":
                raise WebSocketDisconnect()
            else:
                self.buf += data
                await send_raw(self.ws, data)

    async def fetch(self, url, params=None):
        async with httpx.AsyncClient() as client:
            r = await client.get(url, headers={"X-Auth-Token": self.key},
                                 params=params, timeout=15)
        if r.status_code == 403: raise ValueError("Invalid API key.")
        if r.status_code != 200: raise ValueError(f"API error {r.status_code}")
        return r.json()

    async def load_teams(self):
        data = await self.fetch(f"{API_BASE}/competitions/PL/teams")
        self.teams = data.get("teams", [])

    async def get_matches(self, team_id):
        data = await self.fetch(f"{API_BASE}/teams/{team_id}/matches",
                                params={"status": "FINISHED", "limit": 40})
        return data.get("matches", [])

    async def pick_team(self, prompt):
        while True:
            query = (await self.readline(f"\r\n{BOLD}{prompt}{RESET} ")).lower()
            if not query: continue
            found = [t for t in self.teams if query in t["name"].lower()
                     or query in t.get("shortName","").lower()]
            if not found:
                await send(self.ws, f"  {RD}No team found. Try again.{RESET}")
                continue
            if len(found) == 1:
                await send(self.ws, f"  {GR}✓ {found[0]['name']}{RESET}")
                return found[0]
            for i, t in enumerate(found, 1):
                await send(self.ws, f"  {i}. {t['name']}")
            try:
                n = int(await self.readline("  Pick a number: ")) - 1
                if 0 <= n < len(found): return found[n]
            except ValueError: pass
            await send(self.ws, f"  {RD}Invalid.{RESET}")

    async def print_bayes_section(self, team_name, col, bayes, team_stats):
        """Print the full Bayes' Theorem educational breakdown for one team."""
        ws = self.ws
        state_idx  = bayes["current_state"]
        state_name = STATE_NAMES[state_idx]
        state_cols = [GR, GY, RD]
        sc         = state_cols[state_idx]

        p_win   = bayes["p_win"]
        p_sg_w  = bayes["p_state_given_win"]
        p_s     = bayes["p_state"]
        p_wg_s  = bayes["p_win_given_state"]

        await send(ws, f"\r\n  {col}{BOLD}{team_name}{RESET}  {GY}current state →{RESET} {sc}{BOLD}{state_name}{RESET}")

        # Formula line
        await send(ws, f"")
        await send(ws, f"  {GY}  Bayes' Theorem:{RESET}")
        await send(ws, f"  {WH}  P(win | {state_name}) = P({state_name} | win) × P(win) / P({state_name}){RESET}")
        await send(ws, f"")

        # Values
        num_str   = f"{p_sg_w:.3f} × {p_win:.3f}"
        denom_str = f"{p_s:.3f}"
        post_str  = f"{p_wg_s:.3f}  ({p_wg_s*100:.1f}%)"
        prior_str = f"{p_win:.3f}  ({p_win*100:.1f}%)"

        await send(ws, f"  {GY}  P(win)              [prior]      = {WH}{prior_str}{RESET}")
        await send(ws, f"  {GY}  P({state_name:<7} | win)  [likelihood] = {WH}{p_sg_w:.3f}  ({p_sg_w*100:.1f}%){RESET}")
        await send(ws, f"  {GY}  P({state_name:<7})         [marginal]   = {WH}{p_s:.3f}  ({p_s*100:.1f}%){RESET}")
        await send(ws, f"  {GY}  ─────────────────────────────────{RESET}")

        # Show the arithmetic
        if p_s > 1e-9:
            product = p_sg_w * p_win
            await send(ws, f"  {GY}    ({num_str}) / {denom_str}{RESET}")
            await send(ws, f"  {GY}  = {product:.4f} / {denom_str}{RESET}")
        await send(ws, f"  {GY}  P(win | {state_name:<7}) [posterior]  = {GR}{BOLD}{post_str}{RESET}")

        # Prior → Posterior update visual
        delta = p_wg_s - p_win
        arrow = f"{GR}▲ +{delta*100:.1f}%" if delta > 0.005 else (f"{RD}▼ {delta*100:.1f}%" if delta < -0.005 else f"{GY}≈ ~0%")
        await send(ws, f"  {GY}  Belief update: {WH}{p_win*100:.1f}%{GY} → {GR}{p_wg_s*100:.1f}%{GY}  [{arrow}{GY}]{RESET}")

        # Table: P(win | state) for all 3 states
        await send(ws, f"")
        await send(ws, f"  {GY}  All posterior win probabilities:{RESET}")
        await send(ws, f"  {GY}  {'State':<10}  {'P(win|state)':>14}  {'bar'}{RESET}")
        for s, (sn, sc2) in enumerate(zip(STATE_NAMES, state_cols)):
            pws = bayes["all_posteriors"][s]
            bar = mini_bar(pws, 16)
            marker = f" {WH}← current{RESET}" if s == state_idx else ""
            await send(ws, f"  {sc2}  {sn:<10}{RESET}  {WH}{pws*100:>6.1f}%{RESET}  {sc2}{bar}{RESET}{marker}")

    async def print_result(self, home_name, away_name, hs, as_, result):
        ws = self.ws
        ph = result["home_win"]*100; pd = result["draw"]*100; pa = result["away_win"]*100

        await send(ws, "")
        await send(ws, divider())
        await send(ws, f"  {c(CY,home_name,True)}  vs  {c(RD,away_name,True)}")
        await send(ws, divider())

        # ── Win probability ───────────────────────────────────
        await send(ws, f"\r\n  {BOLD}{WH}WIN PROBABILITY{RESET}")
        await send(ws, f"  {prob_bar(ph, pd, pa)}")
        await send(ws, f"  {CY}{home_name:<24}{ph:>5.1f}%{RESET}")
        await send(ws, f"  {GY}{'Draw':<24}{pd:>5.1f}%{RESET}")
        await send(ws, f"  {RD}{away_name:<24}{pa:>5.1f}%{RESET}")

        # ── Prediction ────────────────────────────────────────
        await send(ws, f"\r\n  {BOLD}{WH}PREDICTION{RESET}")
        if result["winner"]=="home": label=f"{GR}{BOLD}{home_name}{RESET}"
        elif result["winner"]=="away": label=f"{GR}{BOLD}{away_name}{RESET}"
        else: label=f"{YL}{BOLD}Draw{RESET}"
        await send(ws, f"  {label}  {GY}({result['confidence']*100:.1f}% confidence){RESET}")

        # ── Conditional probability factors ───────────────────
        await send(ws, f"\r\n  {BOLD}{WH}CONDITIONAL PROBABILITY FACTORS{RESET}  {GY}(weights: base 25% · form 20% · bayes 20% · xG 20% · h2h 15%){RESET}")
        cols = [CY, GR, PU, YL, GR]
        for (fname, (fh, fa)), col in zip(result["factors"].items(), cols):
            arrow = f"{CY}←{RESET}" if fh>fa else (f"{RD}→{RESET}" if fa>fh else "=")
            hbar=mini_bar(fh,10); abar=mini_bar(fa,10)
            await send(ws, f"  {col}{fname:<20}{RESET}  {CY}{hbar} {fh*100:>4.1f}%{RESET}  {arrow}  {RD}{abar} {fa*100:>4.1f}%{RESET}")

        # ── Bayes' Theorem breakdown ──────────────────────────
        await send(ws, f"\r\n  {BOLD}{WH}BAYES' THEOREM  —  P(win | Markov form state){RESET}")
        await send(ws, f"  {GY}Using form state as *evidence* to update the win probability prior.{RESET}")
        await send(ws, f"  {GY}Formula:  P(win|state) = P(state|win) × P(win)  /  P(state){RESET}")

        await self.print_bayes_section(home_name, CY, result["bayes_h"], hs)
        await send(ws, thin())
        await self.print_bayes_section(away_name, RD, result["bayes_a"], as_)

        # ── Markov chain model ────────────────────────────────
        await send(ws, f"\r\n  {BOLD}{WH}MARKOV CHAIN FORM MODEL{RESET}")
        await send(ws, f"  {GY}States: {GR}GOOD{GY}=2 consecutive wins  NEUTRAL=mixed  {RD}POOR{GY}=2 consecutive losses{RESET}")
        for name, stats, col in [(home_name, hs, CY), (away_name, as_, RD)]:
            pi = stats["markov_pi"]; T = stats["markov_T"]
            score = max(0.0, min(1.0, (pi[GOOD]-pi[POOR]+1)/2))
            dom = STATE_NAMES[pi.index(max(pi))]
            dom_cols = [GR, GY, RD]
            dom_col = dom_cols[pi.index(max(pi))]
            await send(ws, f"\r\n  {col}{BOLD}{name}{RESET}  {GY}dominant: {dom_col}{dom}{RESET}  {GY}form: {GR}{score*100:.1f}%{RESET}")
            for i, (sn, sc) in enumerate(zip(STATE_NAMES, [GR,GY,RD])):
                bar = mini_bar(pi[i], 18)
                await send(ws, f"    {sc}{sn:<8}{RESET}  {sc}{bar}{RESET}  {pi[i]*100:>5.1f}%")
            await send(ws, f"    {GY}Transition matrix{RESET}")
            await send(ws, f"    {GY}{'':10}  {'GOOD':>8}  {'NEUTRAL':>8}  {'POOR':>8}{RESET}")
            for ri, (row, sn, sc) in enumerate(zip(T, STATE_NAMES, [GR,GY,RD])):
                max_v = max(row)
                cells = []
                for ci, v in enumerate(row):
                    val_str = f"{v*100:.0f}%"
                    c_col = [GR,GY,RD][ci]
                    if v == max_v: cells.append(f"{c_col}{BOLD}{val_str:>8}{RESET}")
                    else: cells.append(f"{GY}{val_str:>8}{RESET}")
                await send(ws, f"    {sc}{sn:<10}{RESET}  {'  '.join(cells)}")

        # ── Recent form ───────────────────────────────────────
        await send(ws, f"\r\n  {BOLD}{WH}RECENT FORM{RESET}  {GY}(last 10){RESET}")
        await send(ws, f"  {CY}{home_name[:20]:<20}{RESET}  {' '.join(form_char(r) for r in hs['recent_form'])}")
        await send(ws, f"  {RD}{away_name[:20]:<20}{RESET}  {' '.join(form_char(r) for r in as_['recent_form'])}")

        # ── Stats summary ─────────────────────────────────────
        await send(ws, f"\r\n  {BOLD}{WH}STATS{RESET}  {GY}(last {hs['total_games']} / {as_['total_games']} games){RESET}")
        rows = [
            ("Win Rate",      f"{hs['win_rate']*100:.0f}%",     f"{as_['win_rate']*100:.0f}%"),
            ("Goals For",     f"{hs['avg_goals_scored']:.2f}",   f"{as_['avg_goals_scored']:.2f}"),
            ("Goals Against", f"{hs['avg_goals_conceded']:.2f}", f"{as_['avg_goals_conceded']:.2f}"),
            ("xG",            f"{result['xg_h']:.2f}",           f"{result['xg_a']:.2f}"),
            ("H2H Record",    result["h2h"],                     f"({result['h2h_games']} games)"),
        ]
        for lbl, hv, av in rows:
            await send(ws, f"  {GY}{lbl:<16}{RESET}  {CY}{hv:>8}{RESET}    {RD}{av:>8}{RESET}")

        await send(ws, "")
        await send(ws, divider())

    async def run(self):
        ws = self.ws
        await send(ws, f"\r\n{CY}{BOLD}  Premier League Match Predictor{RESET}")
        await send(ws, f"{GY}  Markov chain form · Bayes' Theorem · last {GAME_WINDOW} games{RESET}\r\n")
        await send(ws, f"{GY}  Loading teams...{RESET}")
        try:
            await self.load_teams()
        except ValueError as e:
            await send(ws, f"{RD}  Error: {e}{RESET}")
            return
        await send(ws, f"\r  {GR}✓ {len(self.teams)} teams loaded{RESET}          ")

        while True:
            home = await self.pick_team("Home team:")
            away = await self.pick_team("Away team:")
            if home["id"] == away["id"]:
                await send(ws, f"  {RD}Pick two different teams.{RESET}")
                continue
            await send(ws, f"\r\n{GY}  Fetching match data...{RESET}")
            home_matches = await self.get_matches(home["id"])
            away_matches = await self.get_matches(away["id"])
            await send(ws, f"\r  {GR}✓ Done{RESET}                    ")
            hs  = compute_stats(home_matches, home["id"])
            as_ = compute_stats(away_matches, away["id"])
            if not hs or not as_:
                await send(ws, f"  {RD}Not enough match history.{RESET}")
                continue
            h2h = [m for m in away_matches if m["id"] in {x["id"] for x in home_matches}]
            result = predict(hs, as_, h2h, home["id"])
            await self.print_result(home["name"], away["name"], hs, as_, result)
            again = await self.readline(f"  Predict another? (y/n): ")
            if again.lower() != "y":
                await send(ws, f"\r\n{GY}  Goodbye.{RESET}\r\n")
                break


# ── Routes ────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = Session(websocket)
    try:
        await session.run()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(f"\r\n{RD}Error: {e}{RESET}\r\n")
        except: pass


# ── HTML ──────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PL Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/xterm@5.3.0/css/xterm.min.css" />
<script src="https://cdn.jsdelivr.net/npm/xterm@5.3.0/lib/xterm.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/xterm-addon-fit@0.8.0/lib/xterm-addon-fit.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { width: 100%; height: 100%; background: #0a0a0a; overflow: hidden; }
  body { display: flex; flex-direction: column; align-items: center; justify-content: center; font-family: 'Share Tech Mono', monospace; }
  #header {
    width: 100%; max-width: 900px;
    padding: 18px 28px 10px;
    display: flex; align-items: center; gap: 12px;
  }
  #header .badge {
    font-size: 11px; letter-spacing: 0.15em; text-transform: uppercase;
    color: #00e5ff; border: 1px solid #00e5ff33; padding: 3px 10px;
    background: #00e5ff0a;
  }
  #header .title { color: #555; font-size: 11px; letter-spacing: 0.1em; }
  #terminal-wrap {
    width: 100%; max-width: 900px;
    flex: 1; min-height: 0;
    padding: 0 20px 20px;
  }
  #terminal {
    width: 100%; height: 100%;
    border: 1px solid #1e1e1e;
    background: #0d0d0d;
  }
  .xterm-viewport { background: transparent !important; }
  .xterm { padding: 14px; }
</style>
</head>
<body>
<div id="header">
  <span class="badge">PL Predictor</span>
  <span class="title">football-data.org · Markov Chain · Bayes' Theorem</span>
</div>
<div id="terminal-wrap">
  <div id="terminal"></div>
</div>
<script>
const term = new Terminal({
  fontFamily: "'Share Tech Mono', monospace",
  fontSize: 14,
  lineHeight: 1.4,
  theme: {
    background:   '#0d0d0d',
    foreground:   '#c8c8c8',
    cursor:       '#00e5ff',
    cursorAccent: '#0d0d0d',
    black:        '#1a1a1a',
    brightBlack:  '#444',
    cyan:         '#00e5ff',
    brightCyan:   '#00e5ff',
    green:        '#00e676',
    brightGreen:  '#00e676',
    red:          '#ff1744',
    brightRed:    '#ff1744',
    yellow:       '#ffd600',
    brightYellow: '#ffd600',
    magenta:      '#e040fb',
    brightMagenta:'#e040fb',
    white:        '#c8c8c8',
    brightWhite:  '#ffffff',
  },
  cursorBlink: true,
  allowTransparency: true,
  convertEol: true,
});

const fitAddon = new FitAddon.FitAddon();
term.loadAddon(fitAddon);
term.open(document.getElementById('terminal'));
fitAddon.fit();
window.addEventListener('resize', () => fitAddon.fit());

const proto = location.protocol === 'https:' ? 'wss' : 'ws';
const ws = new WebSocket(`${proto}://${location.host}/ws`);

ws.onmessage = e => term.write(e.data);
ws.onclose   = () => term.write('\\r\\n\\x1b[90m  Connection closed.\\x1b[0m\\r\\n');
ws.onerror   = () => term.write('\\r\\n\\x1b[91m  Connection error.\\x1b[0m\\r\\n');

term.onData(data => {
  if (ws.readyState === WebSocket.OPEN) ws.send(data);
});
</script>
</body>
</html>
"""