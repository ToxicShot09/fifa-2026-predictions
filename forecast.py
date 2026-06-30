"""
2026 FIFA World Cup forecast — LIVE / IN-TOURNAMENT edition (post group stage).
- Strength model: current World Football Elo (June 2026) where known, else mapped from FIFA rank.
- LIVE: every match already played (wc_2026_results.csv) is ingested two ways:
    (1) each result nudges that team's Elo via the standard World Football Elo update
        (goal-difference-weighted K, +100 home advantage for hosts; penalty shoot-outs
        count as draws), so current form drives the knockout-stage simulations.
    (2) results already decided are LOCKED — played knockout ties keep their real winner;
        only matches not yet played are simulated.
- Bracket: the REAL official 2026 knockout bracket (Round of 32 -> R16 -> QF -> SF -> Final),
  reconstructed from the actual group-stage qualifiers.
- Monte Carlo: the remaining bracket is played out 50,000 times with a Poisson goals model.
"""
import numpy as np
import pandas as pd
import os
from collections import defaultdict

rng = np.random.default_rng(42)
N = 50_000

teams = pd.read_csv("wc_2026_teams.csv")

# --- Current World Football Elo (June 2026), from eloratings / worldcupelo ---
ELO_KNOWN = {
    "Spain": 2157, "Argentina": 2115, "France": 2063, "England": 2024,
    "Brazil": 1991, "Portugal": 1989, "Colombia": 1982, "Netherlands": 1959,
    "Croatia": 1933, "Germany": 1910, "Belgium": 1849, "Morocco": 1860,
    "Uruguay": 1890, "Switzerland": 1855, "Japan": 1838, "USA": 1790,
    "Senegal": 1815, "Mexico": 1790, "Ecuador": 1800, "Austria": 1790,
}

# Map FIFA rank -> Elo for teams without a known value (Elo ~= 2262 - 151*ln(rank)).
def rank_to_elo(rank):
    return 2262 - 151 * np.log(rank)

def elo_for(row):
    return ELO_KNOWN.get(row["team"], rank_to_elo(row["fifa_rank"]))

teams["elo"] = teams.apply(elo_for, axis=1)

HOSTS = {"USA", "Mexico", "Canada"}
HOST_BONUS = 35  # modest home advantage in the goals engine

team_elo = dict(zip(teams["team"], teams["elo"]))
team_conf = dict(zip(teams["team"], teams["confederation"]))
team_group = dict(zip(teams["team"], teams["group"]))
elo_pre = dict(team_elo)  # snapshot before live updates, for reporting

# --- LIVE RESULTS INGESTION -------------------------------------------------
RESULTS_FILE = "wc_2026_results.csv"
if os.path.exists(RESULTS_FILE):
    results = pd.read_csv(RESULTS_FILE).fillna("")
else:
    results = pd.DataFrame(columns=["stage", "team1", "score1", "score2", "team2", "winner", "date"])

# (1) Dynamic Elo update from every played result — standard World Football Elo method.
ELO_K0 = 60.0
ELO_HOME_ADV = 100.0
def _margin_k(k0, gd):
    gd = abs(gd)
    if gd <= 1:  return k0
    if gd == 2:  return k0 * 1.5
    if gd == 3:  return k0 * 1.75
    return k0 * (1.75 + (gd - 3) / 8.0)

for _, r in results.iterrows():
    a, b = r["team1"], r["team2"]
    ga, gb = int(r["score1"]), int(r["score2"])
    ra = team_elo[a] + (ELO_HOME_ADV if a in HOSTS else 0.0)
    rb = team_elo[b] + (ELO_HOME_ADV if b in HOSTS else 0.0)
    we_a = 1.0 / (1.0 + 10 ** (-(ra - rb) / 400.0))
    wa = 1.0 if ga > gb else (0.0 if ga < gb else 0.5)  # pens count as a draw for Elo
    k = _margin_k(ELO_K0, ga - gb)
    delta = k * (wa - we_a)
    team_elo[a] += delta
    team_elo[b] -= delta

# (2) Locked knockout outcomes: which already-played tie sent whom through.
played_ko = {}   # frozenset({a,b}) -> advancing team
for _, r in results.iterrows():
    if r["stage"] != "Group" and not str(r["stage"]).startswith("Group"):
        w = r["winner"] if r["winner"] else (r["team1"] if int(r["score1"]) > int(r["score2"]) else r["team2"])
        played_ko[frozenset((r["team1"], r["team2"]))] = w

# --- REAL 2026 KNOCKOUT BRACKET ---------------------------------------------
# Round of 32 in fold order: each consecutive pair is a tie, and winners fold upward
# (adjacent pairs) through R16 -> QF -> SF -> Final exactly as the official bracket.
# Top half (Final left side) = indices 0-15 ; bottom half = 16-31.
R32 = [
    ("Germany", "Paraguay"),                 # M75  -> R16 vs France/Sweden
    ("France", "Sweden"),                    # M77
    ("South Africa", "Canada"),              # M73  -> R16 vs Netherlands/Morocco
    ("Netherlands", "Morocco"),              # M76
    ("Spain", "Austria"),                    # M84  -> R16 vs Portugal/Croatia
    ("Portugal", "Croatia"),                 # M83
    ("Belgium", "Senegal"),                  # M82  -> R16 vs USA/Bosnia
    ("USA", "Bosnia and Herzegovina"),       # M81
    ("Brazil", "Japan"),                     # M74  -> R16 vs Ivory Coast/Norway
    ("Ivory Coast", "Norway"),               # M78
    ("Mexico", "Ecuador"),                   # M79  -> R16 vs England/DR Congo
    ("England", "DR Congo"),                 # M80
    ("Australia", "Egypt"),                  # M87  -> R16 vs Argentina/Cape Verde
    ("Argentina", "Cape Verde"),             # M88
    ("Switzerland", "Algeria"),              # M86  -> R16 vs Colombia/Ghana
    ("Colombia", "Ghana"),                   # M85
]

GOAL_BASE = 1.36
ELO_SCALE = 0.0019

def sim_ko_pair(a, b, n):
    """Single knockout tie between fixed teams a,b -> array of winners (n sims)."""
    ra = team_elo[a] + (HOST_BONUS if a in HOSTS else 0)
    rb = team_elo[b] + (HOST_BONUS if b in HOSTS else 0)
    la = GOAL_BASE * np.exp(ELO_SCALE * (ra - rb))
    lb = GOAL_BASE * np.exp(ELO_SCALE * (rb - ra))
    ga = rng.poisson(la, n); gb = rng.poisson(lb, n)
    p_a = np.clip(0.5 + (ra - rb) / 4000.0, 0.15, 0.85)
    coin = rng.random(n) < p_a
    a_adv = (ga > gb) | ((ga == gb) & coin)
    return np.where(a_adv, a, b)

host_arr = np.array(list(HOSTS))
def run_round(bk, n):
    """Fold a (2m, n) array of team rows into (m, n) winners. Adjacent pairs meet."""
    velo = np.vectorize(lambda t: team_elo[t])
    out = []
    for k in range(bk.shape[0] // 2):
        a_row, b_row = bk[2*k], bk[2*k + 1]
        ea = velo(a_row) + np.where(np.isin(a_row, host_arr), HOST_BONUS, 0)
        eb = velo(b_row) + np.where(np.isin(b_row, host_arr), HOST_BONUS, 0)
        ga = rng.poisson(GOAL_BASE * np.exp(ELO_SCALE * (ea - eb)))
        gb = rng.poisson(GOAL_BASE * np.exp(ELO_SCALE * (eb - ea)))
        p_a = np.clip(0.5 + (ea - eb) / 4000.0, 0.15, 0.85)
        coin = rng.random(n) < p_a
        a_adv = (ga > gb) | ((ga == gb) & coin)
        out.append(np.where(a_adv, a_row, b_row))
    return np.stack(out, axis=0)

def resolve_r32(n):
    """Resolve the 16 R32 ties: locked winner if already played, else simulate."""
    winners = []
    for a, b in R32:
        key = frozenset((a, b))
        if key in played_ko:
            winners.append(np.full(n, played_ko[key]))
        else:
            winners.append(sim_ko_pair(a, b, n))
    return np.stack(winners, axis=0)   # (16, n)

# Counters
win_counts = defaultdict(int)     # champion
final_counts = defaultdict(int)   # reached final
semi_counts = defaultdict(int)    # reached semi-finals (top 4)
qf_counts = defaultdict(int)      # reached quarter-finals (top 8)
r16_counts = defaultdict(int)     # reached round of 16 (top 16)

BATCH = 2000
done = 0
while done < N:
    n = min(BATCH, N - done)
    done += n

    r16 = resolve_r32(n)        # (16, n) — Round-of-16 participants
    for row in r16:
        v, c = np.unique(row, return_counts=True)
        for x, y in zip(v, c): r16_counts[x] += y
    r8 = run_round(r16, n)      # (8, n) — quarter-finalists
    for row in r8:
        v, c = np.unique(row, return_counts=True)
        for x, y in zip(v, c): qf_counts[x] += y
    r4 = run_round(r8, n)       # (4, n) — semi-finalists (TOP 4)
    for row in r4:
        v, c = np.unique(row, return_counts=True)
        for x, y in zip(v, c): semi_counts[x] += y
    r2 = run_round(r4, n)       # (2, n) — finalists
    for row in r2:
        v, c = np.unique(row, return_counts=True)
        for x, y in zip(v, c): final_counts[x] += y
    champ = run_round(r2, n)[0] # champion
    v, c = np.unique(champ, return_counts=True)
    for x, y in zip(v, c): win_counts[x] += y

def pct(d, t):
    return 100.0 * d.get(t, 0) / N

# Only teams still alive (in R32 and not already knocked out) get a meaningful forecast.
eliminated_ko = {a if played_ko[frozenset((a, b))] == b else b
                 for a, b in R32 if frozenset((a, b)) in played_ko}
alive = sorted({t for pair in R32 for t in pair} - eliminated_ko)
rows = []
for t in alive:
    rows.append((t, team_conf[t], team_group[t], int(round(team_elo[t])),
                 pct(win_counts, t), pct(final_counts, t), pct(semi_counts, t),
                 pct(qf_counts, t), pct(r16_counts, t)))
res = pd.DataFrame(rows, columns=["team", "conf", "grp", "elo",
                                  "win%", "final%", "semi%", "qf%", "r16%"])
res = res.sort_values("win%", ascending=False).reset_index(drop=True)

pd.set_option("display.width", 220)

print("\n=== LIVE STATE INGESTED ===")
n_ko = len(played_ko)
print(f"Matches played and locked: {len(results)}  ({len(results)-n_ko} group + {n_ko} knockout)")
print("Eliminated already: Germany & Netherlands (R32, on penalties), plus all non-qualifiers.")
if len(results):
    mov = sorted(((t, team_elo[t] - elo_pre[t]) for t in elo_pre), key=lambda x: -abs(x[1]))
    print("Biggest cumulative Elo swings from live form (post − pre), teams still alive:")
    shown = 0
    for t, d in mov:
        if t in set(alive) and abs(d) >= 0.5:
            print(f"   {t:<14} {elo_pre[t]:7.0f} -> {team_elo[t]:7.0f}  ({d:+6.1f})")
            shown += 1
        if shown >= 12:
            break

print("\n=== 2026 WORLD CUP — MONTE CARLO FORECAST ({} sims, live, real bracket) ===".format(N))
print(res.head(16).to_string(index=False, float_format=lambda x: f"{x:5.1f}"))
print("\nTotal win% (sanity, should be ~100):", round(res["win%"].sum(), 1))

# ---------------- HISTORICAL CONTEXT ----------------
ed = pd.read_csv("wc_all_editions.csv")
print("\n\n=== HISTORICAL PATTERNS (1930-2022, 22 editions) ===")
champs = ed["champion"].replace({"West Germany": "Germany"}).value_counts()
print("\nTitles by nation:")
print(champs.to_string())
host_won = (ed["host_won"] == "Yes").sum()
print(f"\nHost nation won: {host_won}/{len(ed)} = {100*host_won/len(ed):.0f}%")

# ---------------- VERDICT ----------------
FLAG = {
 "Spain": "🇪🇸", "Argentina": "🇦🇷", "France": "🇫🇷", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Portugal": "🇵🇹",
 "Brazil": "🇧🇷", "Colombia": "🇨🇴", "Croatia": "🇭🇷", "Morocco": "🇲🇦", "Belgium": "🇧🇪",
 "Mexico": "🇲🇽", "USA": "🇺🇸", "Switzerland": "🇨🇭", "Paraguay": "🇵🇾", "Canada": "🇨🇦"}

top4 = res.sort_values("semi%", ascending=False).head(4)["team"].tolist()
finalists = res.sort_values("final%", ascending=False).head(2)["team"].tolist()
champ = res.sort_values("win%", ascending=False).head(1)["team"].iloc[0]

print("\n\n=== VERDICT (model, live, real bracket) ===")
print("Predicted SEMI-FINALISTS (top 4):")
for t in top4:
    print(f"   {FLAG.get(t,'  ')}  {t:<11} ({pct(semi_counts,t):4.1f}% to reach semis)")
print("\nPredicted FINAL:")
print(f"   {FLAG.get(finalists[0])}  {finalists[0]}  vs  {FLAG.get(finalists[1])}  {finalists[1]}")
print("\nPredicted CHAMPION:")
print(f"   {FLAG.get(champ)}  {champ}  ({pct(win_counts,champ):.1f}% win probability)")

# ---------------- COMPARISON vs USER PREDICTION ----------------
USER_SEMIS = ["Argentina", "Spain", "France", "England"]
USER_FINAL = ["Spain", "Argentina"]
USER_CHAMP = "Spain"

print("\n\n=== YOUR PREDICTION vs MODEL ===")
print("Your semi-finalists:", ", ".join(USER_SEMIS))
print("Model semi-finalists:", ", ".join(top4))
hit = [t for t in USER_SEMIS if t in top4]
print(f"   -> agree on {len(hit)}/4: {', '.join(hit)}")
print("   Bracket note: France & Spain share the TOP half (meet in SF1 if both win);")
print("                 England & Argentina share the BOTTOM half (meet in SF2 if both win).")
for t in USER_SEMIS:
    print(f"      {t:<11} reach-semi {pct(semi_counts,t):4.1f}% | reach-final {pct(final_counts,t):4.1f}% | title {pct(win_counts,t):4.1f}%")
print(f"\nYour final:  {USER_FINAL[0]} vs {USER_FINAL[1]}   (model final: {finalists[0]} vs {finalists[1]})")
print(f"Your champion:  {USER_CHAMP} ({pct(win_counts,USER_CHAMP):.1f}% in model)  |  Model champion: {champ} ({pct(win_counts,champ):.1f}%)")
print(f"   -> Your champion pick {'ALIGNS with' if champ == USER_CHAMP else 'DIFFERS from'} the model.")
