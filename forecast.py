"""
2026 FIFA World Cup forecast — LIVE / IN-TOURNAMENT edition.
- Historical analysis from wc_all_editions.csv
- Strength model: current World Football Elo (June 2026) where known, else mapped from FIFA rank
- LIVE: actual matchday results (wc_2026_results.csv) are ingested two ways:
    (1) played group matches are LOCKED to their real scoreline (only unplayed games simulate)
    (2) every played result nudges that team's Elo via the standard World Football Elo
        update (goal-difference-weighted K, home advantage for hosts) so current form
        propagates into the knockout-stage simulations, not just the group table.
- Monte Carlo: real group draw -> top2 + 8 best thirds -> seeded knockout, Poisson goals model
"""
import numpy as np
import pandas as pd
import os
from collections import defaultdict

rng = np.random.default_rng(42)
N = 50_000

teams = pd.read_csv("wc_2026_teams.csv")

# --- Current World Football Elo (June 2026), from eloratings / worldcupelo, supplemented per user approval ---
ELO_KNOWN = {
    "Spain": 2157, "Argentina": 2115, "France": 2063, "England": 2024,
    "Brazil": 1991, "Portugal": 1989, "Colombia": 1982, "Netherlands": 1959,
    "Croatia": 1933, "Germany": 1910, "Belgium": 1849, "Morocco": 1860,
    "Uruguay": 1890, "Switzerland": 1855, "Japan": 1838, "USA": 1790,
    "Senegal": 1815, "Mexico": 1790, "Ecuador": 1800, "Austria": 1790,
}

# Map FIFA rank -> Elo for teams without a known value, calibrated to the anchors above.
# Elo ~= 2262 - 151*ln(rank)
def rank_to_elo(rank):
    return 2262 - 151 * np.log(rank)

def elo_for(row):
    if row["team"] in ELO_KNOWN:
        return ELO_KNOWN[row["team"]]
    return rank_to_elo(row["fifa_rank"])

teams["elo"] = teams.apply(elo_for, axis=1)

HOSTS = {"USA", "Mexico", "Canada"}
HOST_BONUS = 35  # modest home advantage, applied in matches

team_elo = dict(zip(teams["team"], teams["elo"]))
elo_pre = dict(team_elo)  # snapshot before live updates, for reporting

# --- LIVE RESULTS INGESTION -------------------------------------------------
# wc_2026_results.csv holds every match already played (group,team1,score1,score2,team2,date).
# Missing file -> behaves exactly like the pre-tournament forecast.
RESULTS_FILE = "wc_2026_results.csv"
if os.path.exists(RESULTS_FILE):
    results = pd.read_csv(RESULTS_FILE)
else:
    results = pd.DataFrame(columns=["group", "team1", "score1", "score2", "team2", "date"])

# Index played results by unordered group pair so the group sim can look them up and lock them.
played = {}   # frozenset({a,b}) -> (team1, score1, score2, team2)
for _, r in results.iterrows():
    played[frozenset((r["team1"], r["team2"]))] = (r["team1"], int(r["score1"]),
                                                   int(r["score2"]), r["team2"])

# (1) Dynamic Elo update from played results — standard World Football Elo method.
#     We_A = 1 / (1 + 10^(-dr/400)); dr = Elo gap (+100 home advantage for a host at home).
#     K = 60 (World Cup weight) scaled up by margin of victory; zero-sum between the two teams.
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
    wa = 1.0 if ga > gb else (0.0 if ga < gb else 0.5)
    k = _margin_k(ELO_K0, ga - gb)
    delta = k * (wa - we_a)
    team_elo[a] += delta
    team_elo[b] -= delta

teams["elo"] = teams["team"].map(team_elo)
team_group = dict(zip(teams["team"], teams["group"]))
team_conf = dict(zip(teams["team"], teams["confederation"]))
groups = defaultdict(list)
for _, r in teams.iterrows():
    groups[r["group"]].append(r["team"])

GOAL_BASE = 1.36
ELO_SCALE = 0.0019  # goals sensitivity to Elo diff

def sim_match_goals(a, b, n):
    """Vectorized: return (goals_a, goals_b) arrays for n sims of a vs b."""
    ra = team_elo[a] + (HOST_BONUS if a in HOSTS else 0)
    rb = team_elo[b] + (HOST_BONUS if b in HOSTS else 0)
    la = GOAL_BASE * np.exp(ELO_SCALE * (ra - rb))
    lb = GOAL_BASE * np.exp(ELO_SCALE * (rb - ra))
    return rng.poisson(la, n), rng.poisson(lb, n)

def knockout_winner(a, b, n):
    """Return array of winners (a or b) for n sims; draws decided by Elo-weighted shootout."""
    ga, gb = sim_match_goals(a, b, n)
    ra, rb = team_elo[a], team_elo[b]
    p_a = 0.5 + (ra - rb) / 4000.0
    p_a = np.clip(p_a, 0.15, 0.85)
    coin = rng.random(n) < p_a
    win_a = (ga > gb) | ((ga == gb) & coin)
    return np.where(win_a, a, b)

# Counters
win_counts = defaultdict(int)
final_counts = defaultdict(int)
semi_counts = defaultdict(int)   # reached semis (top 4)
qf_counts = defaultdict(int)

# Run sims in batches for memory
BATCH = 2000
done = 0
while done < N:
    n = min(BATCH, N - done)
    done += n

    # --- GROUP STAGE ---
    # standings[group] -> dict team -> [pts(n), gd(n), gf(n)]
    pts = {t: np.zeros(n, dtype=int) for t in team_elo}
    gd = {t: np.zeros(n, dtype=int) for t in team_elo}
    gf = {t: np.zeros(n, dtype=int) for t in team_elo}

    for g, gteams in groups.items():
        for i in range(len(gteams)):
            for j in range(i + 1, len(gteams)):
                a, b = gteams[i], gteams[j]
                key = frozenset((a, b))
                if key in played:
                    # LIVE: lock the real scoreline across every simulation.
                    t1, s1, s2, t2 = played[key]
                    sa, sb = (s1, s2) if a == t1 else (s2, s1)
                    ga = np.full(n, sa, dtype=int)
                    gb = np.full(n, sb, dtype=int)
                else:
                    ga, gb = sim_match_goals(a, b, n)
                a_win = ga > gb; b_win = gb > ga; draw = ga == gb
                pts[a] += np.where(a_win, 3, np.where(draw, 1, 0))
                pts[b] += np.where(b_win, 3, np.where(draw, 1, 0))
                gd[a] += ga - gb; gd[b] += gb - ga
                gf[a] += ga; gf[b] += gb

    # rank within each group -> winners, runners, thirds
    # score key: pts*1e6 + gd*1e3 + gf + tiny random tiebreak
    rand_tb = {t: rng.random(n) for t in team_elo}
    winners = {}   # group -> team array
    runners = {}
    thirds_pool = []  # list of (team, key) for third-placed
    for g, gteams in groups.items():
        keys = {t: pts[t]*1_000_000 + gd[t]*1000 + gf[t] + rand_tb[t] for t in gteams}
        arr = np.stack([keys[t] for t in gteams], axis=0)  # (4, n)
        order = np.argsort(-arr, axis=0)  # best first
        gt = np.array(gteams)
        w = gt[order[0]]; rsec = gt[order[1]]; third = gt[order[2]]
        winners[g] = w; runners[g] = rsec
        third_key = np.take_along_axis(arr, order[2:3], axis=0)[0]
        thirds_pool.append((g, third, third_key))

    glist = sorted(groups.keys())  # A..L

    third_teams = np.stack([tp[1] for tp in thirds_pool], axis=0)  # (12, n)
    third_keys = np.stack([tp[2] for tp in thirds_pool], axis=0)   # (12, n)
    third_order = np.argsort(-third_keys, axis=0)  # (12, n)
    top8_third_idx = third_order[:8]  # (8, n)
    third_q = np.take_along_axis(third_teams, top8_third_idx, axis=0)  # (8, n) best->worst third

    elo_lookup = team_elo
    velo = np.vectorize(lambda t: elo_lookup[t])

    W = {g: winners[g] for g in glist}   # group -> team array
    R = {g: runners[g] for g in glist}
    T = [third_q[i] for i in range(8)]   # 8 best thirds (ranked)

    # --- FIXED, strength-INDEPENDENT bracket by group position (realistic draw luck) ---
    # Winning your group earns a softer R32 opponent (a third or a low runner-up),
    # but the slot is fixed before strengths are known, so strong teams can still collide early.
    # 16 R32 matches laid out so consecutive pairs meet; tree folds by adjacent pairing.
    GA,GB,GC,GD,GE,GF,GG,GH,GI,GJ,GK,GL = glist
    bracket = np.stack([
        # ---- TOP HALF ----
        W[GA], T[0],          # M1
        R[GE], R[GF],         # M2
        W[GC], T[2],          # M3
        W[GI], R[GB],         # M4
        W[GE], T[4],          # M5
        R[GG], R[GH],         # M6
        W[GG], T[6],          # M7
        W[GK], R[GD],         # M8
        # ---- BOTTOM HALF ----
        W[GB], T[1],          # M9
        R[GI], R[GJ],         # M10
        W[GD], T[3],          # M11
        W[GJ], R[GA],         # M12
        W[GF], T[5],          # M13
        R[GK], R[GL],         # M14
        W[GH], T[7],          # M15
        W[GL], R[GC],         # M16
    ], axis=0)  # (32, n)

    host_arr = np.array(list(HOSTS))
    def run_round(bk):
        m = bk.shape[0]
        winners_r = []
        for k in range(m // 2):
            a_row = bk[2*k]
            b_row = bk[2*k + 1]
            va = velo(a_row); vb = velo(b_row)
            ea = va + np.where(np.isin(a_row, host_arr), HOST_BONUS, 0)
            eb = vb + np.where(np.isin(b_row, host_arr), HOST_BONUS, 0)
            la = GOAL_BASE * np.exp(ELO_SCALE * (ea - eb))
            lb = GOAL_BASE * np.exp(ELO_SCALE * (eb - ea))
            ga = rng.poisson(la); gb = rng.poisson(lb)
            p_a = np.clip(0.5 + (ea - eb) / 4000.0, 0.15, 0.85)
            coin = rng.random(n) < p_a
            a_adv = (ga > gb) | ((ga == gb) & coin)
            winners_r.append(np.where(a_adv, a_row, b_row))
        return np.stack(winners_r, axis=0)

    r16 = run_round(bracket)     # 16
    # record QF entrants? record after each round
    for row in r16:
        pass
    r8 = run_round(r16)          # 8 (quarterfinalists are r8 entrants = r16 winners)
    # r16 (16 teams) are Round-of-16 winners = quarterfinalists
    for row in r16:
        vals, counts = np.unique(row, return_counts=True)
        for v, c in zip(vals, counts):
            qf_counts[v] += c
    r4 = run_round(r8)           # 4 semifinalists
    for row in r8:
        vals, counts = np.unique(row, return_counts=True)
        for v, c in zip(vals, counts):
            semi_counts[v] += c
    r2 = run_round(r4)           # 2 finalists
    for row in r4:
        vals, counts = np.unique(row, return_counts=True)
        for v, c in zip(vals, counts):
            final_counts[v] += c
    champ = run_round(r2)[0]     # champion
    vals, counts = np.unique(champ, return_counts=True)
    for v, c in zip(vals, counts):
        win_counts[v] += c

def pct(d, t):
    return 100.0 * d.get(t, 0) / N

rows = []
for t in team_elo:
    rows.append((t, team_conf[t], team_group[t], int(round(team_elo[t])),
                 pct(win_counts, t), pct(final_counts, t), pct(semi_counts, t), pct(qf_counts, t)))
res = pd.DataFrame(rows, columns=["team","conf","grp","elo","win%","final%","semi%","qf%"])
res = res.sort_values("win%", ascending=False).reset_index(drop=True)

pd.set_option("display.width", 200)

# ---------------- LIVE INGESTION REPORT ----------------
print("\n=== LIVE RESULTS INGESTED ===")
print(f"Matches played and locked in: {len(results)} (group stage)")
if len(results):
    mov = sorted(((t, team_elo[t] - elo_pre[t]) for t in elo_pre),
                 key=lambda x: -abs(x[1]))
    print("Biggest Elo swings from live form (post − pre):")
    for t, d in mov[:14]:
        if abs(d) < 0.5:
            continue
        print(f"   {t:<14} {elo_pre[t]:7.0f} -> {team_elo[t]:7.0f}  ({d:+5.1f})")

_mode = "live-updated" if len(results) else "pre-tournament"
print("\n=== 2026 WORLD CUP — MONTE CARLO FORECAST ({} sims, {}) ===".format(N, _mode))
print(res.head(16).to_string(index=False, float_format=lambda x: f"{x:5.1f}"))

print("\nTotal win% (sanity, should be ~100):", round(res["win%"].sum(),1))

# ---------------- HISTORICAL CONTEXT ----------------
ed = pd.read_csv("wc_all_editions.csv")
print("\n\n=== HISTORICAL PATTERNS (1930-2022, 22 editions) ===")
champs = ed["champion"].replace({"West Germany":"Germany"}).value_counts()
print("\nTitles by nation:")
print(champs.to_string())

CONF = {  # nation -> confederation, for champions
 "Brazil":"CONMEBOL","Germany":"UEFA","Italy":"UEFA","Argentina":"CONMEBOL",
 "France":"UEFA","Uruguay":"CONMEBOL","England":"UEFA","Spain":"UEFA"}
conf_titles = champs.rename(index=CONF).groupby(level=0).sum()
print("\nTitles by confederation:")
print(conf_titles.to_string())

host_won = (ed["host_won"]=="Yes").sum()
print(f"\nHost nation won: {host_won}/{len(ed)} = {100*host_won/len(ed):.0f}%")

# European champions on American soil
americas_hosts = ["Uruguay","Brazil","Chile","Mexico","Argentina","United States"]
am = ed[ed["host"].isin(americas_hosts)]
euro_champ_in_am = am[am["champion"].replace({"West Germany":"Germany"}).map(lambda c: CONF.get(c)=="UEFA")]
print(f"\nWorld Cups hosted in the Americas: {len(am)}")
print(f"  ...won by a European team: {len(euro_champ_in_am)}  ->  "
      f"{'NONE — every Americas WC won by a CONMEBOL side' if len(euro_champ_in_am)==0 else euro_champ_in_am['champion'].tolist()}")

# ---------------- FLAGGED VERDICT ----------------
FLAG = {
 "Spain":"🇪🇸","Argentina":"🇦🇷","France":"🇫🇷","England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","Portugal":"🇵🇹",
 "Brazil":"🇧🇷","Colombia":"🇨🇴","Netherlands":"🇳🇱","Germany":"🇩🇪","Croatia":"🇭🇷",
 "Uruguay":"🇺🇾","Belgium":"🇧🇪","Morocco":"🇲🇦","USA":"🇺🇸","Mexico":"🇲🇽"}

top4 = res.sort_values("semi%", ascending=False).head(4)["team"].tolist()
finalists = res.sort_values("final%", ascending=False).head(2)["team"].tolist()
champ = res.sort_values("win%", ascending=False).head(1)["team"].iloc[0]

print("\n\n=== VERDICT (model, live-updated) ===")
print("Predicted SEMI-FINALISTS (top 4):")
for t in top4:
    print(f"   {FLAG.get(t,'  ')}  {t:<11} ({pct(semi_counts,t):4.1f}% to reach semis)")
print("\nPredicted FINAL:")
print(f"   {FLAG.get(finalists[0])}  {finalists[0]}  vs  {FLAG.get(finalists[1])}  {finalists[1]}")
print("\nPredicted CHAMPION:")
print(f"   {FLAG.get(champ)}  {champ}  ({pct(win_counts,champ):.1f}% win probability)")

# ---------------- COMPARISON vs USER PREDICTION ----------------
USER_SEMIS = ["Spain", "Argentina", "France", "England"]
USER_FINAL = ["Spain", "Argentina"]
USER_CHAMP = "Spain"

print("\n\n=== YOUR PREDICTION vs MODEL ===")
print("Your semi-finalists:", ", ".join(USER_SEMIS))
print("Model semi-finalists:", ", ".join(top4))
hit = set(USER_SEMIS) & set(top4)
print(f"   -> agree on {len(hit)}/4: {', '.join(sorted(hit))}")
for t in USER_SEMIS:
    print(f"      {t:<11} model semi-prob {pct(semi_counts,t):4.1f}% | win-title {pct(win_counts,t):4.1f}%")
print(f"\nYour final:  {USER_FINAL[0]} vs {USER_FINAL[1]}")
print(f"Model final: {finalists[0]} vs {finalists[1]}")
print(f"\nYour champion:  {USER_CHAMP} ({pct(win_counts,USER_CHAMP):.1f}% in model)")
print(f"Model champion: {champ} ({pct(win_counts,champ):.1f}%)")
verdict = "ALIGNS with the model" if champ == USER_CHAMP else "DIFFERS from the model"
print(f"   -> Your champion pick {verdict}.")
