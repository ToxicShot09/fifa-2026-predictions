# 🏆 2026 FIFA World Cup Forecast

A Monte Carlo simulation that predicts the 2026 FIFA World Cup by playing the entire
tournament **50,000 times** using an Elo-powered Poisson goals model on the real group draw.

> **TL;DR** With the group stage complete and the Round of 32 underway, 🇦🇷 **Argentina**
> is the most likely champion (~24%), narrowly ahead of 🇪🇸 **Spain** (~22%) and 🇫🇷 **France**
> (~20%) — the two of whom share a loaded top half. Football is high-variance — even the
> favourite wins only ~1 time in 4 — so treat this as a probability ranking, not a prophecy.

---

## 🎯 Current predictions (live — post group stage, R32 underway)

### Predicted Champion
🇦🇷 **Argentina** — 24.4% win probability

### Predicted Final
🇦🇷 Argentina  vs  🇫🇷 France

### Predicted Top 4 (Semi-finalists)
| | Country | Reaches semis |
|---|---|---|
| 🇫🇷 | France | 61% |
| 🇦🇷 | Argentina | 61% |
| 🇪🇸 | Spain | 53% |
| 🇧🇷 | Brazil | 45% |

### Win probability — contenders
| Country | Elo | Win title | Reach final |
|---|---|---|---|
| 🇦🇷 Argentina | 2141 | **24.4%** | 41% |
| 🇪🇸 Spain | 2147 | **21.6%** | 35% |
| 🇫🇷 France | 2101 | **19.8%** | 35% |
| 🇧🇷 Brazil | 2023 | 9.5% | 21% |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | 2041 | 8.0% | 17% |
| 🇨🇴 Colombia | 2000 | 5.0% | 12% |

> Already eliminated: 🇩🇪 Germany and 🇳🇱 Netherlands (both lost the R32 on penalties).

---

## 🧠 How it works

The script does **not** predict a winner directly — it simulates the whole tournament and
counts who wins most often.

1. **Strength rating (Elo).** Every team gets a single number — its World Football Elo
   rating. Live June-2026 Elo is used for ~20 top teams; the rest are mapped from their
   FIFA rank via `Elo ≈ 2262 − 151·ln(rank)`. Hosts (USA/Mexico/Canada) get a +35 home bonus.

2. **Match engine (Poisson goals).** For a match A vs B, each team's expected goals depends
   on the Elo gap, and the actual scoreline is drawn from independent Poisson distributions:

   ```
   λ_A = 1.36 · exp(0.0019 · (Elo_A − Elo_B))
   λ_B = 1.36 · exp(0.0019 · (Elo_B − Elo_A))
   ```

   This is the standard Maher / Dixon-Coles family of football models. Knockout draws go to
   an Elo-weighted penalty shootout.

3. **Tournament structure.** The real 12-group draw → round-robin → top 2 + 8 best
   third-placed teams → fixed group-position knockout bracket (R32 → R16 → QF → SF → Final).

4. **Monte Carlo.** The remaining bracket is simulated 50,000 times with fresh randomness.
   `Argentina 24.4%` means Argentina won ~12,200 of the 50,000 simulated tournaments.

### Why this method?
| Approach | Verdict |
|---|---|
| Pure historical frequency ("Brazil won most → Brazil") | ❌ Predicts by reputation, ignores current squads |
| Regression / ML classifier | ❌ Only 22 editions — far too little data, overfits |
| **Elo + Poisson + Monte Carlo** | ✅ Captures current strength *and* propagates football's randomness |

---

## 📂 Dataset

Source: *FIFA World Cup Complete Dataset (1930–2026)*.

| File | Contents |
|---|---|
| `wc_all_editions.csv` | 22 editions — host, champion, runner-up, top scorer, goals, attendance, format |
| `wc_all_matches.csv` | Curated marquee matches (finals & key knockouts) with scores |
| `wc_top_scorers.csv` | Golden Boot winners per edition |
| `wc_2026_teams.csv` | 48 qualified teams — group, confederation, FIFA rank, coach |
| `wc_2026_fixtures.csv` | 2026 schedule — group, stage, venue, kickoff, rankings |

The historical files are used for **context and sanity-checks** (host-advantage rate,
confederation dominance, the "European team in the Americas" pattern), not as direct
predictors.

---

## 🚀 Usage

```bash
# Requires Python 3 with numpy and pandas
pip install numpy pandas

python3 forecast.py
```

Outputs the Monte Carlo win/final/semi probabilities, historical patterns, and a flagged verdict.

---

## 🔴 Live update (in-tournament — post group stage, Round of 32 underway)

`forecast.py` ingests every match already played from `wc_2026_results.csv`
(`stage,team1,score1,score2,team2,winner,date`) — **76 matches** as of June 30 (all 72
group games + the first 4 Round-of-32 ties).

Live results are used **two ways**:

1. **Locked outcomes.** A played knockout tie keeps its real winner in all 50,000
   simulations (penalty results respected); only matches not yet played are simulated.
2. **Dynamic Elo.** Each played result nudges that team's Elo via the standard World
   Football Elo update (`We = 1/(1+10^(−Δ/400))`, K=60 scaled by goal margin, +100 home
   advantage for hosts, shoot-outs counted as draws), so current form drives the sims.

The simulation now runs over the **real official 2026 bracket** (R32 → R16 → QF → SF →
Final), rebuilt from the actual qualifiers, rather than a representative draw.

**State that reshaped the forecast:** Germany and Netherlands are already out (both lost
R32 on penalties, to Paraguay and Morocco). France stormed its group (+38 Elo) and Spain
landed in the same half — so the model now reads **Argentina ~24% champion, Spain ~22%,
France ~20%**, with Brazil edging England for the 4th semi-final slot because the two share
a quarter-final.

## ⚠️ Caveats

- **High variance.** A ~24% favourite *loses* the tournament ~75% of the time. This is a
  ranked probability list, not a guarantee.
- **Model assumptions.** Poisson goals are treated as independent; real matches aren't
  (injuries, momentum, red cards, fatigue). Poisson also slightly underestimates draws.

---

## 📊 Possible extensions

- Feed in live results to update probabilities mid-tournament
- Back-test the model on past World Cups (e.g. 2022) for calibration
- Swap in FIFA's exact official bracket
- Plot probability bars as a chart

---

*Built with an Elo + Poisson + Monte Carlo pipeline. Strength data: [World Football Elo Ratings](https://www.eloratings.net/2026_World_Cup).*
