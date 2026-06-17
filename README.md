# 🏆 2026 FIFA World Cup Forecast

A Monte Carlo simulation that predicts the 2026 FIFA World Cup by playing the entire
tournament **50,000 times** using an Elo-powered Poisson goals model on the real group draw.

> **TL;DR** 🇪🇸 **Spain** comes out as the most likely champion (~24.7%), edging
> 🇦🇷 **Argentina** in a projected final. But football is high-variance — even the
> favourite loses ~75% of the time, so treat this as a probability ranking, not a prophecy.

---

## 🎯 Results

### Predicted Champion
🇪🇸 **Spain** — 24.7% win probability

### Predicted Final
🇪🇸 Spain  vs  🇦🇷 Argentina

### Predicted Top 4 (Semi-finalists)
| | Country | Reaches semis |
|---|---|---|
| 🇦🇷 | Argentina | 73% |
| 🇪🇸 | Spain | 68% |
| 🇫🇷 | France | 57% |
| 🇵🇹 | Portugal | 46% |

### Win probability — Top 10
| Country | Elo | Win title | Reach final |
|---|---|---|---|
| 🇪🇸 Spain | 2157 | **24.7%** | 51% |
| 🇦🇷 Argentina | 2115 | **23.3%** | 60% |
| 🇫🇷 France | 2063 | **13.6%** | 41% |
| 🇵🇹 Portugal | 1989 | 5.6% | 25% |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | 2024 | 5.4% | 19% |
| 🇨🇴 Colombia | 1982 | 5.4% | 24% |
| 🇧🇷 Brazil | 1991 | 5.3% | 23% |
| 🇳🇱 Netherlands | 1959 | 3.6% | 16% |
| 🇩🇪 Germany | 1910 | 2.4% | 17% |
| 🇺🇾 Uruguay | 1890 | 1.8% | 14% |

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

4. **Monte Carlo.** The entire tournament is simulated 50,000 times with fresh randomness.
   `Spain 24.7%` means Spain won ~12,350 of the 50,000 simulated tournaments.

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

## ⚠️ Caveats

- **High variance.** A 24.7% favourite *loses* the tournament ~75% of the time. This is a
  ranked probability list, not a guarantee.
- **Pre-tournament strength.** The forecast is based on current team strength and does not
  ingest matchdays already played once the tournament is underway.
- **Approximate bracket.** Uses a representative group-position knockout bracket, not FIFA's
  exact official slotting — fine for probabilities, but a specific team's path may differ.
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
