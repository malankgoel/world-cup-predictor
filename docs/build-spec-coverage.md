# Build Spec Coverage

How the original build spec maps onto the current implementation.

| Requirement | Status |
|---|---|
| Strong historical prior and time decay | Implemented |
| Minutes-weighted player roll-up | Implemented with annual historical snapshots |
| Injury/suspension availability | Implemented |
| Position-specific attack/defense and separate goalkeeper | Implemented |
| Squad depth and same-club chemistry | Implemented |
| Attack/defense means plus uncertainty | Implemented as approximate Bayesian state |
| Symmetric goal model with learned overdispersion | Implemented |
| Dixon-Coles correction | Implemented and fitted |
| W/D/L, score grid, totals | Implemented |
| Between-match Bayesian updating | Implemented |
| Monte Carlo bracket simulation | Implemented |
| RPS, log loss, Brier, calibration | Implemented |
| Elo baseline | Implemented |
| Historical FIFA/player snapshots | Implemented with reproducible build scripts |
| Full rolling model refit after every validation match | Not used; temporal holdout is cheaper and reproducible |
| Bookmaker closing-odds baseline | Not included because no stable free licensed feed is bundled |
| True in-play prediction | Out of scope; requires minute, score, card, substitution, and live-event feeds |
| Full MCMC Bayesian model | Not used; the lightweight state update is easier to audit and operate |
