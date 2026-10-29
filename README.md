# Peer-to-peer electricity trading and low-voltage compliance on Dutch LV feeders

The simulation model of the MSc thesis *When Does Regulation Reach the Wire? Peer-to-Peer
Electricity Trading and Low-Voltage Compliance on Dutch LV Feeders* (Thijmen ter Linden,
MSc Energy Science, Utrecht University, 2026), published for transparency.

One file, `code/p2p_voltage.py`, holds the whole calculation: every input as a constant with its
source, the model in its six steps, the two scenarios of the thesis and the decomposition of
Scenario 2. With the three datasets of section 3, which are included, it reproduces the output
files from which every number of the thesis was taken; the seeds are fixed. The statistics and
the figures made from those files are not included. The repository ships no run output and no
thesis text.

## 1. What the model does

One Dutch low-voltage feeder (urban, suburban or rural, with 30, 25 or 15 households) is solved
with a time-series power flow for one week at a time in 15-minute steps. Every household carries
a standard load profile; a share of them a solar system, a home charger or a heat pump. A local
price follows the irradiance: low in the sunny hours, high in the hours without sun. The
households that take part in the trade move the flexible part of their demand towards the hours
in which the price is below its daily mean, with the energy of every day unchanged.
Each run is solved twice on the same population, once without the trade (pre-2027) and once
with it (post-2027), and the voltages are checked against the Dutch voltage band.

Scenario 1 samples the participation share, the price elasticity and the platform fee in 1000
Latin-hypercube runs for each of 36 combinations (3 archetypes x 3 demand levels x 4 weeks) at
present solar and device levels. Scenario 2 raises the solar size (4 to 14 kWp) and the adoption
of chargers and heat pumps (0.15 to 0.75) step by step on every archetype and week at three
elasticities, 2772 grid points, at the high demand level with a participation share of 0.80 and
a fee of 0.01 EUR/kWh. In its post-2027 regime the flexible part of the charging and heat-pump
load is spread evenly over each day before the price response acts (the pool, which stands for
smart charging). The decomposition solves every grid point in four demand states (pre-2027, the
pool alone, the price response alone, both) to separate the two mechanisms.

## 2. How the file is organised

| Section | Content |
|---|---|
| 1 Constants | every input of the model with its source: the data file paths, the three feeder archetypes, the household mix, the demand levels, the solar system, the prices, the sampled ranges, the voltage band, the seeds, the charging and heat-pump parameters, the Scenario 2 grid |
| 2 The model in six steps | step 1 `build_feeder`; step 2 `build_population`, `assign_nodes`, `load_irradiance`, `load_demand_shapes`, `pv_series`, `build_addon`, `base_demand`; step 3 `price_signal`; step 4 `reshape_flexible`, `build_demand`, `net_power`, `fixed_count_participation`; step 5 `solve_voltages`; step 6 `compute_metrics`, `ep_stats`, `ep_crossings`. One run is `prepare_cell` (steps 1, 2 and the pre-2027 solve) followed by `post_run` (steps 3 to 6); `sample_lhs` draws the inputs of Scenario 1 |
| 3 The flexible load by source | the same arithmetic as steps 2 and 4, kept per source (residential, charging, heat pump) for output file 05 |
| 4 Scenario 1 | `run_part1`: 36 combinations x 1000 runs; writes files 01 to 05 |
| 5 Scenario 2 | `run_part2`: the 2772 grid points; writes file 06 |
| 6 The decomposition | `run_decomposition`: the four demand states per grid point; writes file 07 |
| 7 Command line | `--smoke`, `--full`, `--decomposition` |

The values in section 1 are those of the frozen input workbook that produced the thesis results
(run of 22 September 2026), written exactly as the model read them. The flexible share `PHI` is
0.25; the sensitivity run of the thesis sets it to 0.50 and runs the file again.

## 3. Datasets

The three time series the model reads are included under `data/`, exactly as the thesis
runs read them. The SHA-256 checksum of each file is given so that a reader can confirm the
input is unchanged before running. Each file is the property of its source and is included
unchanged, with the attribution the source asks for.

| Dataset | Used in | File | Source | SHA-256 |
|---|---|---|---|---|
| KNMI hourly observations, station 260 De Bilt, column `Q` (global radiation), year 2025 | step 2 (solar generation), step 3 (price) | `data/uurgeg_260_2021-2030.txt` (7.4 MB) | KNMI, *Uurgegevens van het weer in Nederland*, https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens, downloaded 10 July 2026; reuse permitted with the attribution "bron: KNMI" (https://www.knmi.nl/copyright) | `9b7af9ca2868228a7daad86dc5e42b758dea99cb060dfa1a0ce66b7f0297e2fd` |
| Standard electricity profiles 2025, version 1.00, categories E1A, E1B, E1C (E2A, E2B carry a zero share) | step 2 (base demand) | `data/Standaardprofielen elektriciteit 2025 versie 1.00.csv` (13.4 MB) | Het Normo, *Profielen elektriciteit 2025*, version 1.00 (5 June 2024), Energiedatawijzer, https://energiedatawijzer.nl/documenten/profielen-elektriciteit-2025/, downloaded 10 July 2026 | `4aa69a15b0a192400c35c28091aa2a90df3c63294fea7e99e6c6c91cfed1334f` |
| ElaadNL home-charging profiles, ten profiles, year 2025, 15 minutes, kW | step 2 (charging load) | `data/profile.csv` (2.0 MB) | ElaadNL, *Charging Profiles Generator*, https://charging.elaad.nl/, generated and downloaded 17 August 2026 | `24ed5a9510f4f9f03cb094f13171fec5e22ed3f4473dfb69b71a09ae250d621f` |

The sources of the constants are given next to each value in section 1 of the file: CBS tables
81528NED (annual consumption per household type), 85592NED (consumer tariffs), 85523NED (heat
pumps), 85005NED (solar capacity), CBS *Klimaatverandering en energietransitie 2023* (solar
households by urbanisation), MFFBAS *Jaarverslag profielallocatie elektriciteit 2024* (the
category mix), NAL/RVO (home charge points), Nijhuis, Gibescu and Cobben (2015) and Phase to
Phase (2001) (the feeders), Netbeheer Nederland (2024) (the busbar voltage), Chwieduk and
Chwieduk (2021) (heat-pump energy per winter day), Faruqui and Sergici (2010) (the elasticity
range), Staatsblad 2025, 17 (the feed-in floor).

## 4. Running it

Python 3.12 with the package versions the thesis runs were made with (pyarrow is optional: it
only adds the parquet copy of file 03):

    python -m venv .venv && source .venv/bin/activate
    pip install numpy==2.5.1 pandas==3.0.3 scipy==1.18.0 power-grid-model==1.13.107 pyarrow==24.0.0

Then, from the root of the repository:

    python code/p2p_voltage.py --smoke          # one combination with 5 runs, 4 grid points and their decomposition; about 10 s; writes smoke/
    python code/p2p_voltage.py --full           # Scenario 1 and Scenario 2; about 40 minutes on a laptop; writes output/ (about 230 MB)
    python code/p2p_voltage.py --decomposition  # the four demand states of every Scenario 2 point; about 5 minutes; writes output/

Output files, all CSV, one directory:

| File | One row per | Content |
|---|---|---|
| `01_run_summary` | Scenario 1 run (36000) | pi, epsilon, tau, participants; the endpoint and whole-feeder maximum, minimum, mean and 95th percentile pre and post in volts and their change in millivolts; VCR; the absolute criterion; the crossings of 253 V and 207 V at the endpoint |
| `02_cell_summary` | combination (36) | the pre-2027 extremes, the median and 5th/95th percentiles of the four changes, the worst post-2027 values, and the two extreme runs: the run with the largest fall of the maximum and the run with the largest rise of the minimum |
| `03_voltage_weeks` (also `.parquet`) | node and timestep (1.76 million) | the voltage pre and post for the mean over the runs and for the two extreme runs of every combination |
| `04_quarterhour_voltage_profiles` | quarter-hour of the day and node | the change of the mean post-2027 voltage over the runs, in millivolts, as its mean over the days of the week with their percentiles and extremes |
| `05_power_price_profiles` | quarter-hour of the day, three series per combination | the feeder's solar generation, the inflexible and flexible load by source (residential, charging, heat pump) pre and post, and the price, for the mean over the runs (the price then being the mean price) and for the two extreme runs |
| `06_scenario_grid` | Scenario 2 point (2772) | the endpoint extremes pre and post, their change, the crossings, VCR pre and post, the compliance state (0 no breach, 1 breach in both regimes, 2 cleared post-2027, 3 new post-2027) |
| `07_sweep_decomposition` | Scenario 2 point (2772) | the endpoint extremes, crossings, VCR, self-consumption, feeder export and peak import and export in the four states A (pre-2027), B (pool alone), D (price response alone), C (both), the pairwise states and changes, and the interaction C - B - D + A |

The voltage columns are in volts, the change columns in millivolts; the thesis applies the 95 %
requirement, the absolute requirement and the endpoint check to these files.

## 5. Software and versions

| Component | Version |
|---|---|
| Python | 3.12.7 |
| power-grid-model (the power-flow solver, Newton-Raphson) | 1.13.107 |
| numpy / pandas / scipy | 2.5.1 / 3.0.3 / 1.18.0 |
| pyarrow (the parquet copy of file 03 only) | 24.0.0 |
| Operating system of the thesis runs | macOS 15 (Darwin 24.6) |

AI assistance: the author used Claude Code (Anthropic) during the development of the code and
the audits of its numbers, with the Claude Opus 5, Claude Opus 5.5, Claude Sonnet 5 and Claude
Fable 5.1 models. The AI statement handed in with the thesis is the authoritative account.

## 6. Relation to the code that produced the thesis results

The thesis results were produced by the same calculation spread over several files, with an
Excel workbook as the parameter source and with validation checks, a run manifest, statistics
and figure stages around it. This file keeps the calculation and leaves the rest out: the
workbook reader is replaced by the constants of section 1; the checks, the manifest, the
statistics and the figures are gone; the step functions, the two scenario loops and the
decomposition are the original code. The file was checked against the stored run of 22
September 2026: the smoke run, one full combination of 1000 runs (all five Scenario 1 output
files), a Scenario 2 subset of 77 grid points (231 rows) and a decomposition subset of 18 grid
points (54 rows) come out identical to every decimal written.

## 7. Citation

ter Linden, T. (2026). *When Does Regulation Reach the Wire? Peer-to-Peer Electricity Trading
and Low-Voltage Compliance on Dutch LV Feeders* [MSc thesis, Utrecht University].
