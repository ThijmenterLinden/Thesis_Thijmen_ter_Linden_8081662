#!/usr/bin/env python3
"""p2p_voltage.py: the simulation model of the MSc thesis "When Does Regulation Reach the Wire?
Peer-to-Peer Electricity Trading and Low-Voltage Compliance on Dutch LV Feeders"
(Thijmen ter Linden, Utrecht University, 2026), in one file.

The model builds one Dutch low-voltage feeder, gives every household a load and a share of them
a solar system, a charger or a heat pump, builds a local price from the irradiance, lets the
participating households move the flexible part of their demand towards the cheap hours, solves
the power flow for one week in 15-minute steps and checks the voltages against the Dutch band.
Every run is solved twice on the same population: pre-2027 (no trade) and post-2027 (trade).

    python code/p2p_voltage.py --smoke          one combination, 5 runs, 4 grid points and their decomposition; ~10 s; writes smoke/
    python code/p2p_voltage.py --full           Scenario 1 (36 combinations x 1000 runs) and Scenario 2 (2772 grid points); ~40 min; writes output/
    python code/p2p_voltage.py --decomposition  the four-state decomposition of Scenario 2 (Table 4.9 of the thesis); ~5 min; writes output/

Inputs: the constants of section 1 and the three data files under data/ (README.md).
Outputs, CSV: 01_run_summary (one row per Scenario 1 run), 02_cell_summary (one row per
combination), 03_voltage_weeks (the voltage at every node and timestep for the mean and the two
extreme runs; also as parquet), 04_quarterhour_voltage_profiles and 05_power_price_profiles
(quarter-hour means of the voltage change, the loads and the price), 06_scenario_grid (one row
per Scenario 2 point), 07_sweep_decomposition (one row per Scenario 2 point, four demand states).
"""
from __future__ import annotations
import argparse, dataclasses as dc, hashlib, math, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from power_grid_model import CalculationMethod, PowerGridModel, initialize_array
from power_grid_model.enum import LoadGenType
from power_grid_model.errors import PowerGridBatchError
from scipy.stats import qmc

# =============================================================================================
# 1. CONSTANTS. Every input of the model with its source. These are the values of the frozen
#    input workbook that produced the thesis results (run of 22 September 2026), written here
#    exactly as the model read them.
# =============================================================================================
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
KNMI_FILE = DATA / "uurgeg_260_2021-2030.txt"     # KNMI hourly observations, station 260 De Bilt, UTC; column Q = global radiation in J/cm2 per hour
NEDU_FILE = DATA / "Standaardprofielen elektriciteit 2025 versie 1.00.csv"   # MFFBAS/NEDU standard profiles 2025 v1.00: fraction of annual consumption per 15 min; column 0 is CET (UTC+1)
ELAAD_FILE = DATA / "profile.csv"                 # ElaadNL home-charging profiles 2025: ten profiles, 15 min, kW, peak 11 kW

ARCHETYPES = ("urban", "suburban", "rural")
DEMAND_LEVELS = ("low", "medium", "high")
WEEK_KEYS = ("summer", "winter", "spring", "autumn")
LOCAL_TZ = "Europe/Amsterdam"
IRRADIANCE_YEAR = 2025                                                  # the year of the KNMI, NEDU and ElaadNL series
ISO_WEEK = {"summer": 25, "winter": 49, "spring": 13, "autumn": 37}     # 2025: the sunniest week, the darkest week, two near-median weeks


@dataclass(frozen=True)
class Feeder:
    cable_length_m: float; cable_xsec_mm2: float; r_ohm_per_km: float; x_ohm_per_km: float; c_farad_per_km: float
    n_households: int; pv_penetration: float; ev_share: float; hp_share: float


@dataclass(frozen=True)
class Demand:
    low_kwh: float; medium_kwh: float; high_kwh: float; spread_min: float; spread_max: float; phi: float


@dataclass(frozen=True)
class PV:
    kwp_mean: float; kwp_std: float; eta_inv: float; g_stc: float


@dataclass(frozen=True)
class Prices:
    p_retail: float; p_floor: float


@dataclass(frozen=True)
class Sampling:
    participation_range: tuple; epsilon_range: tuple; tau_range: tuple; n_runs_per_cell: int


@dataclass(frozen=True)
class Network:
    timestep_min: int; u_nom: float; band: float; abs_lower_band: float; u_rated_ll: float; source_u_pu: float; source_sk_va: float


@dataclass(frozen=True)
class Seeds:
    population_seed: int; placement_seed: int; participation_seed_base: int; lhs_seed: int


@dataclass(frozen=True)
class EVHP:
    ev_flex_share: float; hp_flex_share: float; hp_energy_kwh: float
    hp_morning_start: int; hp_morning_end: int; hp_evening_start: int; hp_evening_end: int; hp_seasonal: dict


@dataclass(frozen=True)
class Config:
    feeders: dict; household_mix: dict; demand: Demand; pv: PV; prices: Prices; sampling: Sampling
    network: Network; seeds: Seeds; ev_hp: EVHP


EV_SHARE = 0.10278261904761905   # share of households with a home charger: 863374 home charge points end 2025 (NAL/RVO) / 8.4 million households
HP_SHARE = 0.09878285714285714   # share of households with a heat pump: 829776 heat pumps at households end 2024 (CBS 85523NED) / 8.4 million households
FEEDERS = {
    # The three Dutch feeder archetypes. Length, cross-section and household count: Nijhuis, Gibescu & Cobben (2015),
    # Fig. 1, the distributions of 88000 Liander feeders. Resistance: Phase to Phase (2001), Table 1.1, column R_ac,30,
    # for 4x150, 4x95 and 4x50 Al. Reactance 0.08 ohm/km: Phase to Phase, Table 4.3. Capacitance 0. Share of households
    # with solar panels by urbanisation: CBS (2023), Klimaatverandering en energietransitie, chapter 4.
    #                 length m  mm2     R ohm/km  X ohm/km  C F/km  households  solar   EV        HP
    "urban":    Feeder(  200.0, 150.0,  0.214,    0.08,     0.0,    30,         0.20,   EV_SHARE, HP_SHARE),
    "suburban": Feeder(  500.0,  95.0,  0.333,    0.08,     0.0,    25,         0.35,   EV_SHARE, HP_SHARE),
    "rural":    Feeder( 1000.0,  50.0,  0.667,    0.08,     0.0,    15,         0.54,   EV_SHARE, HP_SHARE),
}
# Share of the households per MFFBAS profile category, the same on every archetype: MFFBAS, Jaarverslag profielallocatie
# elektriciteit 2024, connections per category in 2024; E1C is rounded so that the three sum to one. Keep the literals
# as written and the keys in this order: the random draw of step 2 uses these exact values in this order.
HOUSEHOLD_MIX = {a: {"E1A": 0.095, "E1B": 0.674, "E1C": 0.231, "E2A": 0.0, "E2B": 0.0} for a in ARCHETYPES}
PHI = 0.25   # flexible share of the base demand (the results of the thesis); 0.50 for its sensitivity run

CFG = Config(
    feeders=FEEDERS, household_mix=HOUSEHOLD_MIX,
    demand=Demand(low_kwh=1830.0, medium_kwh=2550.0, high_kwh=3010.0,  # annual consumption per household, kWh: CBS 81528NED, apartment / terraced house / owner-occupied
                  spread_min=0.85, spread_max=1.15,                    # uniform spread of the annual consumption around the level (Smit)
                  phi=PHI),
    pv=PV(kwp_mean=3.97,    # solar size per prosumer, kWp: CBS 85005NED, 12088287 kWp on households in 2025 / 3041895 households with solar
          kwp_std=1.5,      # standard deviation of the solar size, own assumption; the draw is clipped at 1.0 kWp (step 2)
          eta_inv=0.96,     # inverter efficiency
          g_stc=1000.0),    # standard test irradiance, W/m2
    prices=Prices(p_retail=0.25434999999999997,   # retail tariff, EUR/kWh: bare supply tariff 0.1435 + energy tax 0.11085, incl. VAT, CBS 85592NED, mean January to July 2026
                  p_floor=0.07175),               # feed-in floor, EUR/kWh: half the bare supply tariff, Stb. 2025, 17 (from 1 January 2027)
    sampling=Sampling(participation_range=(0.05, 0.8),         # participation share pi
                      epsilon_range=(-0.5, -0.05),             # price elasticity epsilon, wider than Faruqui & Sergici (2010)
                      tau_range=(0.01, 0.09129999999999999),   # platform fee tau, EUR/kWh, up to half the spread (p_retail - p_floor) / 2
                      n_runs_per_cell=1000),                   # Latin-hypercube runs per combination
    network=Network(timestep_min=15, u_nom=230.0,
                    band=0.10,             # EN 50160: 95 % of the week within +-10 % of 230 V (207 to 253 V)
                    abs_lower_band=0.15,   # Netcode: no interval below -15 % (195.5 V)
                    u_rated_ll=400.0,      # 400/230 V system; households and solar at unity power factor
                    source_u_pu=1.02,      # the busbar is held at 1.02 p.u. (Netbeheer Nederland 2024; Smit)
                    source_sk_va=1e12),    # near-infinite short-circuit power: an ideal busbar
    seeds=Seeds(population_seed=42, placement_seed=7, participation_seed_base=1000, lhs_seed=2024),   # fixed, so every run reproduces
    ev_hp=EVHP(ev_flex_share=0.8, hp_flex_share=0.5,             # flexible part of the charging load and of the heat-pump load
               hp_energy_kwh=15.0,                               # heat-pump electricity per winter day, kWh: Chwieduk & Chwieduk (2021), 2185 kWh over 147 days
               hp_morning_start=6, hp_morning_end=9, hp_evening_start=17, hp_evening_end=22,   # the two heating windows, local hours
               hp_seasonal={"summer": 0.225, "winter": 1.0, "spring": 0.928, "autumn": 0.445}),  # 0.22 + 0.78 x HDD_week / HDD_winter; daily degree-days to 18 C from KNMI De Bilt
)
EV_ACTIVE, HP_ACTIVE = True, True

# Scenario 2: the solar size x adoption grid at a strong price signal, demand level high, the pool on in the post-2027 regime
PI, TAU = 0.80, 0.01
EPS_GRID = [-0.10, -0.35, -0.50]
KWPS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
SHARES = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
SEASONS = ["winter", "spring", "summer", "autumn"]
DL2 = "high"
V4, D3 = 4, 6   # decimals written to the output files: voltages in V, voltage changes in mV

# =============================================================================================
# 2. THE MODEL IN SIX STEPS (Method, section 3.1 of the thesis). One run of one combination is
#    prepare_cell (steps 1 and 2 and the pre-2027 solve, done once) followed by post_run
#    (steps 3 to 6 for one draw of pi, epsilon and tau).
# =============================================================================================
def stable_seed(base, *keys):
    h = hashlib.sha256("|".join(map(str, keys)).encode()).hexdigest()[:8]
    return int(base) + int(h, 16)


# --- Step 1. Build the feeder: one radial cable, node 0 the busbar, one connection per household -----
SOURCE_NODE_ID, LINE_ID_BASE, SOURCE_ID, SYM_LOAD_ID_BASE = 0, 1000, 2000, 3000


def build_feeder(cfg, archetype):
    fc = cfg.feeders[archetype]; n = fc.n_households
    seg_len_km = (fc.cable_length_m / n) / 1000.0
    node = initialize_array("input", "node", n + 1)
    node["id"] = np.arange(0, n + 1); node["u_rated"] = cfg.network.u_rated_ll
    line = initialize_array("input", "line", n)
    line["id"] = LINE_ID_BASE + np.arange(n); line["from_node"] = np.arange(0, n); line["to_node"] = np.arange(1, n + 1)
    line["from_status"] = 1; line["to_status"] = 1
    line["r1"] = fc.r_ohm_per_km * seg_len_km; line["x1"] = fc.x_ohm_per_km * seg_len_km
    line["c1"] = fc.c_farad_per_km * seg_len_km; line["tan1"] = 0.0
    source = initialize_array("input", "source", 1)
    source["id"] = SOURCE_ID; source["node"] = SOURCE_NODE_ID; source["status"] = 1
    source["u_ref"] = cfg.network.source_u_pu; source["sk"] = cfg.network.source_sk_va
    sym_load = initialize_array("input", "sym_load", n)
    sym_load["id"] = SYM_LOAD_ID_BASE + np.arange(n); sym_load["node"] = np.arange(1, n + 1); sym_load["status"] = 1
    sym_load["type"] = LoadGenType.const_power; sym_load["p_specified"] = 0.0; sym_load["q_specified"] = 0.0
    return {"node": node, "line": line, "source": source, "sym_load": sym_load}


# --- Step 2. Build the households and their loads: category, annual consumption, solar size, place on the
#     feeder; the week's irradiance and standard load shapes; the charging and heat-pump loads ------------
def build_population(cfg, archetype, demand_level):
    fc = cfg.feeders[archetype]; n = fc.n_households; solar_share = fc.pv_penetration
    mix = cfg.household_mix[archetype]; categories = list(mix.keys())
    shares = np.array([mix[c] for c in categories], dtype=float); shares = shares / shares.sum()
    level_kwh = float(getattr(cfg.demand, f"{demand_level}_kwh"))
    rng = np.random.default_rng(stable_seed(cfg.seeds.population_seed, archetype, demand_level))
    category = rng.choice(categories, size=n, p=shares)
    E_h_year_kwh = level_kwh * rng.uniform(cfg.demand.spread_min, cfg.demand.spread_max, size=n)
    n_prosumers = int(math.ceil(round(solar_share * n, 9)))
    chosen = rng.choice(n, size=n_prosumers, replace=False)
    is_prosumer = np.zeros(n, dtype=bool); is_prosumer[chosen] = True
    kwp = np.where(is_prosumer, np.clip(rng.normal(cfg.pv.kwp_mean, cfg.pv.kwp_std, n), 1.0, None), 0.0)
    return pd.DataFrame({"household_id": np.arange(n, dtype=int), "category": category.astype(str),
                         "E_h_year_kWh": E_h_year_kwh.astype(float), "phi": np.full(n, cfg.demand.phi, dtype=float),
                         "kWp": kwp.astype(float), "is_prosumer": is_prosumer.astype(bool)})


def assign_nodes(population, cfg, archetype, demand_level):
    n = len(population)
    rng = np.random.default_rng(stable_seed(cfg.seeds.placement_seed, archetype, demand_level))
    out = population.copy(); out["node_id"] = rng.permutation(np.arange(1, n + 1)).astype(int)
    return out


def week_utc_index(cfg, week_key):
    monday_local = pd.Timestamp.fromisocalendar(IRRADIANCE_YEAR, ISO_WEEK[week_key], 1); next_monday = monday_local + pd.Timedelta(days=7)
    utc_start = monday_local.tz_localize(LOCAL_TZ).tz_convert("UTC"); utc_end = next_monday.tz_localize(LOCAL_TZ).tz_convert("UTC")
    return pd.date_range(utc_start, utc_end, freq=f"{cfg.network.timestep_min}min", tz="UTC")[1:]


def load_irradiance(cfg, week_key):
    """Global radiation of the week in W/m2 on the 15-minute UTC axis. KNMI labels the hour ending at HH; the value is
    placed at the centre of the hour and interpolated in time onto the 15-minute stamps."""
    idx = week_utc_index(cfg, week_key)
    rows = []
    with open(KNMI_FILE, "r", encoding="latin-1") as fh:
        for line in fh:
            if line.lstrip().startswith("260,"):
                parts = [p.strip() for p in line.split(",")]
                rows.append((parts[1], parts[2], parts[11]))
    raw = pd.DataFrame(rows, columns=["date", "hour", "Q"])
    ts_utc = (pd.to_datetime(raw["date"], format="%Y%m%d")
              + pd.to_timedelta(pd.to_numeric(raw["hour"]) - 0.5, unit="h")).dt.tz_localize("UTC")
    q_wm2 = pd.to_numeric(raw["Q"], errors="coerce").to_numpy() * (10000.0 / 3600.0)
    g = pd.Series(q_wm2, index=pd.DatetimeIndex(ts_utc)).sort_index()
    g_week = g.reindex(g.index.union(idx)).interpolate("time").reindex(idx).clip(lower=0)
    g_week.name = "G_wm2"; g_week.index = idx
    return g_week


def _nedu_category_columns(categories):
    meta = pd.read_csv(NEDU_FILE, sep=";", header=None, nrows=7, dtype=str, encoding="latin-1")
    rows_by_label = {}
    for i in range(len(meta)):
        label = str(meta.iloc[i, 2]).strip()
        if label in ("Categoriecode", "Type", "Richting"):
            rows_by_label[label] = meta.iloc[i]
    cat_row, type_row, ric_row = rows_by_label["Categoriecode"], rows_by_label["Type"], rows_by_label["Richting"]
    col_for = {}
    for cat in categories:
        for j in range(3, meta.shape[1]):
            if (str(cat_row.iloc[j]).strip() == cat and str(type_row.iloc[j]).strip() == "AZI"
                    and str(ric_row.iloc[j]).strip() == "A"):
                col_for[cat] = j; break
    return col_for


def load_demand_shapes(cfg, week_key):
    """Fraction of the annual consumption in every 15-minute interval of the week, per category (AZI, direction A).
    Every category's column is scaled to sum to one over the year: the file gives the double-tariff categories
    (E1B, E1C, E2B) per tariff period, each period summing to one."""
    idx = week_utc_index(cfg, week_key)
    categories = sorted({cat for mix in cfg.household_mix.values() for cat in mix}); col_for = _nedu_category_columns(categories)
    data = pd.read_csv(NEDU_FILE, sep=";", header=None, skiprows=7, dtype=str, encoding="latin-1")
    ts_cet = pd.to_datetime(data.iloc[:, 0], format="%Y-%m-%d %H:%M")
    ts_utc = pd.DatetimeIndex(ts_cet - pd.Timedelta(hours=1)).tz_localize("UTC")
    values = {cat: pd.to_numeric(data.iloc[:, col_for[cat]], errors="coerce").to_numpy() for cat in categories}
    values = {cat: v / np.nansum(v) for cat, v in values.items()}
    return pd.DataFrame(values, index=ts_utc).reindex(idx)


def pv_series(g, kwp, cfg):
    kwp_arr = np.asarray(kwp, dtype=float)
    pv = (g.to_numpy()[:, None] / cfg.pv.g_stc * cfg.pv.eta_inv) * kwp_arr[None, :]
    columns = kwp.index if isinstance(kwp, pd.Series) else range(len(kwp_arr))
    return pd.DataFrame(pv, index=g.index, columns=columns)


def _fixed_count_homes(share, n, tag, archetype, demand_level, pop_seed):
    n_dev = int(math.ceil(round(share * n, 9)))
    rng = np.random.default_rng(stable_seed(pop_seed, tag, archetype, demand_level))
    return rng.choice(n, size=n_dev, replace=False)


_EV_PROFILES = {}


def load_ev_profiles():
    if "profiles" not in _EV_PROFILES:
        raw = pd.read_csv(ELAAD_FILE); cols = [c for c in raw.columns if c.startswith("profile_")]
        _EV_PROFILES["profiles"] = pd.DataFrame(raw[cols].astype(float).to_numpy(),
                                                index=pd.DatetimeIndex(pd.to_datetime(raw["datetime"], utc=True)), columns=range(len(cols)))
    return _EV_PROFILES["profiles"]


def ev_week_profiles(cfg, week):
    return load_ev_profiles().reindex(week_utc_index(cfg, week))


def _assign_profiles(n_ev, n_profiles, archetype, demand_level, cfg):
    """Which of the ten measured profiles each charging household gets; drawn with replacement once a feeder
    has more charging households than there are profiles."""
    rng = np.random.default_rng(stable_seed(cfg.seeds.population_seed, "ev_profile", archetype, demand_level))
    if n_ev <= n_profiles:
        return rng.choice(n_profiles, size=n_ev, replace=False)
    return rng.choice(n_profiles, size=n_ev, replace=True)


def _hp_base_shape(cfg, local_hour, week):
    hp = cfg.ev_hp; factor = cfg.ev_hp.hp_seasonal[week]
    window_hours = (hp.hp_morning_end - hp.hp_morning_start) + (hp.hp_evening_end - hp.hp_evening_start)
    window_kw = hp.hp_energy_kwh * factor / window_hours
    morning = (local_hour >= hp.hp_morning_start) & (local_hour < hp.hp_morning_end)
    evening = (local_hour >= hp.hp_evening_start) & (local_hour < hp.hp_evening_end)
    return np.where(morning | evening, window_kw, 0.0)


def build_addon(population, cfg, archetype, demand_level, week, flexible_free_pool=False):
    """The charging and heat-pump loads, split into a flexible and an inflexible part (kW per household and
    timestep). With flexible_free_pool=True the flexible part is spread evenly over each local calendar day,
    with the day's energy unchanged: the pool of Scenario 2, post-2027 only."""
    ev = cfg.ev_hp
    idx = week_utc_index(cfg, week); local_hour = idx.tz_convert(LOCAL_TZ).hour.to_numpy()
    n = len(population); household_ids = population["household_id"].to_numpy()
    flex = np.zeros((len(idx), n), dtype=float); inflex = np.zeros((len(idx), n), dtype=float)
    if EV_ACTIVE:
        ev_homes = _fixed_count_homes(cfg.feeders[archetype].ev_share, n, "ev", archetype, demand_level, cfg.seeds.population_seed)
        week_prof = ev_week_profiles(cfg, week).to_numpy()
        assigned = _assign_profiles(len(ev_homes), week_prof.shape[1], archetype, demand_level, cfg)
        for slot, home in enumerate(ev_homes):
            p = week_prof[:, assigned[slot]]
            flex[:, home] += ev.ev_flex_share * p; inflex[:, home] += (1.0 - ev.ev_flex_share) * p
    if HP_ACTIVE:
        hp_homes = _fixed_count_homes(cfg.feeders[archetype].hp_share, n, "hp", archetype, demand_level, cfg.seeds.population_seed)
        hp_base = _hp_base_shape(cfg, local_hour, week)
        flex[:, hp_homes] += (ev.hp_flex_share * hp_base)[:, None]
        inflex[:, hp_homes] += ((1 - ev.hp_flex_share) * hp_base)[:, None]
    if flexible_free_pool:
        local_day = idx.tz_convert(LOCAL_TZ).normalize()
        flex = pd.DataFrame(flex, index=idx).groupby(local_day).transform("mean").to_numpy()
    return pd.DataFrame(flex, index=idx, columns=household_ids), pd.DataFrame(inflex, index=idx, columns=household_ids)


def base_demand(population, demand_shapes, cfg):
    """Base residential demand in kW: annual consumption x the fraction per interval x 4 intervals per hour."""
    steps_per_hour = 60 / cfg.network.timestep_min
    categories = population["category"].tolist(); annual_kwh = population["E_h_year_kWh"].to_numpy()
    household_ids = population["household_id"].to_numpy()
    shape_mat = demand_shapes[categories].to_numpy()
    d_base = shape_mat * annual_kwh[np.newaxis, :] * steps_per_hour
    return pd.DataFrame(d_base, index=demand_shapes.index, columns=household_ids)


# --- Step 3. Construct the local price: low in the sunny hours, high without irradiance; post-2027 only --------
def price_signal(g, cfg, tau):
    p_retail = cfg.prices.p_retail; p_floor = cfg.prices.p_floor
    g_max = g.max()
    if g_max <= 0:
        return pd.Series(p_retail - tau, index=g.index, name="p_eff")
    span = max(p_retail - p_floor - 2 * tau, 0.0)
    p_eff = (p_floor + tau) + span * (1 - g / g_max); p_eff.name = "p_eff"
    return p_eff


# --- Step 4. Apply the demand response: participants reweight their flexible load by (price / daily mean)^epsilon,
#     the energy of every local day unchanged; post-2027 only ---------------------------------------------------
def reshape_flexible(flex_base, price, epsilon):
    day = price.index.tz_convert(LOCAL_TZ).normalize()
    daymean = price.groupby(day).transform("mean"); r = (price / daymean) ** epsilon
    flex_tent = flex_base.mul(r, axis=0)
    scale = flex_base.groupby(day).transform("sum") / flex_tent.groupby(day).transform("sum")
    return flex_tent * scale


def build_demand(population, demand_shapes, price, epsilon, participation_mask, cfg, flexible_addon=None):
    """Demand in kW per household and timestep. With price=None the pre-2027 demand; otherwise the post-2027
    demand, in which the participating households' flexible part follows the price."""
    phi = cfg.demand.phi; d_base = base_demand(population, demand_shapes, cfg)
    inflex = (1 - phi) * d_base; flex_base = phi * d_base
    if flexible_addon is not None:
        addon_flex, addon_inflex = flexible_addon; inflex = inflex + addon_inflex; flex_base = flex_base + addon_flex
    if price is None:
        return d_base if flexible_addon is None else (inflex + flex_base)
    mask_bool = np.zeros(len(population), dtype=bool) if participation_mask is None else np.asarray(participation_mask, dtype=bool)
    flex_reshaped = reshape_flexible(flex_base, price, epsilon); flex_final = flex_base.copy()
    flex_final[flex_base.columns[mask_bool]] = flex_reshaped[flex_base.columns[mask_bool]]
    return inflex + flex_final


def net_power(pv, demand):
    return pv - demand


def fixed_count_participation(pi, n, archetype, demand_level, run_index, cfg):
    """Which households take part in the trade: ceil(pi x n) of them, drawn with a seed per combination and run."""
    n_part = int(math.ceil(round(pi * n, 9)))
    rng = np.random.default_rng(stable_seed(cfg.seeds.participation_seed_base, archetype, demand_level, run_index))
    chosen = rng.choice(n, size=n_part, replace=False)
    mask = np.zeros(n, dtype=bool); mask[chosen] = True
    return mask


# --- Step 5. Solve the power flow: the net power of every connection to the Power Grid Model, Newton-Raphson,
#     one batch over the week's timesteps; returns the voltage at every node in p.u. ---------------------------
def solve_voltages(feeder_dataset, net_power_df, population, cfg):
    n_steps = len(net_power_df); load_ids = feeder_dataset["sym_load"]["id"]; load_nodes = feeder_dataset["sym_load"]["node"]
    n_loads = len(load_ids)
    node_of_household = population.set_index("household_id")["node_id"]
    household_at_node = {int(node): int(hh) for hh, node in node_of_household.items()}
    p_specified = np.empty((n_steps, n_loads), dtype=float)
    for j, node in enumerate(load_nodes):
        p_specified[:, j] = -net_power_df[household_at_node[int(node)]].to_numpy() * 1000.0
    update_sym_load = initialize_array("update", "sym_load", (n_steps, n_loads))
    update_sym_load["id"] = load_ids[np.newaxis, :]; update_sym_load["p_specified"] = p_specified
    model = PowerGridModel(feeder_dataset)
    out = model.calculate_power_flow(update_data={"sym_load": update_sym_load}, symmetric=True,
                                     calculation_method=CalculationMethod.newton_raphson)
    return pd.DataFrame(out["node"]["u_pu"], index=net_power_df.index, columns=feeder_dataset["node"]["id"])


# --- Step 6. Check compliance: the 95 % requirement (VCR, every node within +-10 %), the absolute requirement
#     (no node below -15 %, none above +10 %), and the endpoint extremes and crossings in volts -----------------
def compute_metrics(u_pu, cfg):
    band = cfg.network.band; abs_lower_band = cfg.network.abs_lower_band
    over_lim = 1.0 + band; under_lim = 1.0 - band; abs_lower_lim = 1.0 - abs_lower_band
    u = u_pu.to_numpy()
    vcr = float(((u >= under_lim) & (u <= over_lim)).all(axis=1).mean())
    return {"vcr": vcr, "u_min": float(u.min()), "u_max": float(u.max()), "u_mean": float(u.mean()),
            "p95": float(np.quantile(u, 0.95)),
            "abs_compliant": bool(u.max() <= over_lim and u.min() >= abs_lower_lim)}


def ep_stats(u, node, UN):
    v = u[node].to_numpy() * UN
    return v.max(), v.min(), v.mean(), np.quantile(v, 0.95)


def all_stats(u, UN):
    a = u.to_numpy() * UN
    return a.max(), a.min(), a.mean(), np.quantile(a, 0.95)


def ep_crossings(u, node, UN, band):
    v = u[node].to_numpy() * UN
    over = (1.0 + band) * UN; under = (1.0 - band) * UN
    return int((v > over).sum()), int((v < under).sum())


def quarter_hour_slots(index):
    loc = index.tz_convert(LOCAL_TZ)
    return (loc.hour.to_numpy() * 4 + loc.minute.to_numpy() // 15).astype(int)


# --- One combination and one run ---------------------------------------------------------------------------
def sample_lhs(n_runs, cfg):
    """The Latin-hypercube draws of pi, epsilon and tau for one combination (the same draws in every combination)."""
    sampler = qmc.LatinHypercube(d=3, seed=cfg.seeds.lhs_seed); unit = sampler.random(n=n_runs)
    lows = [cfg.sampling.participation_range[0], cfg.sampling.epsilon_range[0], cfg.sampling.tau_range[0]]
    highs = [cfg.sampling.participation_range[1], cfg.sampling.epsilon_range[1], cfg.sampling.tau_range[1]]
    return qmc.scale(unit, lows, highs)


def prepare_cell(cfg, archetype, demand_level, week, flexible_free_pool=False):
    """Steps 1 and 2 and the pre-2027 solve, once per combination. The pool (flexible_free_pool) applies to the
    post-2027 loads only; the pre-2027 regime always keeps the measured charging profile and the heating windows."""
    population = assign_nodes(build_population(cfg, archetype, demand_level), cfg, archetype, demand_level)
    feeder_dataset = build_feeder(cfg, archetype)
    g = load_irradiance(cfg, week); shapes = load_demand_shapes(cfg, week)
    pv = pv_series(g, population.set_index("household_id")["kWp"], cfg)
    addon_pre = build_addon(population, cfg, archetype, demand_level, week, flexible_free_pool=False)
    addon = build_addon(population, cfg, archetype, demand_level, week, flexible_free_pool=flexible_free_pool)
    net_pre = net_power(pv, build_demand(population, shapes, None, 0.0, None, cfg, flexible_addon=addon_pre))
    u_pre = solve_voltages(feeder_dataset, net_pre, population, cfg)
    return {"population": population, "feeder": feeder_dataset, "g": g, "shapes": shapes, "pv": pv,
            "addon": addon, "u_pre": u_pre, "m_pre": compute_metrics(u_pre, cfg)}


def post_run(cell, cfg, archetype, demand_level, pi, eps, tau, run_index):
    """Steps 3 to 6 for one run: the participants of this run, the price at this fee, the response at this
    elasticity, the post-2027 power flow and its metrics."""
    pop = cell["population"]; mask = fixed_count_participation(pi, len(pop), archetype, demand_level, run_index, cfg)
    price = price_signal(cell["g"], cfg, tau)
    net_po = net_power(cell["pv"], build_demand(pop, cell["shapes"], price, eps, mask, cfg, flexible_addon=cell["addon"]))
    u_po = solve_voltages(cell["feeder"], net_po, pop, cfg)
    return u_po, compute_metrics(u_po, cfg), int(mask.sum())


# =============================================================================================
# 3. THE FLEXIBLE LOAD BY SOURCE (residential, charging, heat pump), used for output file 05:
#    the same arithmetic as build_addon and reshape_flexible, kept per component.
# =============================================================================================
def addon_components(pop, cfg, arch, dl, week, free_pool):
    e = cfg.ev_hp
    idx = week_utc_index(cfg, week); local_hour = idx.tz_convert(LOCAL_TZ).hour.to_numpy()
    n = len(pop); hh = pop["household_id"].to_numpy()
    ev_f = np.zeros((len(idx), n)); ev_i = np.zeros((len(idx), n))
    hp_f = np.zeros((len(idx), n)); hp_i = np.zeros((len(idx), n))
    if EV_ACTIVE:
        ev_homes = _fixed_count_homes(cfg.feeders[arch].ev_share, n, "ev", arch, dl, cfg.seeds.population_seed)
        week_prof = ev_week_profiles(cfg, week).to_numpy()
        assigned = _assign_profiles(len(ev_homes), week_prof.shape[1], arch, dl, cfg)
        for slot, home in enumerate(ev_homes):
            pcol = week_prof[:, assigned[slot]]
            ev_f[:, home] += e.ev_flex_share * pcol; ev_i[:, home] += (1.0 - e.ev_flex_share) * pcol
    if HP_ACTIVE:
        hp_homes = _fixed_count_homes(cfg.feeders[arch].hp_share, n, "hp", arch, dl, cfg.seeds.population_seed)
        hp_base = _hp_base_shape(cfg, local_hour, week)
        hp_f[:, hp_homes] += (e.hp_flex_share * hp_base)[:, None]
        hp_i[:, hp_homes] += ((1.0 - e.hp_flex_share) * hp_base)[:, None]
    if free_pool:
        day = idx.tz_convert(LOCAL_TZ).normalize()
        ev_f = pd.DataFrame(ev_f, index=idx).groupby(day).transform("mean").to_numpy()
        hp_f = pd.DataFrame(hp_f, index=idx).groupby(day).transform("mean").to_numpy()
    mk = lambda a: pd.DataFrame(a, index=idx, columns=hh)
    return mk(ev_f), mk(hp_f), mk(ev_i), mk(hp_i)


_SHAPE_CACHE = {}


def cfg_shapes(cfg, week):
    if week not in _SHAPE_CACHE:
        _SHAPE_CACHE[week] = load_demand_shapes(cfg, week)
    return _SHAPE_CACHE[week]


def component_bases(pop, cfg, arch, dl, week, free_pool):
    base = base_demand(pop, cfg_shapes(cfg, week), cfg)
    ev_f, hp_f, ev_i, hp_i = addon_components(pop, cfg, arch, dl, week, free_pool)
    res_f = cfg.demand.phi * base
    flex_base = res_f + ev_f + hp_f
    inflex = (1.0 - cfg.demand.phi) * base + ev_i + hp_i
    return dict(res_f=res_f, ev_f=ev_f, hp_f=hp_f, flex_base=flex_base, inflex=inflex)


def post_ratio(flex_base, price, eps, mask):
    """The multiplier the response applies to each household's flexible load; 1 where not participating."""
    reshaped = reshape_flexible(flex_base, price, eps)
    ratio = np.ones_like(flex_base.to_numpy())
    cols = np.asarray(mask, dtype=bool)
    denom = flex_base.to_numpy()[:, cols]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio[:, cols] = np.where(denom > 0, reshaped.to_numpy()[:, cols] / denom, 1.0)
    return pd.DataFrame(ratio, index=flex_base.index, columns=flex_base.columns)


def decompose(pop, cfg, arch, dl, week, free_pool, price, eps, mask):
    b = component_bases(pop, cfg, arch, dl, week, free_pool)
    ratio = post_ratio(b["flex_base"], price, eps, mask)
    return dict(**b, res_f_post=b["res_f"] * ratio, ev_f_post=b["ev_f"] * ratio,
                hp_f_post=b["hp_f"] * ratio, flex_post=b["flex_base"] * ratio)


# =============================================================================================
# 4. SCENARIO 1: 36 combinations (3 archetypes x 3 demand levels x 4 weeks) x 1000 Latin-hypercube
#    runs of pi, epsilon and tau at present solar and device levels. Writes files 01 to 05.
# =============================================================================================
RUN_COLS = ["archetype", "demand_level", "week", "run_index", "pi", "epsilon", "tau", "n_participants",
            "pre_max_ep", "pre_min_ep", "pre_mean_ep", "pre_p95_ep", "post_max_ep", "post_min_ep", "post_mean_ep", "post_p95_ep",
            "delta_max_ep", "delta_min_ep", "delta_mean_ep", "delta_p95_ep",
            "pre_max_all", "pre_min_all", "pre_mean_all", "pre_p95_all", "post_max_all", "post_min_all", "post_mean_all", "post_p95_all",
            "delta_max_all", "delta_min_all", "delta_mean_all", "delta_p95_all",
            "vcr_pre", "vcr_post", "abs_compliant_pre", "abs_compliant_post",
            "crossings_above_253_pre", "crossings_above_253_post", "crossings_below_207_pre", "crossings_below_207_post", "n_timesteps"]
CELL_COLS = ["archetype", "demand_level", "week", "n_households", "endpoint_node", "n_timesteps", "n_runs",
             "pre_max_ep", "pre_min_ep", "pre_mean_ep", "pre_p95_ep", "pre_max_all", "pre_min_all", "vcr_pre",
             "abs_compliant_pre", "post_max_ep_worst", "post_min_ep_worst",
             "delta_max_ep_median", "delta_max_ep_p05", "delta_max_ep_p95", "delta_min_ep_median", "delta_min_ep_p05", "delta_min_ep_p95",
             "delta_mean_ep_median", "delta_mean_ep_p05", "delta_mean_ep_p95", "delta_p95_ep_median", "delta_p95_ep_p05", "delta_p95_ep_p95",
             "n_runs_with_any_breach", "vcr_post_min", "vcr_post_median",
             "run_index_max_delta_max", "pi_max_delta_max", "epsilon_max_delta_max", "tau_max_delta_max", "delta_max_ep_best",
             "run_index_max_delta_min", "pi_max_delta_min", "epsilon_max_delta_min", "tau_max_delta_min", "delta_min_ep_best"]
QH_COLS = ["archetype", "demand_level", "week", "quarter_hour_local", "node_id", "mean_delta_mV", "p05_delta_mV", "p95_delta_mV",
           "min_delta_mV", "max_delta_mV", "n_days_contributing"]
PP_COLS = ["archetype", "demand_level", "week", "series_type", "run_index", "pi", "epsilon", "tau", "quarter_hour_local",
           "n_days_contributing", "pv_generation_kw", "demand_inflexible_kw", "demand_flexible_pre_kw", "demand_flexible_post_kw",
           "residential_flexible_pre_kw", "residential_flexible_post_kw", "ev_flexible_pre_kw", "ev_flexible_post_kw",
           "hp_flexible_pre_kw", "hp_flexible_post_kw", "price_eur_per_kwh"]


def _qh(series_steps, slots):
    df = pd.DataFrame({"v": np.asarray(series_steps), "slot": slots})
    g = df.groupby("slot")["v"]
    return g.mean(), g.size()


def _qh_stats(series_steps, slots):
    df = pd.DataFrame({"v": np.asarray(series_steps), "slot": slots})
    g = df.groupby("slot")["v"]
    return (g.mean(), g.quantile(.05), g.quantile(.95), g.min(), g.max(), g.size())


def _vw_frame(arch, dl, week, series, dials, v_pre_2d, v_post_2d, ts, nodes):
    nT = len(ts); nN = len(nodes)
    ri, pi, eps, tau = (dials if dials else (None, None, None, None))
    return pd.DataFrame({
        "archetype": arch, "demand_level": dl, "week": week, "series_type": series,
        "run_index": ri, "pi": pi, "epsilon": eps, "tau": tau,
        "timestamp_utc": np.repeat(ts, nN), "node_id": np.tile(nodes, nT),
        "v_pre": np.round(v_pre_2d.reshape(-1), V4), "v_post": np.round(v_post_2d.reshape(-1), V4)})


def _pp_rows(arch, dl, week, series, dials, slots, pv_feeder, pre, post):
    ri, pi, eps, tau = (dials if dials else (None, None, None, None))
    qh = lambda a: _qh(a, slots)
    pv_m, cnt = qh(pv_feeder); inflex_m, _ = qh(pre["inflex"])
    rfp, _ = qh(pre["res"]); efp, _ = qh(pre["ev"]); hfp, _ = qh(pre["hp"]); flp, _ = qh(pre["flex"])
    rfq, _ = qh(post["res_post"]); efq, _ = qh(post["ev_post"]); hfq, _ = qh(post["hp_post"]); flq, _ = qh(post["flex_post"])
    price_m, _ = qh(post["price"])
    rows = []
    for slot in pv_m.index:
        rows.append([arch, dl, week, series, ri, pi, eps, tau, int(slot), int(cnt[slot]),
                     round(float(pv_m[slot]), 4), round(float(inflex_m[slot]), 4),
                     round(float(flp[slot]), 4), round(float(flq[slot]), 4),
                     round(float(rfp[slot]), 4), round(float(rfq[slot]), 4),
                     round(float(efp[slot]), 4), round(float(efq[slot]), 4),
                     round(float(hfp[slot]), 4), round(float(hfq[slot]), 4), round(float(price_m[slot]), 6)])
    return rows


def run_part1(cfg, cells, n_runs, outdir):
    UN = cfg.network.u_nom; band = cfg.network.band
    paths = {k: outdir / f"{k}.csv" for k in ["01_run_summary", "02_cell_summary", "04_quarterhour_voltage_profiles", "05_power_price_profiles"]}
    vw_csv = outdir / "03_voltage_weeks.csv"
    for p in list(paths.values()) + [vw_csv, outdir / "03_voltage_weeks.parquet"]:
        if p.exists():
            p.unlink()
    cell_rows, qh_rows, pp_rows, vw_frames = [], [], [], []
    hdr01 = True; nonconv = []

    for ci, (arch, dl, week) in enumerate(cells, 1):
        cell = prepare_cell(cfg, arch, dl, week)
        u_pre = cell["u_pre"]; pop = cell["population"]; g = cell["g"]
        node = cfg.feeders[arch].n_households; nodes = list(u_pre.columns)
        idx = u_pre.index; nT = len(idx); slots = quarter_hour_slots(idx)
        m_pre = compute_metrics(u_pre, cfg)
        pre_ep = ep_stats(u_pre, node, UN); pre_all = all_stats(u_pre, UN)
        pre_cross = ep_crossings(u_pre, node, UN, band)

        pv_feeder = cell["pv"].to_numpy().sum(axis=1)
        cb = component_bases(pop, cfg, arch, dl, week, False)
        res_np, ev_np, hp_np = cb["res_f"].to_numpy(), cb["ev_f"].to_numpy(), cb["hp_f"].to_numpy()
        flex_np = cb["flex_base"].to_numpy()
        pre_comp = {"res": res_np.sum(1), "ev": ev_np.sum(1), "hp": hp_np.sum(1),
                    "flex": flex_np.sum(1), "inflex": cb["inflex"].to_numpy().sum(1)}
        draws = sample_lhs(n_runs, cfg)

        sum_u_post = np.zeros((nT, len(nodes)))
        acc = {k: np.zeros(nT) for k in ["res_post", "ev_post", "hp_post", "flex_post", "price"]}
        best = {"dmax": (np.inf, None), "dmin": (-np.inf, None)}   # the run with the most negative change of the maximum, the most positive change of the minimum
        run01 = []; d_ep = {"max": [], "min": [], "mean": [], "p95": []}
        worst_post_max = -np.inf; worst_post_min = np.inf; vcr_posts = []; n_breach = 0

        for ri in range(n_runs):
            pi, eps, tau = float(draws[ri][0]), float(draws[ri][1]), float(draws[ri][2])
            try:
                u_po, m_po, npart = post_run(cell, cfg, arch, dl, pi, eps, tau, ri)
            except PowerGridBatchError:
                nonconv.append((arch, dl, week, ri)); continue
            post_ep = ep_stats(u_po, node, UN); post_all = all_stats(u_po, UN)
            post_cross = ep_crossings(u_po, node, UN, band)
            dmax = (post_ep[0] - pre_ep[0]) * 1000; dmin = (post_ep[1] - pre_ep[1]) * 1000
            dmean = (post_ep[2] - pre_ep[2]) * 1000; dp95 = (post_ep[3] - pre_ep[3]) * 1000
            run01.append([arch, dl, week, ri, pi, eps, tau, npart,
                          *[round(x, V4) for x in pre_ep], *[round(x, V4) for x in post_ep],
                          round(dmax, D3), round(dmin, D3), round(dmean, D3), round(dp95, D3),
                          *[round(x, V4) for x in pre_all], *[round(x, V4) for x in post_all],
                          round((post_all[0] - pre_all[0]) * 1000, D3), round((post_all[1] - pre_all[1]) * 1000, D3),
                          round((post_all[2] - pre_all[2]) * 1000, D3), round((post_all[3] - pre_all[3]) * 1000, D3),
                          round(m_pre["vcr"], 6), round(m_po["vcr"], 6), m_pre["abs_compliant"], m_po["abs_compliant"],
                          pre_cross[0], post_cross[0], pre_cross[1], post_cross[1], nT])
            for k, v in zip(["max", "min", "mean", "p95"], [dmax, dmin, dmean, dp95]):
                d_ep[k].append(v)
            worst_post_max = max(worst_post_max, post_ep[0]); worst_post_min = min(worst_post_min, post_ep[1])
            vcr_posts.append(m_po["vcr"]); n_breach += int(post_cross[0] > 0 or post_cross[1] > 0)
            sum_u_post += u_po.to_numpy()
            mask = fixed_count_participation(pi, len(pop), arch, dl, ri, cfg)
            price = price_signal(g, cfg, tau)
            rn = post_ratio(cb["flex_base"], price, eps, mask).to_numpy()
            acc["res_post"] += (res_np * rn).sum(1); acc["ev_post"] += (ev_np * rn).sum(1)
            acc["hp_post"] += (hp_np * rn).sum(1); acc["flex_post"] += (flex_np * rn).sum(1)
            acc["price"] += price.to_numpy()
            if dmax < best["dmax"][0]:
                best["dmax"] = (dmax, (ri, pi, eps, tau))
            if dmin > best["dmin"][0]:
                best["dmin"] = (dmin, (ri, pi, eps, tau))

        nR = len(run01)
        pd.DataFrame(run01, columns=RUN_COLS).to_csv(paths["01_run_summary"], mode="a", header=hdr01, index=False)
        hdr01 = False
        mean_u_post = sum_u_post / nR

        # 03: the voltage at every node and timestep, pre and post, for the mean over the runs and the two extreme runs
        ts = idx.strftime("%Y-%m-%dT%H:%M:%SZ")
        vw_frames.append(_vw_frame(arch, dl, week, "mean", None, u_pre.to_numpy() * UN, mean_u_post * UN, ts, nodes))
        for tag, key in [("max_delta_max", "dmax"), ("max_delta_min", "dmin")]:
            ri, pi, eps, tau = best[key][1]
            u_po, _, _ = post_run(cell, cfg, arch, dl, pi, eps, tau, ri)
            vw_frames.append(_vw_frame(arch, dl, week, tag, (ri, pi, eps, tau), u_pre.to_numpy() * UN, u_po.to_numpy() * UN, ts, nodes))

        # 04: the mean voltage change per quarter-hour of the day and node, in mV
        delta_step = (mean_u_post - u_pre.to_numpy()) * UN * 1000
        for j, nd in enumerate(nodes):
            md, p05, p95, mn, mx, cnt = _qh_stats(delta_step[:, j], slots)
            for slot in md.index:
                qh_rows.append([arch, dl, week, int(slot), int(nd),
                                round(float(md[slot]), 4), round(float(p05[slot]), 4), round(float(p95[slot]), 4),
                                round(float(mn[slot]), 4), round(float(mx[slot]), 4), int(cnt[slot])])

        # 05: the feeder's solar generation, loads by source and price per quarter-hour, mean over the runs and the two extreme runs
        mn = {k: acc[k] / nR for k in acc}
        pp_rows += _pp_rows(arch, dl, week, "mean", None, slots, pv_feeder, pre_comp, mn)
        for tag, key in [("max_delta_max", "dmax"), ("max_delta_min", "dmin")]:
            ri, pi, eps, tau = best[key][1]
            mask = fixed_count_participation(pi, len(pop), arch, dl, ri, cfg)
            price = price_signal(g, cfg, tau)
            dec = decompose(pop, cfg, arch, dl, week, False, price, eps, mask)
            post = {"res_post": dec["res_f_post"].to_numpy().sum(1), "ev_post": dec["ev_f_post"].to_numpy().sum(1),
                    "hp_post": dec["hp_f_post"].to_numpy().sum(1), "flex_post": dec["flex_post"].to_numpy().sum(1),
                    "price": price.to_numpy()}
            pp_rows += _pp_rows(arch, dl, week, tag, (ri, pi, eps, tau), slots, pv_feeder, pre_comp, post)

        # 02: one row per combination
        qd = lambda a, q: round(float(np.quantile(a, q)), D3)
        cell_rows.append([arch, dl, week, cfg.feeders[arch].n_households, node, nT, nR,
                          *[round(x, V4) for x in pre_ep], round(pre_all[0], V4), round(pre_all[1], V4),
                          round(m_pre["vcr"], 6), m_pre["abs_compliant"],
                          round(worst_post_max, V4), round(worst_post_min, V4),
                          qd(d_ep["max"], .5), qd(d_ep["max"], .05), qd(d_ep["max"], .95),
                          qd(d_ep["min"], .5), qd(d_ep["min"], .05), qd(d_ep["min"], .95),
                          qd(d_ep["mean"], .5), qd(d_ep["mean"], .05), qd(d_ep["mean"], .95),
                          qd(d_ep["p95"], .5), qd(d_ep["p95"], .05), qd(d_ep["p95"], .95),
                          n_breach, round(float(np.min(vcr_posts)), 6), round(float(np.median(vcr_posts)), 6),
                          best["dmax"][1][0], best["dmax"][1][1], best["dmax"][1][2], best["dmax"][1][3], round(best["dmax"][0], D3),
                          best["dmin"][1][0], best["dmin"][1][1], best["dmin"][1][2], best["dmin"][1][3], round(best["dmin"][0], D3)])
        print(f"  [Scenario 1] combination {ci}/{len(cells)}  {arch}/{dl}/{week}  runs={nR}", flush=True)

    pd.DataFrame(cell_rows, columns=CELL_COLS).to_csv(paths["02_cell_summary"], index=False)
    pd.DataFrame(qh_rows, columns=QH_COLS).to_csv(paths["04_quarterhour_voltage_profiles"], index=False)
    pd.DataFrame(pp_rows, columns=PP_COLS).to_csv(paths["05_power_price_profiles"], index=False)
    vw = pd.concat(vw_frames, ignore_index=True)
    vw.to_csv(vw_csv, index=False)
    try:
        vw.to_parquet(outdir / "03_voltage_weeks.parquet", index=False)
    except Exception:
        pass
    return nonconv


# =============================================================================================
# 5. SCENARIO 2: the solar size (4 to 14 kWp) x adoption (0.15 to 0.75) grid on every archetype
#    and week, at three elasticities, pi 0.80 and tau 0.01, the pool on post-2027. Writes file 06.
# =============================================================================================
GRID_COLS = ["season", "archetype", "demand_level", "epsilon", "pi", "tau", "phi", "kwp_mean", "kwp_std", "ev_hp_share",
             "n_households", "endpoint_node", "n_timesteps", "pre_max_ep", "pre_min_ep", "post_max_ep", "post_min_ep",
             "delta_max_mV", "delta_min_mV", "crossings_above_253_pre", "crossings_above_253_post",
             "crossings_below_207_pre", "crossings_below_207_post", "vcr_pre", "vcr_post", "compliance_state",
             "ev_share_used", "hp_share_used", "free_pool_on"]


def _cfg_for(cfg, arch, kwp, share):
    fe = dict(cfg.feeders); fe[arch] = dc.replace(cfg.feeders[arch], ev_share=share, hp_share=share)
    return dc.replace(cfg, pv=dc.replace(cfg.pv, kwp_mean=kwp), feeders=fe)


def run_part2(cfg, archs, seasons, kwps, shares, epss, outdir):
    UN = cfg.network.u_nom; band = cfg.network.band
    rows = []; nonconv = []; total = len(archs) * len(seasons) * len(kwps) * len(shares); done = 0
    for arch in archs:
        for season in seasons:
            for kwp in kwps:
                for share in shares:
                    cfg_s = _cfg_for(cfg, arch, kwp, share)
                    cell = prepare_cell(cfg_s, arch, DL2, season, flexible_free_pool=True)
                    u_pre = cell["u_pre"]; node = cfg_s.feeders[arch].n_households; nT = len(u_pre)
                    pre_ep = ep_stats(u_pre, node, UN); pre_cross = ep_crossings(u_pre, node, UN, band)
                    vcr_pre = compute_metrics(u_pre, cfg_s)["vcr"]
                    for eps in epss:
                        try:
                            u_po, m_po, _ = post_run(cell, cfg_s, arch, DL2, PI, eps, TAU, 0)
                        except PowerGridBatchError:
                            nonconv.append((season, arch, kwp, share, eps)); continue
                        post_ep = ep_stats(u_po, node, UN); post_cross = ep_crossings(u_po, node, UN, band)
                        breach_pre = pre_cross[0] > 0 or pre_cross[1] > 0
                        breach_post = post_cross[0] > 0 or post_cross[1] > 0
                        state = (0 if not breach_pre and not breach_post else 1 if breach_pre and breach_post
                                 else 2 if breach_pre and not breach_post else 3)   # 0 no breach, 1 both, 2 cleared post-2027, 3 new post-2027
                        rows.append([season, arch, DL2, eps, PI, TAU, cfg.demand.phi, kwp, cfg.pv.kwp_std, share,
                                     cfg_s.feeders[arch].n_households, node, nT,
                                     round(pre_ep[0], 4), round(pre_ep[1], 4), round(post_ep[0], 4), round(post_ep[1], 4),
                                     round((post_ep[0] - pre_ep[0]) * 1000, 3), round((post_ep[1] - pre_ep[1]) * 1000, 3),
                                     pre_cross[0], post_cross[0], pre_cross[1], post_cross[1],
                                     round(vcr_pre, 6), round(m_po["vcr"], 6), state, cfg_s.feeders[arch].ev_share, cfg_s.feeders[arch].hp_share, True])
                    done += 1
                    if done % 10 == 0:
                        print(f"  [Scenario 2] {done}/{total} points", flush=True)
    pd.DataFrame(rows, columns=GRID_COLS).to_csv(outdir / "06_scenario_grid.csv", index=False)
    return nonconv


# =============================================================================================
# 6. THE DECOMPOSITION OF SCENARIO 2 (Table 4.9): every grid point solved in four demand states.
#    A pre-2027 (no pool, no price response), B the pool alone, D the price response alone,
#    C both (= the post-2027 regime of Scenario 2). Writes file 07.
# =============================================================================================
STEP_H = 0.25


def energy_metrics(pv, d):
    pv_tot = pv.sum() * STEP_H; d_tot = d.sum() * STEP_H
    sc_h = np.minimum(pv, d).sum() * STEP_H
    pv_f = pv.sum(axis=1); d_f = d.sum(axis=1); sc_f = np.minimum(pv_f, d_f).sum() * STEP_H
    surplus = pv_f - d_f
    return {"pv_kwh": pv_tot, "demand_kwh": d_tot,
            "scr_house": sc_h / pv_tot if pv_tot > 0 else np.nan,
            "scr_feeder": sc_f / pv_tot if pv_tot > 0 else np.nan,
            "ssr_house": sc_h / d_tot, "ssr_feeder": sc_f / d_tot,
            "feeder_export_kwh": np.clip(surplus, 0, None).sum() * STEP_H,
            "feeder_peak_export_kw": float(np.clip(surplus, 0, None).max()),
            "feeder_peak_import_kw": float(np.clip(-surplus, 0, None).max())}


def voltage_metrics(u, node, UN, band, cfg):
    mx, mn, mean, p95 = ep_stats(u, node, UN); over, under = ep_crossings(u, node, UN, band)
    return {"max_ep": mx, "min_ep": mn, "cross_above": over, "cross_below": under, "vcr": compute_metrics(u, cfg)["vcr"]}


def state(pre_over, pre_under, post_over, post_under):
    bp = pre_over > 0 or pre_under > 0; bq = post_over > 0 or post_under > 0
    return 0 if not bp and not bq else 1 if bp and bq else 2 if bp and not bq else 3


def run_decomposition(cfg, archs, seasons, kwps, shares, epss, outdir):
    UN = cfg.network.u_nom; band = cfg.network.band
    shapes_cache = {}; rows = []; nonconv = []; done = 0; total = len(archs) * len(seasons) * len(kwps) * len(shares) * len(epss)
    for arch in archs:
        for season in seasons:
            if season not in shapes_cache:
                shapes_cache[season] = (load_irradiance(cfg, season), load_demand_shapes(cfg, season))
            g, shapes = shapes_cache[season]
            for kwp in kwps:
                for share in shares:
                    cfg_s = _cfg_for(cfg, arch, kwp, share)
                    pop = assign_nodes(build_population(cfg_s, arch, DL2), cfg_s, arch, DL2)
                    feeder = build_feeder(cfg_s, arch); node = cfg_s.feeders[arch].n_households
                    pv_df = pv_series(g, pop.set_index("household_id")["kWp"], cfg_s); pv = pv_df.to_numpy(float)
                    add_off = build_addon(pop, cfg_s, arch, DL2, season, flexible_free_pool=False)
                    add_on = build_addon(pop, cfg_s, arch, DL2, season, flexible_free_pool=True)
                    mask = fixed_count_participation(PI, len(pop), arch, DL2, 0, cfg_s)
                    price = price_signal(g, cfg_s, TAU)
                    dA = build_demand(pop, shapes, None, 0.0, None, cfg_s, flexible_addon=add_off)
                    dB = build_demand(pop, shapes, None, 0.0, None, cfg_s, flexible_addon=add_on)
                    try:
                        uA = solve_voltages(feeder, net_power(pv_df, dA), pop, cfg_s)
                        uB = solve_voltages(feeder, net_power(pv_df, dB), pop, cfg_s)
                    except PowerGridBatchError:
                        nonconv.append((arch, season, kwp, share, "A/B")); continue
                    vA = voltage_metrics(uA, node, UN, band, cfg_s); vB = voltage_metrics(uB, node, UN, band, cfg_s)
                    eA = energy_metrics(pv, dA.to_numpy(float)); eB = energy_metrics(pv, dB.to_numpy(float))
                    for eps in epss:
                        dD = build_demand(pop, shapes, price, eps, mask, cfg_s, flexible_addon=add_off)
                        dC = build_demand(pop, shapes, price, eps, mask, cfg_s, flexible_addon=add_on)
                        try:
                            uD = solve_voltages(feeder, net_power(pv_df, dD), pop, cfg_s)
                            uC = solve_voltages(feeder, net_power(pv_df, dC), pop, cfg_s)
                        except PowerGridBatchError:
                            nonconv.append((arch, season, kwp, share, eps)); continue
                        vD = voltage_metrics(uD, node, UN, band, cfg_s); vC = voltage_metrics(uC, node, UN, band, cfg_s)
                        eD = energy_metrics(pv, dD.to_numpy(float)); eC = energy_metrics(pv, dC.to_numpy(float))
                        row = {"archetype": arch, "season": season, "kwp_mean": kwp, "ev_hp_share": share,
                               "epsilon": eps, "pi": PI, "tau": TAU, "n_households": len(pop),
                               "n_participants": int(mask.sum()), "endpoint_node": node}
                        for tag, v, e in (("A", vA, eA), ("B", vB, eB), ("D", vD, eD), ("C", vC, eC)):
                            row.update({f"{k}_{tag}": val for k, val in v.items()})
                            row.update({f"{k}_{tag}": val for k, val in e.items()})
                        row["state_AC"] = state(vA["cross_above"], vA["cross_below"], vC["cross_above"], vC["cross_below"])
                        row["state_AB"] = state(vA["cross_above"], vA["cross_below"], vB["cross_above"], vB["cross_below"])
                        row["state_AD"] = state(vA["cross_above"], vA["cross_below"], vD["cross_above"], vD["cross_below"])
                        row["dmax_AC_mV"] = (vC["max_ep"] - vA["max_ep"]) * 1000; row["dmin_AC_mV"] = (vC["min_ep"] - vA["min_ep"]) * 1000
                        row["dmax_AB_mV"] = (vB["max_ep"] - vA["max_ep"]) * 1000; row["dmin_AB_mV"] = (vB["min_ep"] - vA["min_ep"]) * 1000
                        row["dmax_AD_mV"] = (vD["max_ep"] - vA["max_ep"]) * 1000; row["dmin_AD_mV"] = (vD["min_ep"] - vA["min_ep"]) * 1000
                        rows.append(row); done += 1
                        if done % 60 == 0:
                            print(f"  [decomposition] {done}/{total}  {arch} {season} kwp={kwp} share={share}", flush=True)
    df = pd.DataFrame(rows)
    df["dmin_interaction_mV"] = df.dmin_AC_mV - df.dmin_AB_mV - df.dmin_AD_mV   # C - B - D + A: the interaction of the two mechanisms
    df["dmax_interaction_mV"] = df.dmax_AC_mV - df.dmax_AB_mV - df.dmax_AD_mV
    df.to_csv(outdir / "07_sweep_decomposition.csv", index=False)
    return nonconv


# =============================================================================================
# 7. COMMAND LINE
# =============================================================================================
def main():
    ap = argparse.ArgumentParser(description="The model of the thesis: Scenario 1, Scenario 2 and the decomposition of Scenario 2.")
    ap.add_argument("--smoke", action="store_true", help="one combination with 5 runs, 4 grid points and their decomposition; writes smoke/")
    ap.add_argument("--full", action="store_true", help="Scenario 1 (36 x 1000 runs) and Scenario 2 (2772 grid points); writes output/")
    ap.add_argument("--decomposition", action="store_true", help="the four-state decomposition of the 2772 grid points; writes output/")
    a = ap.parse_args()
    if not (a.smoke or a.full or a.decomposition):
        ap.error("choose --smoke, --full or --decomposition")
    outdir = HERE.parent / ("smoke" if a.smoke else "output"); outdir.mkdir(exist_ok=True)
    t0 = time.time()
    if a.smoke:
        cells, n_runs = [("rural", "high", "summer")], 5
        grid = dict(archs=["rural"], seasons=["summer"], kwps=[6, 7], shares=[0.35, 0.45], epss=[-0.35])
    else:
        cells = [(x, d, w) for x in ARCHETYPES for d in DEMAND_LEVELS for w in WEEK_KEYS]; n_runs = CFG.sampling.n_runs_per_cell
        grid = dict(archs=list(ARCHETYPES), seasons=SEASONS, kwps=KWPS, shares=SHARES, epss=EPS_GRID)
    nonconv = []
    if a.smoke or a.full:
        nonconv += run_part1(CFG, cells, n_runs, outdir)
        nonconv += run_part2(CFG, outdir=outdir, **grid)
    if a.smoke or a.decomposition:
        nonconv += run_decomposition(CFG, outdir=outdir, **grid)
    print(f"done in {time.time() - t0:.0f} s -> {outdir}; power flows that did not converge: {len(nonconv)}")


if __name__ == "__main__":
    main()
