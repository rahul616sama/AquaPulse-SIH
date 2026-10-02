import time
import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import chirp, spectrogram, correlate

st.set_page_config(
    page_title="AquaPulse",
    page_icon="🌊",
    layout="wide"
)

FS_DEFAULT = 10_000_000

# ------------------------------------------------------------
# VISUAL STYLE
# ------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.3rem; padding-bottom: 2.5rem;}
.main-title {font-size: 3.0rem; font-weight: 850; line-height: 1; margin-bottom: .25rem;}
.subtitle {font-size: 1.04rem; opacity: .78; margin-bottom: .15rem;}
.kicker {font-size: .78rem; font-weight: 800; letter-spacing: .08em; opacity: .68;}
.ap-card {
    border: 1px solid rgba(120,120,120,.24);
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 12px;
    background: rgba(127,127,127,.035);
}
.ap-good {border-left: 4px solid #2e8b57;}
.ap-warn {border-left: 4px solid #d99000;}
.ap-bad {border-left: 4px solid #c0392b;}
.small {font-size: .88rem; opacity: .78;}
.stage {
    padding: 10px 14px;
    border: 1px solid rgba(50,160,200,.35);
    border-radius: 14px;
    background: rgba(50,160,200,.07);
    margin: 8px 0 14px 0;
}
.chain-note {
    padding: 12px 14px;
    border-radius: 12px;
    background: rgba(127,127,127,.07);
    border: 1px solid rgba(127,127,127,.18);
}
div[data-testid="stMetric"] {
    border: 1px solid rgba(127,127,127,.18);
    border-radius: 14px;
    padding: 10px 12px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# PROFILES / DEMO SCENARIOS
# ------------------------------------------------------------
PROFILE_MAP = {
    "Long-range narrowband": {"fc_khz": 150.0, "bw_khz": 100.0, "max_vpk": 28.0, "max_duty": 12.0, "z_ohm": 70.0},
    "Mid-band imaging": {"fc_khz": 300.0, "bw_khz": 180.0, "max_vpk": 24.0, "max_duty": 10.0, "z_ohm": 60.0},
    "High-resolution band": {"fc_khz": 430.0, "bw_khz": 120.0, "max_vpk": 18.0, "max_duty": 8.0, "z_ohm": 50.0},
}

MISSION_WEIGHTS = {
    "High Resolution": (0.20, 0.60, 0.20),
    "Balanced": (0.35, 0.35, 0.30),
    "Long Range": (0.60, 0.20, 0.20),
    "Eco": (0.20, 0.15, 0.65),
}

DEMO_STAGES = [
    ("Balanced Mission", {
        "temperature": 12.0, "salinity": 35.0, "depth": 100, "ph": 8.0, "turbidity": 10,
        "sea": 2, "mission": "Balanced", "req_range": 120, "req_res": 0.20,
        "battery_wh": 120.0, "reserve_wh": 30.0, "pings_remaining": 20000,
        "profile": "Mid-band imaging", "fault_mode": "Normal load",
    }, "AquaPulse balances modeled range margin, resolution and electrical energy."),
    ("High Resolution", {
        "temperature": 24.0, "salinity": 35.0, "depth": 40, "ph": 8.05, "turbidity": 8,
        "sea": 1, "mission": "High Resolution", "req_range": 80, "req_res": 0.08,
        "battery_wh": 180.0, "reserve_wh": 30.0, "pings_remaining": 18000,
        "profile": "High-resolution band", "fault_mode": "Normal load",
    }, "The optimizer prioritizes wider bandwidth while staying inside the selected transducer band."),
    ("Long Range", {
        "temperature": 8.0, "salinity": 35.0, "depth": 600, "ph": 8.0, "turbidity": 15,
        "sea": 2, "mission": "Long Range", "req_range": 200, "req_res": 0.50,
        "battery_wh": 220.0, "reserve_wh": 40.0, "pings_remaining": 20000,
        "profile": "Long-range narrowband", "fault_mode": "Normal load",
    }, "The optimizer moves toward the lower usable band and prioritizes positive modeled range margin."),
    ("Low Battery", {
        "temperature": 12.0, "salinity": 35.0, "depth": 100, "ph": 8.0, "turbidity": 12,
        "sea": 2, "mission": "Eco", "req_range": 100, "req_res": 0.35,
        "battery_wh": 20.0, "reserve_wh": 8.0, "pings_remaining": 40000,
        "profile": "Mid-band imaging", "fault_mode": "Normal load",
    }, "Energy receives the largest score penalty, so lower-energy valid pings are preferred."),
    ("Load Fault", {
        "temperature": 12.0, "salinity": 35.0, "depth": 100, "ph": 8.0, "turbidity": 10,
        "sea": 2, "mission": "Balanced", "req_range": 120, "req_res": 0.20,
        "battery_wh": 120.0, "reserve_wh": 30.0, "pings_remaining": 20000,
        "profile": "Mid-band imaging", "fault_mode": "Impedance mismatch",
    }, "V/I feedback detects an abnormal load and flags the next ping for protection."),
    ("Recovery", {
        "temperature": 12.0, "salinity": 35.0, "depth": 100, "ph": 8.0, "turbidity": 10,
        "sea": 2, "mission": "Balanced", "req_range": 120, "req_res": 0.20,
        "battery_wh": 120.0, "reserve_wh": 30.0, "pings_remaining": 20000,
        "profile": "Mid-band imaging", "fault_mode": "Normal load",
    }, "Normal load is restored and the transmitter returns to a valid adaptive operating point."),
]

DEFAULTS = {
    "temperature": 12.0,
    "salinity": 35.0,
    "depth": 100,
    "ph": 8.0,
    "turbidity": 10,
    "sea": 2,
    "mission": "Balanced",
    "req_range": 120,
    "req_res": 0.20,
    "battery_wh": 120.0,
    "reserve_wh": 30.0,
    "pings_remaining": 20000,
    "profile": "Mid-band imaging",
    "max_vpk": 24.0,
    "max_duty": 10.0,
    "dac_msps": 10.0,
    "dac_bits": 10,
    "driver_eff": 82.0,
    "z_ohm": 60.0,
    "ping_rate": 5.0,
    "sl": 190.0,
    "nl": 70.0,
    "ts": -25.0,
    "di": 12.0,
    "dt": 10.0,
    "pre_eq": True,
    "stability": True,
    "fault_mode": "Normal load",
    "demo_running": False,
    "demo_stage": 0,
    "scope_running": True,
    "last_choice": None,
    "last_scope": None,
    "pending_scenario": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

def apply_profile(name):
    p = PROFILE_MAP[name]
    st.session_state.max_vpk = p["max_vpk"]
    st.session_state.max_duty = p["max_duty"]
    st.session_state.z_ohm = p["z_ohm"]

def apply_scenario(values):
    for k, v in values.items():
        st.session_state[k] = v
    if "profile" in values and values["profile"] in PROFILE_MAP:
        apply_profile(values["profile"])

# Automatic demo stages are queued at the end of a run, then applied
# before any widgets are instantiated on the next rerun.
if st.session_state.pending_scenario is not None:
    queued = dict(st.session_state.pending_scenario)
    st.session_state.pending_scenario = None
    apply_scenario(queued)

# ------------------------------------------------------------
# PHYSICS / DECISION MODEL
# ------------------------------------------------------------
def clamp(x, low, high):
    return max(low, min(high, x))

def sound_speed_mackenzie(T, S, D):
    return (
        1448.96
        + 4.591 * T
        - 5.304e-2 * T * T
        + 2.374e-4 * T * T * T
        + 1.340 * (S - 35)
        + 1.630e-2 * D
        + 1.675e-7 * D * D
        - 1.025e-2 * T * (S - 35)
        - 7.139e-13 * T * D * D * D
    )

def absorption_db_per_km(f_khz, T, S, depth_m, pH):
    # Lightweight decision-model approximation (Ainslie/McColm-style structure).
    depth_km = depth_m / 1000.0
    f1 = 0.78 * math.sqrt(max(S, 1.0) / 35.0) * math.exp(T / 26.0)
    f2 = 42.0 * math.exp(T / 17.0)
    A = 0.106 * math.exp((pH - 8.0) / 0.56)
    B = 0.52 * (1.0 + T / 43.0) * (S / 35.0) * math.exp(-depth_km / 6.0)
    C = 0.00049 * math.exp(-(T / 27.0 + depth_km / 17.0))
    f2sq = f_khz * f_khz
    return (
        A * f1 * f2sq / (f1 * f1 + f2sq)
        + B * f2 * f2sq / (f2 * f2 + f2sq)
        + C * f2sq
    )

def transmission_loss_db(range_m, alpha_db_km):
    return 20.0 * math.log10(max(range_m, 1.0)) + alpha_db_km * (range_m / 1000.0)

def electrical_energy_j(amp, duration_ms, max_vpk, z_ohm, efficiency):
    vrms = (amp * max_vpk) / math.sqrt(2.0)
    p_load = (vrms * vrms) / max(z_ohm, 1.0)
    p_electrical = p_load / max(efficiency, 0.05)
    return p_electrical * (duration_ms / 1000.0)

def modeled_range_margin_db(cand, env):
    tl = transmission_loss_db(env["req_range"], cand["alpha"])
    amp_db = 20.0 * math.log10(max(cand["amp"], 0.01))
    processing_gain = 10.0 * math.log10(max((cand["bw_khz"] * 1000.0) * (cand["dur_ms"] / 1000.0), 1.0))
    # Turbidity is retained only as a small secondary reverberation/scattering proxy.
    turbidity_penalty = 0.02 * env["turbidity"]
    received_index = (
        env["sl"] + amp_db - 2.0 * tl + env["ts"] - env["nl"]
        - env["sea"] - turbidity_penalty + env["di"] + processing_gain
    )
    return received_index - env["dt"]

def make_candidates(env, hw):
    fc_min = max(100.0, hw["tfc_khz"] - hw["tbw_khz"] / 2.0)
    fc_max = min(500.0, hw["tfc_khz"] + hw["tbw_khz"] / 2.0)

    fcs = [100, 150, 200, 250, 300, 350, 400, 450, 500]
    bws = [40, 80, 120, 180]
    durs = [0.5, 1.0, 2.0, 3.0]
    amps = [0.30, 0.50, 0.70, 0.90]

    available_wh = max(0.0, env["battery_wh"] - env["reserve_wh"])
    budget_j = available_wh * 3600.0 / max(env["pings_remaining"], 1)

    reasons = {
        "Transducer band": 0,
        "DAC sample quality": 0,
        "Resolution requirement": 0,
        "Range / link margin": 0,
        "Energy budget": 0,
        "Duty-cycle limit": 0,
    }

    all_candidates = []
    feasible = []
    rejected_examples = []

    for fc in fcs:
        for bw in bws:
            for dur in durs:
                for amp in amps:
                    low = fc - bw / 2.0
                    high = fc + bw / 2.0
                    cand = {
                        "fc_khz": float(fc),
                        "bw_khz": float(bw),
                        "dur_ms": float(dur),
                        "amp": float(amp),
                    }
                    all_candidates.append(cand)
                    reject_reason = None

                    if low < fc_min or high > fc_max:
                        reject_reason = "Transducer band"
                    else:
                        spc = (hw["fs_msps"] * 1000.0) / max(high, 1.0)
                        if spc < 12.0:
                            reject_reason = "DAC sample quality"
                        else:
                            duty_pct = (dur / 1000.0) * hw["ping_rate_hz"] * 100.0
                            if duty_pct > hw["max_duty"]:
                                reject_reason = "Duty-cycle limit"

                    if reject_reason is None:
                        cand["alpha"] = absorption_db_per_km(
                            fc, env["temperature"], env["salinity"], env["depth"], env["ph"]
                        )
                        cand["resolution_m"] = env["c"] / (2.0 * bw * 1000.0)
                        if cand["resolution_m"] > env["req_res"]:
                            reject_reason = "Resolution requirement"

                    if reject_reason is None:
                        cand["margin_db"] = modeled_range_margin_db(cand, env)
                        if cand["margin_db"] < 0.0:
                            reject_reason = "Range / link margin"

                    if reject_reason is None:
                        cand["energy_j"] = electrical_energy_j(
                            amp, dur, hw["max_vpk"], hw["z_ohm"], hw["efficiency"]
                        )
                        if cand["energy_j"] > budget_j:
                            reject_reason = "Energy budget"

                    if reject_reason is not None:
                        reasons[reject_reason] += 1
                        if len(rejected_examples) < 18:
                            rejected_examples.append({
                                **cand,
                                "reason": reject_reason,
                            })
                        continue

                    feasible.append(cand)

    if feasible:
        max_margin = max(max(c["margin_db"] for c in feasible), 1e-6)
        max_energy = max(max(c["energy_j"] for c in feasible), 1e-9)
        min_res = min(c["resolution_m"] for c in feasible)
        w_range, w_quality, w_energy = MISSION_WEIGHTS[env["mission"]]

        for cand in feasible:
            rhat = clamp(cand["margin_db"] / max_margin, 0.0, 1.0)
            qhat = clamp(min_res / max(cand["resolution_m"], 1e-9), 0.0, 1.0)
            ehat = clamp(cand["energy_j"] / max_energy, 0.0, 1.0)
            cand["score"] = w_range * rhat + w_quality * qhat - w_energy * ehat

        feasible.sort(key=lambda x: x["score"], reverse=True)

    return {
        "all": all_candidates,
        "feasible": feasible,
        "reasons": reasons,
        "budget_j": budget_j,
        "rejected_examples": rejected_examples,
        "band": (fc_min, fc_max),
    }

def enrich_choice(cand, mission):
    if cand is None:
        return None
    out = dict(cand)
    out["waveform"] = "Phase-Coded Pulse" if mission == "Eco" else "LFM Chirp"
    out["window"] = (
        "Hamming" if mission == "Long Range"
        else "Blackman" if mission == "High Resolution"
        else "Hann"
    )
    return out

def generate_waveform(choice, fs_hz):
    if choice is None:
        return np.array([]), np.array([])
    duration_s = choice["dur_ms"] / 1000.0
    n = max(16, int(duration_s * fs_hz))
    t = np.arange(n) / fs_hz
    f0 = (choice["fc_khz"] - choice["bw_khz"] / 2.0) * 1000.0
    f1 = (choice["fc_khz"] + choice["bw_khz"] / 2.0) * 1000.0

    if choice["waveform"] == "LFM Chirp":
        sig = chirp(t, f0=f0, f1=f1, t1=duration_s, method="linear")
    else:
        fc = choice["fc_khz"] * 1000.0
        code = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=float)
        sig = np.zeros_like(t)
        edges = np.linspace(0, n, len(code) + 1, dtype=int)
        for i, bit in enumerate(code):
            a, b = edges[i], edges[i + 1]
            sig[a:b] = bit * np.sin(2 * np.pi * fc * t[a:b])

    if choice["window"] == "Blackman":
        window = np.blackman(n)
    elif choice["window"] == "Hamming":
        window = np.hamming(n)
    else:
        window = np.hanning(n)

    return t, sig * window * choice["amp"]

def pulse_compression_metrics(signal, fs_hz, choice):
    if choice is None or signal.size == 0:
        return None
    # Deterministic light noise keeps the demo repeatable.
    seed = int(choice["fc_khz"] * 10 + choice["bw_khz"])
    rng = np.random.default_rng(seed)
    loopback = signal * 0.92 + rng.normal(0.0, 0.015, signal.size)
    corr = correlate(loopback, signal, mode="full", method="fft")
    mag = np.abs(corr)
    mag /= max(np.max(mag), 1e-12)
    lags = np.arange(-signal.size + 1, signal.size) / fs_hz

    peak_idx = int(np.argmax(mag))
    half = 0.5
    left = peak_idx
    right = peak_idx
    while left > 0 and mag[left] >= half:
        left -= 1
    while right < len(mag) - 1 and mag[right] >= half:
        right += 1
    width_s = max((right - left) / fs_hz, 1.0 / fs_hz)

    guard = max(8, int((right - left) * 1.5))
    mask = np.ones_like(mag, dtype=bool)
    mask[max(0, peak_idx - guard):min(len(mag), peak_idx + guard + 1)] = False
    sidelobe = np.max(mag[mask]) if np.any(mask) else 1e-6
    pslr_db = 20.0 * np.log10(max(sidelobe, 1e-9))

    bt = choice["bw_khz"] * 1000.0 * (choice["dur_ms"] / 1000.0)
    return {
        "lags": lags,
        "mag": mag,
        "bt": bt,
        "width_us": width_s * 1e6,
        "pslr_db": pslr_db,
    }

def baseline_metrics(env, hw):
    baseline = {
        "fc_khz": hw["tfc_khz"],
        "bw_khz": min(120.0, hw["tbw_khz"]),
        "dur_ms": 2.0,
        "amp": 0.80,
    }
    low = baseline["fc_khz"] - baseline["bw_khz"] / 2.0
    high = baseline["fc_khz"] + baseline["bw_khz"] / 2.0
    band_low = hw["tfc_khz"] - hw["tbw_khz"] / 2.0
    band_high = hw["tfc_khz"] + hw["tbw_khz"] / 2.0
    baseline["valid_transducer"] = low >= band_low and high <= band_high
    baseline["alpha"] = absorption_db_per_km(
        baseline["fc_khz"], env["temperature"], env["salinity"], env["depth"], env["ph"]
    )
    baseline["resolution_m"] = env["c"] / (2.0 * baseline["bw_khz"] * 1000.0)
    baseline["margin_db"] = modeled_range_margin_db(baseline, env)
    baseline["energy_j"] = electrical_energy_j(
        baseline["amp"], baseline["dur_ms"], hw["max_vpk"], hw["z_ohm"], hw["efficiency"]
    )
    return baseline

def fault_status(mode, expected_z):
    if mode == "Normal load":
        return {"health": "OK", "measured_z": expected_z * 1.03, "amp_error": 2.1, "action": "Transmit allowed"}
    if mode == "Open load":
        return {"health": "OPEN LOAD", "measured_z": 1e6, "amp_error": 100.0, "action": "TX INHIBIT"}
    if mode == "Impedance mismatch":
        return {"health": "MISMATCH", "measured_z": expected_z * 2.4, "amp_error": 17.5, "action": "Flag + inhibit next ping"}
    if mode == "Overcurrent":
        return {"health": "OVERCURRENT", "measured_z": expected_z * 0.35, "amp_error": 28.0, "action": "DRIVER OFF"}
    return {"health": "SENSOR FALLBACK", "measured_z": expected_z, "amp_error": 3.0, "action": "Use last-valid/safe inputs"}

# ------------------------------------------------------------
# HEADER / DEMO CONTROLS
# ------------------------------------------------------------
st.markdown('<div class="kicker">SMART INDIA HACKATHON · ADAPTIVE SONAR TRANSMITTER</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">AquaPulse</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Mission-Aware · Energy-Budgeted · Transducer-Aware · Self-Verifying</div>',
    unsafe_allow_html=True
)
st.caption("Digital testbed for a bench-scale software-defined AUV sonar transmitter. Acoustic range values below are model outputs, not tank-validated measurements.")

d1, d2, d3, d4 = st.columns([1.3, 1, 1, 1])
if d1.button("▶ Start Judge Demo", use_container_width=True):
    st.session_state.demo_running = True
    st.session_state.demo_stage = 0
    apply_scenario(DEMO_STAGES[0][1])
    st.rerun()
if d2.button("⏸ Pause Demo", use_container_width=True):
    st.session_state.demo_running = False
if d3.button("⏭ Next Stage", use_container_width=True):
    st.session_state.demo_running = False
    st.session_state.demo_stage = (st.session_state.demo_stage + 1) % len(DEMO_STAGES)
    apply_scenario(DEMO_STAGES[st.session_state.demo_stage][1])
    st.rerun()
if d4.button("↻ Reset", use_container_width=True):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    apply_profile(st.session_state.profile)
    st.rerun()

stage_name, _, stage_note = DEMO_STAGES[st.session_state.demo_stage]
st.markdown(
    f'<div class="stage"><b>Demo stage:</b> {stage_name}<br><span class="small">{stage_note}</span></div>',
    unsafe_allow_html=True
)

st.subheader("Quick Mission Scenarios")
q1, q2, q3, q4 = st.columns(4)
if q1.button("High Resolution Reef", use_container_width=True):
    apply_scenario(DEMO_STAGES[1][1]); st.session_state.demo_stage = 1; st.rerun()
if q2.button("Long Range Survey", use_container_width=True):
    apply_scenario(DEMO_STAGES[2][1]); st.session_state.demo_stage = 2; st.rerun()
if q3.button("Low Battery Return", use_container_width=True):
    apply_scenario(DEMO_STAGES[3][1]); st.session_state.demo_stage = 3; st.rerun()
if q4.button("Load Fault", use_container_width=True):
    apply_scenario(DEMO_STAGES[4][1]); st.session_state.demo_stage = 4; st.rerun()

# ------------------------------------------------------------
# SIDEBAR INPUTS
# ------------------------------------------------------------
st.sidebar.header("Mission & Environment")
temperature = st.sidebar.slider("Temperature (°C)", -2.0, 35.0, step=0.5, key="temperature")
salinity = st.sidebar.slider("Salinity (PSU)", 25.0, 40.0, step=0.5, key="salinity")
depth = st.sidebar.slider("Depth (m)", 0, 2000, step=10, key="depth")
ph = st.sidebar.slider("pH (model input)", 7.7, 8.3, step=0.05, key="ph")
turbidity = st.sidebar.slider("Turbidity proxy (NTU)", 0, 100, step=1, key="turbidity")
sea = st.sidebar.slider("Sea-state / bubble penalty (dB)", 0, 20, step=1, key="sea")
mission = st.sidebar.selectbox("Mission Mode", list(MISSION_WEIGHTS.keys()), key="mission")
req_range = st.sidebar.slider("Required range (m)", 10, 500, step=5, key="req_range")
req_res = st.sidebar.slider("Required resolution (m)", 0.02, 1.00, step=0.01, key="req_res")

st.sidebar.header("Energy Budget")
battery_wh = st.sidebar.slider("Remaining energy (Wh)", 5.0, 500.0, step=5.0, key="battery_wh")
reserve_wh = st.sidebar.slider("Reserved energy (Wh)", 0.0, 150.0, step=5.0, key="reserve_wh")
pings_remaining = st.sidebar.slider("Planned pings remaining", 100, 100000, step=100, key="pings_remaining")

st.sidebar.header("Transmitter Hardware")
old_profile = st.session_state.get("_profile_shadow", st.session_state.profile)
profile = st.sidebar.selectbox("Transducer profile", list(PROFILE_MAP.keys()), key="profile")
if profile != old_profile:
    apply_profile(profile)
st.session_state["_profile_shadow"] = profile

pdef = PROFILE_MAP[profile]
st.sidebar.caption(f"Profile band: {pdef['fc_khz'] - pdef['bw_khz']/2:.0f}–{pdef['fc_khz'] + pdef['bw_khz']/2:.0f} kHz")
max_vpk = st.sidebar.number_input("Max drive (Vpk)", 1.0, 100.0, key="max_vpk")
max_duty = st.sidebar.number_input("Max duty cycle (%)", 1.0, 50.0, key="max_duty")
dac_msps = st.sidebar.number_input("DAC rate (MS/s)", 2.0, 100.0, step=1.0, key="dac_msps")
dac_bits = st.sidebar.number_input("DAC bits", 8, 16, step=1, key="dac_bits")
driver_eff = st.sidebar.slider("Driver efficiency (%)", 20.0, 98.0, step=1.0, key="driver_eff")
z_ohm = st.sidebar.number_input("Expected load impedance (Ω)", 5.0, 500.0, step=1.0, key="z_ohm")
ping_rate = st.sidebar.slider("Ping rate (Hz)", 1.0, 20.0, step=1.0, key="ping_rate")
pre_eq = st.sidebar.checkbox("Pre-equalization (simulation)", key="pre_eq")
stability = st.sidebar.checkbox("EMA / hysteresis behavior", key="stability")

st.sidebar.header("Fault Injection")
fault_mode = st.sidebar.selectbox(
    "Electrical / sensor condition",
    ["Normal load", "Open load", "Impedance mismatch", "Overcurrent", "Sensor failure"],
    key="fault_mode"
)

with st.sidebar.expander("Advanced acoustic scenario assumptions"):
    sl = st.number_input("Source level assumption (dB)", 140.0, 230.0, key="sl")
    nl = st.number_input("Noise level assumption (dB)", 20.0, 120.0, key="nl")
    ts = st.number_input("Target strength assumption (dB)", -80.0, 20.0, key="ts")
    di = st.number_input("Directivity gain assumption (dB)", 0.0, 40.0, key="di")
    dt = st.number_input("Detection threshold (dB)", 0.0, 30.0, key="dt")
    st.caption("These are scenario assumptions for the transmitter decision model, not onboard measurements.")

# ------------------------------------------------------------
# RUN OPTIMIZER
# ------------------------------------------------------------
c = sound_speed_mackenzie(temperature, salinity, depth)
env = {
    "temperature": temperature, "salinity": salinity, "depth": depth, "ph": ph,
    "turbidity": turbidity, "sea": sea, "mission": mission,
    "req_range": req_range, "req_res": req_res,
    "battery_wh": battery_wh, "reserve_wh": reserve_wh, "pings_remaining": pings_remaining,
    "sl": st.session_state.sl, "nl": st.session_state.nl, "ts": st.session_state.ts,
    "di": st.session_state.di, "dt": st.session_state.dt, "c": c,
}
hw = {
    "tfc_khz": pdef["fc_khz"], "tbw_khz": pdef["bw_khz"],
    "max_vpk": max_vpk, "max_duty": max_duty,
    "fs_msps": dac_msps, "bits": dac_bits,
    "efficiency": driver_eff / 100.0, "z_ohm": z_ohm,
    "ping_rate_hz": ping_rate,
}
result = make_candidates(env, hw)
choice = enrich_choice(result["feasible"][0], mission) if result["feasible"] else None

# Simple hysteresis: retain a nearby prior valid choice only if it remains physically valid and useful.
if stability and choice and st.session_state.last_choice:
    prev = st.session_state.last_choice
    if abs(choice["fc_khz"] - prev["fc_khz"]) <= 50 and prev.get("margin_db", -99) >= 0:
        same_band = (
            prev["fc_khz"] - prev["bw_khz"]/2 >= result["band"][0]
            and prev["fc_khz"] + prev["bw_khz"]/2 <= result["band"][1]
        )
        if same_band:
            choice = prev
if choice:
    st.session_state.last_choice = choice

fault = fault_status(fault_mode, z_ohm)
hard_inhibit = fault_mode in ("Open load", "Overcurrent")
tx_ok = choice is not None and not hard_inhibit

# ------------------------------------------------------------
# TOP STATUS / SELECTED PING
# ------------------------------------------------------------
left, right = st.columns([1.05, 1.45])

with left:
    st.markdown("### Mission Snapshot")
    a, b = st.columns(2)
    a.metric("Required Range", f"{req_range} m")
    b.metric("Required Resolution", f"{req_res:.2f} m")
    a.metric("Sound Speed", f"{c:.1f} m/s")
    alpha_ref = absorption_db_per_km(pdef["fc_khz"], temperature, salinity, depth, ph)
    b.metric("Absorption @ profile fc", f"{alpha_ref:.1f} dB/km")
    st.caption("Turbidity is treated only as a secondary scattering/reverberation proxy; it does not directly choose the transmit band.")

with right:
    st.markdown("### Selected Ping")
    if not tx_ok:
        if choice is None:
            st.error("TX INHIBIT — no candidate satisfies all mission, energy and hardware constraints.")
        else:
            st.error(f"{fault['action']} — {fault['health']}.")
    else:
        st.success("FEASIBLE — selected after hard-constraint rejection and mission-weighted scoring.")

    if choice:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Centre Frequency", f"{choice['fc_khz']:.0f} kHz")
        r2.metric("Bandwidth", f"{choice['bw_khz']:.0f} kHz")
        r3.metric("Pulse", f"{choice['dur_ms']:.2f} ms")
        r4.metric("Drive", f"{choice['amp']*100:.0f}%")
        r5, r6, r7, r8 = st.columns(4)
        r5.metric("Waveform", choice["waveform"])
        r6.metric("Resolution", f"{choice['resolution_m']*100:.1f} cm")
        r7.metric("Energy / Ping", f"{choice['energy_j']*1000:.2f} mJ")
        r8.metric("Modeled Range Margin", f"{choice['margin_db']:+.1f} dB")
    else:
        st.info("Try relaxing range/resolution, choosing a compatible transducer, or increasing hardware capability.")

# ------------------------------------------------------------
# WHY THIS WAVEFORM?
# ------------------------------------------------------------
st.markdown("## Why this waveform?")
why1, why2 = st.columns([1.2, 1])

with why1:
    if choice:
        band_low, band_high = result["band"]
        st.markdown(
            f"""
<div class="ap-card ap-good">
<b>Selected candidate</b><br>
{choice['fc_khz']:.0f} kHz centre · {choice['bw_khz']:.0f} kHz bandwidth ·
{choice['dur_ms']:.2f} ms · {choice['amp']*100:.0f}% drive<br><br>
<span class="small">
✓ Inside transducer band ({band_low:.0f}–{band_high:.0f} kHz)<br>
✓ Meets {req_res:.2f} m resolution requirement<br>
✓ Positive modeled margin at {req_range} m<br>
✓ Below the per-ping electrical energy budget<br>
✓ Compatible with {dac_msps:.0f} MS/s DAC sample-quality rule
</span>
</div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="ap-card ap-bad"><b>No valid waveform</b><br><span class="small">Every candidate was rejected by at least one hard constraint.</span></div>',
            unsafe_allow_html=True
        )

with why2:
    st.markdown("**Hard-constraint rejection counts**")
    total = max(len(result["all"]), 1)
    for name, count in result["reasons"].items():
        st.write(f"{name}: **{count}** ({100*count/total:.1f}%)")

st.markdown("**Example rejected candidates**")
examples = result["rejected_examples"][:6]
if examples:
    cols = st.columns(min(3, len(examples)))
    for i, ex in enumerate(examples):
        with cols[i % len(cols)]:
            st.markdown(
                f"""
<div class="ap-card ap-warn">
<b>{ex['reason']}</b><br>
<span class="small">{ex['fc_khz']:.0f} kHz · {ex['bw_khz']:.0f} kHz BW · {ex['dur_ms']:.1f} ms · {ex['amp']*100:.0f}%</span>
</div>
                """,
                unsafe_allow_html=True
            )

# ------------------------------------------------------------
# FIXED VS ADAPTIVE
# ------------------------------------------------------------
st.markdown("## Fixed Sonar vs AquaPulse")
baseline = baseline_metrics(env, hw)
available_j = max(0.0, battery_wh - reserve_wh) * 3600.0
adaptive_pings = int(available_j / max(choice["energy_j"], 1e-9)) if choice else 0
baseline_pings = int(available_j / max(baseline["energy_j"], 1e-9))

f1, f2 = st.columns(2)
with f1:
    st.markdown("### Fixed Baseline")
    b1, b2, b3 = st.columns(3)
    b1.metric("Energy / Ping", f"{baseline['energy_j']*1000:.2f} mJ")
    b2.metric("Resolution", f"{baseline['resolution_m']*100:.1f} cm")
    b3.metric("Modeled Margin", f"{baseline['margin_db']:+.1f} dB")
    st.write(f"Transducer-valid: **{'Yes' if baseline['valid_transducer'] else 'No'}**")
    st.write(f"Estimated electrical pings from non-reserve energy: **{baseline_pings:,}**")

with f2:
    st.markdown("### AquaPulse Adaptive")
    if choice:
        a1, a2, a3 = st.columns(3)
        a1.metric("Energy / Ping", f"{choice['energy_j']*1000:.2f} mJ")
        a2.metric("Resolution", f"{choice['resolution_m']*100:.1f} cm")
        a3.metric("Modeled Margin", f"{choice['margin_db']:+.1f} dB")
        st.write("Transducer-valid: **Yes**")
        st.write(f"Estimated electrical pings from non-reserve energy: **{adaptive_pings:,}**")
    else:
        st.write("No valid adaptive candidate.")

st.caption("Comparison is an electrical/acoustic decision simulation. It is not a measured underwater endurance or range result.")

# ------------------------------------------------------------
# BATTERY FORECAST
# ------------------------------------------------------------
st.markdown("## Mission Energy Forecast")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Per-Ping Budget", f"{result['budget_j']*1000:.2f} mJ")
m2.metric("Selected Energy", f"{choice['energy_j']*1000:.2f} mJ" if choice else "—")
m3.metric("Planned Pings", f"{pings_remaining:,}")
m4.metric("Reserve", f"{reserve_wh:.0f} Wh")
if choice:
    budget_use = 100.0 * choice["energy_j"] / max(result["budget_j"], 1e-9)
    st.progress(min(budget_use / 100.0, 1.0), text=f"Selected ping uses {budget_use:.3f}% of the current per-ping budget")
    if battery_wh <= reserve_wh:
        st.warning("No non-reserve energy remains. TX should be inhibited by mission policy.")

# ------------------------------------------------------------
# SIGNAL VISUALIZATION
# ------------------------------------------------------------
st.markdown("## Signal Lab")
sc1, sc2, sc3 = st.columns(3)
if sc1.button("▶ Run Scope", use_container_width=True):
    st.session_state.scope_running = True
if sc2.button("⏸ Pause Scope", use_container_width=True):
    st.session_state.scope_running = False
single_ping = sc3.button("● Single Ping", use_container_width=True)

fs_hz = dac_msps * 1_000_000.0
if choice and (st.session_state.scope_running or single_ping or st.session_state.last_scope is None):
    t, signal = generate_waveform(choice, fs_hz)
    st.session_state.last_scope = (t, signal, dict(choice))
elif st.session_state.last_scope is not None:
    t, signal, scope_choice = st.session_state.last_scope
else:
    t, signal = np.array([]), np.array([])

tabs = st.tabs(["Waveform", "Spectrum", "Spectrogram", "Transducer Response", "Pulse Compression"])

with tabs[0]:
    fig = go.Figure()
    if signal.size:
        stride = max(1, len(signal) // 6000)
        fig.add_trace(go.Scatter(x=t[::stride]*1000, y=signal[::stride], mode="lines", name="Electrical waveform"))
    fig.update_layout(height=360, xaxis_title="Time (ms)", yaxis_title="Normalized amplitude", margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    fig = go.Figure()
    if signal.size:
        fft_data = np.fft.rfft(signal)
        freq = np.fft.rfftfreq(len(signal), 1/fs_hz)
        mag = np.abs(fft_data)
        mask = freq <= 700000
        fig.add_trace(go.Scatter(x=freq[mask]/1000, y=mag[mask], mode="lines"))
    fig.update_layout(height=360, xaxis_title="Frequency (kHz)", yaxis_title="Magnitude", margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    if signal.size >= 128:
        nperseg = min(1024, max(128, len(signal)//20))
        noverlap = int(nperseg * 0.75)
        f, tspec, sxx = spectrogram(signal, fs_hz, nperseg=nperseg, noverlap=noverlap)
        mask = f <= 600000
        fig = go.Figure(data=go.Heatmap(
            x=tspec*1000, y=f[mask]/1000, z=10*np.log10(sxx[mask] + 1e-12),
            colorscale="Viridis"
        ))
        fig.update_layout(height=390, xaxis_title="Time (ms)", yaxis_title="Frequency (kHz)", margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No waveform available.")

with tabs[3]:
    faxis = np.linspace(100, 500, 500)
    x = (faxis - pdef["fc_khz"]) / max(pdef["bw_khz"]/2, 1)
    response = np.exp(-1.4*x*x)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=faxis, y=response, mode="lines", name="Illustrative load response"))
    if choice:
        fig.add_vrect(
            x0=choice["fc_khz"]-choice["bw_khz"]/2,
            x1=choice["fc_khz"]+choice["bw_khz"]/2,
            opacity=0.15, line_width=0, annotation_text="Selected band"
        )
    fig.update_layout(height=360, xaxis_title="Frequency (kHz)", yaxis_title="Relative response", margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Illustrative electrical response only. Replace with measured transducer transfer function for real calibration.")

with tabs[4]:
    pcm = pulse_compression_metrics(signal, fs_hz, choice) if signal.size and choice else None
    if pcm:
        fig = go.Figure()
        center_mask = np.abs(pcm["lags"]) <= max(choice["dur_ms"]/1000.0 * 0.08, 150e-6)
        fig.add_trace(go.Scatter(x=pcm["lags"][center_mask]*1e6, y=pcm["mag"][center_mask], mode="lines"))
        fig.update_layout(height=360, xaxis_title="Delay (µs)", yaxis_title="Normalized correlation", margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Time-Bandwidth Product", f"{pcm['bt']:.1f}")
        p2.metric("Theoretical Resolution", f"{choice['resolution_m']*100:.1f} cm")
        p3.metric("Sim. Compressed Width", f"{pcm['width_us']:.2f} µs")
        p4.metric("Sim. PSLR", f"{pcm['pslr_db']:.1f} dB")
        st.caption("Matched-filter metrics are simulated from the generated/loopback signal and are not tank measurements.")
    else:
        st.info("No valid ping available for pulse-compression analysis.")

# ------------------------------------------------------------
# CANDIDATE EXPLORER
# ------------------------------------------------------------
st.markdown("## Candidate Explorer")
if result["feasible"]:
    topn = result["feasible"][:120]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[c["energy_j"]*1000 for c in topn],
        y=[c["score"] for c in topn],
        mode="markers",
        marker=dict(
            size=[11 if i == 0 else 7 for i in range(len(topn))],
            color=[c["margin_db"] for c in topn],
            colorscale="Turbo",
            showscale=True,
            colorbar=dict(title="Margin dB"),
        ),
        text=[
            f"{c['fc_khz']:.0f} kHz | {c['bw_khz']:.0f} kHz BW | {c['dur_ms']:.1f} ms | {c['amp']*100:.0f}%"
            for c in topn
        ],
        hovertemplate="%{text}<br>Energy=%{x:.3f} mJ<br>Score=%{y:.3f}<extra></extra>",
    ))
    fig.update_layout(height=390, xaxis_title="Electrical Energy / Ping (mJ)", yaxis_title="Optimizer Score", margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)

    inspect_labels = [
        f"#{i+1}: {c['fc_khz']:.0f} kHz · {c['bw_khz']:.0f} kHz BW · {c['dur_ms']:.1f} ms · {c['amp']*100:.0f}%"
        for i, c in enumerate(result["feasible"][:12])
    ]
    selected_label = st.selectbox("Inspect a feasible candidate", inspect_labels)
    idx = inspect_labels.index(selected_label)
    ins = result["feasible"][idx]
    st.write(
        f"Score **{ins['score']:.3f}** · Resolution **{ins['resolution_m']*100:.1f} cm** · "
        f"Modeled margin **{ins['margin_db']:+.1f} dB** · Energy **{ins['energy_j']*1000:.2f} mJ**"
    )
else:
    st.info("No feasible candidates to inspect.")

# ------------------------------------------------------------
# OUTPUT HEALTH / FAULT INJECTION
# ------------------------------------------------------------
st.markdown("## Closed-Loop Output Health")
h1, h2, h3, h4 = st.columns(4)
h1.metric("Expected Z", f"{z_ohm:.0f} Ω")
h2.metric("Measured Z", ">1 MΩ" if fault["measured_z"] >= 1e5 else f"{fault['measured_z']:.0f} Ω")
h3.metric("Amplitude Error", f"{fault['amp_error']:.1f}%")
h4.metric("Health", fault["health"])

if fault_mode == "Normal load":
    st.success(f"Action: {fault['action']}")
elif fault_mode == "Sensor failure":
    st.warning(f"Action: {fault['action']}")
else:
    st.error(f"Action: {fault['action']}")

# ------------------------------------------------------------
# INTERACTIVE HARDWARE CHAIN
# ------------------------------------------------------------
st.markdown("## Interactive Hardware Chain")
chain = ["STM32H7", "Timer / DMA", "High-Speed DAC", "Reconstruction Filter", "Low-Voltage Driver", "Transducer Load", "V/I Feedback"]
stage = st.radio("Inspect stage", chain, horizontal=True)

stage_text = {
    "STM32H7": f"Runs the candidate search between pings and prepares the next waveform buffer. Current mission: {mission}.",
    "Timer / DMA": f"Streams samples without CPU-per-sample work. Target DAC rate: {dac_msps:.1f} MS/s.",
    "High-Speed DAC": f"External DAC target: {dac_bits}-bit at {dac_msps:.1f} MS/s. At 500 kHz this is {dac_msps*1000/500:.1f} samples/cycle.",
    "Reconstruction Filter": "Removes DAC images before the driver. Final cutoff/order must be verified with the actual analog chain.",
    "Low-Voltage Driver": f"Bench-safe driver model. Maximum configured drive: {max_vpk:.1f} Vpk; efficiency assumption: {driver_eff:.0f}%.",
    "Transducer Load": f"Selected profile: {profile}. Usable band: {result['band'][0]:.0f}–{result['band'][1]:.0f} kHz; expected load ≈ {z_ohm:.0f} Ω.",
    "V/I Feedback": f"Current injected condition: {fault_mode}. Health result: {fault['health']}. Protective action: {fault['action']}.",
}
st.markdown(f'<div class="chain-note"><b>{stage}</b><br><span class="small">{stage_text[stage]}</span></div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# FEASIBILITY GATES
# ------------------------------------------------------------
st.markdown("## Feasibility Gates")
g1 = "PASS" if dac_msps >= 8 else "RISK"
g2 = "PASS" if choice else "BLOCKED"
g3 = "PASS" if result["feasible"] else "BLOCKED"
g4 = "PASS" if choice and choice["energy_j"] <= result["budget_j"] else "BLOCKED"
g5 = "PASS" if fault_mode == "Normal load" else "FAULT DETECTED"
g6 = "PASS" if choice and choice["bw_khz"] >= 40 else "—"
gcols = st.columns(6)
for col, (name, val) in zip(gcols, [
    ("G1 Digital Stream", g1), ("G2 Analog Chirp", g2), ("G3 Adaptation", g3),
    ("G4 Energy", g4), ("G5 Feedback", g5), ("G6 Compression", g6)
]):
    col.metric(name, val)

st.divider()
st.markdown("### Model Boundary")
st.info(
    "Sound speed, absorption, transmission loss, target/noise assumptions and link margin are used only as a decision simulation. "
    "The SIH bench prototype can validate waveform generation, electrical energy, load health and pulse compression. "
    "Real underwater source level, efficiency and acoustic range require a characterized transducer plus water-tank calibration."
)

st.caption("AquaPulse · Sense → Decide → Generate → Transmit → Verify")

# ------------------------------------------------------------
# AUTOMATIC JUDGE DEMO ADVANCE
# ------------------------------------------------------------
if st.session_state.demo_running:
    time.sleep(2.6)
    next_stage = st.session_state.demo_stage + 1
    if next_stage >= len(DEMO_STAGES):
        st.session_state.demo_running = False
        st.session_state.demo_stage = len(DEMO_STAGES) - 1
    else:
        st.session_state.demo_stage = next_stage
        st.session_state.pending_scenario = DEMO_STAGES[next_stage][1]
        st.rerun()
