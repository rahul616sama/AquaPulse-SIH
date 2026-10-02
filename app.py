import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import chirp

st.set_page_config(page_title="AquaPulse", page_icon="🌊", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 1.5rem;}
.ap-title {font-size: 2.8rem; font-weight: 850; line-height: 1; margin-bottom: .15rem;}
.ap-sub {font-size: 1rem; opacity: .72; margin-bottom: .4rem;}
.ap-note {
    padding: 12px 14px;
    border: 1px solid rgba(120,120,120,.22);
    border-radius: 14px;
    background: rgba(127,127,127,.035);
}
.ap-good {border-left: 4px solid #2e8b57;}
.ap-bad {border-left: 4px solid #c0392b;}
div[data-testid="stMetric"] {
    border: 1px solid rgba(127,127,127,.18);
    border-radius: 13px;
    padding: 8px 10px;
}
</style>
""", unsafe_allow_html=True)

PROFILES = {
    "Long-range": {"fc": 150.0, "bw": 100.0, "vpk": 28.0, "z": 70.0},
    "Balanced": {"fc": 300.0, "bw": 180.0, "vpk": 24.0, "z": 60.0},
    "High-resolution": {"fc": 430.0, "bw": 120.0, "vpk": 18.0, "z": 50.0},
}

WEIGHTS = {
    "High Resolution": (0.20, 0.60, 0.20),
    "Balanced": (0.35, 0.35, 0.30),
    "Long Range": (0.60, 0.20, 0.20),
    "Eco": (0.20, 0.15, 0.65),
}

DEFAULTS = {
    "mission": "Balanced",
    "req_range": 120,
    "req_res": 0.20,
    "battery": 120.0,
    "reserve": 30.0,
    "pings": 20000,
    "profile": "Balanced",
    "temperature": 12.0,
    "salinity": 35.0,
    "depth": 100,
    "turbidity": 10,
    "ph": 8.0,
    "sea": 2,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def apply_preset(name):
    if name == "High Resolution":
        vals = dict(mission="High Resolution", req_range=80, req_res=0.08, battery=180.0,
                    reserve=30.0, pings=18000, profile="High-resolution",
                    temperature=24.0, salinity=35.0, depth=40, turbidity=8)
    elif name == "Long Range":
        vals = dict(mission="Long Range", req_range=200, req_res=0.50, battery=220.0,
                    reserve=40.0, pings=20000, profile="Long-range",
                    temperature=8.0, salinity=35.0, depth=600, turbidity=15)
    elif name == "Eco":
        vals = dict(mission="Eco", req_range=100, req_res=0.35, battery=20.0,
                    reserve=8.0, pings=40000, profile="Balanced",
                    temperature=12.0, salinity=35.0, depth=100, turbidity=12)
    else:
        vals = dict(mission="Balanced", req_range=120, req_res=0.20, battery=120.0,
                    reserve=30.0, pings=20000, profile="Balanced",
                    temperature=12.0, salinity=35.0, depth=100, turbidity=10)
    for k, v in vals.items():
        st.session_state[k] = v

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def sound_speed(T, S, D):
    return (
        1448.96 + 4.591*T - 5.304e-2*T*T + 2.374e-4*T*T*T
        + 1.340*(S-35) + 1.630e-2*D + 1.675e-7*D*D
        - 1.025e-2*T*(S-35) - 7.139e-13*T*D*D*D
    )

def absorption(f_khz, T, S, Dm, pH):
    D = Dm / 1000.0
    f1 = 0.78 * math.sqrt(max(S, 1.0)/35.0) * math.exp(T/26.0)
    f2 = 42.0 * math.exp(T/17.0)
    A = 0.106 * math.exp((pH-8.0)/0.56)
    B = 0.52 * (1.0 + T/43.0) * (S/35.0) * math.exp(-D/6.0)
    C = 0.00049 * math.exp(-(T/27.0 + D/17.0))
    f2sq = f_khz * f_khz
    return A*f1*f2sq/(f1*f1 + f2sq) + B*f2*f2sq/(f2*f2 + f2sq) + C*f2sq

def energy_j(amp, dur_ms, max_vpk, z_ohm, eff=0.82):
    vrms = amp * max_vpk / math.sqrt(2.0)
    power = (vrms*vrms) / max(z_ohm, 1.0)
    return (power / max(eff, 0.05)) * (dur_ms/1000.0)

def range_margin(cand, env):
    tl = 20.0 * math.log10(max(env["range"], 1.0)) + cand["alpha"] * (env["range"]/1000.0)
    amp_db = 20.0 * math.log10(max(cand["amp"], 0.01))
    pg = 10.0 * math.log10(max(cand["bw"]*1000.0*(cand["dur"]/1000.0), 1.0))
    turbidity_penalty = 0.02 * env["turbidity"]
    received = 190.0 + amp_db - 2.0*tl - 25.0 - 70.0 - env["sea"] - turbidity_penalty + 12.0 + pg
    return received - 10.0

def optimize(env, hw):
    fcs = [100,150,200,250,300,350,400,450,500]
    bws = [40,80,120,180]
    durs = [0.5,1.0,2.0,3.0]
    amps = [0.3,0.5,0.7,0.9]

    band_lo = max(100.0, hw["fc"] - hw["bw"]/2.0)
    band_hi = min(500.0, hw["fc"] + hw["bw"]/2.0)
    budget = max(0.0, env["battery"] - env["reserve"]) * 3600.0 / max(env["pings"], 1)

    reasons = {"Transducer band": 0, "Resolution": 0, "Range margin": 0, "Energy budget": 0, "DAC sampling": 0}
    feasible = []

    for fc in fcs:
        for bw in bws:
            for dur in durs:
                for amp in amps:
                    lo, hi = fc-bw/2.0, fc+bw/2.0
                    if lo < band_lo or hi > band_hi:
                        reasons["Transducer band"] += 1
                        continue
                    if 10_000.0 / max(hi,1.0) < 12.0:
                        reasons["DAC sampling"] += 1
                        continue

                    c = {"fc":float(fc), "bw":float(bw), "dur":float(dur), "amp":float(amp)}
                    c["alpha"] = absorption(fc, env["T"], env["S"], env["D"], env["pH"])
                    c["resolution"] = env["c"] / (2.0*bw*1000.0)
                    if c["resolution"] > env["resolution"]:
                        reasons["Resolution"] += 1
                        continue

                    c["margin"] = range_margin(c, env)
                    if c["margin"] < 0:
                        reasons["Range margin"] += 1
                        continue

                    c["energy"] = energy_j(amp, dur, hw["vpk"], hw["z"])
                    if c["energy"] > budget:
                        reasons["Energy budget"] += 1
                        continue

                    feasible.append(c)

    if not feasible:
        return None, budget, reasons, (band_lo, band_hi), []

    max_margin = max(c["margin"] for c in feasible)
    max_energy = max(c["energy"] for c in feasible)
    min_res = min(c["resolution"] for c in feasible)
    wr, wq, we = WEIGHTS[env["mission"]]

    for c in feasible:
        rhat = clamp(c["margin"]/max(max_margin,1e-9), 0, 1)
        qhat = clamp(min_res/max(c["resolution"],1e-9), 0, 1)
        ehat = clamp(c["energy"]/max(max_energy,1e-9), 0, 1)
        c["score"] = wr*rhat + wq*qhat - we*ehat

    feasible.sort(key=lambda x: x["score"], reverse=True)
    return feasible[0], budget, reasons, (band_lo, band_hi), feasible

def make_wave(choice):
    if choice is None:
        return np.array([]), np.array([])
    fs = 10_000_000
    n = int(choice["dur"]/1000.0 * fs)
    t = np.arange(n)/fs
    f0 = (choice["fc"] - choice["bw"]/2.0)*1000.0
    f1 = (choice["fc"] + choice["bw"]/2.0)*1000.0
    sig = chirp(t, f0=f0, f1=f1, t1=choice["dur"]/1000.0, method="linear")
    sig *= np.hanning(len(sig))
    sig *= choice["amp"]
    return t, sig

st.markdown('<div class="ap-title">AquaPulse</div>', unsafe_allow_html=True)
st.markdown('<div class="ap-sub">A clean interactive demo of an adaptive, energy-aware sonar transmitter for AUVs.</div>', unsafe_allow_html=True)

p1, p2, p3, p4 = st.columns(4)
if p1.button("Balanced", use_container_width=True):
    apply_preset("Balanced"); st.rerun()
if p2.button("High Resolution", use_container_width=True):
    apply_preset("High Resolution"); st.rerun()
if p3.button("Long Range", use_container_width=True):
    apply_preset("Long Range"); st.rerun()
if p4.button("Eco / Low Battery", use_container_width=True):
    apply_preset("Eco"); st.rerun()

st.sidebar.header("Mission Setup")
mission = st.sidebar.selectbox("Mission", list(WEIGHTS.keys()), key="mission")
req_range = st.sidebar.slider("Required range (m)", 10, 500, step=5, key="req_range")
req_res = st.sidebar.slider("Required resolution (m)", 0.02, 1.00, step=0.01, key="req_res")

st.sidebar.header("Energy")
battery = st.sidebar.slider("Remaining energy (Wh)", 5.0, 500.0, step=5.0, key="battery")
reserve = st.sidebar.slider("Reserve energy (Wh)", 0.0, 150.0, step=5.0, key="reserve")
pings = st.sidebar.slider("Planned pings", 100, 100000, step=100, key="pings")

st.sidebar.header("Hardware")
profile_name = st.sidebar.selectbox("Transducer profile", list(PROFILES.keys()), key="profile")
profile = PROFILES[profile_name]

with st.sidebar.expander("Environment"):
    temperature = st.slider("Temperature (°C)", -2.0, 35.0, step=0.5, key="temperature")
    salinity = st.slider("Salinity (PSU)", 25.0, 40.0, step=0.5, key="salinity")
    depth = st.slider("Depth (m)", 0, 2000, step=10, key="depth")
    turbidity = st.slider("Turbidity proxy", 0, 100, step=1, key="turbidity")

c = sound_speed(temperature, salinity, depth)
env = {
    "T": temperature, "S": salinity, "D": depth, "pH": st.session_state.ph,
    "turbidity": turbidity, "sea": st.session_state.sea,
    "mission": mission, "range": req_range, "resolution": req_res,
    "battery": battery, "reserve": reserve, "pings": pings, "c": c,
}
choice, budget, reasons, band, feasible = optimize(env, profile)

if choice:
    st.markdown('<div class="ap-note ap-good"><b>FEASIBLE</b> · AquaPulse found a valid waveform for the current mission.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="ap-note ap-bad"><b>TX INHIBIT</b> · No waveform currently satisfies all mission, energy and hardware limits.</div>', unsafe_allow_html=True)

m1, m2, m3, m4, m5, m6 = st.columns(6)
if choice:
    m1.metric("Centre", f"{choice['fc']:.0f} kHz")
    m2.metric("Bandwidth", f"{choice['bw']:.0f} kHz")
    m3.metric("Pulse", f"{choice['dur']:.1f} ms")
    m4.metric("Drive", f"{choice['amp']*100:.0f}%")
    m5.metric("Energy / Ping", f"{choice['energy']*1000:.2f} mJ")
    m6.metric("Modeled Margin", f"{choice['margin']:+.1f} dB")
else:
    for col, label in zip([m1,m2,m3,m4,m5,m6], ["Centre","Bandwidth","Pulse","Drive","Energy / Ping","Modeled Margin"]):
        col.metric(label, "—")

left, right = st.columns([1.45, 1])

with left:
    tabs = st.tabs(["Waveform", "Spectrum"])
    t, sig = make_wave(choice)

    with tabs[0]:
        fig = go.Figure()
        if sig.size:
            stride = max(1, len(sig)//5000)
            fig.add_trace(go.Scatter(x=t[::stride]*1000, y=sig[::stride], mode="lines"))
        fig.update_layout(height=340, margin=dict(l=20,r=20,t=20,b=20), xaxis_title="Time (ms)", yaxis_title="Normalized amplitude", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        fig = go.Figure()
        if sig.size:
            fft_data = np.fft.rfft(sig)
            freq = np.fft.rfftfreq(len(sig), 1/10_000_000)
            mag = np.abs(fft_data)
            mask = freq <= 600000
            fig.add_trace(go.Scatter(x=freq[mask]/1000, y=mag[mask], mode="lines"))
        fig.update_layout(height=340, margin=dict(l=20,r=20,t=20,b=20), xaxis_title="Frequency (kHz)", yaxis_title="Magnitude", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("### Why this waveform?")
    if choice:
        st.write(f"**{choice['fc']:.0f} kHz / {choice['bw']:.0f} kHz BW / {choice['dur']:.1f} ms** was selected because it:")
        st.write(f"✓ stays inside the **{band[0]:.0f}–{band[1]:.0f} kHz** transducer band")
        st.write(f"✓ meets the **{req_res:.2f} m** resolution target")
        st.write(f"✓ keeps a positive modeled margin at **{req_range} m**")
        st.write("✓ stays within the available energy-per-ping budget")
        st.write(f"✓ is the highest-scoring valid candidate in **{mission}** mode")
    else:
        st.write("No candidate is currently valid.")
        st.write("Try lowering required range, relaxing resolution, increasing energy, or changing transducer profile.")

    st.markdown("### Fixed vs AquaPulse")
    if choice:
        fixed = {"fc": profile["fc"], "bw": min(120.0, profile["bw"]), "dur": 2.0, "amp": 0.8}
        fixed["alpha"] = absorption(fixed["fc"], temperature, salinity, depth, st.session_state.ph)
        fixed["resolution"] = c/(2.0*fixed["bw"]*1000.0)
        fixed["margin"] = range_margin(fixed, env)
        fixed["energy"] = energy_j(fixed["amp"], fixed["dur"], profile["vpk"], profile["z"])
        c1, c2 = st.columns(2)
        c1.metric("Fixed Energy", f"{fixed['energy']*1000:.2f} mJ")
        c2.metric("AquaPulse Energy", f"{choice['energy']*1000:.2f} mJ")
        c1.metric("Fixed Resolution", f"{fixed['resolution']*100:.1f} cm")
        c2.metric("AquaPulse Resolution", f"{choice['resolution']*100:.1f} cm")

with st.expander("Technical details"):
    d1, d2, d3 = st.columns(3)
    d1.metric("Sound Speed", f"{c:.1f} m/s")
    d2.metric("Per-Ping Budget", f"{budget*1000:.2f} mJ")
    d3.metric("Feasible Candidates", f"{len(feasible)}")
    st.write("**Rejected candidates**")
    st.write(" · ".join([f"{k}: {v}" for k, v in reasons.items()]))
    if choice:
        st.write(
            f"Selected score: **{choice['score']:.3f}** · "
            f"Estimated resolution: **{choice['resolution']*100:.1f} cm** · "
            f"Absorption: **{choice['alpha']:.1f} dB/km**"
        )

st.caption(
    "Simulation only: acoustic range and efficiency are modeled, not measured. "
    "Real performance requires a characterized transducer and water-tank validation."
)
