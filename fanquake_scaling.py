#!/usr/bin/python

"""
Hacky single-station "energy" metrics over a short window, per component (Z/N/E),
using ObsPy + Matplotlib.

Time conventions:
  - Plot window: [input_start - plot_before_seconds, input_end]
  - Define t=0 at (input_start - plot_before_seconds)
  - The envelope (and shading) is NOT plotted before input_start (i.e., before t=plot_before_seconds)
  - We do NOT draw a vertical line marking input_start.

Noise level:
  - We ALWAYS compute the 98th percentile (abs amplitude) in the first noise_window seconds
    starting at plot start (t=0), and report it as "98th=...".
  - If background_noise_level_user is None: noise_level used for thresholding = that percentile.
  - Else: noise_level used for thresholding = background_noise_level_user (ALWAYS interpreted
    in mm/s^2 units, i.e., final ACC units when convert_acc_to_percent_g=False).

Defaults updated per request.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pytz
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from obspy import Stream, Trace, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.core.inventory.inventory import Inventory
import time

from scipy.signal import hilbert

time.sleep(0.1)

# ----------------------------
# User inputs / defaults
# ----------------------------

# Choose and load the PNSN and event logos
PNSN_logo = mpimg.imread("PNSNLogo_RGB_Main.png")
Hawks_logo = mpimg.imread("seahawks_logo.png")


def normalize_rgb(r: int, g: int, b: int) -> Tuple[float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0)


# Define station details (single-station)
network = "UW"
station = "KDK"
location = "--"  # "--" --> ""
channel = "HNZ"

# Titles
Top_banner_font_size = 80
Top_banner_line = "BANNER TEXT"
title_line1 = "Upper title line          "
# title_line2 auto: "{network}.{station}  {local starttime of first sample plotted}"

# Local time window (metric window)
pacific = pytz.timezone("America/Los_Angeles")
#plot_start_time_local = pacific.localize(dt.datetime(2026, 1, 17, 19, 11, 17))
#plot_end_time_local = pacific.localize(dt.datetime(2026, 1, 17, 19, 11, 28))
playname= "shaheed"
if playname == "shaheed":
    plot_start_time_local = pacific.localize(dt.datetime(2026, 1, 17, 17, 21, 17))
    plot_end_time_local = pacific.localize(dt.datetime(2026, 1, 17, 17, 22, 27))
    Top_banner_line="SPEEDQUAKE"
elif playname == "JSN":
    plot_start_time_local = pacific.localize(dt.datetime(2026, 1, 17, 17, 51, 40))
    plot_end_time_local = pacific.localize(dt.datetime(2026, 1, 17, 17, 52, 50))
    Top_banner_line="JSN Touchdown vs 49ers 2026"
elif playname == "Beastquake":
    plot_start_time_local = pacific.localize(dt.datetime(2011,1,8,16,42,21))
    plot_end_time_local = pacific.localize(dt.datetime(2011,1,8,16,43,31))
    channel = "ENZ"
    Top_banner_line="BEASTQUAKE"
elif playname == "MarshawnPackers":
    plot_start_time_local = pacific.localize(dt.datetime(2015,1,18,15,8,37))
    plot_end_time_local = pacific.localize(dt.datetime(2015,1,18,15,9,47))
    channel = "ENZ"
    Top_banner_line="Beastmode TD in Green Bay 2015"
elif playname == "Marshawn49ers":
    plot_start_time_local = pacific.localize(dt.datetime(2014,1,19,17,31,37))
    plot_end_time_local = pacific.localize(dt.datetime(2014,1,19,17,32,47))
    channel = "ENZ"
    Top_banner_line="BEASTMODE Touchdown vs 49ers 2014"
elif playname == "MarshawnSaints":
    plot_start_time_local = pacific.localize(dt.datetime(2014,1,11,16,26,27))
    plot_end_time_local = pacific.localize(dt.datetime(2014,1,11,16,27,37))
    channel = "ENZ"
    Top_banner_line="BEASTMODE Touchdown vs Saints 2014"
elif playname == "Kam":
    plot_start_time_local = pacific.localize(dt.datetime(2015,1,10,20,1,20))
    plot_end_time_local = pacific.localize(dt.datetime(2015,1,10,20,2,30))
    channel = "ENZ"
    Top_banner_line="Bam Bam Kam 90yd Pick 6 vs Panthers 2015"
elif playname == "Sherman":
    plot_start_time_local = pacific.localize(dt.datetime(2014,1,19,18,49,5))
    plot_end_time_local = pacific.localize(dt.datetime(2014,1,19,18,50,15))
    channel = "ENZ"
    Top_banner_line="SHERMAN TIP vs 49ers 2014"
elif playname == "Kearse":
    plot_start_time_local = pacific.localize(dt.datetime(2015,1,18,15,28,52))
    plot_end_time_local = pacific.localize(dt.datetime(2015,1,18,15,30,2))
    channel = "ENZ"
    Top_banner_line="Kearse Touchdown vs Green Bay 2015"


ylim = 5
threshold_factor = 1.05
background_noise_level_user: Optional[float] = 0.2
env_power = 1.5

# Updated default: plot this many seconds BEFORE the input start time
plot_before_seconds = 0

# Ground motion type.  ACC, VEL, NONE
ground_motion_type = "ACC"
# Updated default
convert_acc_to_percent_g = False
G0 = 9.80665

# Response-removal and filtering
initial_highpass = 0.02
pre_filt = (0.05, 0.1, 30.0, 35.0)
water_level = 60.0  # optional; set None to let ObsPy default
freqmin = 0.1
if ground_motion_type=="VEL":
    freqmin=1.0
# Updated default
freqmax = 30.0
if freqmax > 48: 
    freqmax=48.

# Colors & linewidth
trace_color = normalize_rgb(0, 34, 68)
text_color = normalize_rgb(105, 190, 40)
banner_color = normalize_rgb(0, 34, 68)
trace_linewidth = 0.4

# Shaded region color
shade_color = normalize_rgb(105, 190, 40)
shade_alpha = 0.8

# Envelope display toggle
show_envelope = True

# Plot geometry + logos
top_banner_frac = 0.12
bottom_banner_frac = 0.06
logo_height = 0.08  # fraction of figure height (in figure coords)

# Download padding
padding_in_sec = 100.0

# Envelope / metrics settings
smoothing_in_sec = 2.0
smoothing_kernel = "gaussian"  # "boxcar" or "gaussian"
# Updated default
noise_window = 15.0
noise_percentile = 98.0
# Updated default
#threshold_factor = 1.05

# Updated default: user override noise level (always interpreted as mm/s^2)
#background_noise_level_user: Optional[float] = 0.3

# Optional normalization reference for area
area_reference: Optional[float] = 50
if freqmax==10:
    area_reference=53
else:
    area_reference=59
if ground_motion_type=="VEL":
    if freqmax==10:
        area_reference=2.1
    else:
        area_reference=2.15


# Output figure file (no plt.show())
#fig_name = "energy_metrics.png"
fig_name = "Fanquake_" + playname + "_" + ground_motion_type + "_" + str(background_noise_level_user) + \
            "_" + str(freqmax) + "Hz_thresh" + str(threshold_factor) + "_N" + str(env_power) + \
            "_smoo" + str(smoothing_in_sec) + "s.png" 
if ground_motion_type=="VEL":
    background_noise_level_user=int(100000*background_noise_level_user/37.)/100000.
    ylim=ylim/37.

fig_dpi = 200


# ----------------------------
# Implementation
# ----------------------------

@dataclass
class Metrics:
    peak: float
    duration_s: float
    area: float
    area_norm: Optional[float]
    noise_level: float
    noise_p98: float
    threshold_amp: float
    seg_plot_i0: Optional[int]
    seg_plot_i1: Optional[int]


def _to_utc(aware_dt: dt.datetime) -> UTCDateTime:
    """Convert tz-aware datetime to ObsPy UTCDateTime."""
    return UTCDateTime(aware_dt.astimezone(pytz.utc))


def _utc_to_local_datetime(utc: UTCDateTime, tz: pytz.BaseTzInfo) -> dt.datetime:
    """Convert ObsPy UTCDateTime to tz-aware datetime in tz."""
    dtu = utc.datetime.replace(tzinfo=pytz.utc)
    return dtu.astimezone(tz)


def _format_local_dt_no_fraction(dtl: dt.datetime) -> str:
    """
    Format: 'January 15, 2026  8:30:01pm'
    - local time
    - no decimal seconds
    - lower-case am/pm
    """
    month = dtl.strftime("%B")
    day = dtl.day
    year = dtl.year
    hour12 = dtl.strftime("%I").lstrip("0") or "0"
    mmss = dtl.strftime(":%M:%S")
    ampm = dtl.strftime("%p").lower()
    return "{} {}, {}  {}{}{}".format(month, day, year, hour12, mmss, ampm)


def _location_code(loc: str) -> str:
    """ObsPy/FDSN uses empty string for '--' location."""
    if loc.strip() == "--":
        return ""
    return loc.strip()


def _channel_base(ch: str) -> str:
    """Given a channel code like HNZ, return base prefix 'HN'."""
    ch = ch.strip()
    if len(ch) < 2:
        raise ValueError("channel must be at least 2 chars, got: {}".format(ch))
    return ch[:2]


def _download_three_components(
    client: Client,
    net: str,
    sta: str,
    loc: str,
    chan_base: str,
    t0: UTCDateTime,
    t1: UTCDateTime,
) -> Stream:
    """Download Z/N/E components for base prefix (e.g., 'HN' -> HNZ/HNN/HNE)."""
    st = Stream()
    for comp in ("Z", "N", "E"):
        cha = "{}{}".format(chan_base, comp)
        try:
            st_comp = client.get_waveforms(
                network=net,
                station=sta,
                location=loc,
                channel=cha,
                starttime=t0,
                endtime=t1,
                attach_response=False,
            )
            st += st_comp
        except Exception:
            pass
    return st


def _get_inventory_response(
    client: Client,
    net: str,
    sta: str,
    loc: str,
    chan_base: str,
    t0: UTCDateTime,
    t1: UTCDateTime,
) -> Optional[Inventory]:
    """Fetch station metadata including response for base + '?' (e.g., HN?)."""
    try:
        inv = client.get_stations(
            network=net,
            station=sta,
            location=loc if loc != "" else None,
            channel="{}?".format(chan_base),
            starttime=t0,
            endtime=t1,
            level="response",
        )
        return inv
    except Exception:
        return None


def _smooth_1d(x: np.ndarray, sr: float, smoothing_s: float, kernel: str) -> np.ndarray:
    """Moving-window smoothing for 1D arrays."""
    if smoothing_s <= 0.0:
        return x.copy()

    win = int(round(smoothing_s * sr))
    win = max(win, 1)

    if kernel.lower() == "boxcar":
        k = np.ones(win, dtype=float)
        k /= np.sum(k)
        return np.convolve(x, k, mode="same")

    if kernel.lower() == "gaussian":
        sigma = max(win / 6.0, 1e-9)
        n = np.arange(win) - (win - 1) / 2.0
        k = np.exp(-0.5 * (n / sigma) ** 2)
        k /= np.sum(k)
        return np.convolve(x, k, mode="same")

    raise ValueError("Unknown smoothing kernel: {}".format(kernel))


def _process_trace_in_place(
    tr: Trace,
    inv: Optional[Inventory],
    gm_type: str,
    do_acc_percent_g: bool,
    g0: float,
    hp: float,
    pre_filt_tuple: Tuple[float, float, float, float],
    water_level_val: Optional[float],
    fmin: float,
    fmax: float,
) -> Tuple[str, str]:
    """
    Process trace in place. Returns (y_units_label, gm_mode_used).
    gm_mode_used may differ from gm_type if response removal fails.

    Units:
      - VEL => mm/s
      - ACC => %g if do_acc_percent_g else mm/s^2
      - NONE => Counts
    """
    tr.detrend("demean")
    if hp is not None and hp > 0.0:
        tr.filter("highpass", freq=hp, corners=2, zerophase=True)
    tr.detrend("demean")

    gm_mode_used = gm_type.upper().strip()
    y_units = "Counts"

    if gm_mode_used != "NONE":
        if inv is None:
            gm_mode_used = "NONE"
        else:
            try:
                remove_kwargs = {"inventory": inv, "pre_filt": pre_filt_tuple}
                if water_level_val is not None:
                    remove_kwargs["water_level"] = float(water_level_val)

                if gm_mode_used == "VEL":
                    tr.remove_response(output="VEL", **remove_kwargs)
                    tr.data = tr.data.astype(np.float64) * 1000.0  # m/s -> mm/s
                    y_units = "mm/s"
                elif gm_mode_used == "ACC":
                    tr.remove_response(output="ACC", **remove_kwargs)
                    if do_acc_percent_g:
                        tr.data = tr.data.astype(np.float64) / float(g0) * 100.0
                        y_units = "%g"
                    else:
                        tr.data = tr.data.astype(np.float64) * 1000.0  # m/s^2 -> mm/s^2
                        y_units = "mm/s$^2$"
                else:
                    gm_mode_used = "NONE"
                    y_units = "Counts"
            except Exception:
                gm_mode_used = "NONE"
                y_units = "Counts"

    tr.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
    return y_units, gm_mode_used


def _compute_noise_percentile_from_plot_start(
    y_plot: np.ndarray,
    sr: float,
    noise_window_s: float,
    pct: float,
) -> float:
    """
    Compute percentile(abs(amplitude)) from the first noise_window seconds of the processed trace
    starting at plot start (t=0) == (input_start - plot_before_seconds).
    """
    n = int(round(noise_window_s * sr))
    n = max(n, 1)
    x0 = y_plot[: min(n, y_plot.size)]
    return float(np.percentile(np.abs(x0.astype(np.float64)), pct))


def _compute_envelope_smoothed(data: np.ndarray, sr: float) -> np.ndarray:
    """Envelope from Hilbert magnitude, then smoothed."""
    env = np.abs(hilbert(data.astype(np.float64)))
    return _smooth_1d(env, sr=sr, smoothing_s=smoothing_in_sec, kernel=smoothing_kernel)


def _compute_metrics_from_metric_window(
    y_plot: np.ndarray,
    env_plot: np.ndarray,
    t_rel_plot: np.ndarray,
    sr: float,
    noise_level: float,
    noise_p98: float,
    threshold_factor_val: float,
    area_ref: Optional[float],
    i_metric0: int,
    i_metric1: int,
) -> Metrics:
    """Compute metrics ONLY on indices [i_metric0, i_metric1] within plot arrays."""
    y_m = y_plot[i_metric0:i_metric1 + 1]
    env_m = env_plot[i_metric0:i_metric1 + 1]
    t_m = t_rel_plot[i_metric0:i_metric1 + 1]

    peak = float(np.max(np.abs(y_m)))
    i_peak_m = int(np.argmax(np.abs(y_m)))

    threshold_amp = float(noise_level) * float(threshold_factor_val)

    above = env_m > threshold_amp
    if not np.any(above):
        return Metrics(
            peak=peak,
            duration_s=0.0,
            area=0.0,
            area_norm=(0.0 / area_ref) if area_ref else None,
            noise_level=float(noise_level),
            noise_p98=float(noise_p98),
            threshold_amp=float(threshold_amp),
            seg_plot_i0=None,
            seg_plot_i1=None,
        )

    idx = np.where(above)[0]
    breaks = np.where(np.diff(idx) > 1)[0]
    seg_starts = [idx[0]] + [idx[b + 1] for b in breaks]
    seg_ends = [idx[b] for b in breaks] + [idx[-1]]

    chosen0_m = None
    chosen1_m = None
    for s0, s1 in zip(seg_starts, seg_ends):
        if s0 <= i_peak_m <= s1:
            chosen0_m = int(s0)
            chosen1_m = int(s1)
            break

    if chosen0_m is None:
        return Metrics(
            peak=peak,
            duration_s=0.0,
            area=0.0,
            area_norm=(0.0 / area_ref) if area_ref else None,
            noise_level=float(noise_level),
            noise_p98=float(noise_p98),
            threshold_amp=float(threshold_amp),
            seg_plot_i0=None,
            seg_plot_i1=None,
        )

    duration_s = float((chosen1_m - chosen0_m + 1) / sr)

    seg_env = env_m[chosen0_m:chosen1_m + 1] - threshold_amp
    seg_env = np.clip(seg_env, 0.0, None)
    seg_t = t_m[chosen0_m:chosen1_m + 1]

    if hasattr(np, "trapezoid"):
        area = float(np.trapezoid(seg_env, seg_t))
    elif hasattr(np, "trapz"):
        area = float(np.trapz(seg_env, seg_t))
    else:
        area = float(np.sum(0.5 * (seg_env[1:] + seg_env[:-1]) * np.diff(seg_t)))

    area_norm = (area / float(area_ref)) if area_ref else None

    seg_plot_i0 = i_metric0 + chosen0_m
    seg_plot_i1 = i_metric0 + chosen1_m

    return Metrics(
        peak=peak,
        duration_s=duration_s,
        area=area,
        area_norm=area_norm,
        noise_level=float(noise_level),
        noise_p98=float(noise_p98),
        threshold_amp=float(threshold_amp),
        seg_plot_i0=seg_plot_i0,
        seg_plot_i1=seg_plot_i1,
    )


def _add_banners_and_logos(
    fig: plt.Figure,
    top_frac: float,
    bottom_frac: float,
    banner_rgb: Tuple[float, float, float],
    top_text: str,
    top_font_size: float,
    logo_left: np.ndarray,
    logo_right: np.ndarray,
    logo_h: float,
) -> None:
    """Add top/bottom solid-color banners and left/right logos just below the top banner."""
    ax_top = fig.add_axes([0.0, 1.0 - top_frac, 1.0, top_frac])
    ax_top.set_facecolor(banner_rgb)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for spine in ax_top.spines.values():
        spine.set_visible(False)

    ax_top.text(
        0.5, 0.5, top_text,
        ha="center", va="center",
        color="white",
        fontsize=top_font_size,
        transform=ax_top.transAxes,
    )

    ax_bot = fig.add_axes([0.0, 0.0, 1.0, bottom_frac])
    ax_bot.set_facecolor(banner_rgb)
    ax_bot.set_xticks([])
    ax_bot.set_yticks([])
    for spine in ax_bot.spines.values():
        spine.set_visible(False)

    y0 = 1.0 - top_frac - logo_h - 0.005
    y0 = max(y0, bottom_frac + 0.005)

    ax_ll = fig.add_axes([0.01, y0, logo_h * 1.6, logo_h])
    ax_ll.imshow(logo_left)
    ax_ll.axis("off")

    ax_lr = fig.add_axes([1.0 - 0.01 - logo_h * 1.6, y0, logo_h * 1.6, logo_h])
    ax_lr.imshow(logo_right)
    ax_lr.axis("off")


def main() -> None:
    client = Client("IRIS")

    loc_code = _location_code(location)
    chan_base = _channel_base(channel)

    # Input metric window (start/end)
    t_input0 = _to_utc(plot_start_time_local)
    t_input1 = _to_utc(plot_end_time_local)
    metric_len = float(t_input1 - t_input0)

    # Plot window starts earlier; THIS is "zero time" (t=0)
    t_plot0 = t_input0 - float(plot_before_seconds)
    t_plot1 = t_input1
    plot_len = float(t_plot1 - t_plot0)

    # Download window includes padding on both sides of the plot window
    t_req0 = t_plot0 - float(padding_in_sec)
    t_req1 = t_plot1 + float(padding_in_sec)

    st = _download_three_components(
        client=client,
        net=network,
        sta=station,
        loc=loc_code,
        chan_base=chan_base,
        t0=t_req0,
        t1=t_req1,
    )

    inv = _get_inventory_response(
        client=client,
        net=network,
        sta=station,
        loc=loc_code,
        chan_base=chan_base,
        t0=t_req0,
        t1=t_req1,
    )

    if len(st) > 0:
        st.merge(method=1, fill_value="interpolate")

    traces_by_comp: Dict[str, Optional[Trace]] = {"Z": None, "N": None, "E": None}
    for comp in ("Z", "N", "E"):
        cha = "{}{}".format(chan_base, comp)
        tr = st.select(channel=cha)
        if len(tr) > 0:
            traces_by_comp[comp] = tr[0].copy()

    results: Dict[str, Dict[str, object]] = {}

    # For auto title_line2 (local time of earliest plotted first sample across comps)
    time_of_first_sample_plotted_utc: Optional[UTCDateTime] = None

    for comp in ("Z", "N", "E"):
        tr0 = traces_by_comp[comp]
        if tr0 is None:
            results[comp] = {"ok": False, "msg": "No data for {}{}".format(chan_base, comp)}
            continue

        tr_padded = tr0.copy()

        y_units, gm_used = _process_trace_in_place(
            tr=tr_padded,
            inv=inv,
            gm_type=ground_motion_type,
            do_acc_percent_g=convert_acc_to_percent_g,
            g0=G0,
            hp=initial_highpass,
            pre_filt_tuple=pre_filt,
            water_level_val=water_level,
            fmin=freqmin,
            fmax=freqmax,
        )

        # Trim to the PLOT window (starts at input_start - plot_before_seconds)
        tr_plot = tr_padded.copy()
        tr_plot.trim(starttime=t_plot0, endtime=t_plot1, pad=False)
        if tr_plot.stats.npts < 2:
            results[comp] = {"ok": False, "msg": "Trim produced empty plot trace for {}".format(comp)}
            continue

        if time_of_first_sample_plotted_utc is None:
            time_of_first_sample_plotted_utc = tr_plot.stats.starttime
        else:
            time_of_first_sample_plotted_utc = min(time_of_first_sample_plotted_utc, tr_plot.stats.starttime)

        y_plot = tr_plot.data.astype(np.float64)
        sr = float(tr_plot.stats.sampling_rate)
        npts = int(tr_plot.stats.npts)

        # Time vector: seconds since plot start (t_plot0) => t=0 at (input_start - plot_before_seconds)
        start_offset = float(tr_plot.stats.starttime - t_plot0)
        t_rel = start_offset + (np.arange(npts, dtype=float) / sr)

        # Always compute the percentile in the noise window for reporting
        noise_p98 = _compute_noise_percentile_from_plot_start(
            y_plot=y_plot,
            sr=sr,
            noise_window_s=noise_window,
            pct=noise_percentile,
        )

        # Noise level used for thresholding
        if background_noise_level_user is None:
            noise_level = noise_p98
        else:
            noise_level = float(background_noise_level_user)

        # Envelope over full plotted window; but display (and shade) only for t >= input_start
        env_plot = _compute_envelope_smoothed(y_plot, sr=sr)
        print("-------{} {}: max|y|={:.6g}  max(env)={:.6g}  ratio env/|y|={:.3f}".format(
        station, comp, np.max(np.abs(y_plot)), np.max(env_plot), np.max(env_plot) / np.max(np.abs(y_plot))
        ))
        # Fudge for values less than one
        if ground_motion_type=="VEL":
            env_plot = env_plot * 37.
            env_plot = np.power(env_plot,env_power)
            env_plot = env_plot / 37.
        else:
            env_plot = np.power(env_plot,env_power)

        # Metric window in this time system:
        # input start is at t = plot_before_seconds
        t_metric0 = float(plot_before_seconds)
        t_metric1 = float(plot_before_seconds) + metric_len

        i_metric = np.where((t_rel >= t_metric0) & (t_rel <= t_metric1))[0]
        if i_metric.size < 2:
            results[comp] = {"ok": False, "msg": "Metric window not found in plot window for {}".format(comp)}
            continue

        i_metric0 = int(i_metric[0])
        i_metric1 = int(i_metric[-1])

        metrics = _compute_metrics_from_metric_window(
            y_plot=y_plot,
            env_plot=env_plot,
            t_rel_plot=t_rel,
            sr=sr,
            noise_level=noise_level,
            noise_p98=noise_p98,
            threshold_factor_val=threshold_factor,
            area_ref=area_reference,
            i_metric0=i_metric0,
            i_metric1=i_metric1,
        )

        results[comp] = {
            "ok": True,
            "t_rel": t_rel,
            "y_plot": y_plot,
            "env_plot": env_plot,
            "metrics": metrics,
            "y_units": y_units,
            "gm_used": gm_used,
        }

    # title_line2 formatting in local time, no fractional seconds
    if time_of_first_sample_plotted_utc is None:
        title_line2_auto = "{}.{}  NO_DATA".format(network, station)
    else:
        dt_local = _utc_to_local_datetime(time_of_first_sample_plotted_utc, pacific)
        dt_local = dt_local.replace(microsecond=0)
        title_line2_auto = "{}  {}.{}  {}  {}  noisethresh={}  {}Hz  threshfactor {}  N={}  smoo={}s".format \
            (playname, network, station, _format_local_dt_no_fraction(dt_local), ground_motion_type, \
            background_noise_level_user, freqmax, threshold_factor, env_power, smoothing_in_sec)

    # ----------------------------
    # Plot
    # ----------------------------
    fig = plt.figure(figsize=(22, 14))

    _add_banners_and_logos(
        fig=fig,
        top_frac=top_banner_frac,
        bottom_frac=bottom_banner_frac,
        banner_rgb=banner_color,
        top_text=Top_banner_line,
        top_font_size=Top_banner_font_size,
        logo_left=PNSN_logo,
        logo_right=Hawks_logo,
        logo_h=logo_height,
    )

    main_bottom = bottom_banner_frac + 0.02
    main_top = 1.0 - top_banner_frac - logo_height - 0.03
    main_height = max(main_top - main_bottom, 0.2)

    gs = fig.add_gridspec(
        nrows=3,
        ncols=1,
        left=0.06,
        right=0.98,
        bottom=main_bottom,
        top=main_bottom + main_height,
        hspace=0.10,
    )

    axes = []
    input_start_t_rel = float(plot_before_seconds)

    for i, comp in enumerate(("Z", "N", "E")):
        ax = fig.add_subplot(gs[i, 0])
        axes.append(ax)

        ax.text(
            0.01, 0.98, comp,
            transform=ax.transAxes,
            ha="left", va="top",
            color=text_color,
            fontsize=22,
            fontweight="bold",
        )

        if comp == "Z":
            ax.set_title(
                "{}\n{}".format(title_line1.rstrip(), title_line2_auto.rstrip()),
                color=text_color,
                fontsize=24,
                pad=14,
            )

        info = results.get(comp, {})
        if not info.get("ok", False):
            ax.text(
                0.5, 0.5,
                info.get("msg", "Missing data"),
                transform=ax.transAxes,
                ha="center", va="center",
                color="red",
                fontsize=16,
            )
            ax.grid(True, linewidth=0.5, alpha=0.2)
            ax.axvline(0.0, linewidth=1.0, color="black", alpha=0.2)
            continue

        t_rel = info["t_rel"]              # type: ignore[assignment]
        y_plot = info["y_plot"]            # type: ignore[assignment]
        env_plot = info["env_plot"]        # type: ignore[assignment]
        metrics: Metrics = info["metrics"]  # type: ignore[assignment]
        y_units = str(info["y_units"])
        gm_used = str(info["gm_used"])

        # Seismogram
        ax.plot(t_rel, y_plot, linewidth=trace_linewidth, color=trace_color)
        ax.set_ylim(-1*ylim,ylim)

        # Envelope: optional, and do not display before input start time
        if show_envelope:
            env_for_plot = env_plot.copy()
            env_for_plot[t_rel < input_start_t_rel] = np.nan
            ax.plot(t_rel, env_for_plot, linewidth=0.9, color="red")

        # Short red line for chosen background noise level: from plot start (t=0) to 5% of total x-range
        x0 = float(t_rel[0])
        x1 = float(t_rel[-1])
        x_span = x1 - x0
        x_short_end = 0.0 + 0.05 * x_span
        ax.plot([0.0, x_short_end], [metrics.noise_level, metrics.noise_level],
                linewidth=2.0, color="red")

        # Shade ONLY the single segment used in the calculations (and never before input start)
        if metrics.seg_plot_i0 is not None and metrics.seg_plot_i1 is not None:
            i0s = int(metrics.seg_plot_i0)
            i1s = int(metrics.seg_plot_i1)
            seg_t = t_rel[i0s:i1s + 1]
            seg_env = env_plot[i0s:i1s + 1]
            thr = float(metrics.threshold_amp)

            seg_above = (seg_env > thr) & (seg_t >= input_start_t_rel)
            ax.fill_between(
                seg_t,
                thr,
                seg_env,
                where=seg_above,
                alpha=shade_alpha,
                color=shade_color,
                linewidth=0.0,
            )

        # Upper right metrics text block
        if metrics.area_norm is None:
            area_text = "area = {:.4g}".format(metrics.area)
        else:
            area_text = "area = {:.4g} (norm={:.4g})".format(metrics.area, metrics.area_norm)

        text_lines = [
            "peak = {:.4g}".format(metrics.peak),
            "dur = {:.3f} s".format(metrics.duration_s),
            area_text,
        ]
        ax.text(
            0.99, 0.98,
            "\n".join(text_lines),
            transform=ax.transAxes,
            ha="right", va="top",
            color=text_color,
            fontsize=16,
            bbox=dict(facecolor="white", edgecolor="none", alpha=1.0),
        )

        ax.set_ylabel(y_units, color=text_color)
        ax.tick_params(axis="x", colors=text_color)
        ax.tick_params(axis="y", colors=text_color)

        ax.axhline(0.0, linewidth=0.6, color="black", alpha=0.2)
        ax.grid(True, linewidth=0.5, alpha=0.2)

        # Beastquakes annotations
        ax.text(0.07, 0.7,"{:.2g} Beastquakes".format(metrics.area_norm),
            transform=ax.transAxes,
            ha="left", va="bottom",
            color=text_color,
            fontsize=30,fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=1.0),
        )

        #ax.text(
        #    0.01, 0.02,
        #    "mode={}".format(gm_used),
        #    transform=ax.transAxes,
        #    ha="left", va="bottom",
        #    color=text_color,
        #    fontsize=12,
        #    bbox=dict(facecolor="white", edgecolor="none", alpha=1.0),
        #)

        # Mark ONLY plot start (t=0)
        ax.axvline(0.0, linewidth=1.0, color="black", alpha=0.2)

    axes[-1].set_xlabel(
        "Seconds since (input start - plot_before_seconds) (t=0)", color=text_color
    )

    for ax in axes:
        ax.set_xlim(0.0, float(plot_len))

    # Print summary to stdout
    for comp in ("Z", "N", "E"):
        info = results.get(comp, {})
        if not info.get("ok", False):
            print("{}: {}".format(comp, info.get("msg", "missing")))
            continue

        m: Metrics = info["metrics"]  # type: ignore[assignment]

        if m.area_norm is None:
            print(
                "{}: peak={:.3f} dur={:.3f}s area={:.3g} noise={:.4g} thr={:.4g}  98th={:.4g}".format(
                    comp, m.peak, m.duration_s, m.area, m.noise_level, m.threshold_amp, m.noise_p98
                )
            )
        else:
            print(
                "{}: peak={:.3f} dur={:.3f}s area={:.3g} (norm={:.3g}) noise={:.4g} thr={:.4g}  98th={:.4g}".format(
                    comp, m.peak, m.duration_s, m.area, m.area_norm, m.noise_level, m.threshold_amp, m.noise_p98
                )
            )

    fig.savefig(fig_name, dpi=fig_dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()


