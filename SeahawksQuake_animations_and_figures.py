#!/home/ahutko/miniconda3/envs/seahawks/bin/python

"""
PNSN-style seismogram PNGs + MP4 animations with progressive reveal.

Fixes in THIS version (based on your latest message):
- AutoDateLocator/AutoDateFormatter import fixed (from matplotlib.dates).
- Default fps = 60 (if input_video present and fps is not explicitly set differently).
- Animation header logos are *forced* hard-left / hard-right and never overlap:
    - We compute a fixed pixel height for the logo band, then place logos in pixel-space,
      converting to figure fractions so the placement is stable.
- Overlay animations: the seismogram box/spines are ALWAYS fully present (not revealed).
    - We render the box-only RGBA and composite it every frame.
- Progress printing every 10% with flush=True for *all* phases.

Notes:
- Your Matplotlib "tostring_rgb" issue is avoided by using fig.canvas.buffer_rgba().
- Social-media universal: MP4 with yuv420p; codec preference is libopenh264 then mpeg4.
  (Your conda ffmpeg has libopenh264 but not libx264; that's fine for compatibility.)
"""

from __future__ import annotations

import os
import time
import shutil
import subprocess
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytz

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.core.inventory import Inventory, read_inventory


# ----------------------------
# User parameters (edit these)
# ----------------------------

def normalize_rgb(r: int, g: int, b: int) -> Tuple[float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0)


# Layout tweaks to seismogram portion to provide extra whitespace.
margin_bottom_extra = 0.12  # room for angled x tick labels
margin_left_extra = 0.02
margin_right_extra = 0.02

# Choose and load the PNSN and event logos
PNSN_logo = mpimg.imread("PNSNLogo_RGB_Main.png")
Hawks_logo = mpimg.imread("seahawks_logo.png")

# Default per request
logo_height = 0.08  # vertical fraction between top banner and seismogram panel reserved for logos

# Titles
Top_banner_line = "SEAHAWKS GOING TO DARK SIDE!"
title_line1 = "Ground motion at Lumen Field from UW.HWK5          "
title_line2 = "NFL Divisional Playoff vs 49ers Jan 17, 2026          "

# Input video (set None for blank-video MP4s)
#input_video = None  # e.g. "SeahawksRun.mov"
input_video = "SeahawksRun.mov"

# Audio handling
audio_off = False  # default False: keep audio if present in input video

# Frames per second:
# fps = 0 -> MP4s with ONE frame (final)
# fps < 0 -> use input video's native fps (if video present)
# fps > 0 -> use that fps
fps = 60  # default per request

# Chop input video
video_start = 2
video_end = 9

# Overlay appearance
animation_transparency = 0.5
animation_transparency_color = "white"

# Define station details (single-station)
network = "UW"
station = "HWK5"
location = "--"
channel = "HNE"

# Multi-station list (1-6). Format: "STA.CHA" or "NET.STA.LOC.CHA"
station_channels = ["HWK5.HNE", "HWK4.HNE", "HWK3.HNZ"]

# Seismogram timing controls
seismogram_buffer_plot_before = 30
seismogram_buffer_plot_end = 30
seismogram_padding = 20

# Response-removal and filtering
pre_filt = (0.05, 0.1, 30.0, 35.0)
water_level = 60.0  # optional; set None to let ObsPy default
filter_type = "bandpass"
freqmin = 0.1
freqmax = 10.0

# Ground motion type
ground_motion_type = "ACC"
convert_acc_to_percent_g = True
G0 = 9.80665

# Colors & linewidth
trace_color = normalize_rgb(0, 34, 68)
text_color = normalize_rgb(105, 190, 40)
border_color = normalize_rgb(0, 34, 68)
trace_linewidth = 0.4

# ylimits behavior:
# None => ylim = 1.02 * max(abs) after offsets
# ylimits < 0 => normalize each trace to 1 and hide y-axis; spacing between traces = abs(ylimits)
ylimits: Optional[Union[float, Tuple[float, float]]] = -2.03

# Local time window (the “reveal” portion)
pacific = pytz.timezone("America/Los_Angeles")
plot_start_time_local = pacific.localize(dt.datetime(2017, 1, 8, 1, 49, 23))
plot_end_time_local = pacific.localize(dt.datetime(2017, 1, 8, 1, 49, 30))

# PNG layout controls
figsize_inches = (8.0, 8.0)
top_banner_frac = 0.12
bottom_banner_frac = 0.06
margin_left = 0.11
margin_right = 0.06
margin_between_banner_and_plot = 0.02

# PNG x ticks
x_tick_rotation_deg = 45
x_tick_fontsize = 10

# Animation geometry
anim_out_width_px = 1280
blank_video_height_px = 720
anim_stacked_seismo_height_frac = 0.28
anim_overlay_height_frac = 0.22

anim_header_title_fontsize = 22
anim_subtitle_fontsize = 14

# Outputs
out_png_single = "Seahawks_seismograms_{sta}.{cha}.png".format(sta=station, cha=channel)
out_png_multi = "Seahawks_seismograms_multi_station.png"

out_mp4_stacked_single = "Seahawks_seismograms_animation_stacked_single_station.mp4"
out_mp4_stacked_multi = "Seahawks_seismograms_animation_stacked_multi_station.mp4"
out_mp4_overlay_single = "Seahawk_seismograms_animation_overlay_single_station.mp4"
out_mp4_overlay_multi = "Seahawks_seismograms_animation_overlay_multi_station.mp4"

# Optional: force ffmpeg binary
FFMPEG_BIN = None


# ----------------------------
# Types / helpers
# ----------------------------

@dataclass
class TraceSpec:
    network: str
    station: str
    location: str
    channel: str


def _find_ffmpeg() -> str:
    if FFMPEG_BIN:
        return str(FFMPEG_BIN)
    p = shutil.which("ffmpeg")
    if not p:
        raise RuntimeError("ffmpeg not found on PATH.")
    return p


def _find_ffprobe() -> Optional[str]:
    return shutil.which("ffprobe")


def _print_progress_factory(total: int, label: str):
    """
    Prints at 0%, 10%, ..., 100%, flush immediately.
    """
    next_pct = {"val": 0}
    print("{}: 0%".format(label), flush=True)

    def update(i: int) -> None:
        if total <= 0:
            return
        pct = int(round(100.0 * (i + 1) / float(total)))
        while pct >= next_pct["val"] and next_pct["val"] <= 100:
            print("{}: {}%".format(label, next_pct["val"]), flush=True)
            next_pct["val"] += 10

    return update


def _dt_to_utc_obspy(t_local: dt.datetime) -> UTCDateTime:
    if t_local.tzinfo is None:
        raise ValueError("plot_start_time_local/plot_end_time_local must be timezone-aware.")
    t_utc = t_local.astimezone(pytz.utc)
    return UTCDateTime(t_utc.replace(tzinfo=None))


def _format_local_tick(x, pos=None, tz=pacific) -> str:
    d = mdates.num2date(x)
    d_local = d.astimezone(tz)
    return d_local.strftime("%-I:%M:%S %p")


def _ground_motion_label_units(gm_type: str) -> Tuple[str, str]:
    if gm_type == "ACC":
        return ("Acceleration", "% g" if convert_acc_to_percent_g else "m/s^2")
    if gm_type == "VEL":
        return ("Velocity", "m/s")
    if gm_type == "DISP":
        return ("Displacement", "m")
    if gm_type == "DEF":
        return ("Deformation", "")
    return ("Ground motion", "")


def _parse_station_channels(items: List[str], default_net: str, default_loc: str) -> List[TraceSpec]:
    specs: List[TraceSpec] = []
    for s in items:
        parts = s.split(".")
        if len(parts) == 2:
            sta, cha = parts
            specs.append(TraceSpec(default_net, sta, default_loc, cha))
        elif len(parts) == 4:
            net, sta, loc, cha = parts
            specs.append(TraceSpec(net, sta, loc, cha))
        else:
            raise ValueError("Bad station_channels entry: {} (use STA.CHA or NET.STA.LOC.CHA)".format(s))
    return specs


def _compute_windows() -> Dict[str, object]:
    plot_window_start_local = plot_start_time_local - dt.timedelta(seconds=seismogram_buffer_plot_before)
    plot_window_end_local = plot_end_time_local + dt.timedelta(seconds=seismogram_buffer_plot_end)

    plot_window_start_utc = _dt_to_utc_obspy(plot_window_start_local)
    plot_window_end_utc = _dt_to_utc_obspy(plot_window_end_local)

    download_start_utc = plot_window_start_utc - float(seismogram_padding)
    download_end_utc = plot_window_end_utc + float(seismogram_padding)

    reveal_start_utc = _dt_to_utc_obspy(plot_start_time_local)
    reveal_end_utc = _dt_to_utc_obspy(plot_end_time_local)
    reveal_duration = float(reveal_end_utc - reveal_start_utc)

    return {
        "plot_window_start_utc": plot_window_start_utc,
        "plot_window_end_utc": plot_window_end_utc,
        "download_start_utc": download_start_utc,
        "download_end_utc": download_end_utc,
        "reveal_start_utc": reveal_start_utc,
        "reveal_end_utc": reveal_end_utc,
        "reveal_duration": reveal_duration,
    }


def _compute_main_panel_rect_png() -> List[float]:
    left = margin_left + float(margin_left_extra)
    right = 1.0 - (margin_right + float(margin_right_extra))
    top = 1.0 - top_banner_frac - logo_height - margin_between_banner_and_plot
    bottom = bottom_banner_frac + float(margin_bottom_extra)
    return [left, bottom, right - left, top - bottom]


def _ylimits_is_negative(yl: Optional[Union[float, Tuple[float, float]]]) -> bool:
    if yl is None or isinstance(yl, tuple):
        return False
    return float(yl) < 0.0


def _apply_normalization_if_negative_ylimits(ys: List[np.ndarray]) -> Tuple[List[np.ndarray], bool, float]:
    if not _ylimits_is_negative(ylimits):
        return ys, False, 0.0
    spacing = abs(float(ylimits)) if ylimits is not None else 1.03
    if spacing <= 0.0:
        spacing = 1.03

    ys_out: List[np.ndarray] = []
    for y in ys:
        m = float(np.max(np.abs(y))) if y.size else 1.0
        if m <= 0.0:
            m = 1.0
        ys_out.append(y / m)
    return ys_out, True, spacing


def _compute_offsets(ys: List[np.ndarray], normalize_mode: bool, norm_spacing: float) -> np.ndarray:
    n = len(ys)
    if n == 0:
        return np.array([], dtype=np.float64)

    if normalize_mode:
        spacing = norm_spacing
    else:
        maxamp = 0.0
        for y in ys:
            if y.size:
                maxamp = max(maxamp, float(np.max(np.abs(y))))
        if maxamp <= 0.0:
            maxamp = 1.0
        spacing = maxamp

    return (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * spacing


def _expand_ylim_for_offsets(ys_off: List[np.ndarray]) -> Tuple[float, float]:
    maxabs = 0.0
    for y in ys_off:
        if y.size:
            maxabs = max(maxabs, float(np.max(np.abs(y))))
    if maxabs <= 0.0:
        maxabs = 1.0
    lim = 1.02 * maxabs
    return (-lim, lim)


def _try_load_inventory_local_first(spec: TraceSpec) -> Optional[Inventory]:
    fname = "Station_{net}_{sta}.xml".format(net=spec.network, sta=spec.station)
    if os.path.exists(fname):
        try:
            return read_inventory(fname)
        except Exception:
            return None
    return None


def _get_inventory(client: Client, spec: TraceSpec, start_utc: UTCDateTime, end_utc: UTCDateTime) -> Optional[Inventory]:
    inv = _try_load_inventory_local_first(spec)
    if inv is not None:
        return inv
    try:
        return client.get_stations(
            network=spec.network,
            station=spec.station,
            location=spec.location,
            channel=spec.channel,
            starttime=start_utc,
            endtime=end_utc,
            level="response",
        )
    except Exception:
        return None


def _download_and_process(
    client: Client,
    spec: TraceSpec,
    download_start_utc: UTCDateTime,
    download_end_utc: UTCDateTime,
    plot_start_utc: UTCDateTime,
    plot_end_utc: UTCDateTime,
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    st = client.get_waveforms(
        network=spec.network,
        station=spec.station,
        location=spec.location,
        channel=spec.channel,
        starttime=download_start_utc,
        endtime=download_end_utc,
        attach_response=False,
    )
    st.merge(method=1, fill_value="interpolate")
    st.detrend("demean")

    inv = _get_inventory(client, spec, download_start_utc, download_end_utc)
    yaxis_ok = True
    if inv is not None:
        rr_kwargs = dict(
            inventory=inv,
            output=ground_motion_type,
            pre_filt=pre_filt,
            plot=False,
        )
        if water_level is not None:
            rr_kwargs["water_level"] = float(water_level)
        try:
            st.remove_response(**rr_kwargs)
        except Exception:
            yaxis_ok = False
    else:
        yaxis_ok = False

    if filter_type:
        f = filter_type.lower()
        if f == "bandpass":
            st.filter("bandpass", freqmin=freqmin, freqmax=freqmax)
        elif f == "highpass":
            st.filter("highpass", freq=freqmin)
        elif f == "lowpass":
            st.filter("lowpass", freq=freqmax)
        else:
            raise ValueError("Unsupported filter_type: {}".format(filter_type))

    st.trim(starttime=plot_start_utc, endtime=plot_end_utc, pad=False)
    if len(st) == 0:
        raise RuntimeError("No data after trimming for {}.{}".format(spec.station, spec.channel))

    tr = st[0]
    fs = float(tr.stats.sampling_rate)
    y = tr.data.astype(np.float64)

    if yaxis_ok and ground_motion_type == "ACC" and convert_acc_to_percent_g:
        y = 100.0 * (y / G0)

    t0 = tr.stats.starttime.timestamp
    n = tr.stats.npts
    t = t0 + np.arange(n, dtype=np.float64) / fs
    return t, y, fs, yaxis_ok


# ----------------------------
# Matplotlib canvas helpers
# ----------------------------

def _fig_to_rgba(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return rgba.copy()


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    rgba = _fig_to_rgba(fig)
    return rgba[:, :, :3].copy()


# ----------------------------
# PNG rendering
# ----------------------------

def _style_axes_png(ax: plt.Axes, show_y: bool) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color(border_color)
        spine.set_linewidth(1.0)

    ax.tick_params(axis="x", colors=text_color)
    ax.tick_params(axis="y", colors=text_color)

    if show_y:
        gm_label, gm_units = _ground_motion_label_units(ground_motion_type)
        ax.set_ylabel(
            "{}{}".format(gm_label, (" " + gm_units) if gm_units else ""),
            color=text_color,
            fontweight="bold",
        )
    else:
        ax.set_ylabel("")
        ax.set_yticks([])

    for tick in ax.get_xticklabels():
        tick.set_rotation(x_tick_rotation_deg)
        tick.set_ha("right")
        tick.set_fontsize(x_tick_fontsize)
        tick.set_fontweight("bold")


def _add_banners_and_logos_png(fig: plt.Figure) -> None:
    top_ax = fig.add_axes([0.0, 1.0 - top_banner_frac, 1.0, top_banner_frac])
    top_ax.set_facecolor(border_color)
    top_ax.set_xticks([]); top_ax.set_yticks([])
    for spine in top_ax.spines.values():
        spine.set_visible(False)
    top_ax.text(
        0.5, 0.55, Top_banner_line,
        ha="center", va="center",
        color="white",
        fontsize=18,
        fontweight="bold",
        family="sans-serif",
        transform=top_ax.transAxes,
    )

    bottom_ax = fig.add_axes([0.0, 0.0, 1.0, bottom_banner_frac])
    bottom_ax.set_facecolor(border_color)
    bottom_ax.set_xticks([]); bottom_ax.set_yticks([])
    for spine in bottom_ax.spines.values():
        spine.set_visible(False)

    white_top = 1.0 - top_banner_frac
    white_bottom = max(white_top - float(logo_height), bottom_banner_frac)
    band_h = white_top - white_bottom
    if band_h <= 0:
        return

    # Scale logos to fill band height; keep left/right placement
    fig_w_in = float(fig.get_figwidth())
    fig_h_in = float(fig.get_figheight())
    fig_hw_ratio = fig_h_in / fig_w_in

    def place_logo(img: np.ndarray, x_left: float, y_bottom: float, h_rel: float) -> float:
        ih, iw = img.shape[0], img.shape[1]
        aspect = float(iw) / float(ih) if ih > 0 else 1.0
        w_rel = h_rel * aspect * fig_hw_ratio
        axl = fig.add_axes([x_left, y_bottom, w_rel, h_rel])
        axl.imshow(img)
        axl.axis("off")
        return w_rel

    logo_h = 0.85 * band_h
    y0 = white_bottom + 0.5 * (band_h - logo_h)

    place_logo(Hawks_logo, x_left=0.03, y_bottom=y0, h_rel=logo_h)

    ih, iw = PNSN_logo.shape[0], PNSN_logo.shape[1]
    aspect = float(iw) / float(ih) if ih > 0 else 1.0
    w_rel = logo_h * aspect * fig_hw_ratio
    x_left = 0.97 - w_rel
    place_logo(PNSN_logo, x_left=x_left, y_bottom=y0, h_rel=logo_h)


def _plot_png_one_panel(
    series: List[Tuple[TraceSpec, np.ndarray, np.ndarray, bool]],
    outpath: str,
    show_station_labels_right: bool,
) -> None:
    fig = plt.figure(figsize=figsize_inches, dpi=150)
    fig.patch.set_facecolor("white")
    _add_banners_and_logos_png(fig)

    rect = _compute_main_panel_rect_png()
    ax = fig.add_axes(rect)

    ax.set_title(
        "{}\n{}".format(title_line1, title_line2),
        loc="center",
        color=text_color,
        fontweight="bold",
        fontsize=11,
        pad=8,
        family="sans-serif",
    )

    yaxis_ok_all = all(ok for (_spec, _t, _y, ok) in series)

    specs = [spec for (spec, _t, _y, _ok) in series]
    ts = [t for (_spec, t, _y, _ok) in series]
    ys = [y for (_spec, _t, y, _ok) in series]

    ys, normalize_mode, norm_spacing = _apply_normalization_if_negative_ylimits(ys)
    offsets = _compute_offsets(ys, normalize_mode=normalize_mode, norm_spacing=norm_spacing)
    ys_off = [y + offsets[i] for i, y in enumerate(ys)]

    x_mpl_list: List[np.ndarray] = []
    for t in ts:
        dt_utc = [dt.datetime.fromtimestamp(float(x), tz=dt.timezone.utc) for x in t]
        x_mpl_list.append(mdates.date2num(dt_utc))

    for x_mpl, y_plot in zip(x_mpl_list, ys_off):
        ax.plot(x_mpl, y_plot, color=trace_color, linewidth=trace_linewidth)

    locator = AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: _format_local_tick(x, pos, tz=pacific)))

    show_yaxis = bool(yaxis_ok_all) and (not normalize_mode)
    _style_axes_png(ax, show_y=show_yaxis)

    ax.set_xlim(x_mpl_list[0][0], x_mpl_list[0][-1])
    ylo, yhi = _expand_ylim_for_offsets(ys_off)
    ax.set_ylim((ylo, yhi))

    if show_station_labels_right:
        x0, x1 = ax.get_xlim()
        dx = (x1 - x0) * 0.01
        x_text = x1 + dx
        for i, spec in enumerate(specs):
            ax.text(
                x_text, offsets[i],
                spec.station,
                ha="left", va="center",
                color=text_color,
                fontweight="bold",
                fontsize=11,
                family="sans-serif",
                clip_on=False,
            )

    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ----------------------------
# FFmpeg IO
# ----------------------------

def _ensure_even_dims_rgb(frame_rgb: np.ndarray) -> np.ndarray:
    h, w, c = frame_rgb.shape
    if c != 3:
        raise ValueError("Expected RGB with 3 channels.")
    h2 = h - (h % 2)
    w2 = w - (w % 2)
    return frame_rgb[:h2, :w2, :]


def _write_mp4_ffmpeg_pipe(frames_rgb: List[np.ndarray], fps_out: int, outpath: str) -> str:
    if len(frames_rgb) == 0:
        raise ValueError("No frames to write.")

    ffmpeg = _find_ffmpeg()
    fr0 = _ensure_even_dims_rgb(frames_rgb[0].astype(np.uint8))
    h, w, _ = fr0.shape
    frames_fixed = [fr0] + [_ensure_even_dims_rgb(fr.astype(np.uint8)) for fr in frames_rgb[1:]]

    nframes = len(frames_fixed)
    prog = _print_progress_factory(nframes, "ffmpeg write {}".format(outpath))

    last_err = None
    candidates = [
        ("libopenh264", ["-c:v", "libopenh264", "-b:v", "6M"]),
        ("mpeg4", ["-c:v", "mpeg4", "-q:v", "3"]),
    ]

    for codec_name, codec_args in candidates:
        cmd = [
            ffmpeg, "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", "{}x{}".format(w, h),
            "-r", str(int(fps_out)),
            "-i", "-",
            "-an",
        ] + codec_args + [
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            outpath,
        ]

        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert proc.stdin is not None
            broken = False
            for i, fr in enumerate(frames_fixed):
                try:
                    proc.stdin.write(fr.tobytes())
                except BrokenPipeError:
                    broken = True
                    break
                prog(i)

            try:
                proc.stdin.close()
            except Exception:
                pass

            rc = proc.wait()
            err = proc.stderr.read() if proc.stderr is not None else b""
            if rc == 0 and (not broken):
                return outpath

            last_err = err.decode("utf-8", errors="replace")
            print("ffmpeg failed codec={}, rc={}, broken_pipe={}".format(codec_name, rc, broken), flush=True)
            if last_err.strip():
                print(last_err.strip(), flush=True)

        except Exception as e:
            last_err = str(e)
            print("ffmpeg exception codec={}: {}".format(codec_name, e), flush=True)

    raise RuntimeError("All encoding attempts failed. Last error:\n{}".format(last_err))


def _mux_audio_from_input(out_silent: str, out_final: str, duration_s: float) -> None:
    if audio_off or duration_s <= 0.0:
        shutil.copyfile(out_silent, out_final)
        return
    if not input_video or not os.path.exists(str(input_video)):
        shutil.copyfile(out_silent, out_final)
        return

    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", out_silent,
        "-ss", str(float(video_start)),
        "-t", str(float(duration_s)),
        "-i", str(input_video),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "160k",
        "-shortest",
        "-movflags", "+faststart",
        out_final,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == 0:
        return

    print("Audio mux failed; writing without audio.\n{}".format(proc.stderr.decode("utf-8", "replace")), flush=True)
    shutil.copyfile(out_silent, out_final)


def _get_video_fps(video_path: str) -> float:
    ffprobe = _find_ffprobe()
    if ffprobe:
        cmd = [
            ffprobe, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=nokey=1:noprint_wrappers=1",
            video_path,
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode == 0:
            s = p.stdout.decode("utf-8", "replace").strip()
            if "/" in s:
                num, den = s.split("/", 1)
                try:
                    return float(num) / float(den)
                except Exception:
                    pass
            else:
                try:
                    return float(s)
                except Exception:
                    pass
    return 60.0


def _decode_video_frames_ffmpeg(video_path: str, start_s: float, duration_s: float, out_width: int, fps_out: float) -> Tuple[List[np.ndarray], int, int]:
    ffmpeg = _find_ffmpeg()

    # Probe height
    cmd_one = [
        ffmpeg,
        "-ss", str(float(start_s)),
        "-t", "0.1",
        "-i", video_path,
        "-vf", "scale={}:-2".format(int(out_width)),
        "-frames:v", "1",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1",
    ]
    one = subprocess.run(cmd_one, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if one.returncode != 0 or len(one.stdout) == 0:
        raise RuntimeError("Failed to probe video dimensions:\n{}".format(one.stderr.decode("utf-8", "replace")))

    w = int(out_width)
    h = int(len(one.stdout) // (w * 3))
    if h <= 0:
        raise RuntimeError("Could not infer video height from probe.")
    frame_size = w * h * 3

    cmd = [
        ffmpeg,
        "-ss", str(float(start_s)),
        "-t", str(float(duration_s)),
        "-i", video_path,
        "-vf", "scale={}:-2,fps={}".format(int(out_width), float(fps_out)),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    raw = proc.stdout.read()
    err = proc.stderr.read() if proc.stderr is not None else b""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError("ffmpeg decode failed:\n{}".format(err.decode("utf-8", "replace")))

    n = len(raw) // frame_size
    if n <= 0:
        raise RuntimeError("No frames decoded from video.")
    raw = raw[: n * frame_size]
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((n, h, w, 3))
    return [arr[i].copy() for i in range(n)], w, h


def _make_black_frame(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


# ----------------------------
# Animation rendering
# ----------------------------

def _compute_anim_header_px(video_h: int) -> Tuple[int, int, int, int]:
    # Compute consistent pixel sizes (stable, avoids overlap/centering surprises)
    base_total = float(video_h) + 260.0
    top_px = int(round(top_banner_frac * base_total))
    logo_px = int(round(logo_height * base_total))
    bottom_px = int(round(bottom_banner_frac * base_total))
    total_px = video_h + top_px + logo_px + bottom_px
    return total_px, top_px, logo_px, bottom_px


def _render_header_rgb(width_px: int, top_px: int, logo_px: int) -> np.ndarray:
    """
    Pixel-anchored left/right logos:
    - Left logo box x=0.03W
    - Right logo box ends at x=0.97W
    - Both scaled to fit logo band height with aspect preserved.
    Also includes centered titles.
    """
    dpi = 100
    fig_h_px = int(top_px + logo_px)
    fig = plt.figure(figsize=(width_px / dpi, fig_h_px / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")

    # Top banner
    ax_top = fig.add_axes([0, logo_px / fig_h_px, 1, top_px / fig_h_px])
    ax_top.set_facecolor(border_color)
    ax_top.set_xticks([]); ax_top.set_yticks([])
    for s in ax_top.spines.values():
        s.set_visible(False)
    ax_top.text(
        0.5, 0.55, Top_banner_line,
        ha="center", va="center",
        color="white",
        fontsize=int(anim_header_title_fontsize),
        fontweight="bold",
        family="sans-serif",
        transform=ax_top.transAxes,
    )

    # Logo/title band
    ax_band = fig.add_axes([0, 0, 1, logo_px / fig_h_px])
    ax_band.set_facecolor("white")
    ax_band.set_xticks([]); ax_band.set_yticks([])
    for s in ax_band.spines.values():
        s.set_visible(False)

    ax_band.text(
        0.5, 0.65, title_line1,
        ha="center", va="center",
        color=text_color,
        fontsize=int(anim_subtitle_fontsize),
        fontweight="bold",
        family="sans-serif",
        transform=ax_band.transAxes,
    )
    ax_band.text(
        0.5, 0.30, title_line2,
        ha="center", va="center",
        color=text_color,
        fontsize=int(anim_subtitle_fontsize),
        fontweight="bold",
        family="sans-serif",
        transform=ax_band.transAxes,
    )

    # Convert pixel to figure fractions
    fig_w_px = float(width_px)
    band_h_px = float(logo_px)
    pad_px = 0.03 * fig_w_px
    max_h_px = 0.85 * band_h_px

    def place_logo_pixels(img: np.ndarray, x_left_px: float) -> Tuple[float, float]:
        ih, iw = img.shape[0], img.shape[1]
        aspect = float(iw) / float(ih) if ih > 0 else 1.0
        h_px = max_h_px
        w_px = h_px * aspect
        y_bottom_px = 0.5 * (band_h_px - h_px)
        # Convert to figure coords
        x0 = x_left_px / fig_w_px
        y0 = y_bottom_px / float(fig_h_px)
        ww = w_px / fig_w_px
        hh = h_px / float(fig_h_px)
        axl = fig.add_axes([x0, y0, ww, hh])
        axl.imshow(img)
        axl.axis("off")
        return w_px, h_px

    # Left logo
    _wL, _hL = place_logo_pixels(Hawks_logo, x_left_px=pad_px)

    # Right logo (compute width first, then place so right edge hits 0.97W)
    ih, iw = PNSN_logo.shape[0], PNSN_logo.shape[1]
    aspect = float(iw) / float(ih) if ih > 0 else 1.0
    h_px = max_h_px
    w_px = h_px * aspect
    x_left_px = 0.97 * fig_w_px - w_px
    place_logo_pixels(PNSN_logo, x_left_px=x_left_px)

    rgb = _fig_to_rgb(fig)
    plt.close(fig)
    return rgb


def _render_bottom_banner_rgb(width_px: int, bottom_px: int) -> np.ndarray:
    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, bottom_px / dpi), dpi=dpi)
    fig.patch.set_facecolor(border_color)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(border_color)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    rgb = _fig_to_rgb(fig)
    plt.close(fig)
    return rgb


def _cursor_time_utc(windows: Dict[str, object], i: int, nframes: int, fps_used: float, one_frame_mode: bool) -> float:
    # Always end on plot_window_end_utc so the last frame includes seismogram_buffer_plot_end.
    if one_frame_mode or i == (nframes - 1):
        return float(windows["plot_window_end_utc"])
    return float(windows["reveal_start_utc"]) + float(i) / float(fps_used)


def _cursor_frac(windows: Dict[str, object], t_cursor_utc: float) -> float:
    t0 = float(windows["plot_window_start_utc"])
    t1 = float(windows["plot_window_end_utc"])
    if t1 <= t0:
        return 1.0
    f = (t_cursor_utc - t0) / (t1 - t0)
    return max(0.0, min(1.0, float(f)))


def _mask_reveal_rgb(full_rgb: np.ndarray, frac: float) -> np.ndarray:
    h, w, _ = full_rgb.shape
    x = int(round(frac * w))
    x = max(0, min(w, x))
    out = full_rgb.copy()
    if x < w:
        out[:, x:, :] = 255
    return out


def _mask_reveal_rgba_alpha(src_rgba: np.ndarray, frac: float) -> np.ndarray:
    h, w, _ = src_rgba.shape
    x = int(round(frac * w))
    x = max(0, min(w, x))
    out = src_rgba.copy()
    if x < w:
        out[:, x:, 3] = 0
    return out


def _alpha_composite_over(dst_rgb: np.ndarray, src_rgba: np.ndarray, top: int, left: int) -> None:
    h, w, _ = src_rgba.shape
    roi = dst_rgb[top:top + h, left:left + w, :].astype(np.float32)
    src_rgb = src_rgba[:, :, :3].astype(np.float32)
    a = (src_rgba[:, :, 3:4].astype(np.float32) / 255.0)
    comp = src_rgb * a + roi * (1.0 - a)
    dst_rgb[top:top + h, left:left + w, :] = comp.astype(np.uint8)


def _render_seismo_full_rgb(series: List[Tuple[TraceSpec, np.ndarray, np.ndarray, bool]], width_px: int, height_px: int) -> np.ndarray:
    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")

    ax = fig.add_axes([0.03, 0.08, 0.95, 0.86])
    ax.set_facecolor("white")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(border_color)
        s.set_linewidth(1.0)

    ts = [t for (_spec, t, _y, _ok) in series]
    ys = [y for (_spec, _t, y, _ok) in series]
    ys, normalize_mode, norm_spacing = _apply_normalization_if_negative_ylimits(ys)
    offsets = _compute_offsets(ys, normalize_mode=normalize_mode, norm_spacing=norm_spacing)
    ys_off = [y + offsets[i] for i, y in enumerate(ys)]

    for t, y_plot in zip(ts, ys_off):
        dt_utc = [dt.datetime.fromtimestamp(float(x), tz=dt.timezone.utc) for x in t]
        x_mpl = mdates.date2num(dt_utc)
        ax.plot(x_mpl, y_plot, color=trace_color, linewidth=trace_linewidth)

    dt0 = dt.datetime.fromtimestamp(float(ts[0][0]), tz=dt.timezone.utc)
    dt1 = dt.datetime.fromtimestamp(float(ts[0][-1]), tz=dt.timezone.utc)
    ax.set_xlim(mdates.date2num(dt0), mdates.date2num(dt1))
    ylo, yhi = _expand_ylim_for_offsets(ys_off)
    ax.set_ylim((ylo, yhi))

    rgb = _fig_to_rgb(fig)
    plt.close(fig)
    return rgb


def _render_seismo_lines_rgba(series: List[Tuple[TraceSpec, np.ndarray, np.ndarray, bool]], width_px: int, height_px: int) -> np.ndarray:
    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_alpha(0.0)

    ax = fig.add_axes([0.03, 0.08, 0.95, 0.86])
    ax.set_facecolor((1, 1, 1, 0))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    ts = [t for (_spec, t, _y, _ok) in series]
    ys = [y for (_spec, _t, y, _ok) in series]
    ys, normalize_mode, norm_spacing = _apply_normalization_if_negative_ylimits(ys)
    offsets = _compute_offsets(ys, normalize_mode=normalize_mode, norm_spacing=norm_spacing)
    ys_off = [y + offsets[i] for i, y in enumerate(ys)]

    for t, y_plot in zip(ts, ys_off):
        dt_utc = [dt.datetime.fromtimestamp(float(x), tz=dt.timezone.utc) for x in t]
        x_mpl = mdates.date2num(dt_utc)
        ax.plot(x_mpl, y_plot, color=trace_color, linewidth=trace_linewidth)

    dt0 = dt.datetime.fromtimestamp(float(ts[0][0]), tz=dt.timezone.utc)
    dt1 = dt.datetime.fromtimestamp(float(ts[0][-1]), tz=dt.timezone.utc)
    ax.set_xlim(mdates.date2num(dt0), mdates.date2num(dt1))
    ylo, yhi = _expand_ylim_for_offsets(ys_off)
    ax.set_ylim((ylo, yhi))

    rgba = _fig_to_rgba(fig)
    plt.close(fig)
    return rgba


def _render_seismo_base_rgb(width_px: int, height_px: int) -> np.ndarray:
    """
    White background + full rectangle spines. No wiggles.
    Used for stacked animations so the box is always present.
    """
    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")

    ax = fig.add_axes([0.03, 0.08, 0.95, 0.86])
    ax.set_facecolor("white")
    ax.set_xticks([]); ax.set_yticks([])

    for s in ax.spines.values():
        s.set_color(border_color)
        s.set_linewidth(2.0)
        s.set_alpha(1.0)

    rgb = _fig_to_rgb(fig)
    plt.close(fig)
    return rgb


def _render_seismo_box_rgba(width_px: int, height_px: int) -> np.ndarray:
    """
    Transparent background; only a full rectangle spine/box (always present).
    """
    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_alpha(0.0)

    ax = fig.add_axes([0.03, 0.08, 0.95, 0.86])
    ax.set_facecolor((1, 1, 1, 0))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(border_color)
        s.set_linewidth(2.0)
        s.set_alpha(1.0)

    rgba = _fig_to_rgba(fig)
    plt.close(fig)
    return rgba


def _make_rect_rgba(width_px: int, height_px: int, color: str, alpha: float) -> np.ndarray:
    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((1, 1, 1, 0))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=color, edgecolor="none", alpha=float(alpha)))
    rgba = _fig_to_rgba(fig)
    plt.close(fig)
    return rgba


def _prepare_video_frames_and_fps(windows: Dict[str, object]) -> Tuple[List[np.ndarray], int, int, float, bool, bool]:
    have_video = bool(input_video) and os.path.exists(str(input_video))
    reveal_duration = float(windows["reveal_duration"])
    chopped_duration = float(video_end) - float(video_start)

    if not have_video:
        return [_make_black_frame(int(anim_out_width_px), int(blank_video_height_px))], int(anim_out_width_px), int(blank_video_height_px), 1.0, True, False

    native = _get_video_fps(str(input_video))
    if fps is None:
        fps_used = float(native)
    elif float(fps) < 0:
        fps_used = float(native)
    else:
        fps_used = float(fps)

    one_frame_mode = (int(fps) == 0)

    decode_fps = 1.0 if one_frame_mode else fps_used

    if (not one_frame_mode) and abs(chopped_duration - reveal_duration) > 1e-3:
        raise ValueError(
            "Chopped video duration ({:.3f}s) must equal reveal duration ({:.3f}s). Adjust video_start/video_end or plot times."
            .format(chopped_duration, reveal_duration)
        )

    frames, w, h = _decode_video_frames_ffmpeg(
        video_path=str(input_video),
        start_s=float(video_start),
        duration_s=float(chopped_duration),
        out_width=int(anim_out_width_px),
        fps_out=float(decode_fps),
    )
    if one_frame_mode:
        frames = [frames[-1]]

    return frames, w, h, fps_used, one_frame_mode, True


def _make_stacked_animation(
    windows: Dict[str, object],
    series: List[Tuple[TraceSpec, np.ndarray, np.ndarray, bool]],
    outpath: str,
) -> None:
    vid_frames, vid_w, vid_h, fps_used, one_frame_mode, have_video = _prepare_video_frames_and_fps(windows)
    chopped_duration = float(video_end) - float(video_start)

    _total_px, top_px, logo_px, bottom_px = _compute_anim_header_px(vid_h)
    header_rgb = _render_header_rgb(width_px=vid_w, top_px=top_px, logo_px=logo_px)
    bottom_rgb = _render_bottom_banner_rgb(width_px=vid_w, bottom_px=bottom_px)

    seismo_h = int(round(vid_h * float(anim_stacked_seismo_height_frac)))

    # Base panel: WHITE + FULL BOX (always visible)
    seismo_base_rgb = _render_seismo_base_rgb(width_px=vid_w, height_px=seismo_h)

    # Wiggles only, transparent background
    seismo_lines_rgba_full = _render_seismo_lines_rgba(series, width_px=vid_w, height_px=seismo_h)

    nframes = len(vid_frames)
    build_prog = _print_progress_factory(nframes, "stacked build frames ({})".format(outpath))
    frames_out: List[np.ndarray] = []

    for i in range(nframes):
        # cursor time: force final frame to include buffer_plot_end by ending at plot_window_end_utc
        t_cursor = _cursor_time_utc(windows, i, nframes, fps_used, one_frame_mode)
        frac = _cursor_frac(windows, t_cursor)

        # Reveal only the wiggles (alpha mask); do NOT touch the base box
        lines_rgba = _mask_reveal_rgba_alpha(seismo_lines_rgba_full, frac)

        seismo_rgb = seismo_base_rgb.copy()
        _alpha_composite_over(seismo_rgb, lines_rgba, top=0, left=0)

        frame = np.vstack([header_rgb, vid_frames[i], seismo_rgb, bottom_rgb])
        frames_out.append(_ensure_even_dims_rgb(frame))
        build_prog(i)

    out_silent = outpath + ".silent.mp4"
    _write_mp4_ffmpeg_pipe(
        frames_out,
        fps_out=(1 if one_frame_mode else int(round(fps_used))),
        outpath=out_silent,
    )

    if audio_off or one_frame_mode or (not have_video):
        shutil.move(out_silent, outpath)
    else:
        _mux_audio_from_input(out_silent, outpath, duration_s=float(chopped_duration))
        os.remove(out_silent)


def _make_overlay_animation(
    windows: Dict[str, object],
    series: List[Tuple[TraceSpec, np.ndarray, np.ndarray, bool]],
    outpath: str,
) -> None:
    vid_frames, vid_w, vid_h, fps_used, one_frame_mode, have_video = _prepare_video_frames_and_fps(windows)
    chopped_duration = float(video_end) - float(video_start)

    _total_px, top_px, logo_px, bottom_px = _compute_anim_header_px(vid_h)
    header_rgb = _render_header_rgb(width_px=vid_w, top_px=top_px, logo_px=logo_px)
    bottom_rgb = _render_bottom_banner_rgb(width_px=vid_w, bottom_px=bottom_px)

    overlay_h = int(round(vid_h * float(anim_overlay_height_frac)))
    overlay_w = vid_w
    overlay_top_in_video = vid_h - overlay_h
    overlay_top_in_frame = header_rgb.shape[0] + overlay_top_in_video

    rect_rgba = _make_rect_rgba(overlay_w, overlay_h, color=animation_transparency_color, alpha=float(animation_transparency))
    box_rgba = _render_seismo_box_rgba(overlay_w, overlay_h)          # ALWAYS present
    lines_rgba_full = _render_seismo_lines_rgba(series, overlay_w, overlay_h)

    nframes = len(vid_frames)
    build_prog = _print_progress_factory(nframes, "overlay build frames ({})".format(outpath))
    frames_out: List[np.ndarray] = []

    for i in range(nframes):
        base = np.vstack([header_rgb, vid_frames[i], bottom_rgb]).astype(np.uint8)

        t_cursor = _cursor_time_utc(windows, i, nframes, fps_used, one_frame_mode)
        frac = _cursor_frac(windows, t_cursor)

        lines_rgba = _mask_reveal_rgba_alpha(lines_rgba_full, frac)

        _alpha_composite_over(base, rect_rgba, top=overlay_top_in_frame, left=0)
        _alpha_composite_over(base, box_rgba, top=overlay_top_in_frame, left=0)   # FULL rectangle every frame
        _alpha_composite_over(base, lines_rgba, top=overlay_top_in_frame, left=0)

        frames_out.append(_ensure_even_dims_rgb(base))
        build_prog(i)

    out_silent = outpath + ".silent.mp4"
    _write_mp4_ffmpeg_pipe(frames_out, fps_out=(1 if one_frame_mode else int(round(fps_used))), outpath=out_silent)

    if audio_off or one_frame_mode or (not have_video):
        shutil.move(out_silent, outpath)
    else:
        _mux_audio_from_input(out_silent, outpath, duration_s=float(chopped_duration))
        os.remove(out_silent)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    t0 = time.perf_counter()

    windows = _compute_windows()
    client = Client("IRIS")

    specs_multi = _parse_station_channels(station_channels, default_net=network, default_loc=location)

    todo = 1 + len(specs_multi)
    prog = _print_progress_factory(todo, "download/process")

    spec_single = TraceSpec(network=network, station=station, location=location, channel=channel)
    t1, y1, _fs1, ok1 = _download_and_process(
        client=client,
        spec=spec_single,
        download_start_utc=windows["download_start_utc"],
        download_end_utc=windows["download_end_utc"],
        plot_start_utc=windows["plot_window_start_utc"],
        plot_end_utc=windows["plot_window_end_utc"],
    )
    series_single = [(spec_single, t1, y1, ok1)]
    prog(0)

    series_multi: List[Tuple[TraceSpec, np.ndarray, np.ndarray, bool]] = []
    for i, spec in enumerate(specs_multi, start=1):
        t, y, _fs, ok = _download_and_process(
            client=client,
            spec=spec,
            download_start_utc=windows["download_start_utc"],
            download_end_utc=windows["download_end_utc"],
            plot_start_utc=windows["plot_window_start_utc"],
            plot_end_utc=windows["plot_window_end_utc"],
        )
        series_multi.append((spec, t, y, ok))
        prog(i)

    _plot_png_one_panel(series_single, outpath=out_png_single, show_station_labels_right=False)
    print("Wrote {}".format(out_png_single), flush=True)

    _plot_png_one_panel(series_multi, outpath=out_png_multi, show_station_labels_right=True)
    print("Wrote {}".format(out_png_multi), flush=True)

    _make_stacked_animation(windows, series_single, out_mp4_stacked_single)
    print("Wrote {}".format(out_mp4_stacked_single), flush=True)

    _make_stacked_animation(windows, series_multi, out_mp4_stacked_multi)
    print("Wrote {}".format(out_mp4_stacked_multi), flush=True)

    _make_overlay_animation(windows, series_single, out_mp4_overlay_single)
    print("Wrote {}".format(out_mp4_overlay_single), flush=True)

    _make_overlay_animation(windows, series_multi, out_mp4_overlay_multi)
    print("Wrote {}".format(out_mp4_overlay_multi), flush=True)

    t1 = time.perf_counter()
    print("Total runtime: {:.2f} seconds".format(t1 - t0), flush=True)


if __name__ == "__main__":
    main()



