"""Regenerate fig03 (Framingham validation case) and fig04 (temporal distribution)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

OUT = Path(__file__).resolve().parents[1]

GEN = "#0E7C7B"
REFC = "#E07B39"
INK = "#333333"
MUTED = "#6B7280"
PAPER = "#F7F4EC"
GREEN = "#2E8B57"
BROWN = "#8B6B4A"
RED = "#C0392B"
BLUE = "#3465A4"
ORANGE = "#E07B39"


def style(base=13):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
    })


def save(fig, stem):
    for ext in (".png", ".pdf"):
        fig.savefig(OUT / f"{stem}{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", stem)


def draw_house(ax, x, y, s=1.0, color=RED, z=5):
    w, h = 0.016 * s, 0.014 * s
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=color, edgecolor="white", lw=0.5, zorder=z))
    ax.add_patch(Polygon([(x - w * 0.72, y + h / 2), (x + w * 0.72, y + h / 2), (x, y + h * 1.25)],
                         facecolor=color, edgecolor="white", lw=0.5, zorder=z))


def draw_multifamily(ax, x, y, s=1.0, z=5):
    w, h = 0.016 * s, 0.030 * s
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=BLUE, edgecolor="white", lw=0.5, zorder=z))
    for r in range(3):
        for c in range(2):
            ax.add_patch(Rectangle((x - w * 0.31 + c * w * 0.38, y - h * 0.34 + r * h * 0.3),
                                   w * 0.24, h * 0.16, facecolor="white", edgecolor="none", zorder=z + 1))


def draw_borefield(ax, x, y, label, s=1.0, z=6):
    for radius, alpha in ((0.020, 0.28), (0.013, 0.55), (0.006, 1.0)):
        ax.add_patch(Circle((x, y), radius * s, facecolor=ORANGE, edgecolor="none", alpha=alpha, zorder=z))
    if label:
        ax.text(x, y - 0.032 * s, label, ha="center", va="top", fontsize=10.5, color="#8A4A0E",
                fontweight="bold", zorder=z + 1, linespacing=1.05)


def fig03():
    style()
    fig, ax = plt.subplots(figsize=(16.4, 8.3))
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0.02, 0.995)
    ax.set_ylim(0.062, 0.935)
    ax.axis("off")

    # ------------------------------------------------ map panel
    ax.add_patch(FancyBboxPatch((0.03, 0.175), 0.59, 0.735,
                                boxstyle="round,pad=0.006,rounding_size=0.012",
                                facecolor=PAPER, edgecolor="#D8D5C8", lw=1.2, zorder=0))
    # roads
    roads = [
        [(0.055, 0.80), (0.16, 0.79), (0.32, 0.79), (0.50, 0.78), (0.60, 0.80)],
        [(0.21, 0.79), (0.21, 0.62), (0.245, 0.50), (0.245, 0.36), (0.41, 0.37), (0.435, 0.47), (0.535, 0.43), (0.56, 0.55), (0.56, 0.79)],
        [(0.435, 0.47), (0.50, 0.45), (0.555, 0.43), (0.595, 0.46)],
        [(0.055, 0.30), (0.16, 0.25), (0.27, 0.30), (0.41, 0.315), (0.585, 0.21)],
        [(0.065, 0.66), (0.10, 0.56), (0.115, 0.42), (0.115, 0.26)],
    ]
    for road in roads:
        xs, ys = zip(*road)
        ax.plot(xs, ys, color="#7D7D73", lw=1.1, alpha=0.7, solid_capstyle="round", zorder=1)

    route = [(0.21, 0.79), (0.38, 0.79), (0.56, 0.77), (0.56, 0.55), (0.535, 0.43),
             (0.435, 0.47), (0.41, 0.37), (0.245, 0.36), (0.245, 0.50), (0.21, 0.62), (0.21, 0.79)]
    xs, ys = zip(*route)
    ax.plot(xs, ys, color=GREEN, lw=7, alpha=0.85, solid_capstyle="round", zorder=2)
    ax.plot(xs, ys, color="white", lw=2.2, alpha=0.7, solid_capstyle="round", zorder=3)

    road_labels = [
        ("Flagg Dr", 0.085, 0.735), ("Rose Kennedy Ln", 0.30, 0.735),
        ("Concord St", 0.505, 0.725), ("Normandy Rd", 0.45, 0.845),
        ("Hampden Rd", 0.315, 0.585), ("Hampshire Rd", 0.41, 0.545),
        ("Berkshire Rd", 0.10, 0.50), ("Lindbergh Rd", 0.50, 0.485),
        ("Hartford St", 0.578, 0.565), ("Burdett Ave", 0.53, 0.245),
        ("Prindiville Ave", 0.10, 0.335), ("Denison Ave", 0.09, 0.225),
    ]
    for text, x, y in road_labels:
        ax.text(x, y, text, ha="center", va="center", fontsize=10.5, color="#5A5A52",
                zorder=4, style="italic")

    # participant buildings (named, fully inside the panel)
    participants = [
        (0.065, 0.585, 0.115, 0.075, "Framingham\nHousing Authority"),
        (0.24, 0.82, 0.155, 0.055, "Framingham Public Schools\nWelcome Center"),
        (0.445, 0.585, 0.115, 0.075, "Framingham\nFire Department"),
        (0.455, 0.685, 0.10, 0.055, "Municipal\nbuilding"),
        (0.235, 0.405, 0.10, 0.05, "Commercial\nbuilding"),
    ]
    for x, y, w, h, label in participants:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.006",
                                    facecolor=BROWN, edgecolor="white", lw=1.0, zorder=5))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.3,
                color="white", fontweight="bold", zorder=6, linespacing=1.15)

    # residential
    houses = [(0.145, 0.385), (0.175, 0.41), (0.20, 0.345), (0.245, 0.325), (0.285, 0.33),
              (0.325, 0.335), (0.365, 0.34), (0.40, 0.41), (0.435, 0.395), (0.47, 0.415),
              (0.50, 0.375), (0.535, 0.385), (0.185, 0.34)]
    for x, y in houses:
        draw_house(ax, x, y, 1.25)
    for x, y in [(0.155, 0.545), (0.155, 0.505), (0.155, 0.465), (0.185, 0.525)]:
        draw_multifamily(ax, x, y, 1.1)

    draw_borefield(ax, 0.26, 0.665, "B1", 1.15)
    draw_borefield(ax, 0.415, 0.655, "B2", 1.15)
    draw_borefield(ax, 0.33, 0.525, "B3", 1.15)

    # legend strip under the map
    ly = 0.122
    ax.add_patch(FancyBboxPatch((0.03, 0.085), 0.59, 0.075,
                                boxstyle="round,pad=0.006,rounding_size=0.01",
                                facecolor="white", edgecolor="#D8D5C8", lw=1.0, zorder=0))
    ax.plot([0.045, 0.082], [ly, ly], color=GREEN, lw=7, solid_capstyle="round", zorder=8)
    ax.text(0.09, ly, "GEN loop route", va="center", fontsize=10.5)
    draw_borefield(ax, 0.205, ly, "", 0.8, z=8)
    ax.text(0.228, ly, "Borefields", va="center", fontsize=10.5)
    draw_house(ax, 0.318, ly - 0.004, 1.0, z=8)
    ax.text(0.334, ly, "Residential", va="center", fontsize=10.5)
    draw_multifamily(ax, 0.428, ly, 0.85, z=8)
    ax.text(0.443, ly, "FHA multifamily", va="center", fontsize=10.5)
    sq = 0.016
    ax.add_patch(Rectangle((0.528, ly - sq / 2), sq, sq, facecolor=BROWN, edgecolor="white", lw=0.7, zorder=8))
    ax.text(0.549, ly + 0.012, "Commercial /", va="center", fontsize=8.7)
    ax.text(0.549, ly - 0.014, "municipal", va="center", fontsize=8.7)
    ax.text(0.605, 0.192, "Schematic — not to scale", ha="right", va="center", fontsize=9.5,
            color=MUTED, style="italic", zorder=7)

    # ------------------------------------------------ right info panels
    def panel(y0, h, header, header_color, lines):
        ax.add_patch(FancyBboxPatch((0.645, y0), 0.335, h,
                                    boxstyle="round,pad=0.006,rounding_size=0.012",
                                    facecolor="white", edgecolor="#D7DCE0", lw=1.2, zorder=2))
        ax.add_patch(FancyBboxPatch((0.645, y0 + h - 0.052), 0.335, 0.052,
                                    boxstyle="round,pad=0.006,rounding_size=0.012",
                                    facecolor=header_color, edgecolor="none", zorder=3))
        ax.add_patch(Rectangle((0.645, y0 + h - 0.052), 0.335, 0.026, facecolor=header_color,
                               edgecolor="none", zorder=3))
        ax.text(0.8125, y0 + h - 0.026, header, ha="center", va="center", fontsize=12.3,
                fontweight="bold", color="white", zorder=4)
        yy = y0 + h - 0.085
        for text, big, bold in lines:
            ax.text(0.662, yy, text, ha="left", va="center",
                    fontsize=14.5 if big else 11.3, fontweight="bold" if bold else "normal",
                    color=INK, zorder=4)
            yy -= 0.052 if big else 0.043
        return yy

    panel(0.672, 0.238, "Common service basis (functional unit)", GEN, [
        ("89.375 GWh$_{\\mathrm{th}}$ delivered service", True, True),
        ("space heating, cooling & domestic hot water", False, False),
        ("50 years (2025–2075) = 1.787 GWh$_{\\mathrm{th}}$ yr$^{-1}$", False, False),
        ("identical denominator for GEN and REF", False, False),
    ])
    panel(0.40, 0.245, "System composition", "#4A6FA5", [
        ("37 modeled buildings", True, True),
        ("23 single-family residential", False, False),
        ("9 FHA multifamily  ·  5 commercial / municipal", False, False),
        ("3 borefields  ·  115 boreholes", True, True),
    ])
    panel(0.085, 0.275, "Modeling basis", "#B07A4A", [
        ("HEATNETS + URBANopt loads", True, True),
        ("project specification, engineering assumptions,", False, False),
        ("and modeled loads on one weather basis", False, False),
        ("GEN operation rerun through modified HEATNETS;", False, False),
        ("REF reconstructed on the same delivered load", False, False),
    ])
    save(fig, "fig03_framingham_case")


def fig04():
    style()
    fig, ax = plt.subplots(figsize=(17.6, 9.2))
    fig.subplots_adjust(left=0.155, right=0.99, top=0.945, bottom=0.105)
    ax.set_xlim(2021, 2126)
    ax.set_ylim(-0.15, 6.35)
    ax.set_xlabel("Calendar year", fontsize=20, labelpad=10)
    rows = {
        "reference\nperiod": 5.66, "Stage A": 4.85, "B6 operation": 4.00,
        "B2–B4\nreplacements": 3.05, "Stage C": 2.18, "future\nbackgrounds": 1.36,
        "climate response": 0.48,
    }
    ax.set_yticks(list(rows.values()))
    ax.set_yticklabels(list(rows.keys()), fontsize=18, fontweight="bold", color=INK)
    ax.set_xticks([2025, 2045, 2065, 2085, 2105, 2125])
    ax.grid(True, axis="x", color="#E4E8EC", lw=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=17)

    ax.axvspan(2025, 2075, ymin=0.05, ymax=0.95, color="#EFF4F8", zorder=0)
    ax.axvspan(2075, 2125, ymin=0.05, ymax=0.95, color="#FBF5E8", zorder=0)
    ax.axvline(2025, color="#B03A2E", lw=1.4)
    ax.axvline(2075, color="#B03A2E", lw=1.2, ls="--")
    ax.text(2024.2, 6.22, "start\n(year 0)", color="#B03A2E", fontsize=11.5, ha="right", va="center", linespacing=1.05)
    ax.text(2050, 6.22, "50-year study period for GEN and REF", ha="center", va="center",
            fontsize=16.5, color=INK, fontweight="bold")
    ax.text(2100, 6.22, "dynamic climate-response horizon", ha="center", va="center",
            fontsize=16.5, color=INK, fontweight="bold")

    ax.hlines(rows["reference\nperiod"], 2025, 2075, color="#5D6D7E", lw=6)

    # Stage A pulses
    ax.scatter(2025, 5.02, marker="D", s=120, color=GEN, edgecolor="white", lw=1.1, zorder=6)
    ax.text(2027.5, 5.02, "GEN construction (year 0)", ha="left", va="center", fontsize=14,
            style="italic", color=GEN, fontweight="bold")
    ax.scatter(2025, 4.68, marker="D", s=120, color=REFC, edgecolor="white", lw=1.1, zorder=6)
    ax.text(2027.5, 4.68, "REF equipment & gas/oil infrastructure (year 0)", ha="left", va="center",
            fontsize=14, style="italic", color=REFC, fontweight="bold")

    # B6 operation
    ax.hlines(4.14, 2026, 2075, color=GEN, lw=15)
    ax.hlines(3.80, 2026, 2075, color=REFC, lw=15)
    ax.text(2050, 4.14, "GEN annual electricity (years 1–50)", ha="center", va="center",
            fontsize=13, color="white", fontweight="bold")
    ax.text(2050, 3.80, "REF annual gas, oil & electricity (years 1–50)", ha="center", va="center",
            fontsize=13, color="white", fontweight="bold")

    # replacements
    ax.text(2026.5, 3.44, "GEN circulation pumps (yr 15, 30, 45)", ha="left", va="center",
            fontsize=13.5, color=GEN, fontweight="bold")
    for year in [2040, 2055, 2070]:
        ax.hlines(3.24, year - 5, year + 5, color="#BCE0DE", lw=6)
        ax.scatter(year, 3.24, marker="D", s=95, color=GEN, edgecolor="white", lw=1.0, zorder=6)
    ax.text(2026.5, 2.99, "GEN heat pumps (yr 25)", ha="left", va="center", fontsize=13.5,
            color=GEN, fontweight="bold")
    ax.hlines(2.82, 2045, 2055, color="#BCE0DE", lw=6)
    ax.scatter(2050, 2.82, marker="D", s=95, color=GEN, edgecolor="white", lw=1.0, zorder=6)
    ax.text(2026.5, 2.57, "REF furnaces & AC (yr 20, 40)", ha="left", va="center", fontsize=13.5,
            color=REFC, fontweight="bold")
    for year in [2045, 2065]:
        ax.hlines(2.42, year - 5, year + 5, color="#F5D5BC", lw=6)
        ax.scatter(year, 2.42, marker="D", s=95, color=REFC, edgecolor="white", lw=1.0, zorder=6)
    ax.text(2071.0, 2.66, "replacement timing varied ±5 yr (sensitivity)", ha="center", va="center",
            fontsize=11.5, color=MUTED, style="italic")

    # Stage C
    ax.scatter(2075, rows["Stage C"], marker="D", s=130, color="#B03A2E", edgecolor="white",
               lw=1.0, zorder=6)
    ax.text(2072.5, rows["Stage C"], "GEN & REF end-of-life (yr 50)", ha="right", va="center",
            fontsize=14, style="italic", color="#B03A2E", fontweight="bold")

    # future backgrounds
    for year in [2025, 2040, 2045, 2050, 2055]:
        ax.scatter(year, rows["future\nbackgrounds"], s=95, color="#2E86AB", edgecolor="white",
                   lw=0.9, zorder=6)
        ax.text(year, rows["future\nbackgrounds"] + 0.16, str(year), rotation=90, ha="center",
                va="bottom", fontsize=12, color="#2E86AB")
    ax.text(2062, rows["future\nbackgrounds"] + 0.05,
            "premise snapshots (SSP2-PkBudg1000 / SSP2-NPi),\ninterpolated between years",
            ha="left", va="center", fontsize=13.5, color="#2E86AB", style="italic", linespacing=1.2)

    # climate response
    ax.hlines(rows["climate response"], 2026, 2125, color="#CFC7EA", lw=12)
    ax.text(2028, rows["climate response"] + 0.30,
            "dynamic GWP100 & radiative forcing tracked over the 100-year climate-response horizon",
            ha="left", va="center", fontsize=14.5, color="#51488F")

    # >50-yr callout
    box = FancyBboxPatch((2078.5, 3.22), 46.5, 1.42, boxstyle="round,pad=0.02,rounding_size=0.02",
                         facecolor="white", edgecolor="#D6DBDF", lw=1.0, zorder=4)
    ax.add_patch(box)
    ax.text(2081, 4.42, "Components likely to exceed 50 years", ha="left", va="center",
            fontsize=13, fontweight="bold", color=INK, zorder=5)
    for i, item in enumerate([
        "• ground loop / borefield: 50+ yr",
        "• buried HDPE piping: up to ~100 yr",
        "• heat pumps & pumps: replaced within 50 yr",
        "Core case: no residual-life or recycling credit",
    ]):
        ax.text(2081.5, 4.13 - i * 0.245, item, ha="left", va="center",
                fontsize=11.6, color=INK if i < 3 else MUTED, zorder=5)
    ax.text(2100, 2.18, "long-lived GEN infrastructure may retain residual\nlife beyond year 50 (Module D, sensitivity S6)",
            ha="center", va="center", fontsize=12.5, color=GEN, style="italic")
    save(fig, "fig04_temporal_distribution")


if __name__ == "__main__":
    fig03()
    fig04()
