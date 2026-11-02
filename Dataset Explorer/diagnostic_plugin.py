"""
diagnostic_plugin.py
────────────────────
DiagnosticPlugin — file-to-file visual comparison tool.

Unlike every other plugin this one reads NO HDF5 data.  Its data dict is
populated by the orchestrator directly from files the user loaded through
Options ▸ Compare arrays from files (advanced).  It is never registered in REGISTRY (no tree entry).

Data dict keys
──────────────
  "diag_A"       ndarray   — first loaded file  (required)
  "diag_B"       ndarray   — second loaded file (optional; None = solo view)

Params dict keys
────────────────
  "diag_name_A"  str       — basename of file A
  "diag_name_B"  str       — basename of file B
  "diag_z_swap"  bool      — False: A(teal) on top / True: B(coral) on top

Tabs
────
  Overlay : the primary view — line plot (1-D) or heatmap (2-D)
  Matrix tab is handled by the orchestrator directly (bypasses compute).

Colour convention
─────────────────
  File A → teal  (#4ecdc4)  primary / reference
  File B → coral (#e07b54)  comparison

  Default z-order (z_swap=False): A on top, B behind.
  After swap (z_swap=True):       B on top, A behind.

1-D draw order  → controlled by CurveSpec list order (last = top layer)
2-D draw order  → controlled by ImageSpec.z_value (higher = top layer)

Origin anchoring
────────────────
  Both images / curves are anchored at (0, 0).
  If A is 128×128 and B is 256×256, A occupies the bottom-left quadrant.
  No resampling or interpolation is ever applied.

Alpha transparency (2-D)
────────────────────────
  The data minimum maps to alpha=0 (fully transparent) so the background
  void does not wash out the layer beneath it.  Values rise linearly to
  alpha=255 at the data maximum.
"""

from __future__ import annotations

import numpy as np

from plugin_base import (
    BasePlugin, PlotRequest, CurveSpec, MarkerSpec, ImageSpec,
)

N_DS = 4_000   # min-max downsample budget for 1-D curves

# ── Thin-matrix detection ─────────────────────────────────────────────────────
# A 2-D array with few columns (e.g. N×6 phase-space matrix) is rendered as
# overlaid per-column line plots rather than a useless few-pixel-wide heatmap.
MAX_THIN_COLS = 10

# Distinguishable palette for up to 10 per-column curves
_MULTI_CURVE_PALETTE = [
    "#4ecdc4",   # teal
    "#e07b54",   # coral
    "#a8e06d",   # lime
    "#d4a5f5",   # lavender
    "#f5d76e",   # gold
    "#ff6b8a",   # rose
    "#5bcefa",   # sky
    "#ff9f43",   # tangerine
    "#c8d6e5",   # silver
    "#f368e0",   # magenta
]


class DiagnosticPlugin(BasePlugin):

    GROUP    = "_Diagnostic"
    DATASET  = "Overlay"
    REQUIRES = []                   # no HDF5 reads
    TABS     = ["Overlay"]
    TREE_CONTRIBUTIONS = {}         # does not appear in the file tree
    DISPLAY_NAME    = "Diagnostic Overlay"
    DEFAULT_SCALE_X = "linear"
    DEFAULT_SCALE_Y = "linear"

    # ── Public interface ──────────────────────────────────────────────────────

    def compute(
        self,
        data:    dict[str, np.ndarray],
        attrs:   dict[str, dict],
        header:  dict,
        params:  dict,
        tab:     str,
        scale_x: str,
        scale_y: str,
        window:  tuple[int, int],
    ) -> PlotRequest:

        arr_a  = data.get("diag_A")
        arr_b  = data.get("diag_B")      # None when only one file loaded
        name_a = params.get("diag_name_A", "File A")
        name_b = params.get("diag_name_B", "File B")
        z_swap = bool(params.get("diag_z_swap", False))

        if arr_a is None:
            return self.error_request("No diagnostic data loaded.")

        arr_a = arr_a.astype(np.float64)
        if arr_b is not None:
            arr_b = arr_b.astype(np.float64)

        # ── Dimension mismatch guard ──────────────────────────────────────────
        if arr_b is not None and arr_b.ndim != arr_a.ndim:
            return self.error_request(
                f"Cannot overlay {arr_a.ndim}-D '{name_a}' "
                f"with {arr_b.ndim}-D '{name_b}'.  "
                f"Both files must have the same number of dimensions."
            )

        ndim = arr_a.ndim

        if arr_b is None:
            return self._solo(arr_a, name_a, ndim, scale_x, scale_y)
        else:
            return self._overlay(arr_a, name_a, arr_b, name_b,
                                 ndim, z_swap, scale_x, scale_y)

    # ── Solo (one file) ───────────────────────────────────────────────────────

    def _solo(
        self,
        arr:     np.ndarray,
        name:    str,
        ndim:    int,
        scale_x: str,
        scale_y: str,
    ) -> PlotRequest:

        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        n    = arr.size

        info = (
            f"⚡ Diagnostic  |  {name}  |  "
            f"shape: {arr.shape}  |  dtype: {arr.dtype}  |  "
            f"min: {vmin:.4e}  max: {vmax:.4e}  mean: {float(np.nanmean(arr)):.4e}"
        )

        if ndim == 1:
            return self._solo_1d(arr, name, scale_x, scale_y, vmin, vmax, n, info)
        elif ndim == 2 and arr.shape[1] <= MAX_THIN_COLS:
            return self._solo_multi_col(
                arr, name, scale_x, scale_y, vmin, vmax, info)
        else:
            return self._solo_2d(arr, name, vmin, vmax, n, info)

    def _solo_1d(
        self, arr, name, scale_x, scale_y, vmin, vmax, n, info
    ) -> PlotRequest:

        x_full    = np.arange(n, dtype=np.float64)
        xd, yd    = self._minmax_downsample(arr, N_DS, x_full)
        xd, yd    = self._anchor_origin(xd, yd, arr)

        curve = CurveSpec(
            x     = xd,
            y     = yd,
            color = "#4ecdc4",
            width = 1.0,
            label = name,
        )

        return PlotRequest(
            title     = f"⚡ {name}",
            x_label   = "Index",
            y_label   = "Value",
            curves    = [curve],
            log_x     = (scale_x == "log"),
            log_y     = (scale_y == "log"),
            status    = f"Solo — {n:,} values  |  [{vmin:.3e}, {vmax:.3e}]",
            info      = info,
            tab       = "Overlay",
            n_points  = n,
            dim_label = "D0",
        )

    def _solo_2d(
        self, arr, name, vmin, vmax, n, info
    ) -> PlotRequest:

        image = ImageSpec(
            data     = arr,
            colormap = "teal",
            z_value  = 0,
            label    = name,
        )
        rows, cols = arr.shape

        return PlotRequest(
            title     = f"⚡ {name}",
            x_label   = "Column",
            y_label   = "Row",
            images    = [image],
            log_x     = False,
            log_y     = False,
            status    = (
                f"Solo — {rows}×{cols} = {rows*cols:,} cells  |  "
                f"[{vmin:.3e}, {vmax:.3e}]"
            ),
            info      = info,
            tab       = "Overlay",
            n_points  = rows * cols,
            dim_label = "D0,D1",
        )

    # ── Solo multi-column (thin 2-D) ──────────────────────────────────────────

    def _solo_multi_col(
        self,
        arr:     np.ndarray,
        name:    str,
        scale_x: str,
        scale_y: str,
        vmin:    float,
        vmax:    float,
        info:    str,
    ) -> PlotRequest:
        """
        Render a thin 2-D array (e.g. N×6 phase-space matrix) as overlaid
        per-column line plots.

        Each column becomes a separate labeled curve with a distinct colour
        from _MULTI_CURVE_PALETTE.  This replaces the useless heatmap that
        would otherwise be generated for arrays with very few columns.

        Column labels use a fixed physical convention for 6-column phase-space
        matrices: [x, y, z, v_x, v_y, v_z].  For other widths, generic
        "Col N" labels are used.
        """
        rows, cols = arr.shape

        # Column labels — phase-space convention for 6 cols
        if cols == 6:
            col_labels = ["x", "y", "z", "v_x", "v_y", "v_z"]
        else:
            col_labels = [f"Col {i}" for i in range(cols)]

        curves: list[CurveSpec] = []
        for c in range(cols):
            col_data = arr[:, c]
            x_full   = np.arange(rows, dtype=np.float64)
            xd, yd   = self._minmax_downsample(col_data, N_DS, x_full)
            xd, yd   = self._anchor_origin(xd, yd, col_data)

            color = _MULTI_CURVE_PALETTE[c % len(_MULTI_CURVE_PALETTE)]
            curves.append(CurveSpec(
                x     = xd,
                y     = yd,
                color = color,
                width = 1.0,
                label = col_labels[c],
            ))

        return PlotRequest(
            title     = f"⚡ {name}",
            x_label   = "Index",
            y_label   = "Value",
            curves    = curves,
            log_x     = (scale_x == "log"),
            log_y     = (scale_y == "log"),
            status    = (
                f"Multi-column — {rows:,}×{cols} = {rows*cols:,} values  |  "
                f"[{vmin:.3e}, {vmax:.3e}]  |  {cols} channels"
            ),
            info      = info,
            tab       = "Overlay",
            n_points  = rows,
            dim_label = f"D0×{cols}",
        )

    # ── Overlay (two files) ───────────────────────────────────────────────────

    def _overlay(
        self,
        arr_a:   np.ndarray,
        name_a:  str,
        arr_b:   np.ndarray,
        name_b:  str,
        ndim:    int,
        z_swap:  bool,
        scale_x: str,
        scale_y: str,
    ) -> PlotRequest:

        vmin_a, vmax_a = float(np.nanmin(arr_a)), float(np.nanmax(arr_a))
        vmin_b, vmax_b = float(np.nanmin(arr_b)), float(np.nanmax(arr_b))

        info = (
            f"⚡ Overlay  |  "
            f"A: {name_a} {arr_a.shape} [{vmin_a:.3e}, {vmax_a:.3e}]  |  "
            f"B: {name_b} {arr_b.shape} [{vmin_b:.3e}, {vmax_b:.3e}]  |  "
            f"{'B on top' if z_swap else 'A on top'}  (⇅ to swap)"
        )

        if ndim == 1:
            return self._overlay_1d(
                arr_a, name_a, arr_b, name_b,
                z_swap, scale_x, scale_y,
                vmin_a, vmax_a, vmin_b, vmax_b, info,
            )
        elif (ndim == 2
              and arr_a.shape[1] <= MAX_THIN_COLS
              and arr_b.shape[1] <= MAX_THIN_COLS):
            return self._overlay_multi_col(
                arr_a, name_a, arr_b, name_b,
                z_swap, scale_x, scale_y, info,
            )
        else:
            return self._overlay_2d(
                arr_a, name_a, arr_b, name_b,
                z_swap, vmin_a, vmax_a, vmin_b, vmax_b, info,
            )

    def _overlay_1d(
        self,
        arr_a, name_a, arr_b, name_b,
        z_swap, scale_x, scale_y,
        vmin_a, vmax_a, vmin_b, vmax_b, info,
    ) -> PlotRequest:
        """
        Two 1-D curves on shared axes, anchored at x=0.

        Draw order (last in list = on top):
          z_swap=False  →  [coral(B), teal(A)]   A on top
          z_swap=True   →  [teal(A), coral(B)]   B on top
        """
        n_a = len(arr_a)
        xd_a, yd_a = self._minmax_downsample(arr_a, N_DS,
                                              np.arange(n_a, dtype=np.float64))
        xd_a, yd_a = self._anchor_origin(xd_a, yd_a, arr_a)

        n_b = len(arr_b)
        xd_b, yd_b = self._minmax_downsample(arr_b, N_DS,
                                              np.arange(n_b, dtype=np.float64))
        xd_b, yd_b = self._anchor_origin(xd_b, yd_b, arr_b)

        curve_a = CurveSpec(x=xd_a, y=yd_a, color="#4ecdc4",
                            width=1.0, label=f"A: {name_a}")
        curve_b = CurveSpec(x=xd_b, y=yd_b, color="#e07b54",
                            width=1.0, label=f"B: {name_b}")

        # Last curve in list is drawn on top by pyqtgraph
        curves = [curve_b, curve_a] if not z_swap else [curve_a, curve_b]
        top_label = f"A: {name_a}" if not z_swap else f"B: {name_b}"

        return PlotRequest(
            title     = "⚡ Diagnostic Overlay",
            x_label   = "Index",
            y_label   = "Value",
            curves    = curves,
            log_x     = (scale_x == "log"),
            log_y     = (scale_y == "log"),
            status    = (
                f"1-D overlay  |  A: {n_a:,} pts  B: {n_b:,} pts  |  "
                f"top: {top_label}  (⇅ to swap)"
            ),
            info      = info,
            tab       = "Overlay",
            n_points  = max(n_a, n_b),
            dim_label = "D0",
        )

    def _overlay_2d(
        self,
        arr_a, name_a, arr_b, name_b,
        z_swap, vmin_a, vmax_a, vmin_b, vmax_b, info,
    ) -> PlotRequest:
        """
        Two 2-D heatmaps, origin-anchored at (0,0).

        z_value controls draw order (higher = in front):
          z_swap=False  →  A(teal) z=10, B(coral) z=0   →  A on top
          z_swap=True   →  B(coral) z=10, A(teal) z=0   →  B on top

        Alpha transparency: data minimum → alpha=0 so the background void
        of the top layer is transparent, revealing the layer beneath.
        """
        z_a = 10 if not z_swap else 0
        z_b = 0  if not z_swap else 10

        image_a = ImageSpec(
            data     = arr_a,
            colormap = "teal",
            z_value  = z_a,
            label    = f"A: {name_a}",
        )
        image_b = ImageSpec(
            data     = arr_b,
            colormap = "coral",
            z_value  = z_b,
            label    = f"B: {name_b}",
        )

        rows_a, cols_a = arr_a.shape
        rows_b, cols_b = arr_b.shape
        top_label = f"A: {name_a}" if not z_swap else f"B: {name_b}"

        return PlotRequest(
            title     = "⚡ Diagnostic Overlay",
            x_label   = "Column",
            y_label   = "Row",
            images    = [image_a, image_b],
            log_x     = False,
            log_y     = False,
            status    = (
                f"2-D overlay  |  A: {rows_a}×{cols_a}  B: {rows_b}×{cols_b}  |  "
                f"top: {top_label}  (⇅ to swap)"
            ),
            info      = info,
            tab       = "Overlay",
            n_points  = max(rows_a * cols_a, rows_b * cols_b),
            dim_label = "D0,D1",
        )

    # ── Overlay multi-column (thin 2-D) ───────────────────────────────────────

    def _overlay_multi_col(
        self,
        arr_a:   np.ndarray,
        name_a:  str,
        arr_b:   np.ndarray,
        name_b:  str,
        z_swap:  bool,
        scale_x: str,
        scale_y: str,
        info:    str,
    ) -> PlotRequest:
        """
        Overlay two thin 2-D arrays as interleaved per-column curves.

        File A columns use solid _MULTI_CURVE_PALETTE colours.
        File B columns use the same palette but rendered with dashed lines
        so overlapping channels remain visually distinguishable.

        z_swap controls which file's curves are drawn on top (last in list).
        """
        rows_a, cols_a = arr_a.shape
        rows_b, cols_b = arr_b.shape

        def _make_curves(arr, label_prefix, cols, dash):
            if cols == 6:
                col_labels = ["x", "y", "z", "v_x", "v_y", "v_z"]
            else:
                col_labels = [f"Col {i}" for i in range(cols)]

            out = []
            for c in range(cols):
                col_data = arr[:, c]
                x_full   = np.arange(len(arr), dtype=np.float64)
                xd, yd   = self._minmax_downsample(col_data, N_DS, x_full)
                xd, yd   = self._anchor_origin(xd, yd, col_data)
                color    = _MULTI_CURVE_PALETTE[c % len(_MULTI_CURVE_PALETTE)]
                out.append(CurveSpec(
                    x=xd, y=yd, color=color, width=1.0,
                    label=f"{label_prefix}: {col_labels[c]}",
                    dash=dash,
                ))
            return out

        curves_a = _make_curves(arr_a, f"A:{name_a}", cols_a, dash=False)
        curves_b = _make_curves(arr_b, f"B:{name_b}", cols_b, dash=True)

        # z_swap: last in list = on top in pyqtgraph
        if z_swap:
            curves = curves_a + curves_b
            top = f"B: {name_b}"
        else:
            curves = curves_b + curves_a
            top = f"A: {name_a}"

        return PlotRequest(
            title     = "⚡ Diagnostic Overlay",
            x_label   = "Index",
            y_label   = "Value",
            curves    = curves,
            log_x     = (scale_x == "log"),
            log_y     = (scale_y == "log"),
            status    = (
                f"Multi-col overlay  |  "
                f"A: {rows_a}×{cols_a}  B: {rows_b}×{cols_b}  |  "
                f"top: {top}  (⇅ to swap)"
            ),
            info      = info,
            tab       = "Overlay",
            n_points  = max(rows_a, rows_b),
            dim_label = f"D0×{max(cols_a, cols_b)}",
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _anchor_origin(
        xd: np.ndarray,
        yd: np.ndarray,
        arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Guarantee the downsampled curve starts at x=0.

        _minmax_downsample returns the argmin/argmax of each bucket.
        For large arrays the first bucket's extremum may be at index ~1500,
        not 0, so without this fix the two curves don't share a true origin.
        """
        if xd[0] != 0.0:
            xd = np.concatenate([[0.0],     xd])
            yd = np.concatenate([[arr[0]],  yd])
        return xd, yd