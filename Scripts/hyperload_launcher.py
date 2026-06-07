"""
HYPERLOAD Launcher
==================
Main window called by hyperload.py.
Floating script panels, suite tree, file dock, system stats.
"""

import sys
import json
import traceback
import random
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QToolButton, QLineEdit, QScrollArea,
    QFrame, QProgressBar, QFileDialog, QMessageBox, QSizePolicy,
    QApplication,
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, QPoint, QRect, QPointF,
    QObject, pyqtSignal, QMimeData, QByteArray,
)
from PyQt6.QtGui import (
    QIcon, QColor, QPainter, QBrush, QFont, QDrag, QCursor, QPen,
)

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = '#232323'
PANEL    = '#252525'
SURFACE  = '#2a2a2a'
SURFACE2 = '#2e2e2e'
BORDER   = '#1a1a1a'
BORDER2  = '#1e1e1e'
ACCENT   = '#4a8fd9'
ACC_DIM  = '#1e3050'
TEXT     = '#c8c8c8'
TEXT_DIM = '#888888'
TEXT_FAINT = '#505050'
TEXT_GHOST = '#383838'
INPUT_BG   = '#1e1e1e'
INPUT_TEXT = '#6888b0'

SUITE_COLORS = {
    'image_processing_suite': '#5a9e7e',
    'stacking_suite':         '#4a8fd9',
    'lucky_suite':            '#9070d0',
    'solar_system_suite':     '#c87840',
    'variable_star_suite':    '#b09030',
    'transient_pipeline':     '#c05060',
    'session_qa_suite':       '#4090b0',
    'spectroscopy_suite':     '#8060c0',
    'spectroscopy_pipeline':  '#8060c0',
    'exoplanet_transit_suite':'#3090a0',
}

SUITE_SCRIPTS = {
    'image_processing_suite': ['wavelet_stretch','dark_channel','mask_builder','nb_palette','local_norm','ab_compare','blind_deconv','pattern_noise_corrector'],
    'stacking_suite':         ['VectorisedWinsorizedSigmaStack','subframe_ranking','drizzle_combiner','multi_session_manager'],
    'lucky_suite':            ['lucky_preprocessor','bispectrum_speckle','speckle_holography','thresher_stack','dipli','imagemm','phase_diversity_wavefront','adi_klip'],
    'solar_system_suite':     ['comet_pipeline','comet_coma_analyser','moving_object','planet_derotation','planetary_texture_mapper','orbit_determinator'],
    'variable_star_suite':    ['variable_star','broeg_photometry','extinction_map','period_finder','transit_mcmc','occultation_timing'],
    'transient_pipeline':     ['zogy_subtract','sfft_subtract','transient_detector','transient_vetter'],
    'session_qa_suite':       ['seeing_estimator','psf_heatmap','astrometric_distortion_mapper','proper_motion_finder','ab_compare'],
    'spectroscopy_suite':     ['spectral_extractor','ccd_fringe_corrector','radial_velocity'],
    'spectroscopy_pipeline':  ['spectral_extractor','ccd_fringe_corrector','radial_velocity'],
    'exoplanet_transit_suite':['transit_mcmc','period_finder','variable_star'],
}

SCRIPT_DESCS = {
    'wavelet_stretch':             'Multi-scale wavelet stretch with per-band luminance control.',
    'dark_channel':                'Dark channel prior haze and vignette removal.',
    'mask_builder':                'Star, galaxy and nebula masks via morphological detection.',
    'nb_palette':                  'Narrowband palette mapping: SHO, HOO, HSO, custom.',
    'local_norm':                  'Local normalisation to flatten large-scale gradients.',
    'ab_compare':                  'Side-by-side A/B comparator with sync zoom and pan.',
    'blind_deconv':                'Blind deconvolution via regularised Wiener-Hunt. No PSF needed.',
    'pattern_noise_corrector':     'CMOS/CCD pattern noise: amp glow, banding, Münch destripe.',
    'VectorisedWinsorizedSigmaStack': 'Vectorised Winsorized Sigma-Clipping stacker.',
    'subframe_ranking':            'Rank subframes by FWHM, eccentricity and SNR.',
    'drizzle_combiner':            'Drizzle resampling for undersampled sensors.',
    'multi_session_manager':       'Align and combine FITS from multiple nights.',
    'lucky_preprocessor':          'Select best N% of video frames by Laplacian energy.',
    'bispectrum_speckle':          'Bispectrum speckle imaging for diffraction-limited resolution.',
    'speckle_holography':          'Speckle holography for binary stars and planetary surfaces.',
    'thresher_stack':              'The Thresher: blind deconv using ALL frames without discarding.',
    'dipli':                       'DIPLI Deep Image Prior + SGLD Bayesian U-Net. Singh 2025.',
    'imagemm':                     'ImageMM multi-frame deconvolution with maximum marginalisation.',
    'phase_diversity_wavefront':   'Wavefront sensing from defocused image pairs.',
    'adi_klip':                    'Angular Differential Imaging + KLIP for companion detection.',
    'comet_pipeline':              'Dual-stack comet pipeline: star-aligned and comet-aligned.',
    'comet_coma_analyser':         'Coma profile: Af(ρ), dust production rate, asymmetry.',
    'moving_object':               'Moving object detection and MPC tracklet generation.',
    'planet_derotation':           'Derotates planetary frames for alt-az field rotation.',
    'planetary_texture_mapper':    'Cylindrical projection with limb darkening correction.',
    'orbit_determinator':          'Keplerian orbit from MPC tracklets with MC uncertainty.',
    'variable_star':               'End-to-end variable star pipeline with AAVSO export.',
    'broeg_photometry':            'Broeg 2005 optimal comparison star weighting.',
    'extinction_map':              'Atmospheric extinction from multi-airmass standard stars.',
    'period_finder':               'Lomb-Scargle, ACF and wavelet period analysis.',
    'transit_mcmc':                'Exoplanet transit MCMC: Rp/Rs, impact parameter.',
    'occultation_timing':          'Stellar occultation timing with chord fit.',
    'zogy_subtract':               'ZOGY Fourier-space optimal image subtraction.',
    'sfft_subtract':               'SFFT tiled subtraction. O(N log N). Scales to 16+ Mpx.',
    'transient_detector':          'Source detection on difference images.',
    'transient_vetter':            'Rule-based real/bogus scorer 0-100.',
    'seeing_estimator':            'Atmospheric seeing from FWHM vs airmass. Fried r0.',
    'psf_heatmap':                 'PSF FWHM and ellipticity map across the full field.',
    'astrometric_distortion_mapper':'Distortion map from WCS residuals.',
    'proper_motion_finder':        'Proper motion detection from multi-epoch astrometry.',
    'spectral_extractor':          '1D extraction from 2D spectral images with wavelength cal.',
    'ccd_fringe_corrector':        'CCD fringe correction via Fabry-Pérot etalon model.',
    'radial_velocity':             'Radial velocity via CCF with stellar templates.',
    'psf_photometry':              'PSF-fitting photometry for crowded fields.',
    'sysrem_detrend':              'SYSREM systematic detrending for ensemble light curves.',
    'sersic_profiler':             'Sersic profile fitting and galaxy morphology analysis.',
    'spatially_varying_psf_deconv':'Position-dependent RL deconvolution using PSF heatmap.',
    'sfft_subtract':               'SFFT Fourier-space image subtraction for transient detection.',
    'bias_drift_corrector':        'Bias thermal drift correction for long imaging sessions.',
    'cosmic_ray_rejection':        'Cosmic ray detection and rejection from single frames.',
    'dark_temp_corrector':         'Dark frame temperature scaling for non-matching temperatures.',
    'mosaic_photometric_matching': 'Photometric matching across mosaic panels.',
    'planetary_zone_stacker':      'Zone-based planetary stacking for optimal detail.',
    'satellite_trail_remover':     'Satellite and aircraft trail detection and removal.',
    'software_adc':                'Software atmospheric dispersion corrector for planetary imaging.',
    'bg_gradient':                 'Background gradient removal and sky modelling.',
}

EXAMPLE_FILES = [
    {'name': 'M42_Ha.fits',    'size': '142 MB'},
    {'name': 'M31_RGB.fits',   'size': '380 MB'},
    {'name': 'Jupiter.ser',    'size': '2.1 GB'},
    {'name': 'M57_stack.fits', 'size': '95 MB'},
    {'name': 'LC_data.csv',    'size': '2 MB'},
    {'name': 'Sun_Ha.fits',    'size': '67 MB'},
]

MIME_TYPE = 'application/x-hyperload-file'

STYLESHEET = """
QWidget {
    background-color: #232323;
    color: #c8c8c8;
    font-family: -apple-system, 'Segoe UI', Ubuntu, sans-serif;
    font-size: 15px;
}
QScrollBar:vertical {
    background: #1e1e1e; width: 5px; border: none;
}
QScrollBar::handle:vertical {
    background: #3a3a3a; border-radius: 2px; min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: #484848; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: #1e1e1e; height: 5px; border: none;
}
QScrollBar::handle:horizontal {
    background: #3a3a3a; border-radius: 2px;
}
QLineEdit {
    background: #1e1e1e;
    border: 1px solid #1a1a1a;
    border-radius: 3px;
    padding: 3px 6px;
    color: #6888b0;
    font-family: 'Consolas', 'Fira Code', monospace;
    font-size: 13px;
}
QLineEdit:focus { border-color: #2e4060; }
QPushButton {
    background: #2a2a2a;
    border: 1px solid #1a1a1a;
    border-radius: 3px;
    padding: 3px 8px;
    color: #888888;
    font-size: 13px;
}
QPushButton:hover { background: #333333; color: #aaaaaa; }
QPushButton:pressed { background: #242424; }
QProgressBar {
    background: #1e1e1e;
    border: none;
    border-radius: 1px;
}
QProgressBar::chunk { background: #3d6ea0; border-radius: 1px; }
QToolTip {
    background: #2a2a2a;
    border: 1px solid #1a1a1a;
    color: #c8c8c8;
    padding: 4px 6px;
    font-size: 13px;
}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sep(parent=None):
    """1px horizontal separator line."""
    w = QFrame(parent)
    w.setFrameShape(QFrame.Shape.HLine)
    w.setFixedHeight(1)
    w.setStyleSheet(f'background:{BORDER};border:none;')
    return w


def _sec_label(text, parent=None):
    lbl = QLabel(text.upper(), parent)
    lbl.setStyleSheet(f'color:{TEXT_FAINT};font-size: 12px;letter-spacing:0.08em;')
    return lbl


def _path_edit(value='', parent=None):
    e = QLineEdit(value, parent)
    e.setStyleSheet(f'background:{INPUT_BG};border:1px solid {BORDER2};'
                    f'border-radius:3px;padding:3px 6px;'
                    f'color:{INPUT_TEXT};font-size: 12px;')
    return e


# ── Worker ────────────────────────────────────────────────────────────────────

class ScriptWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal()
    error    = pyqtSignal(str)

    def __init__(self, script_path, lazy_import_fn):
        super().__init__()
        self.path           = script_path
        self.lazy_import_fn = lazy_import_fn

    def run(self):
        try:
            self.progress.emit(10, f'Importing {self.path.stem}…')
            module = self.lazy_import_fn(self.path)
            self.progress.emit(40, f'Starting {self.path.stem}…')
            if hasattr(module, 'main'):
                module.main()
            self.progress.emit(100, 'Done.')
            self.finished.emit()
        except Exception as e:
            self.error.emit(f'{e}\n\n{traceback.format_exc()}')


# ── Resize handle ─────────────────────────────────────────────────────────────

class ResizeHandle(QWidget):
    CURSORS = {
        'n':  Qt.CursorShape.SizeVerCursor,
        's':  Qt.CursorShape.SizeVerCursor,
        'e':  Qt.CursorShape.SizeHorCursor,
        'w':  Qt.CursorShape.SizeHorCursor,
        'ne': Qt.CursorShape.SizeBDiagCursor,
        'sw': Qt.CursorShape.SizeBDiagCursor,
        'nw': Qt.CursorShape.SizeFDiagCursor,
        'se': Qt.CursorShape.SizeFDiagCursor,
    }

    def __init__(self, direction, panel):
        super().__init__(panel)
        self.direction  = direction
        self.panel      = panel
        self._active    = False
        self._start_pos = QPoint()
        self._start_geo = QRect()
        self.setCursor(self.CURSORS[direction])
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._active    = True
            self._start_pos = e.globalPosition().toPoint()
            self._start_geo = self.panel.geometry()

    def mouseMoveEvent(self, e):
        if not self._active:
            return
        delta = e.globalPosition().toPoint() - self._start_pos
        geo   = QRect(self._start_geo)
        MIN_W, MIN_H = 260, 80
        d = self.direction
        if 'e' in d:
            geo.setRight(max(geo.left() + MIN_W, self._start_geo.right() + delta.x()))
        if 's' in d:
            geo.setBottom(max(geo.top() + MIN_H, self._start_geo.bottom() + delta.y()))
        if 'w' in d:
            new_left = min(self._start_geo.right() - MIN_W,
                           self._start_geo.left() + delta.x())
            geo.setLeft(new_left)
        if 'n' in d:
            new_top = min(self._start_geo.bottom() - MIN_H,
                          self._start_geo.top() + delta.y())
            geo.setTop(new_top)
        self.panel.setGeometry(geo)
        self.panel._place_handles()

    def mouseReleaseEvent(self, e):
        self._active = False


# ── File chip (dock) ──────────────────────────────────────────────────────────

class FileChip(QWidget):
    def __init__(self, file_info, parent=None):
        super().__init__(parent)
        self.file_info   = file_info
        self._press_pos  = None
        self.setFixedHeight(36)
        self.setStyleSheet(
            f'background:{SURFACE};border:1px solid {BORDER2};'
            f'border-radius:3px;padding:0 8px;')
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setFixedWidth(max(100, len(file_info['name']) * 6 + 40))

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 6, 0)
        lay.setSpacing(5)
        ico = QLabel('🗂')
        ico.setStyleSheet('font-size: 16px;background:transparent;border:none;')
        name_lbl = QLabel(file_info['name'])
        name_lbl.setStyleSheet(f'color:#686868;font-size: 12px;font-weight:500;'
                                f'background:transparent;border:none;')
        size_lbl = QLabel(file_info['size'])
        size_lbl.setStyleSheet(f'color:{TEXT_GHOST};font-size: 11px;'
                                f'background:transparent;border:none;')
        lay.addWidget(ico)
        lay.addWidget(name_lbl)
        lay.addWidget(size_lbl)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._press_pos = e.position().toPoint()

    def mouseMoveEvent(self, e):
        if self._press_pos is None:
            return
        if (e.position().toPoint() - self._press_pos).manhattanLength() < 6:
            return
        drag  = QDrag(self)
        mime  = QMimeData()
        data  = json.dumps(self.file_info).encode()
        mime.setData(MIME_TYPE, QByteArray(data))
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)
        self._press_pos = None

    def mouseReleaseEvent(self, e):
        self._press_pos = None


# ── File item row (inside panel) ──────────────────────────────────────────────

class FileItemRow(QWidget):
    def __init__(self, file_info, script_id, files_ref, panel):
        super().__init__()
        self.file_info  = file_info
        self.script_id  = script_id
        self.files_ref  = files_ref
        self.panel      = panel
        self.setFixedHeight(26)
        self.setStyleSheet(
            f'background:{INPUT_BG};border:1px solid {BORDER};'
            f'border-radius:3px;')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 6, 0)
        lay.setSpacing(5)
        name = QLabel(file_info['name'])
        name.setStyleSheet(f'color:#6080a0;font-size: 12px;background:transparent;border:none;')
        size = QLabel(file_info['size'])
        size.setStyleSheet(f'color:{TEXT_GHOST};font-size: 11px;background:transparent;border:none;')
        rm = QPushButton('✕')
        rm.setFixedSize(16, 16)
        rm.setStyleSheet(
            f'background:transparent;border:none;color:{TEXT_GHOST};font-size: 13px;'
            f'padding:0;')
        rm.clicked.connect(self._remove)
        lay.addWidget(name, 1)
        lay.addWidget(size)
        lay.addWidget(rm)

    def _remove(self):
        lst = self.files_ref.get(self.script_id, [])
        self.files_ref[self.script_id] = [
            f for f in lst if f['name'] != self.file_info['name']]
        self.panel.refresh_file_list()
        self.panel.main_win.rebuild_tree()


# ── Script panel (floating window) ───────────────────────────────────────────

class ScriptPanel(QFrame):
    HANDLE_SIZE = 6

    def __init__(self, desktop, suite_id, color, script_id,
                 root, config, files_ref, lazy_import_fn, main_win):
        super().__init__(desktop)
        self.desktop        = desktop
        self.suite_id       = suite_id
        self.color          = color
        self.script_id      = script_id
        self.root           = root
        self.config         = config
        self.files_ref      = files_ref
        self.lazy_import_fn = lazy_import_fn
        self.main_win       = main_win
        self._collapsed     = False
        self._pre_h         = 400
        self._drag_off      = None
        self._worker        = None
        self._thread        = None

        # Position with cascade
        n = len(desktop.main_win.windows)
        self.setGeometry(20 + n*24, 20 + n*24, 320, 400)

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName(f'panel_{script_id}')
        self._set_style(focused=True)
        self.setAcceptDrops(True)

        lay = QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)

        self._tb  = self._build_titlebar()
        self._body = self._build_body()
        lay.addWidget(self._tb)
        lay.addWidget(self._body, 1)

        self._handles = {}
        for d in ('n','s','e','w','ne','nw','se','sw'):
            h = ResizeHandle(d, self)
            self._handles[d] = h
        self._place_handles()

        self.show()
        self.raise_()

    # ── Style ─────────────────────────────────────────────────────────────

    def _set_style(self, focused=False):
        border = '#2e3e50' if focused else BORDER
        # Set directly on the widget — no class/id selector needed
        self.setStyleSheet(
            f'background:{SURFACE};border:1px solid {border};border-radius:4px;')

    # ── Title bar ─────────────────────────────────────────────────────────

    def _build_titlebar(self):
        tb = QWidget()
        tb.setFixedHeight(28)
        tb.setStyleSheet(
            f'background:{INPUT_BG};border-bottom:1px solid {BORDER};'
            f'border-radius:0;')
        lay = QHBoxLayout(tb)
        lay.setContentsMargins(7, 0, 5, 0)
        lay.setSpacing(5)

        dot = QLabel()
        dot.setFixedSize(7, 7)
        dot.setStyleSheet(
            f'background:{self.color};border-radius:3px;border:none;')

        title = QLabel(self.script_id.replace('_', ' '))
        title.setStyleSheet(
            f'color:#909090;font-size: 14px;font-weight:600;background:transparent;border:none;')

        suite_name = QLabel(self.suite_id.replace('_', ' '))
        suite_name.setStyleSheet(
            f'color:{TEXT_GHOST};font-size: 12px;background:transparent;border:none;')

        self._collapse_btn = QToolButton()
        self._collapse_btn.setText('⌃')
        self._collapse_btn.setFixedSize(20, 20)
        self._collapse_btn.setStyleSheet(
            f'background:transparent;border:none;color:{TEXT_GHOST};font-size: 15px;')
        self._collapse_btn.clicked.connect(self._toggle_collapse)

        close_btn = QToolButton()
        close_btn.setText('✕')
        close_btn.setFixedSize(20, 20)
        close_btn.setStyleSheet(
            f'background:transparent;border:none;color:{TEXT_GHOST};font-size: 14px;')
        close_btn.clicked.connect(self._close)

        lay.addWidget(dot)
        lay.addWidget(title)
        lay.addWidget(suite_name)
        lay.addStretch()
        lay.addWidget(self._collapse_btn)
        lay.addWidget(close_btn)

        tb.mouseDoubleClickEvent = lambda e: self._toggle_collapse()
        tb.mousePressEvent  = self._tb_press
        tb.mouseMoveEvent   = self._tb_move
        tb.mouseReleaseEvent = self._tb_release
        return tb

    def _tb_press(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_off = e.position().toPoint()
            self.raise_()
            self.main_win.focus_panel(self)

    def _tb_move(self, e):
        if self._drag_off is None:
            return
        desktop_pos = self.desktop.mapFromGlobal(
            e.globalPosition().toPoint()) - self._drag_off
        dw = self.desktop.width()
        dh = self.desktop.height()
        x = max(0, min(dw - self.width(),  desktop_pos.x()))
        y = max(0, min(dh - 28,            desktop_pos.y()))
        self.move(x, y)

    def _tb_release(self, e):
        self._drag_off = None

    # ── Body ──────────────────────────────────────────────────────────────

    def _build_body(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet('border:none;background:transparent;')
        inner = QWidget()
        inner.setStyleSheet(f'background:{SURFACE};')
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # About
        lay.addWidget(_sec_label('About'))
        desc_lbl = QLabel(SCRIPT_DESCS.get(
            self.script_id, 'Standalone script.'))
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet(f'color:{TEXT_DIM};font-size: 13px;'
                                f'background:transparent;')
        lay.addWidget(desc_lbl)

        # Paths
        lay.addWidget(_sep())
        lay.addWidget(_sec_label('Paths'))

        grid = QWidget()
        grid.setStyleSheet('background:transparent;')
        gl = QGridLayout(grid)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setSpacing(4)

        import_dir = self.config.get('import_dir', '')
        export_dir = self.config.get('export_dir', '')
        work_dir   = self.config.get('working_dir', '')

        self._in_edit  = _path_edit(import_dir)
        self._out_edit = _path_edit(
            f'{export_dir}\\{self.script_id}' if export_dir else '')
        self._wd_edit  = _path_edit(work_dir)

        for i, (lbl, widget) in enumerate([
            ('Input',   self._in_edit),
            ('Output',  self._out_edit),
            ('Working', self._wd_edit),
        ]):
            l = QLabel(lbl)
            l.setFixedWidth(68)
            l.setStyleSheet(f'color:{TEXT_FAINT};font-size: 12px;'
                            f'text-transform:uppercase;background:transparent;')
            gl.addWidget(l, i, 0)
            gl.addWidget(widget, i, 1)

        lay.addWidget(grid)

        # Browse buttons
        browse_row = QHBoxLayout()
        browse_row.setSpacing(4)
        for label, edit in [('Browse Input', self._in_edit),
                             ('Browse Output', self._out_edit)]:
            btn = QPushButton(label)
            btn.setFixedHeight(22)
            btn.clicked.connect(lambda _, e=edit: self._browse(e))
            browse_row.addWidget(btn)
        lay.addLayout(browse_row)

        # Run button + progress
        lay.addWidget(_sep())
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        run_btn = QPushButton('▶  Run')
        run_btn.setStyleSheet(
            f'background:{ACC_DIM};border:1px solid #2a4070;'
            f'color:#6898d0;font-weight:600;border-radius:3px;'
            f'padding:4px 12px;font-size: 13px;')
        run_btn.clicked.connect(self._run)

        open_btn = QPushButton('Open full window')
        open_btn.clicked.connect(self._open_full)

        btn_row.addWidget(run_btn)
        btn_row.addWidget(open_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setFixedHeight(3)
        self._progress.setVisible(False)
        lay.addWidget(self._progress)

        # Drop zone
        lay.addWidget(_sep())
        lay.addWidget(_sec_label('Files'))

        self._drop_zone = QLabel('Drop files here  ·  or drag from dock below')
        self._drop_zone.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_zone.setFixedHeight(48)
        self._drop_zone.setStyleSheet(
            f'color:{TEXT_GHOST};font-size: 13px;'
            f'border:1px dashed #2e2e2e;border-radius:3px;background:transparent;')
        self._drop_zone.setAcceptDrops(True)
        self._drop_zone.dragEnterEvent = self._dz_enter
        self._drop_zone.dragLeaveEvent = self._dz_leave
        self._drop_zone.dropEvent      = self._dz_drop
        lay.addWidget(self._drop_zone)

        self._file_list_widget = QWidget()
        self._file_list_widget.setStyleSheet('background:transparent;')
        self._fll = QVBoxLayout(self._file_list_widget)
        self._fll.setContentsMargins(0, 0, 0, 0)
        self._fll.setSpacing(3)
        lay.addWidget(self._file_list_widget)

        lay.addStretch()
        scroll.setWidget(inner)
        self.refresh_file_list()
        return scroll

    def refresh_file_list(self):
        while self._fll.count():
            item = self._fll.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for f in self.files_ref.get(self.script_id, []):
            self._fll.addWidget(
                FileItemRow(f, self.script_id, self.files_ref, self))

    # ── Drag & drop ───────────────────────────────────────────────────────

    def _dz_enter(self, e):
        if e.mimeData().hasFormat(MIME_TYPE):
            e.acceptProposedAction()
            self._drop_zone.setStyleSheet(
                f'color:{ACCENT};font-size: 13px;'
                f'border:1px dashed {ACCENT};border-radius:3px;'
                f'background:{ACC_DIM};')

    def _dz_leave(self, e):
        self._drop_zone.setStyleSheet(
            f'color:{TEXT_GHOST};font-size: 13px;'
            f'border:1px dashed #2e2e2e;border-radius:3px;background:transparent;')

    def _dz_drop(self, e):
        self._dz_leave(e)
        if not e.mimeData().hasFormat(MIME_TYPE):
            return
        data = json.loads(bytes(e.mimeData().data(MIME_TYPE)).decode())
        self._add_file(data)

    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(MIME_TYPE):
            e.acceptProposedAction()

    def dropEvent(self, e):
        if e.mimeData().hasFormat(MIME_TYPE):
            data = json.loads(bytes(e.mimeData().data(MIME_TYPE)).decode())
            self._add_file(data)

    def _add_file(self, f):
        lst = self.files_ref.setdefault(self.script_id, [])
        if not any(x['name'] == f['name'] for x in lst):
            lst.append(f)
        self.refresh_file_list()
        self.main_win.rebuild_tree()
        self.main_win.set_status(f"{f['name']} → {self.script_id.replace('_',' ')}")

    # ── Resize handles ────────────────────────────────────────────────────

    def _place_handles(self):
        w, h = self.width(), self.height()
        s = self.HANDLE_SIZE
        placements = {
            'n':  QRect(s, 0, w - 2*s, s),
            's':  QRect(s, h-s, w - 2*s, s),
            'e':  QRect(w-s, s, s, h - 2*s),
            'w':  QRect(0, s, s, h - 2*s),
            'ne': QRect(w-s, 0, s, s),
            'nw': QRect(0, 0, s, s),
            'se': QRect(w-s, h-s, s, s),
            'sw': QRect(0, h-s, s, s),
        }
        for d, rect in placements.items():
            self._handles[d].setGeometry(rect)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place_handles()

    # ── Collapse ──────────────────────────────────────────────────────────

    def _toggle_collapse(self):
        if self._collapsed:
            self._body.setVisible(True)
            self.resize(self.width(), self._pre_h)
            self._collapsed = False
            self._collapse_btn.setText('⌃')
        else:
            self._pre_h = self.height()
            self._body.setVisible(False)
            self.resize(self.width(), 28)
            self._collapsed = True
            self._collapse_btn.setText('⌄')

    # ── Run ───────────────────────────────────────────────────────────────

    def _browse(self, edit):
        path = QFileDialog.getExistingDirectory(self, 'Select folder')
        if path:
            edit.setText(path)

    def _open_full(self):
        script_path = self._find_script()
        if not script_path:
            self.main_win.set_status(f'Script not found: {self.script_id}')
            return
        self._run_script(script_path)

    def _run(self):
        script_path = self._find_script()
        if not script_path:
            self.main_win.set_status(f'Script not found: {self.script_id}')
            return
        self._run_script(script_path)

    def _find_script(self):
        for folder in ('Scripts', 'Suites'):
            p = self.root / folder / f'{self.script_id}.py'
            if p.exists():
                return p
        return None

    def _run_script(self, path):
        # Launch as a fully independent OS process.
        # Each script creates its own QApplication — running module.main()
        # inside the launcher causes a second QApplication which freezes Qt.
        # subprocess.Popen gives each script its own Python interpreter.
        import subprocess
        import sys as _sys
        try:
            if _sys.platform == 'win32':
                self._proc = subprocess.Popen(
                    [_sys.executable, str(path)],
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    stderr=subprocess.PIPE)
            else:
                self._proc = subprocess.Popen(
                    [_sys.executable, str(path)],
                    stderr=subprocess.PIPE)
            self.main_win.set_status(
                f'{path.stem.replace("_", " ")} launched  ·  PID {self._proc.pid}')
            # Watch for early crash without blocking UI
            QTimer.singleShot(3000, lambda: self._check_proc(path.stem))
        except Exception as e:
            QMessageBox.critical(self, 'Launch Error', str(e))

    def _check_proc(self, name):
        if hasattr(self, '_proc') and self._proc is not None:
            ret = self._proc.poll()
            if ret is not None and ret != 0:
                # Script crashed — read stderr for clue
                try:
                    err = self._proc.stderr.read().decode(errors='replace')[-800:]
                except Exception:
                    err = f'Exit code: {ret}'
                if err.strip():
                    QMessageBox.warning(
                        self, f'{name} — Launch Error',
                        f'Script exited with code {ret}:\n\n{err}')
                    self.main_win.set_status(f'{name} exited with error (code {ret})')

    def _on_progress(self, pct, msg):
        self._progress.setValue(pct)
        self.main_win.set_status(msg, pct)

    def _on_finished(self):
        self._progress.setVisible(False)
        self.main_win.set_status(
            f'{self.script_id.replace("_", " ")} complete.', -1)

    def _on_error(self, msg):
        self._progress.setVisible(False)
        QMessageBox.critical(self, 'Script Error', msg[:800])
        self.main_win.set_status(f'Error in {self.script_id}', -1)

    def _close(self):
        self.main_win.on_panel_closed(self.script_id)
        self.hide()
        self.deleteLater()

    # ── Focus ─────────────────────────────────────────────────────────────

    def mousePressEvent(self, e):
        super().mousePressEvent(e)
        self.raise_()
        self.main_win.focus_panel(self)


# ── Desktop widget ────────────────────────────────────────────────────────────

class DesktopWidget(QWidget):
    def __init__(self, main_win):
        super().__init__()
        self.main_win = main_win
        self._stars   = []
        rng = random.Random(42)
        for _ in range(60):
            self._stars.append((
                rng.random(), rng.random(),
                rng.random() * 0.9 + 0.3,
                rng.random() * 0.08 + 0.02,
            ))
        self.setAcceptDrops(True)

        self.hint = QLabel(self)
        self.hint.setText(
            'Click a script in the tree to open its panel\n'
            'Panels are movable  ·  resizable  ·  collapsible')
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint.setStyleSheet(
            f'color:#404040;font-size: 14px;background:transparent;')

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.hint.setGeometry(0, 0, self.width(), self.height())

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(BG))
        w, h = self.width(), self.height()
        p.setPen(Qt.PenStyle.NoPen)
        for rx, ry, r, a in self._stars:
            p.setBrush(QBrush(QColor(80, 120, 180, int(a * 255))))
            p.drawEllipse(QPointF(rx * w, ry * h), r, r)
        p.end()

    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(MIME_TYPE):
            e.acceptProposedAction()

    def dropEvent(self, e):
        if e.mimeData().hasFormat(MIME_TYPE):
            self.main_win.set_status(
                'Drop onto a script panel or a script row in the tree')


# ── Suite / script rows ───────────────────────────────────────────────────────

class SuiteRow(QWidget):
    def __init__(self, suite, color, loaded, expanded,
                 script_count, main_win):
        super().__init__()
        self.suite_id = suite['id']
        self.color    = color
        self.main_win = main_win
        self.setFixedHeight(38)
        self._bg_default = 'transparent'
        self._bg_open    = '#2a2a2a'

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 8, 0)
        lay.setSpacing(0)

        self._chev = QLabel('▶' if not expanded else '▼')
        self._chev.setFixedWidth(18)
        self._chev.setStyleSheet(
            f'color:{"#999999" if loaded else "#555555"};'
            f'font-size: 15px;background:transparent;')

        dot = QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(
            f'background:{color if loaded else "#2e2e2e"};'
            f'border-radius:4px;border:none;margin:0 6px;')

        name_lbl = QLabel(suite['name'])
        name_lbl.setStyleSheet(
            f'color:{"#909090" if loaded else "#484848"};'
            f'font-size: 14px;font-weight:600;background:transparent;')

        cnt_lbl = QLabel(str(script_count))
        cnt_lbl.setStyleSheet(
            f'color:{TEXT_GHOST};font-size: 12px;margin-right:6px;background:transparent;')

        tog = QPushButton('unload' if loaded else 'load')
        tog.setFixedHeight(18)
        tog.setStyleSheet(
            f'background:#1e1e1e;border:1px solid #1a1a1a;border-radius:2px;'
            f'color:#444444;font-size: 11px;padding:0 5px;')
        tog.clicked.connect(lambda: main_win.toggle_suite(self.suite_id))

        lay.addWidget(self._chev)
        lay.addWidget(dot)
        lay.addWidget(name_lbl, 1)
        lay.addWidget(cnt_lbl)
        lay.addWidget(tog)

        self.setStyleSheet(
            f'background:{"#2a2a2a" if expanded else "transparent"};')
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.main_win.toggle_suite_expand(self.suite_id)


class ScriptRow(QWidget):
    def __init__(self, script_id, suite_id, color, loaded, file_count, main_win):
        super().__init__()
        self.script_id = script_id
        self.suite_id  = suite_id
        self.main_win  = main_win
        self.setFixedHeight(30)
        self.setAcceptDrops(True)
        self.setCursor(
            Qt.CursorShape.PointingHandCursor if loaded
            else Qt.CursorShape.ForbiddenCursor)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(38, 0, 8, 0)
        lay.setSpacing(6)

        dash = QFrame()
        dash.setFixedSize(10, 1)
        dash.setStyleSheet(f'background:#2e2e2e;')

        is_open = script_id in main_win.windows
        name_color = color if is_open else (TEXT_GHOST if loaded else '#2e2e2e')
        name = QLabel(script_id.replace('_', ' '))
        name.setStyleSheet(
            f'color:{name_color};font-size: 13px;background:transparent;'
            f'{"font-weight:600;" if is_open else ""}')

        badge_text = f'{file_count} file{"s" if file_count!=1 else ""}' \
            if file_count > 0 else '—'
        badge_color = '#1e2a3a' if file_count > 0 else INPUT_BG
        badge_border = '#2a3a4a' if file_count > 0 else BORDER2
        badge_text_color = '#4a6888' if file_count > 0 else TEXT_GHOST
        badge = QLabel(badge_text)
        badge.setStyleSheet(
            f'background:{badge_color};border:1px solid {badge_border};'
            f'border-radius:8px;color:{badge_text_color};'
            f'font-size: 11px;padding:0 5px;')

        if not loaded:
            self.setStyleSheet('background:transparent;opacity:0.4;')
        elif is_open:
            self.setStyleSheet(f'background:#1e2a3a;')
        else:
            self.setStyleSheet('background:transparent;')

        lay.addWidget(dash)
        lay.addWidget(name, 1)
        lay.addWidget(badge)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            if self.suite_id not in self.main_win.loaded_suites:
                self.main_win.set_status('Load this suite first')
                return
            self.main_win.open_script_window(self.suite_id, self.script_id)

    def enterEvent(self, e):
        if self.suite_id in self.main_win.loaded_suites:
            self.setStyleSheet(
                f'background:{"#1e2a3a" if self.script_id in self.main_win.windows else "#2a2a2a"};')

    def leaveEvent(self, e):
        is_open = self.script_id in self.main_win.windows
        self.setStyleSheet(
            f'background:{"#1e2a3a" if is_open else "transparent"};')

    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(MIME_TYPE):
            e.acceptProposedAction()
            self.setStyleSheet(
                f'background:#1e2a3a;border-left:2px solid '
                f'{SUITE_COLORS.get(self.suite_id, ACCENT)};')

    def dragLeaveEvent(self, e):
        self.setStyleSheet('background:transparent;')

    def dropEvent(self, e):
        self.setStyleSheet('background:transparent;')
        if not e.mimeData().hasFormat(MIME_TYPE):
            return
        if self.suite_id not in self.main_win.loaded_suites:
            self.main_win.set_status('Load the suite first')
            return
        data = json.loads(bytes(e.mimeData().data(MIME_TYPE)).decode())
        lst = self.main_win.script_files.setdefault(self.script_id, [])
        if not any(x['name'] == data['name'] for x in lst):
            lst.append(data)
        self.main_win.rebuild_tree()
        if self.script_id in self.main_win.windows:
            self.main_win.windows[self.script_id].refresh_file_list()
        self.main_win.set_status(
            f"{data['name']} → {self.script_id.replace('_', ' ')}")


# ── Main window ───────────────────────────────────────────────────────────────

class HyperloadWindow(QMainWindow):
    def __init__(self, root, suites, scripts, config,
                 save_config_fn, lazy_import_fn, logo_path):
        super().__init__()
        self.root           = root
        self.suites         = suites
        self.scripts        = scripts
        self.config         = config
        self.save_config_fn = save_config_fn
        self.lazy_import_fn = lazy_import_fn

        self.script_files   = {}
        self.windows        = {}
        self.open_suites    = {}
        self.loaded_suites  = set(
            config.get('loaded_suites') or [s['id'] for s in suites])

        self._save_timer = QTimer()
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._do_save)

        self._cpu = 23.0
        self._ram = 67.0

        self.setWindowTitle('HYPERLOAD')
        self.setMinimumSize(1000, 680)
        self.resize(1380, 860)
        if logo_path and logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))

        self._build_ui()
        self._stat_timer = QTimer()
        self._stat_timer.timeout.connect(self._update_stats)
        self._stat_timer.start(2200)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_lay = QVBoxLayout(central)
        root_lay.setSpacing(0)
        root_lay.setContentsMargins(0, 0, 0, 0)

        root_lay.addWidget(self._build_topbar())

        body = QWidget()
        body_lay = QHBoxLayout(body)
        body_lay.setSpacing(0)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.addWidget(self._build_sidebar())
        self.desktop = DesktopWidget(self)
        body_lay.addWidget(self.desktop, 1)
        root_lay.addWidget(body, 1)

        root_lay.addWidget(self._build_dock())
        root_lay.addWidget(self._build_statusbar())

    def _build_topbar(self):
        bar = QWidget()
        bar.setFixedHeight(28)
        bar.setStyleSheet(
            f'background:#1e1e1e;border-bottom:1px solid #161616;')
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(10, 0, 10, 0)
        lay.setSpacing(12)

        logo_lbl = QLabel('✦')
        logo_lbl.setStyleSheet(f'color:{ACCENT};font-size: 17px;background:transparent;')
        name_lbl = QLabel('HYPERLOAD')
        name_lbl.setStyleSheet(
            f'color:#a0a0a0;font-size: 15px;font-weight:700;'
            f'letter-spacing:0.15em;background:transparent;')
        ver_lbl = QLabel('v2.0')
        ver_lbl.setStyleSheet(
            f'color:#2e2e2e;font-size: 12px;border:1px solid #2e2e2e;'
            f'border-radius:2px;padding:1px 4px;background:transparent;')

        lay.addWidget(logo_lbl)
        lay.addWidget(name_lbl)
        lay.addWidget(ver_lbl)
        lay.addWidget(_vsep())

        for label, bar_attr, val_attr, color, init_pct in [
            ('CPU', '_cpu_bar', '_cpu_val', '#1a3460', 23),
            ('RAM', '_ram_bar', '_ram_val', '#0c3020', 67),
        ]:
            lay.addWidget(_stat_label(label))
            pb = QProgressBar()
            pb.setFixedSize(52, 3)
            pb.setRange(0, 100)
            pb.setValue(init_pct)
            pb.setTextVisible(False)
            pb.setStyleSheet(
                f'QProgressBar{{background:#0e1117;border:none;border-radius:1px;}}'
                f'QProgressBar::chunk{{background:{color};border-radius:1px;}}')
            setattr(self, bar_attr, pb)
            lay.addWidget(pb)
            vl = _stat_label('23%' if label == 'CPU' else '10.7 GB')
            setattr(self, val_attr, vl)
            lay.addWidget(vl)

        lay.addWidget(_stat_label('DISK'))
        disk_pb = QProgressBar()
        disk_pb.setFixedSize(52, 3)
        disk_pb.setRange(0, 100)
        disk_pb.setValue(45)
        disk_pb.setTextVisible(False)
        disk_pb.setStyleSheet(
            'QProgressBar{background:#0e1117;border:none;border-radius:1px;}'
            'QProgressBar::chunk{background:#3a2800;border-radius:1px;}')
        lay.addWidget(disk_pb)
        lay.addWidget(_stat_label('1.8 TB'))

        lay.addStretch()
        self._led = QLabel()
        self._led.setFixedSize(6, 6)
        self._led.setStyleSheet('background:#2a5020;border-radius:3px;border:none;')
        self._led_txt = _stat_label('IDLE')
        lay.addWidget(self._led)
        lay.addWidget(self._led_txt)
        lay.addWidget(_vsep())
        n_scripts = len(self.scripts)
        lay.addWidget(_stat_label(f'{n_scripts} scripts'))
        return bar

    def _build_sidebar(self):
        sb = QWidget()
        sb.setFixedWidth(310)
        sb.setStyleSheet(
            f'background:{PANEL};border-right:1px solid {BORDER};')
        lay = QVBoxLayout(sb)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)

        # Top paths section
        top = QWidget()
        top.setStyleSheet(f'background:{PANEL};')
        top_lay = QVBoxLayout(top)
        top_lay.setContentsMargins(10, 10, 10, 8)
        top_lay.setSpacing(5)

        top_lay.addWidget(_sec_label('Workspace'))

        self.wd_edit     = _path_edit(self.config.get('working_dir', ''))
        self.import_edit = _path_edit(self.config.get('import_dir',  ''))
        self.export_edit = _path_edit(self.config.get('export_dir',  ''))

        for lbl, widget in [
            ('Working dir', self.wd_edit),
            ('Import',      self.import_edit),
            ('Export',      self.export_edit),
        ]:
            l = _path_label(lbl)
            top_lay.addWidget(l)
            top_lay.addWidget(widget)
            widget.textChanged.connect(self._schedule_save)

        top_lay.addWidget(_sec_label('System'))
        grid = QWidget()
        grid.setStyleSheet('background:transparent;')
        gl = QGridLayout(grid)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setSpacing(4)

        import os
        n_threads = os.cpu_count() or 4
        for col, (label, value, color) in enumerate([
            ('Threads', str(n_threads), '#1e3460'),
            ('VRAM',    '6.2 GB',       '#0c3020'),
        ]):
            box = QWidget()
            box.setStyleSheet(
                f'background:{INPUT_BG};border:1px solid {BORDER2};'
                f'border-radius:3px;')
            bl = QVBoxLayout(box)
            bl.setContentsMargins(6, 4, 6, 4)
            bl.setSpacing(1)
            ll = QLabel(label)
            ll.setStyleSheet(f'color:{TEXT_GHOST};font-size: 11px;background:transparent;')
            vl = QLabel(value)
            vl.setStyleSheet(f'color:{color};font-size: 16px;font-weight:600;background:transparent;')
            bl.addWidget(ll)
            bl.addWidget(vl)
            gl.addWidget(box, 0, col)
        top_lay.addWidget(grid)

        lay.addWidget(top)
        lay.addWidget(_sep())

        # Tree
        tree_scroll = QScrollArea()
        tree_scroll.setWidgetResizable(True)
        tree_scroll.setStyleSheet(
            'border:none;background:transparent;')

        self._tree_inner = QWidget()
        self._tree_inner.setStyleSheet(f'background:{PANEL};')
        self._tree_lay = QVBoxLayout(self._tree_inner)
        self._tree_lay.setContentsMargins(0, 0, 0, 0)
        self._tree_lay.setSpacing(0)
        self._tree_lay.addStretch()
        tree_scroll.setWidget(self._tree_inner)
        lay.addWidget(tree_scroll, 1)

        # Bottom hint
        hint_w = QWidget()
        hint_w.setStyleSheet(
            f'background:{PANEL};border-top:1px solid {BORDER};')
        hint_lay = QVBoxLayout(hint_w)
        hint_lay.setContentsMargins(10, 6, 10, 6)
        self._side_info = QLabel('Click a script to open its panel.')
        self._side_info.setWordWrap(True)
        self._side_info.setStyleSheet(f'color:{TEXT_GHOST};font-size: 12px;background:transparent;')
        hint_lay.addWidget(self._side_info)
        lay.addWidget(hint_w)

        self.rebuild_tree()
        return sb

    def rebuild_tree(self):
        # Clear existing rows (keep the stretch at end)
        while self._tree_lay.count() > 1:
            item = self._tree_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Build suite list from actual discovered suites
        # Use known order, fall back to discovered order
        KNOWN_ORDER = [
            'image_processing_suite', 'stacking_suite', 'lucky_suite',
            'solar_system_suite', 'variable_star_suite', 'transient_pipeline',
            'session_qa_suite', 'spectroscopy_suite', 'spectroscopy_pipeline',
            'exoplanet_transit_suite',
        ]
        ordered = []
        seen = set()
        for sid in KNOWN_ORDER:
            for s in self.suites:
                if s['id'] == sid and sid not in seen:
                    ordered.append(s)
                    seen.add(sid)
        for s in self.suites:
            if s['id'] not in seen:
                ordered.append(s)

        for suite in ordered:
            sid     = suite['id']
            color   = SUITE_COLORS.get(sid, ACCENT)
            loaded  = sid in self.loaded_suites
            expanded = self.open_suites.get(sid, False)
            script_list = SUITE_SCRIPTS.get(sid, [])

            row = SuiteRow(suite, color, loaded, expanded,
                           len(script_list), self)
            # Insert before the stretch
            self._tree_lay.insertWidget(
                self._tree_lay.count() - 1, row)

            if expanded:
                for sc in script_list:
                    fc = len(self.script_files.get(sc, []))
                    sr = ScriptRow(sc, sid, color, loaded, fc, self)
                    self._tree_lay.insertWidget(
                        self._tree_lay.count() - 1, sr)

    def toggle_suite_expand(self, suite_id):
        self.open_suites[suite_id] = not self.open_suites.get(suite_id, False)
        self.rebuild_tree()

    def toggle_suite(self, suite_id):
        if suite_id in self.loaded_suites:
            self.loaded_suites.discard(suite_id)
            # Close open panels for this suite's scripts
            for sc in SUITE_SCRIPTS.get(suite_id, []):
                if sc in self.windows:
                    self.windows[sc]._close()
        else:
            self.loaded_suites.add(suite_id)
        self.rebuild_tree()
        self._schedule_save()

    def open_script_window(self, suite_id, script_id):
        if script_id in self.windows:
            self.windows[script_id].raise_()
            self.focus_panel(self.windows[script_id])
            return
        color = SUITE_COLORS.get(suite_id, ACCENT)
        panel = ScriptPanel(
            self.desktop, suite_id, color, script_id,
            self.root, self.config, self.script_files,
            self.lazy_import_fn, self)
        self.windows[script_id] = panel
        self.desktop.hint.setVisible(False)
        self.rebuild_tree()

    def on_panel_closed(self, script_id):
        self.windows.pop(script_id, None)
        if not self.windows:
            self.desktop.hint.setVisible(True)
        self.rebuild_tree()

    def focus_panel(self, panel):
        for p in self.windows.values():
            p._set_style(focused=False)
        panel._set_style(focused=True)

    def _build_dock(self):
        dock = QWidget()
        dock.setFixedHeight(68)
        dock.setStyleSheet(
            f'background:#1e1e1e;border-top:1px solid #161616;')
        lay = QVBoxLayout(dock)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(4)
        lay.addWidget(_stat_label('FILE DOCK  —  drag files onto script panels or tree rows'))
        inner_scroll = QScrollArea()
        inner_scroll.setWidgetResizable(True)
        inner_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner_scroll.setStyleSheet('border:none;background:transparent;')
        chips_widget = QWidget()
        chips_widget.setStyleSheet('background:transparent;')
        chips_lay = QHBoxLayout(chips_widget)
        chips_lay.setContentsMargins(0, 0, 0, 0)
        chips_lay.setSpacing(5)

        # Add files button
        add_btn = QPushButton('＋  Add files…')
        add_btn.setFixedHeight(36)
        add_btn.setStyleSheet(
            f'background:#2a2a2a;border:1px solid {BORDER2};border-radius:3px;'
            f'color:{TEXT_DIM};font-size: 12px;padding:0 10px;')
        add_btn.clicked.connect(self._add_dock_files)
        chips_lay.addWidget(add_btn)

        clear_btn = QPushButton('✕  Clear')
        clear_btn.setFixedHeight(36)
        clear_btn.setStyleSheet(
            f'background:#2a2a2a;border:1px solid {BORDER2};border-radius:3px;'
            f'color:{TEXT_GHOST};font-size: 12px;padding:0 10px;')
        clear_btn.clicked.connect(self._clear_dock)
        chips_lay.addWidget(clear_btn)

        self._chips_lay = chips_lay
        chips_lay.addStretch()

        inner_scroll.setWidget(chips_widget)
        lay.addWidget(inner_scroll, 1)
        return dock

    def _add_dock_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Add files to dock', '',
            'FITS/SER/CSV (*.fits *.fit *.fts *.ser *.csv);;All (*)')
        for path in paths:
            p = Path(path)
            size_b = p.stat().st_size
            if size_b > 1_073_741_824:
                size_str = f'{size_b/1_073_741_824:.1f} GB'
            elif size_b > 1_048_576:
                size_str = f'{size_b/1_048_576:.0f} MB'
            else:
                size_str = f'{size_b/1024:.0f} KB'
            fi = {'name': p.name, 'size': size_str, 'path': str(p)}
            # Insert before stretch
            self._chips_lay.insertWidget(
                self._chips_lay.count() - 1, FileChip(fi))

    def _clear_dock(self):
        # Remove all FileChip widgets, keep Add and Clear buttons and stretch
        to_remove = []
        for i in range(self._chips_lay.count()):
            item = self._chips_lay.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), FileChip):
                to_remove.append(item.widget())
        for w in to_remove:
            w.deleteLater()

    def _build_statusbar(self):
        sb = QWidget()
        sb.setFixedHeight(20)
        sb.setStyleSheet(
            f'background:#1a1a1a;border-top:1px solid #141414;')
        lay = QHBoxLayout(sb)
        lay.setContentsMargins(8, 0, 8, 0)
        lay.setSpacing(8)
        self._status_lbl = QLabel('Ready.')
        self._status_lbl.setStyleSheet(
            f'color:#444444;font-size: 12px;background:transparent;')
        self._sb_prog = QProgressBar()
        self._sb_prog.setFixedSize(120, 3)
        self._sb_prog.setRange(0, 100)
        self._sb_prog.setTextVisible(False)
        self._sb_prog.setVisible(False)
        self._sb_prog.setStyleSheet(
            'QProgressBar{background:#222;border:none;border-radius:1px;}'
            'QProgressBar::chunk{background:#3d6ea0;border-radius:1px;}')
        arch_lbl = QLabel('Lazy load  ·  memmap FITS  ·  result cache')
        arch_lbl.setStyleSheet(f'color:#2a2a2a;font-size: 12px;background:transparent;')
        lay.addWidget(self._status_lbl, 1)
        lay.addWidget(self._sb_prog)
        lay.addWidget(arch_lbl)
        return sb

    def set_status(self, msg, progress=-1):
        self._status_lbl.setText(msg)
        self._side_info.setText(msg)
        if progress >= 0:
            self._sb_prog.setVisible(True)
            self._sb_prog.setValue(progress)
        else:
            self._sb_prog.setVisible(False)

    def _update_stats(self):
        self._cpu = max(5, min(90, self._cpu + (random.random()-0.5)*9))
        self._ram = max(22, min(86, self._ram + (random.random()-0.5)*3))
        self._cpu_bar.setValue(int(self._cpu))
        self._cpu_val.setText(f'{int(self._cpu)}%')
        self._ram_bar.setValue(int(self._ram))
        self._ram_val.setText(f'{self._ram/100*16:.1f} GB')
        busy = self._cpu > 60
        self._led.setStyleSheet(
            f'background:{"#5a3010" if busy else "#2a5020"};'
            f'border-radius:3px;border:none;')
        self._led_txt.setText('ACTIVE' if busy else 'IDLE')

    def _schedule_save(self):
        self._save_timer.start(2000)

    def _do_save(self):
        self.config['working_dir']   = self.wd_edit.text()
        self.config['import_dir']    = self.import_edit.text()
        self.config['export_dir']    = self.export_edit.text()
        self.config['loaded_suites'] = list(self.loaded_suites)
        self.save_config_fn(self.config)

    def closeEvent(self, e):
        self._do_save()
        e.accept()


# ── Small helpers ─────────────────────────────────────────────────────────────

def _vsep():
    w = QFrame()
    w.setFrameShape(QFrame.Shape.VLine)
    w.setFixedWidth(1)
    w.setStyleSheet(f'background:{BORDER};border:none;')
    return w


def _stat_label(text):
    l = QLabel(text)
    l.setStyleSheet(f'color:{TEXT_GHOST};font-size: 12px;background:transparent;')
    return l


def _path_label(text):
    l = QLabel(text)
    l.setStyleSheet(f'color:{TEXT_FAINT};font-size: 12px;background:transparent;')
    return l


# ── Entry point ───────────────────────────────────────────────────────────────

def run(app, root, suites, scripts, config, save_config_fn,
        lazy_import_fn, logo_path):
    """Called by hyperload.py."""
    app.setStyleSheet(STYLESHEET)
    win = HyperloadWindow(
        root, suites, scripts, config,
        save_config_fn, lazy_import_fn, logo_path)
    win.show()
    sys.exit(app.exec())
