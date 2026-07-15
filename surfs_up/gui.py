"""PyQt GUI for configuring and running SURF workflows."""

import csv
import datetime
import inspect
import json
import re
import sys
import traceback
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

# Support IDEs that launch this file directly instead of using ``python -m``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import astropy.units as u
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import surf
from astropy.io import fits
from astropy.time import Time
from PyQt6.QtCore import QDateTime, QObject, Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QPainter
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDateTimeEdit,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTabBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from sunpy.coordinates import sun

import surf.surf_analysis as sa
import surf.surf_inputs as sin
from surfs_up.core import (
    SimulationRequest,
    build_generated_code,
    build_uniform_boundary_code,
    format_datetime_axis_like_surf,
    plot_custom_timeseries,
    plot_radial as plot_radial_profile,
    run_generated_code,
    timeseries_figsize,
)


EXAMPLE_INPUTS_DIR = Path(surf.__file__).resolve().parent / "data" / "example_inputs"
DONKI_CME_ANALYSIS_URL = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CMEAnalysis"
SUPPORTED_OBSERVERS = {
    "MERCURY", "VENUS", "EARTH", "MARS", "JUPITER", "SATURN", "ACE",
    "PSP", "SOLO", "STA", "STB", "ULYSSES",
}
APP_STYLESHEET = """
QWidget { font-size: 10pt; }
QPushButton[role="primary"] { font-weight: 600; }
QTextEdit[role="console"] { font-family: Consolas, monospace; }
"""


def parse_wsa_start_time(filepath: Path):
    """Infer a WSA map timestamp from FITS metadata or filename text."""
    filepath = Path(filepath)
    try:
        header = fits.getheader(filepath)
        for key in ("DATE-OBS", "DATE_OBS", "DATE", "MAPDATE"):
            if key in header:
                value = str(header[key]).strip()
                for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
                    try:
                        return datetime.datetime.strptime(value[:19], fmt)
                    except ValueError:
                        continue
    except Exception:
        pass
    match = re.search(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", filepath.name)
    if match:
        return datetime.datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return None


def parse_cortom_start_time(filepath: Path):
    """Infer a CorTom timestamp from filename text."""
    match = re.search(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", Path(filepath).name)
    if match:
        return datetime.datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return None


def _wsa_speed_map(filepath: str):
    return sin.get_WSA_maps(filepath)


def _draw_speed_map(ax, speed_map, longitudes, latitudes, extraction_latitude, title):
    """Draw an ambient speed map using the same axis conventions as the web preview."""
    speed_values = speed_map.to_value(u.km / u.s) if hasattr(speed_map, "to_value") else np.asarray(speed_map)
    lon_values = longitudes.to_value(u.deg) if hasattr(longitudes, "to_value") else np.rad2deg(np.asarray(longitudes))
    lat_values = latitudes.to_value(u.deg) if hasattr(latitudes, "to_value") else np.rad2deg(np.asarray(latitudes))
    speed_values = np.asarray(speed_values)
    if speed_values.shape == (len(lon_values), len(lat_values)):
        speed_values = speed_values.T
    mesh = ax.pcolormesh(lon_values, lat_values, speed_values, shading="auto")
    ax.axhline(extraction_latitude, color="white", linewidth=1.2, linestyle="--")
    ax.set_xlim(float(np.nanmin(lon_values)), float(np.nanmax(lon_values)))
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(title)
    plt.colorbar(mesh, ax=ax, label="Vin [km/s]")


class UserSpecifiedAmbientTab(QWidget):
    """Simple user-defined speed profile controls."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        box = QGroupBox("User-specified boundary")
        form = QFormLayout()
        self.speed_profile_edit = QTextEdit()
        self.speed_profile_edit.setPlainText(", ".join(["400"] * 128))
        form.addRow("Speed profile [km/s]", self.speed_profile_edit)
        box.setLayout(form)
        layout.addWidget(box)
        self.setLayout(layout)

    def get_state(self):
        values = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", self.speed_profile_edit.toPlainText())]
        return {"speed_profile_kms": values or [400.0] * 128}


class MasAmbientTab(QWidget):
    """MAS ambient source controls."""

    status_message = pyqtSignal(str)
    error_message = pyqtSignal(str)
    start_time_selected = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.model_inner_boundary_rs = 21.5
        self.model_latitude_deg = 0.0
        self.model_solver = "huxt"
        self.include_bpol = False

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        box = QGroupBox("HelioMAS boundary profile")
        form = QFormLayout()
        self.cr_spin = QSpinBox()
        self.cr_spin.setRange(1, 9999)
        self.cr_spin.setValue(2000)
        self.source_radius_spin = QDoubleSpinBox()
        self.source_radius_spin.setRange(1.0, 1000.0)
        self.source_radius_spin.setDecimals(1)
        self.source_radius_spin.setValue(30.0)
        self.source_radius_spin.setSuffix(" Rs")
        self.decelerate_toggle = QCheckBox("Decelerate to inner boundary")
        self.decelerate_toggle.setChecked(True)
        self.use_map_time_toggle = QCheckBox("Use map time - 5 days as model start time")
        self.use_map_time_toggle.setChecked(True)
        self.use_map_time_toggle.toggled.connect(self._on_use_map_time_toggled)
        self.plot_button = QPushButton("Extract and Plot Vin")
        self.plot_button.setProperty("role", "primary")
        self.plot_button.clicked.connect(self.plot_profile)
        form.addRow("Carrington rotation", self.cr_spin)
        form.addRow("Map radius", self.source_radius_spin)
        form.addRow("", self.decelerate_toggle)
        form.addRow("", self.use_map_time_toggle)
        form.addRow(self.plot_button)
        box.setLayout(form)
        layout.addWidget(box)
        self.setLayout(layout)

    def get_state(self):
        return {
            "cr_num": self.cr_spin.value(),
            "source_radius_rs": self.source_radius_spin.value(),
            "decelerate_to_inner_boundary": self.decelerate_toggle.isChecked(),
        }

    def set_model_inner_boundary(self, rmin_rs: float):
        self.model_inner_boundary_rs = float(rmin_rs)

    def set_model_latitude(self, latitude_deg: float):
        self.model_latitude_deg = float(latitude_deg)

    def set_model_solver(self, solver_name: str):
        self.model_solver = str(solver_name).strip().lower()

    def set_include_bpol(self, include_bpol: bool):
        self.include_bpol = bool(include_bpol)

    def _mas_map_time(self):
        map_time = sun.carrington_rotation_time(float(self.cr_spin.value())).to_datetime()
        return map_time.replace(tzinfo=None) if map_time.tzinfo is not None else map_time

    def emit_map_time_if_enabled(self):
        if self.use_map_time_toggle.isChecked():
            self.start_time_selected.emit(
                self._mas_map_time() - datetime.timedelta(days=5)
            )

    def _on_use_map_time_toggled(self, enabled: bool):
        if enabled:
            self.emit_map_time_if_enabled()

    def plot_profile(self):
        original_text = self.plot_button.text()
        self.plot_button.setText("downloading and processing")
        self.plot_button.setEnabled(False)
        QApplication.processEvents()
        try:
            cr_num = self.cr_spin.value()
            latitude = self.model_latitude_deg * u.deg
            speed_map, longitudes, latitudes = sin.get_MAS_vr_map(cr_num)
            v_orig = sin.get_MAS_long_profile(cr_num, latitude)
            if self.decelerate_toggle.isChecked():
                acc_profile = "huxt" if self.model_solver == "huxt" else "parker"
                v_mapped = sin.map_v_boundary_inwards(
                    v_orig,
                    self.source_radius_spin.value() * u.solRad,
                    self.model_inner_boundary_rs * u.solRad,
                    acc_profile=acc_profile,
                )
            else:
                v_mapped = v_orig
            lon = np.linspace(0.0, 360.0, len(v_orig), endpoint=False)
            fig, (ax_map, ax_v) = plt.subplots(2, 1, figsize=(10, 9))
            _draw_speed_map(ax_map, speed_map, longitudes, latitudes, latitude.value, f"MAS speed map | CR {cr_num}")
            ax_map.set_xlabel("Carrington longitude [deg]")
            ax_v.plot(
                lon,
                v_orig.to_value(u.km / u.s),
                label=f"Original at {self.source_radius_spin.value():.1f} Rs",
            )
            ax_v.plot(lon, v_mapped.to_value(u.km / u.s), linestyle="--", label=f"Mapped to {self.model_inner_boundary_rs:.1f} Rs")
            ax_v.set_xlim(0.0, 360.0)
            ax_v.set_xlabel("Carrington longitude [deg]")
            ax_v.set_ylabel("Vin [km/s]")
            ax_v.grid(True, alpha=0.3)
            ax_v.legend()
            fig.tight_layout()
            plt.show()
            self.status_message.emit("MAS data are available; speed map and profile plotted.")
        except Exception:
            self.error_message.emit(traceback.format_exc())
        finally:
            self.plot_button.setText(original_text)
            self.plot_button.setEnabled(True)


class ModelParametersTab(QWidget):
    """Core model configuration controls."""

    start_datetime_updated = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        now = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=5)).replace(tzinfo=None, microsecond=0)
        cr_num, cr_lon = sin.datetime2surfinputs(now)
        earth_lat = sin.get_earth_lat(now)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        model_box = QGroupBox("Model Parameters")
        form = QFormLayout()
        self.start_datetime = QDateTimeEdit(QDateTime(now))
        self.start_datetime.setCalendarPopup(True)
        self.start_datetime.dateTimeChanged.connect(self._sync_from_datetime)
        self.cr_num_spin = QSpinBox()
        self.cr_num_spin.setRange(1, 9999)
        self.cr_num_spin.setValue(int(cr_num))
        self.cr_lon_init_spin = QDoubleSpinBox()
        self.cr_lon_init_spin.setRange(0.0, 360.0)
        self.cr_lon_init_spin.setValue(cr_lon.to_value(u.deg))
        self.simtime_spin = QDoubleSpinBox()
        self.simtime_spin.setRange(0.01, 1000.0)
        self.simtime_spin.setValue(5.0)
        self.solver_combo = QComboBox()
        self.solver_combo.addItems(["huxt", "hydro", "hydro-pcm"])
        self.include_bpol_toggle = QCheckBox("Include B polarity")
        form.addRow(
            "Model start date (default: 5 days prior to forecast date, to allow CME propagation)",
            self.start_datetime,
        )
        form.addRow("CR", self.cr_num_spin)
        form.addRow("Earth Carr lon at start", self.cr_lon_init_spin)
        form.addRow("Run time", self.simtime_spin)
        form.addRow("Solver", self.solver_combo)
        form.addRow("", self.include_bpol_toggle)
        model_box.setLayout(form)

        advanced_box = QGroupBox("Advanced")
        advanced_form = QFormLayout()
        self.rmin_spin = QDoubleSpinBox(); self.rmin_spin.setRange(1.0, 500.0); self.rmin_spin.setValue(21.5)
        self.rmax_spin = QDoubleSpinBox(); self.rmax_spin.setRange(2.0, 5000.0); self.rmax_spin.setValue(240.0)
        self.lon_min_spin = QDoubleSpinBox(); self.lon_min_spin.setRange(-360.0, 360.0); self.lon_min_spin.setValue(315.0)
        self.lon_max_spin = QDoubleSpinBox(); self.lon_max_spin.setRange(-360.0, 360.0); self.lon_max_spin.setValue(45.0)
        self.latitude_spin = QDoubleSpinBox(); self.latitude_spin.setRange(-90.0, 90.0); self.latitude_spin.setValue(float(earth_lat.to_value(u.deg) if hasattr(earth_lat, "to_value") else earth_lat))
        self.frame_combo = QComboBox(); self.frame_combo.addItem("Sidereal", "sidereal"); self.frame_combo.addItem("Synodic", "synodic")
        self.one_d_toggle = QCheckBox("Run as 1D")
        self.one_d_toggle.toggled.connect(self._on_1d_toggled)
        self.dr_spin = QDoubleSpinBox(); self.dr_spin.setRange(0.01, 100.0); self.dr_spin.setValue(1.5)
        self.nlon_spin = QSpinBox(); self.nlon_spin.setRange(1, 4096); self.nlon_spin.setValue(128)
        self.vmax_spin = QDoubleSpinBox(); self.vmax_spin.setRange(1.0, 10000.0); self.vmax_spin.setValue(3000.0)
        self.streak_lines_toggle = QCheckBox("Enable streak lines")
        self.streak_spacing_spin = QDoubleSpinBox(); self.streak_spacing_spin.setRange(0.1, 360.0); self.streak_spacing_spin.setValue(10.0)
        advanced_form.addRow("rmin", self.rmin_spin)
        advanced_form.addRow("rmax", self.rmax_spin)
        advanced_form.addRow("Longitude start", self.lon_min_spin)
        advanced_form.addRow("Longitude stop", self.lon_max_spin)
        advanced_form.addRow("Model latitude", self.latitude_spin)
        advanced_form.addRow("Frame", self.frame_combo)
        advanced_form.addRow("", self.one_d_toggle)
        advanced_form.addRow("dr", self.dr_spin)
        advanced_form.addRow("nlon", self.nlon_spin)
        advanced_form.addRow("vmax", self.vmax_spin)
        advanced_form.addRow("", self.streak_lines_toggle)
        advanced_form.addRow("Streak spacing", self.streak_spacing_spin)
        advanced_box.setLayout(advanced_form)
        layout.addWidget(model_box)
        layout.addWidget(advanced_box)
        self.setLayout(layout)
        self._on_1d_toggled(False)

    def _sync_from_datetime(self):
        dt = self.start_datetime.dateTime().toPyDateTime()
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        cr_num, cr_lon = sin.datetime2surfinputs(dt)
        self.cr_num_spin.blockSignals(True); self.cr_lon_init_spin.blockSignals(True)
        self.cr_num_spin.setValue(int(cr_num)); self.cr_lon_init_spin.setValue(cr_lon.to_value(u.deg))
        self.cr_num_spin.blockSignals(False); self.cr_lon_init_spin.blockSignals(False)
        self.start_datetime_updated.emit(dt)

    def _on_1d_toggled(self, enabled: bool):
        self.lon_min_spin.setEnabled(not enabled)
        self.lon_max_spin.setEnabled(not enabled)
        self.frame_combo.setCurrentIndex(1 if enabled else 0)

    def set_start_datetime(self, dt: datetime.datetime):
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        self.start_datetime.setDateTime(QDateTime(dt))

    def get_state(self):
        return {
            "solver": self.solver_combo.currentText().lower(),
            "rmin": self.rmin_spin.value(),
            "rmax": self.rmax_spin.value(),
            "lon_min": self.lon_min_spin.value(),
            "lon_max": self.lon_max_spin.value(),
            "latitude": self.latitude_spin.value(),
            "is_1d": self.one_d_toggle.isChecked(),
            "frame": self.frame_combo.currentData(),
            "simtime_days": self.simtime_spin.value(),
            "dr_rs": self.dr_spin.value(),
            "nlon": self.nlon_spin.value(),
            "vmax_kms": self.vmax_spin.value(),
            "streak_lines_enabled": self.streak_lines_toggle.isChecked(),
            "streak_spacing_deg": self.streak_spacing_spin.value(),
            "include_bpol": self.include_bpol_toggle.isChecked(),
            "start_datetime": self.start_datetime.dateTime().toString("yyyy-MM-dd HH:mm:ss"),
            "cr_num": self.cr_num_spin.value(),
            "cr_lon_init_deg": self.cr_lon_init_spin.value(),
        }


class InSituAmbientTab(QWidget):
    """Controls for OMNI backmapped reconstruction or forecast boundary setup."""

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        box = QGroupBox("OMNI backmapped boundary")
        form = QFormLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["reconstruction", "forecast"])

        self.forecast_datetime = QDateTimeEdit()
        self.forecast_datetime.setCalendarPopup(True)
        self.forecast_datetime.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.mode_combo.currentTextChanged.connect(self._sync_forecast_time_enabled)

        info = QLabel(
            "Uses the model start time and run time to initialize the OMNI-based "
            "SURF setup."
        )
        info.setWordWrap(True)

        form.addRow("Mode", self.mode_combo)
        form.addRow("Forecast time", self.forecast_datetime)
        box.setLayout(form)

        layout.addWidget(box)
        layout.addWidget(info)
        self.setLayout(layout)
        self.set_model_start_datetime(
            datetime.datetime.now(datetime.UTC).replace(tzinfo=None, microsecond=0)
        )
        self._sync_forecast_time_enabled(self.mode_combo.currentText())

    def get_state(self):
        """Return current InSitu settings."""
        return {
            "mode": self.mode_combo.currentText(),
            "forecast_datetime": self.forecast_datetime.dateTime().toString(
                "yyyy-MM-dd HH:mm:ss"
            ),
        }

    def set_model_start_datetime(self, dt: datetime.datetime):
        """Keep the forecast time five days after the model start time."""
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        self.forecast_datetime.setDateTime(QDateTime(dt + datetime.timedelta(days=5)))

    def _sync_forecast_time_enabled(self, mode: str):
        self.forecast_datetime.setEnabled(mode == "forecast")


class OmniAmbientTab(QWidget):
    """Controls for OMNI-driven time-dependent boundaries."""

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        box = QGroupBox("OMNI outwards boundary")
        form = QFormLayout()

        self.use_215_inner_boundary_toggle = QCheckBox(
            "Set inner boundary to 215 Rs and run outwards"
        )
        self.use_215_inner_boundary_toggle.setChecked(True)

        info = QLabel(
            "Build time-dependent boundary conditions directly from OMNI observations "
            "for outward model runs."
        )
        info.setWordWrap(True)

        form.addRow("", self.use_215_inner_boundary_toggle)
        box.setLayout(form)

        layout.addWidget(box)
        layout.addWidget(info)
        self.setLayout(layout)

    def get_state(self):
        """Return current OMNI settings."""
        return {
            "use_215_inner_boundary": self.use_215_inner_boundary_toggle.isChecked(),
        }


class FileAmbientTab(QWidget):
    """Shared UI for ambient sources that select a single input file."""

    status_message = pyqtSignal(str)
    error_message = pyqtSignal(str)
    start_time_selected = pyqtSignal(object)

    def __init__(
        self,
        title: str,
        filter_text: str,
        parser,
        description: str,
        include_decelerate_option: bool = False,
        include_wsa_speed_reduction: bool = False,
        include_use_map_time_option: bool = False,
        default_pattern: str = "",
        profile_loader=None,
        br_profile_loader=None,
        speed_map_loader=None,
        source_radius_rs: float | None = None,
    ):
        super().__init__()
        self.filter_text = filter_text
        self.parser = parser
        self.selected_file = ""
        self.include_decelerate_option = include_decelerate_option
        self.include_wsa_speed_reduction = include_wsa_speed_reduction
        self.include_use_map_time_option = include_use_map_time_option
        self.default_pattern = default_pattern
        self.profile_loader = profile_loader
        self.br_profile_loader = br_profile_loader
        self.speed_map_loader = speed_map_loader
        self.source_radius_rs = source_radius_rs
        self.model_inner_boundary_rs = 21.5
        self.model_latitude_deg = 0.0
        self.model_solver = "huxt"
        self.model_start_datetime = datetime.datetime.now(datetime.UTC).replace(tzinfo=None, microsecond=0)
        self.include_bpol = False
        self.last_parsed_time = None

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        box = QGroupBox(title)
        form = QFormLayout()

        file_row = QWidget()
        file_layout = QHBoxLayout()
        file_layout.setContentsMargins(0, 0, 0, 0)

        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        self.select_button = QPushButton("Select file")
        self.select_button.clicked.connect(self.select_file)

        file_layout.addWidget(self.file_edit)
        file_layout.addWidget(self.select_button)
        file_row.setLayout(file_layout)
        self.file_row = file_row

        self.detected_time_label = QLabel("No file selected.")
        self.detected_time_label.setWordWrap(True)

        if self.profile_loader is not None:
            self.plot_button = QPushButton("Extract and Plot Vin")
            self.plot_button.setProperty("role", "primary")
            self.plot_button.clicked.connect(self.plot_profile)

        if self.include_decelerate_option:
            self.decelerate_toggle = QCheckBox("Decelerate to inner boundary")
            self.decelerate_toggle.setChecked(True)

        if self.include_wsa_speed_reduction:
            self.wsa_speed_reduction_toggle = QCheckBox("Apply WSA speed reduction")
            self.wsa_speed_reduction_toggle.setChecked(True)

        if self.include_use_map_time_option:
            self.use_map_time_toggle = QCheckBox("Use map time - 5 days as model start time")
            self.use_map_time_toggle.setChecked(True)
            self.use_map_time_toggle.toggled.connect(self._on_use_map_time_toggled)

        if self.source_radius_rs is not None:
            self.source_radius_spin = QDoubleSpinBox()
            self.source_radius_spin.setRange(1.0, 1000.0)
            self.source_radius_spin.setDecimals(1)
            self.source_radius_spin.setValue(float(self.source_radius_rs))
            self.source_radius_spin.setSuffix(" Rs")

        form.addRow("File", file_row)
        form.addRow("Detected start time", self.detected_time_label)
        if self.source_radius_rs is not None:
            form.addRow("Map radius", self.source_radius_spin)
        if self.include_wsa_speed_reduction:
            form.addRow("", self.wsa_speed_reduction_toggle)
        if self.include_decelerate_option:
            form.addRow("", self.decelerate_toggle)
        if self.include_use_map_time_option:
            form.addRow("", self.use_map_time_toggle)
        if self.profile_loader is not None:
            form.addRow(self.plot_button)
        box.setLayout(form)
        self.file_form = form

        info = QLabel(description)
        info.setWordWrap(True)

        layout.addWidget(box)
        layout.addWidget(info)
        self.setLayout(layout)
        self._load_default_file()

    def select_file(self):
        """Open a file chooser rooted at the SURF example inputs directory."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select file",
            str(EXAMPLE_INPUTS_DIR),
            self.filter_text,
        )
        if filepath:
            self._apply_selected_file(filepath)

    def _apply_selected_file(self, filepath: str):
        """Apply a selected file path and emit a start-time update when possible."""
        try:
            parsed_time = self.parser(Path(filepath))
            self.selected_file = filepath
            self.file_edit.setText(filepath)
            self.last_parsed_time = parsed_time

            if parsed_time is not None:
                self.detected_time_label.setText(parsed_time.strftime("%Y-%m-%d %H:%M:%S UTC"))
                if self._should_apply_map_time():
                    self.start_time_selected.emit(parsed_time - datetime.timedelta(days=5))
                    self.status_message.emit("Run start time set to map time - 5 days.")
                else:
                    self.status_message.emit("File selected. Map time not applied to model start.")
            else:
                self.detected_time_label.setText("Could not determine date from metadata or filename.")
                self.status_message.emit("File selected, but no start time could be inferred.")
        except Exception:
            self.error_message.emit(traceback.format_exc())

    def get_state(self):
        """Return selected file path for code generation."""
        state = {"filepath": self.selected_file}
        if self.source_radius_rs is not None:
            state["source_radius_rs"] = self.source_radius_spin.value()
        if self.include_decelerate_option:
            state["decelerate_to_inner_boundary"] = self.decelerate_toggle.isChecked()
        if self.include_wsa_speed_reduction:
            state["apply_wsa_speed_reduction"] = self.wsa_speed_reduction_toggle.isChecked()
        return state

    def _load_default_file(self):
        """Select a default example file when available on the local machine."""
        if not self.default_pattern:
            return

        matches = sorted(EXAMPLE_INPUTS_DIR.glob(self.default_pattern))
        if not matches:
            return

        self._apply_selected_file(str(matches[0]))

    def set_model_inner_boundary(self, rmin_rs: float):
        """Set model inner boundary radius used for profile comparison mapping."""
        self.model_inner_boundary_rs = float(rmin_rs)

    def set_model_latitude(self, latitude_deg: float):
        """Set model latitude used for ambient profile extraction."""
        self.model_latitude_deg = float(latitude_deg)

    def set_model_solver(self, solver_name: str):
        """Set solver used for solver-specific WSA speed reduction."""
        self.model_solver = str(solver_name).strip().lower()

    def set_model_start_datetime(self, dt: datetime.datetime):
        """Set model start time used by download-backed ambient sources."""
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        self.model_start_datetime = dt

    def set_include_bpol(self, include_bpol: bool):
        """Enable/disable bpol extraction in profile preview plots."""
        self.include_bpol = bool(include_bpol)

    def _should_apply_map_time(self) -> bool:
        """Return True when parsed map time should update model start."""
        if not self.include_use_map_time_option:
            return False
        return self.use_map_time_toggle.isChecked()

    def _on_use_map_time_toggled(self, enabled: bool):
        """Apply stored parsed map time when enabled and available."""
        if enabled and self.last_parsed_time is not None:
            self.start_time_selected.emit(
                self.last_parsed_time - datetime.timedelta(days=5)
            )
            self.status_message.emit("Run start time set to map time - 5 days.")

    def emit_map_time_if_enabled(self):
        """Public helper to apply parsed map time when this option is enabled."""
        if self.last_parsed_time is None:
            return
        if not self._should_apply_map_time():
            return
        self.start_time_selected.emit(
            self.last_parsed_time - datetime.timedelta(days=5)
        )

    def plot_profile(self):
        """Extract and plot source profile together with mapped-to-rmin profile."""
        if self.profile_loader is None:
            return

        if not self.selected_file:
            self.status_message.emit("Select an input file before plotting.")
            return

        original_text = self.plot_button.text()
        original_style = self.plot_button.styleSheet()
        self.plot_button.setText("downloading and processing")
        self.plot_button.setStyleSheet(
            "QPushButton { background-color: #b22222; color: white; font-weight: 600; }"
        )
        self.plot_button.setEnabled(False)
        QApplication.processEvents()

        try:
            latitude = self.model_latitude_deg * u.deg
            map_to_inner = (
                self.decelerate_toggle.isChecked()
                if self.include_decelerate_option
                else True
            )
            v_orig = self.profile_loader(self.selected_file, latitude)
            if self.speed_map_loader is not None:
                speed_map, map_longitudes, map_latitudes = self.speed_map_loader(
                    self.selected_file
                )
            source_radius_rs = (
                21.5
                if self.source_radius_rs is None
                else self.source_radius_spin.value()
            )
            apply_wsa_reduction = (
                self.wsa_speed_reduction_toggle.isChecked()
                if self.include_wsa_speed_reduction
                else False
            )
            if apply_wsa_reduction:
                longitude = np.linspace(
                    0.0, 2.0 * np.pi, len(v_orig), endpoint=False
                ) * u.rad
                mapper = (
                    sin.map_v_inwards
                    if self.model_solver == "huxt"
                    else sin.map_v_inwards_parker
                )
                wsa_reduction = mapper(
                    v_orig,
                    215.0 * u.solRad,
                    longitude,
                    self.model_inner_boundary_rs * u.solRad,
                )
                v_reduced = wsa_reduction[0]
            else:
                v_reduced = v_orig

            include_bpol_plot = self.include_bpol and (self.br_profile_loader is not None)
            acc_profile = "huxt" if self.model_solver == "huxt" else "parker"
            if include_bpol_plot:
                b_orig = self.br_profile_loader(self.selected_file, latitude)
                if map_to_inner:
                    mapped = sin.map_v_boundary_inwards(
                        v_reduced,
                        source_radius_rs * u.solRad,
                        self.model_inner_boundary_rs * u.solRad,
                        acc_profile=acc_profile,
                        b_orig=b_orig,
                    )
                    if isinstance(mapped, tuple):
                        v_mapped, b_mapped = mapped
                    else:
                        v_mapped = mapped
                        b_mapped = np.ones(len(v_orig)) * np.nan
                else:
                    v_mapped = v_reduced
                    b_mapped = b_orig
            else:
                if map_to_inner:
                    v_mapped = sin.map_v_boundary_inwards(
                        v_reduced,
                        source_radius_rs * u.solRad,
                        self.model_inner_boundary_rs * u.solRad,
                        acc_profile=acc_profile,
                    )
                else:
                    v_mapped = v_reduced
            carr_lon = np.linspace(0.0, 360.0, len(v_orig), endpoint=False)

            include_speed_map = self.speed_map_loader is not None
            if include_speed_map and include_bpol_plot:
                fig, (ax_map, ax_v, ax_b) = plt.subplots(
                    3, 1, figsize=(10, 12)
                )
            elif include_speed_map:
                fig, (ax_map, ax_v) = plt.subplots(2, 1, figsize=(10, 9))
            elif include_bpol_plot:
                fig, (ax_v, ax_b) = plt.subplots(2, 1, sharex=True)
            else:
                fig, ax_v = plt.subplots()

            if include_speed_map:
                _draw_speed_map(
                    ax_map,
                    speed_map,
                    map_longitudes,
                    map_latitudes,
                    latitude.value,
                    f"{self.__class__.__name__.replace('AmbientTab', '')} speed map",
                )
                ax_map.set_xlabel("Carrington longitude [deg]")

            ax_v.plot(
                carr_lon,
                v_orig.to_value(u.km / u.s),
                linewidth=1.5,
                label=f"Original at {source_radius_rs:.1f} Rs",
            )
            if apply_wsa_reduction:
                ax_v.plot(
                    carr_lon,
                    v_reduced.to_value(u.km / u.s),
                    linewidth=1.5,
                    linestyle="-.",
                    label=(
                        "WSA speed reduction: "
                        f"215 to {self.model_inner_boundary_rs:.1f} Rs "
                        "(longitude unchanged)"
                    ),
                )
            ax_v.plot(
                carr_lon,
                v_mapped.to_value(u.km / u.s),
                linewidth=1.5,
                linestyle="--",
                label=(
                    f"Mapped to {self.model_inner_boundary_rs:.1f} Rs"
                    if map_to_inner
                    else (
                        "Speed-reduced boundary"
                        if apply_wsa_reduction
                        else "Original (no deceleration mapping)"
                    )
                ),
            )
            ax_v.set_xlim(0.0, 360.0)
            ax_v.set_ylabel("Vin [km/s]")
            ax_v.grid(True, alpha=0.3)
            ax_v.legend()

            if include_bpol_plot:
                ax_b.plot(
                    carr_lon,
                    np.asarray(b_orig),
                    linewidth=1.5,
                    label=f"Original bpol at {source_radius_rs:.1f} Rs",
                )
                ax_b.plot(
                    carr_lon,
                    np.asarray(b_mapped),
                    linewidth=1.5,
                    linestyle="--",
                    label=(
                        f"Mapped bpol to {self.model_inner_boundary_rs:.1f} Rs"
                        if map_to_inner
                        else "Original bpol (no deceleration mapping)"
                    ),
                )
                ax_b.set_xlim(0.0, 360.0)
                ax_b.set_xlabel("Carrington longitude [deg]")
                ax_b.set_ylabel("bpol")
                ax_b.grid(True, alpha=0.3)
                ax_b.legend()
            else:
                ax_v.set_xlabel("Carrington longitude [deg]")

            fig.tight_layout()
            plt.show()
            self.status_message.emit("Ambient profile plotted.")
        except Exception:
            self.error_message.emit(traceback.format_exc())
        finally:
            self.plot_button.setEnabled(True)
            self.plot_button.setText(original_text)
            self.plot_button.setStyleSheet(original_style)


class WsaAmbientTab(FileAmbientTab):
    """WSA boundary tab backed by either a local file or the ISWA archive."""

    def __init__(self, use_iswa_download: bool = False):
        self.use_iswa_download = use_iswa_download
        title = "WSA map from ISWA" if use_iswa_download else "WSA boundary file"
        description = (
            "Download the WSA map required for the selected map date. ISWA uses "
            "the newest available map at or before that time."
            if use_iswa_download
            else "Select a WSA FITS file. The run start time is inferred from FITS metadata when "
            "available, otherwise from the filename."
        )
        super().__init__(
            title,
            "WSA FITS (*.fits)",
            parse_wsa_start_time,
            description,
            include_decelerate_option=True,
            include_wsa_speed_reduction=True,
            include_use_map_time_option=not use_iswa_download,
            default_pattern=None if use_iswa_download else "**/*.fits",
            profile_loader=sin.get_WSA_long_profile,
            br_profile_loader=sin.get_WSA_br_long_profile,
            speed_map_loader=_wsa_speed_map,
            source_radius_rs=21.5,
        )
        if use_iswa_download:
            self.iswa_datetime_edit = QDateTimeEdit()
            self.iswa_datetime_edit.setCalendarPopup(True)
            self.iswa_datetime_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
            self.iswa_datetime_edit.setDateTime(
                QDateTime(self.model_start_datetime + datetime.timedelta(days=5))
            )
            self.iswa_datetime_edit.dateTimeChanged.connect(self._on_iswa_datetime_changed)
            self.use_model_start_for_iswa_toggle = QCheckBox(
                "Use map time - 5 days as model start time"
            )
            self.use_model_start_for_iswa_toggle.setChecked(True)
            self.use_model_start_for_iswa_toggle.toggled.connect(
                self._on_use_model_start_for_iswa_toggled
            )
            iswa_datetime_row = QWidget()
            iswa_datetime_layout = QHBoxLayout(iswa_datetime_row)
            iswa_datetime_layout.setContentsMargins(0, 0, 0, 0)
            iswa_datetime_layout.addWidget(self.iswa_datetime_edit, 1)
            iswa_datetime_layout.addWidget(self.use_model_start_for_iswa_toggle)
            self.file_form.insertRow(0, "Map date/time", iswa_datetime_row)
            self.file_form.setRowVisible(self.file_row, False)
            self.select_button.setText("Download required map")
            self.plot_button.setText("Check data availability and plot")
            self.detected_time_label.setText("No ISWA map downloaded yet.")
            self._sync_iswa_datetime_state()

    def select_file(self):
        """Download the ISWA map or select a local WSA file."""
        if not self.use_iswa_download:
            return super().select_file()

        self.selected_file = ""
        self.file_edit.clear()
        self.last_parsed_time = None
        self.detected_time_label.setText("No valid ISWA map downloaded.")

        try:
            required_for = self._iswa_requested_datetime()
            path = sin.get_WSA_from_ISWA(required_for)
            map_time = parse_wsa_start_time(Path(path))
            if map_time is None:
                raise ValueError(
                    f"Could not determine the timestamp of downloaded WSA map: {path}"
                )

            date_offset = abs((map_time.date() - required_for.date()).days)
            if date_offset > 1:
                raise ValueError(
                    "No WSA map was found within one day of the requested map date "
                    f"{required_for:%Y-%m-%d} UTC. ISWA returned a map dated "
                    f"{map_time:%Y-%m-%d %H:%M:%S} UTC."
                )

            self._apply_selected_file(str(path))
            self.status_message.emit(
                "Required WSA map downloaded from ISWA for map date "
                f"{required_for:%Y-%m-%d} UTC."
            )
        except Exception:
            self.error_message.emit(traceback.format_exc())

    def plot_profile(self):
        """Download the ISWA map on demand before plotting, if needed."""
        if self.use_iswa_download and not self.selected_file:
            self.select_file()
            if not self.selected_file:
                return
        super().plot_profile()

    def set_model_start_datetime(self, dt: datetime.datetime):
        """Track the model start used by other download-backed ambient sources."""
        super().set_model_start_datetime(dt)

    def _set_iswa_datetime(self, dt: datetime.datetime):
        """Set the ISWA request datetime without double-emitting change handling."""
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        self.iswa_datetime_edit.blockSignals(True)
        self.iswa_datetime_edit.setDateTime(QDateTime(dt))
        self.iswa_datetime_edit.blockSignals(False)
        self._on_iswa_datetime_changed()

    def _sync_iswa_datetime_state(self):
        """Apply the selected ISWA map time to the model start when enabled."""
        synced = self.use_model_start_for_iswa_toggle.isChecked()
        self.iswa_datetime_edit.setEnabled(True)
        if synced:
            self.start_time_selected.emit(
                self._iswa_requested_datetime() - datetime.timedelta(days=5)
            )

    def _on_use_model_start_for_iswa_toggled(self, _enabled=None):
        """Update ISWA map time mode when the sync checkbox changes."""
        self._sync_iswa_datetime_state()

    def _on_iswa_datetime_changed(self, _datetime=None):
        """Clear a downloaded map when its requested ISWA date changes."""
        self.selected_file = ""
        self.file_edit.clear()
        self.last_parsed_time = None
        self.detected_time_label.setText("No ISWA map downloaded for this date/time.")
        if self.use_model_start_for_iswa_toggle.isChecked():
            self.start_time_selected.emit(
                self._iswa_requested_datetime() - datetime.timedelta(days=5)
            )

    def _iswa_requested_datetime(self):
        """Return the selected ISWA request datetime."""
        return self.iswa_datetime_edit.dateTime().toPyDateTime().replace(tzinfo=None)

    def get_state(self):
        """Return WSA settings, including the independently selected ISWA map date."""
        state = super().get_state()
        if self.use_iswa_download:
            state["iswa_map_datetime"] = self._iswa_requested_datetime().isoformat()
        return state


class CorTomAmbientTab(FileAmbientTab):
    """CorTom boundary file selection tab."""

    def __init__(self):
        super().__init__(
            "CorTom boundary file",
            "CorTom DAT (*.dat)",
            parse_cortom_start_time,
            "Select a CorTom DAT file. The run start time is inferred from the filename.",
            include_decelerate_option=True,
            include_use_map_time_option=True,
            default_pattern="**/*.dat",
            profile_loader=sin.get_CorTom_long_profile,
            speed_map_loader=sin.get_CorTom_vr_map,
            source_radius_rs=8.0,
        )


class AmbientSolarWindTab(QWidget):
    """Ambient solar wind configuration with source-specific subtabs."""

    status_message = pyqtSignal(str)
    error_message = pyqtSignal(str)
    start_time_selected = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        source_specs = [
            ("User specified", "user_specified"),
            ("MAS", "mas"),
            ("WSA", "wsa"),
            ("WSA (ISWA)", "wsa_iswa"),
            ("OMNI-backmapped", "insitu_backmapped"),
            ("OMNI outwards", "omni"),
            ("CorTom", "cortom"),
        ]

        self.source_tabs = QTabWidget()
        self.user_tab = UserSpecifiedAmbientTab()
        self.mas_tab = MasAmbientTab()
        self.wsa_tab = WsaAmbientTab()
        self.wsa_iswa_tab = WsaAmbientTab(use_iswa_download=True)
        self.insitu_tab = InSituAmbientTab()
        self.omni_tab = OmniAmbientTab()
        self.cortom_tab = CorTomAmbientTab()

        self.mas_tab.status_message.connect(self.status_message.emit)
        self.mas_tab.error_message.connect(self.error_message.emit)
        self.mas_tab.start_time_selected.connect(self.start_time_selected.emit)
        self.wsa_tab.status_message.connect(self.status_message.emit)
        self.wsa_tab.error_message.connect(self.error_message.emit)
        self.wsa_tab.start_time_selected.connect(self.start_time_selected.emit)
        self.wsa_iswa_tab.status_message.connect(self.status_message.emit)
        self.wsa_iswa_tab.error_message.connect(self.error_message.emit)
        self.cortom_tab.status_message.connect(self.status_message.emit)
        self.cortom_tab.error_message.connect(self.error_message.emit)
        self.cortom_tab.start_time_selected.connect(self.start_time_selected.emit)

        self.source_tabs.addTab(self.user_tab, "User specified")
        self.source_tabs.addTab(self.mas_tab, "MAS")
        self.source_tabs.addTab(self.wsa_tab, "WSA")
        self.source_tabs.addTab(self.wsa_iswa_tab, "WSA (ISWA)")
        self.source_tabs.addTab(self.insitu_tab, "OMNI-backmapped")
        self.source_tabs.addTab(self.omni_tab, "OMNI outwards")
        self.source_tabs.addTab(self.cortom_tab, "CorTom")

        self._source_tab_indices = {
            "user_specified": 0,
            "mas": 1,
            "wsa": 2,
            "wsa_iswa": 3,
            "insitu_backmapped": 4,
            "omni": 5,
            "cortom": 6,
        }

        self.source_tabs.setCurrentIndex(self._source_tab_indices["wsa_iswa"])
        self.source_tabs.currentChanged.connect(self._on_source_tab_changed)

        layout.addWidget(self.source_tabs)
        self.setLayout(layout)

    def _on_source_tab_changed(self, _index: int):
        """Use the visible source tab as the ambient source for the next run."""
        self.emit_active_map_time_if_enabled()

    def _update_source_tab_highlight(self):
        """Retained for compatibility; the selected tab now identifies the source."""

    def _selected_source(self) -> str:
        """Return the ambient source represented by the visible source tab."""
        current_index = self.source_tabs.currentIndex()
        for source_key, index in self._source_tab_indices.items():
            if index == current_index:
                return source_key
        return "wsa_iswa"

    def set_selected_source(self, source_key: str):
        """Select the source tab identified by a saved configuration."""
        index = self._source_tab_indices.get(str(source_key), -1)
        if index >= 0:
            self.source_tabs.setCurrentIndex(index)

    def emit_active_map_time_if_enabled(self):
        """Apply selected-source map time to model start when that source enables it."""
        source = self._selected_source()
        if source == "mas":
            self.mas_tab.emit_map_time_if_enabled()
        elif source == "wsa":
            self.wsa_tab.emit_map_time_if_enabled()
        elif source == "cortom":
            self.cortom_tab.emit_map_time_if_enabled()

    def get_state(self):
        """Return the selected run source and its parameters."""
        state = {"source": self._selected_source()}
        if state["source"] == "user_specified":
            state.update(self.user_tab.get_state())
        elif state["source"] == "mas":
            state.update(self.mas_tab.get_state())
        elif state["source"] == "wsa":
            state.update(self.wsa_tab.get_state())
        elif state["source"] == "wsa_iswa":
            state.update(self.wsa_iswa_tab.get_state())
        elif state["source"] == "insitu_backmapped":
            state.update(self.insitu_tab.get_state())
        elif state["source"] == "omni":
            state.update(self.omni_tab.get_state())
        elif state["source"] == "cortom":
            state.update(self.cortom_tab.get_state())
        return state

    def set_model_inner_boundary(self, rmin_rs: float):
        """Propagate current model inner boundary to source tabs for comparison plots."""
        self.mas_tab.set_model_inner_boundary(rmin_rs)
        self.wsa_tab.set_model_inner_boundary(rmin_rs)
        self.wsa_iswa_tab.set_model_inner_boundary(rmin_rs)
        self.cortom_tab.set_model_inner_boundary(rmin_rs)

    def set_model_latitude(self, latitude_deg: float):
        """Propagate model latitude to source tabs that extract ambient profiles."""
        self.mas_tab.set_model_latitude(latitude_deg)
        self.wsa_tab.set_model_latitude(latitude_deg)
        self.wsa_iswa_tab.set_model_latitude(latitude_deg)
        self.cortom_tab.set_model_latitude(latitude_deg)

    def set_model_solver(self, solver_name: str):
        """Propagate solver selection to source tabs with solver-specific loading."""
        self.mas_tab.set_model_solver(solver_name)
        self.wsa_tab.set_model_solver(solver_name)
        self.wsa_iswa_tab.set_model_solver(solver_name)
        self.cortom_tab.set_model_solver(solver_name)

    def set_model_start_datetime(self, dt: datetime.datetime):
        """Propagate model start time to sources that depend on it."""
        self.wsa_tab.set_model_start_datetime(dt)
        self.wsa_iswa_tab.set_model_start_datetime(dt)
        self.insitu_tab.set_model_start_datetime(dt)

    def set_include_bpol(self, include_bpol: bool):
        """Propagate bpol plotting option to relevant source tabs."""
        self.mas_tab.set_include_bpol(include_bpol)
        self.wsa_tab.set_include_bpol(include_bpol)
        self.wsa_iswa_tab.set_include_bpol(include_bpol)


class VisualisationTab(QWidget):
    """Tab containing standard post-run plotting controls."""

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)

        self.plot_panels = QTabWidget()
        self.plot_panels.setDocumentMode(True)

        self.map_box = QGroupBox("2D Map (sa.plot)")
        map_form = QFormLayout()
        self.map_time_spin = QDoubleSpinBox()
        self.map_time_spin.setRange(0.0, 1000.0)
        self.map_time_spin.setSingleStep(0.1)
        self.map_time_spin.setValue(1.5)
        self.map_time_spin.setSuffix(" day")

        self.map_minimalplot_toggle = QCheckBox("Minimal plot")
        self.map_plot_hcs_toggle = QCheckBox("Plot HCS")
        self.map_plot_hcs_toggle.setChecked(True)
        self.map_annotate_toggle = QCheckBox("Annotate plot")
        self.map_annotate_toggle.setChecked(True)
        self.map_trace_earth_toggle = QCheckBox("Trace Earth connection (slow)")

        self.map_limit_rmax_toggle = QCheckBox("Limit outer radius")
        self.map_limit_rmax_toggle.toggled.connect(self._on_map_limit_rmax_toggled)
        self.map_rmax_spin = QDoubleSpinBox()
        self.map_rmax_spin.setRange(1.0, 5000.0)
        self.map_rmax_spin.setSingleStep(5.0)
        self.map_rmax_spin.setValue(240.0)
        self.map_rmax_spin.setSuffix(" Rs")
        self.map_rmax_spin.setEnabled(False)

        rmax_row = QWidget()
        rmax_layout = QHBoxLayout()
        rmax_layout.setContentsMargins(0, 0, 0, 0)
        rmax_layout.addWidget(self.map_limit_rmax_toggle)
        rmax_layout.addWidget(self.map_rmax_spin)
        rmax_layout.addStretch(1)
        rmax_row.setLayout(rmax_layout)

        self.plot_map_button = QPushButton("Plot 2D Map")
        self.plot_map_button.setProperty("role", "primary")
        map_form.addRow("Time", self.map_time_spin)
        map_form.addRow("", self.map_minimalplot_toggle)
        map_form.addRow("", self.map_plot_hcs_toggle)
        map_form.addRow("", self.map_annotate_toggle)
        map_form.addRow("", self.map_trace_earth_toggle)
        map_form.addRow("", rmax_row)
        map_form.addRow(self.plot_map_button)
        self.map_box.setLayout(map_form)

        radial_box = QGroupBox("Radial Profile (sa.plot_radial)")
        radial_form = QFormLayout()
        self.radial_time_spin = QDoubleSpinBox()
        self.radial_time_spin.setRange(0.0, 1000.0)
        self.radial_time_spin.setSingleStep(0.1)
        self.radial_time_spin.setValue(1.5)
        self.radial_time_spin.setSuffix(" day")

        self.radial_lon_spin = QDoubleSpinBox()
        self.radial_lon_spin.setRange(-360.0, 360.0)
        self.radial_lon_spin.setSingleStep(1.0)
        self.radial_lon_spin.setValue(0.0)
        self.radial_lon_spin.setSuffix(" deg")

        self.plot_radial_button = QPushButton("Plot Radial Profile")
        self.plot_radial_button.setProperty("role", "primary")
        radial_form.addRow("Time", self.radial_time_spin)
        radial_form.addRow("Longitude", self.radial_lon_spin)
        radial_form.addRow(self.plot_radial_button)
        radial_box.setLayout(radial_form)

        ts_box = QGroupBox("Time Series (sa.plot_timeseries)")
        ts_form = QFormLayout()
        self.ts_location_combo = QComboBox()
        self.ts_location_combo.addItem("Earth", "Earth")
        self.ts_location_combo.addItem("Mercury", "Mercury")
        self.ts_location_combo.addItem("Venus", "Venus")
        self.ts_location_combo.addItem("Mars", "Mars")
        self.ts_location_combo.addItem("Jupiter", "Jupiter")
        self.ts_location_combo.addItem("Saturn", "Saturn")
        self.ts_location_combo.addItem("ACE", "ACE")
        self.ts_location_combo.addItem("Parker Solar Probe (PSP)", "PSP")
        self.ts_location_combo.addItem("Solar Orbiter (SolO)", "SOLO")
        self.ts_location_combo.addItem("STEREO-A", "STA")
        self.ts_location_combo.addItem("STEREO-B", "STB")
        self.ts_location_combo.addItem("Ulysses", "ULYSSES")
        self.ts_location_combo.addItem("Custom radius / longitude", "custom")
        self.ts_location_combo.currentIndexChanged.connect(
            self._on_timeseries_location_changed
        )

        self.ts_radius_spin = QDoubleSpinBox()
        self.ts_radius_spin.setRange(0.1, 10.0)
        self.ts_radius_spin.setSingleStep(0.1)
        self.ts_radius_spin.setValue(1.0)
        self.ts_radius_spin.setSuffix(" AU")

        self.ts_lon_spin = QDoubleSpinBox()
        self.ts_lon_spin.setRange(-360.0, 360.0)
        self.ts_lon_spin.setSingleStep(1.0)
        self.ts_lon_spin.setValue(0.0)
        self.ts_lon_spin.setSuffix(" deg")

        self.ts_custom_coordinates = QWidget()
        ts_custom_layout = QHBoxLayout()
        ts_custom_layout.setContentsMargins(0, 0, 0, 0)
        ts_custom_layout.addWidget(QLabel("Radius"))
        ts_custom_layout.addWidget(self.ts_radius_spin)
        ts_custom_layout.addWidget(QLabel("Fixed model longitude"))
        ts_custom_layout.addWidget(self.ts_lon_spin)
        self.ts_custom_coordinates.setLayout(ts_custom_layout)

        ts_note = QLabel(
            "Standard locations follow each observer's ephemeris throughout the run. "
            "Custom longitude is fixed on the SURF model grid: HEEQ/model longitude "
            "at run start for sidereal runs, or corotating model longitude for synodic runs."
        )
        ts_note.setWordWrap(True)

        self.ts_csv_output_edit = QLineEdit()
        self.ts_csv_output_edit.setReadOnly(True)
        self.ts_csv_output_edit.setPlaceholderText("Use default SURF figures path (.csv)")
        self.ts_csv_output_button = QPushButton("Select output")
        self.ts_csv_output_button.clicked.connect(self._select_timeseries_csv_output_path)
        self.ts_csv_clear_output_button = QPushButton("Clear")
        self.ts_csv_clear_output_button.clicked.connect(self.ts_csv_output_edit.clear)

        ts_csv_output_row = QWidget()
        ts_csv_output_layout = QHBoxLayout()
        ts_csv_output_layout.setContentsMargins(0, 0, 0, 0)
        ts_csv_output_layout.addWidget(self.ts_csv_output_edit, 1)
        ts_csv_output_layout.addWidget(self.ts_csv_output_button)
        ts_csv_output_layout.addWidget(self.ts_csv_clear_output_button)
        ts_csv_output_row.setLayout(ts_csv_output_layout)

        self.plot_timeseries_button = QPushButton("Plot Time Series")
        self.plot_timeseries_button.setProperty("role", "primary")
        self.export_timeseries_csv_button = QPushButton("Export Time Series CSV")
        ts_form.addRow("Location", self.ts_location_combo)
        ts_form.addRow("", ts_note)
        ts_form.addRow("Custom coordinates", self.ts_custom_coordinates)
        ts_form.addRow("CSV output", ts_csv_output_row)
        ts_form.addRow(self.plot_timeseries_button)
        ts_form.addRow(self.export_timeseries_csv_button)
        ts_box.setLayout(ts_form)

        self.map_panel_index = self.plot_panels.addTab(self.map_box, "2D Map")
        self.plot_panels.addTab(radial_box, "Radial Profile")
        self.plot_panels.addTab(ts_box, "Time Series")
        layout.addWidget(self.plot_panels)
        self.setLayout(layout)
        self._on_timeseries_location_changed()

    def set_1d_mode(self, enabled: bool):
        """Disable 2D map controls when the model is configured for 1D runs."""
        self.map_box.setEnabled(not enabled)
        self.plot_panels.setTabEnabled(self.map_panel_index, not enabled)

    def _on_map_limit_rmax_toggled(self, enabled: bool):
        """Enable outer-radius spin box only when radius limiting is requested."""
        self.map_rmax_spin.setEnabled(enabled)

    def _on_timeseries_location_changed(self, _index=None):
        """Show fixed coordinates only when the custom location is selected."""
        self.ts_custom_coordinates.setVisible(
            self.ts_location_combo.currentData() == "custom"
        )

    def _select_timeseries_csv_output_path(self):
        """Select an optional explicit CSV path for time-series export."""
        start_dir = str(sa.get_figure_dir())
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Select CSV output",
            start_dir,
            "CSV file (*.csv)",
        )
        if filepath:
            if not filepath.lower().endswith(".csv"):
                filepath += ".csv"
            self.ts_csv_output_edit.setText(filepath)


class MoviesTab(QWidget):
    """Tab containing controls for generating post-run SURF animations."""

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.movie_tabs = QTabWidget()
        self.movie_tabs.setDocumentMode(True)

        movie_2d_page = QWidget()
        movie_2d_layout = QVBoxLayout()
        movie_2d_layout.setContentsMargins(0, 0, 0, 0)

        movie_box = QGroupBox("Animation (sa.animate)")
        movie_form = QFormLayout()

        self.movie_tag_edit = QLineEdit("gui")

        self.movie_duration_spin = QDoubleSpinBox()
        self.movie_duration_spin.setRange(1.0, 600.0)
        self.movie_duration_spin.setSingleStep(1.0)
        self.movie_duration_spin.setValue(10.0)
        self.movie_duration_spin.setSuffix(" s")

        self.movie_fps_spin = QSpinBox()
        self.movie_fps_spin.setRange(1, 60)
        self.movie_fps_spin.setValue(5)
        self.movie_fps_spin.setSuffix(" fps")

        self.movie_plot_hcs_toggle = QCheckBox("Plot HCS")
        self.movie_plot_hcs_toggle.setChecked(True)

        self.movie_trace_earth_toggle = QCheckBox("Trace Earth connection (slow)")

        self.movie_limit_rmax_toggle = QCheckBox("Limit outer radius")
        self.movie_limit_rmax_toggle.toggled.connect(self._on_movie_limit_rmax_toggled)
        self.movie_rmax_spin = QDoubleSpinBox()
        self.movie_rmax_spin.setRange(1.0, 5000.0)
        self.movie_rmax_spin.setSingleStep(5.0)
        self.movie_rmax_spin.setValue(240.0)
        self.movie_rmax_spin.setSuffix(" Rs")
        self.movie_rmax_spin.setEnabled(False)

        settings_row = QWidget()
        settings_layout = QHBoxLayout()
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.addWidget(QLabel("Tag"))
        settings_layout.addWidget(self.movie_tag_edit, 1)
        settings_layout.addSpacing(12)
        settings_layout.addWidget(QLabel("Duration"))
        settings_layout.addWidget(self.movie_duration_spin)
        settings_layout.addSpacing(12)
        settings_layout.addWidget(QLabel("Frame rate"))
        settings_layout.addWidget(self.movie_fps_spin)
        settings_row.setLayout(settings_layout)

        toggles_row = QWidget()
        toggles_layout = QHBoxLayout()
        toggles_layout.setContentsMargins(0, 0, 0, 0)
        toggles_layout.addWidget(self.movie_plot_hcs_toggle)
        toggles_layout.addSpacing(12)
        toggles_layout.addWidget(self.movie_trace_earth_toggle)
        toggles_layout.addSpacing(12)
        toggles_layout.addWidget(self.movie_limit_rmax_toggle)
        toggles_layout.addWidget(self.movie_rmax_spin)
        toggles_layout.addStretch(1)
        toggles_row.setLayout(toggles_layout)

        self.movie_output_edit = QLineEdit()
        self.movie_output_edit.setReadOnly(True)
        self.movie_output_edit.setPlaceholderText("Use default SURF figures path (.gif)")
        self.movie_output_button = QPushButton("Select output")
        self.movie_output_button.clicked.connect(self._select_output_path_2d)
        self.movie_clear_output_button = QPushButton("Clear")
        self.movie_clear_output_button.clicked.connect(self.movie_output_edit.clear)

        output_row = QWidget()
        output_layout = QHBoxLayout()
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.movie_output_edit, 1)
        output_layout.addWidget(self.movie_output_button)
        output_layout.addWidget(self.movie_clear_output_button)
        output_row.setLayout(output_layout)

        self.movie_play_on_complete_toggle = QCheckBox("Play movie when complete")
        self.movie_play_on_complete_toggle.setChecked(True)

        self.generate_movie_button = QPushButton("Create 2D Movie")
        self.generate_movie_button.setProperty("role", "primary")

        movie_form.addRow("", settings_row)
        movie_form.addRow("", toggles_row)
        movie_form.addRow("Output", output_row)
        movie_form.addRow("", self.movie_play_on_complete_toggle)
        movie_form.addRow(self.generate_movie_button)
        movie_box.setLayout(movie_form)
        movie_2d_layout.addWidget(movie_box)
        movie_2d_layout.addStretch(1)
        movie_2d_page.setLayout(movie_2d_layout)

        movie_ts_page = QWidget()
        movie_ts_layout = QVBoxLayout()
        movie_ts_layout.setContentsMargins(0, 0, 0, 0)

        movie_ts_box = QGroupBox("Animation With Time Series (sa.animate_with_ts)")
        movie_ts_form = QFormLayout()

        self.movie_ts_tag_edit = QLineEdit("gui")

        self.movie_ts_duration_spin = QDoubleSpinBox()
        self.movie_ts_duration_spin.setRange(1.0, 600.0)
        self.movie_ts_duration_spin.setSingleStep(1.0)
        self.movie_ts_duration_spin.setValue(10.0)
        self.movie_ts_duration_spin.setSuffix(" s")

        self.movie_ts_fps_spin = QSpinBox()
        self.movie_ts_fps_spin.setRange(1, 60)
        self.movie_ts_fps_spin.setValue(5)
        self.movie_ts_fps_spin.setSuffix(" fps")

        self.movie_ts_plot_hcs_toggle = QCheckBox("Plot HCS")
        self.movie_ts_plot_hcs_toggle.setChecked(True)

        self.movie_ts_limit_rmax_toggle = QCheckBox("Limit outer radius")
        self.movie_ts_limit_rmax_toggle.toggled.connect(self._on_movie_ts_limit_rmax_toggled)
        self.movie_ts_rmax_spin = QDoubleSpinBox()
        self.movie_ts_rmax_spin.setRange(1.0, 5000.0)
        self.movie_ts_rmax_spin.setSingleStep(5.0)
        self.movie_ts_rmax_spin.setValue(240.0)
        self.movie_ts_rmax_spin.setSuffix(" Rs")
        self.movie_ts_rmax_spin.setEnabled(False)

        self.movie_ts_field_combo = QComboBox()
        self.movie_ts_field_combo.addItems(["P_DYN", "V", "n", "T"])
        self.movie_ts_field_combo.setCurrentText("V")

        ts_settings_row = QWidget()
        ts_settings_layout = QHBoxLayout()
        ts_settings_layout.setContentsMargins(0, 0, 0, 0)
        ts_settings_layout.addWidget(QLabel("Tag"))
        ts_settings_layout.addWidget(self.movie_ts_tag_edit, 1)
        ts_settings_layout.addSpacing(12)
        ts_settings_layout.addWidget(QLabel("Duration"))
        ts_settings_layout.addWidget(self.movie_ts_duration_spin)
        ts_settings_layout.addSpacing(12)
        ts_settings_layout.addWidget(QLabel("Frame rate"))
        ts_settings_layout.addWidget(self.movie_ts_fps_spin)
        ts_settings_row.setLayout(ts_settings_layout)

        ts_toggles_row = QWidget()
        ts_toggles_layout = QHBoxLayout()
        ts_toggles_layout.setContentsMargins(0, 0, 0, 0)
        ts_toggles_layout.addWidget(self.movie_ts_plot_hcs_toggle)
        ts_toggles_layout.addSpacing(12)
        ts_toggles_layout.addWidget(self.movie_ts_limit_rmax_toggle)
        ts_toggles_layout.addWidget(self.movie_ts_rmax_spin)
        ts_toggles_layout.addStretch(1)
        ts_toggles_row.setLayout(ts_toggles_layout)

        ts_field_row = QWidget()
        ts_field_layout = QHBoxLayout()
        ts_field_layout.setContentsMargins(0, 0, 0, 0)
        ts_field_layout.addWidget(QLabel("Field"))
        ts_field_layout.addWidget(self.movie_ts_field_combo)
        ts_field_layout.addStretch(1)
        ts_field_row.setLayout(ts_field_layout)

        self.movie_ts_output_edit = QLineEdit()
        self.movie_ts_output_edit.setReadOnly(True)
        self.movie_ts_output_edit.setPlaceholderText("Use default SURF figures path (.gif)")
        self.movie_ts_output_button = QPushButton("Select output")
        self.movie_ts_output_button.clicked.connect(self._select_output_path_ts)
        self.movie_ts_clear_output_button = QPushButton("Clear")
        self.movie_ts_clear_output_button.clicked.connect(self.movie_ts_output_edit.clear)

        ts_output_row = QWidget()
        ts_output_layout = QHBoxLayout()
        ts_output_layout.setContentsMargins(0, 0, 0, 0)
        ts_output_layout.addWidget(self.movie_ts_output_edit, 1)
        ts_output_layout.addWidget(self.movie_ts_output_button)
        ts_output_layout.addWidget(self.movie_ts_clear_output_button)
        ts_output_row.setLayout(ts_output_layout)

        self.generate_movie_with_ts_button = QPushButton("Create Movie With Time Series")
        self.generate_movie_with_ts_button.setProperty("role", "primary")

        movie_ts_form.addRow("", ts_settings_row)
        movie_ts_form.addRow("", ts_toggles_row)
        movie_ts_form.addRow("", ts_field_row)
        movie_ts_form.addRow("Output", ts_output_row)
        movie_ts_form.addRow(self.generate_movie_with_ts_button)
        movie_ts_box.setLayout(movie_ts_form)
        movie_ts_layout.addWidget(movie_ts_box)
        movie_ts_layout.addStretch(1)
        movie_ts_page.setLayout(movie_ts_layout)

        self.movie_tabs.addTab(movie_2d_page, "2D movie")
        self.movie_tabs.addTab(movie_ts_page, "movie with time series")
        layout.addWidget(self.movie_tabs)
        self.setLayout(layout)

    def _on_movie_limit_rmax_toggled(self, enabled: bool):
        """Enable outer-radius spin box only when radius limiting is requested."""
        self.movie_rmax_spin.setEnabled(enabled)

    def _on_movie_ts_limit_rmax_toggled(self, enabled: bool):
        """Enable outer-radius spin box for time-series movie requests."""
        self.movie_ts_rmax_spin.setEnabled(enabled)

    def set_solver(self, solver_name: str):
        """Restrict time-series field choices based on active solver."""
        solver_key = str(solver_name).strip().lower()
        is_huxt = solver_key == "huxt"

        field_model = self.movie_ts_field_combo.model()
        for index in range(self.movie_ts_field_combo.count()):
            field_name = self.movie_ts_field_combo.itemText(index)
            allow_field = (not is_huxt) or field_name == "V"
            item = field_model.item(index)
            if item is not None:
                item.setEnabled(allow_field)

        if is_huxt and self.movie_ts_field_combo.currentText() != "V":
            self.movie_ts_field_combo.setCurrentText("V")

    def _select_output_path_for(self, target_edit: QLineEdit):
        """Select an optional explicit GIF/MP4 path for movie output."""
        start_dir = str(sa.get_figure_dir())
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Select movie output",
            start_dir,
            "GIF animation (*.gif);;MP4 video (*.mp4)",
        )
        if filepath:
            lower_path = filepath.lower()
            if not lower_path.endswith((".gif", ".mp4")):
                if "MP4" in selected_filter:
                    filepath += ".mp4"
                else:
                    filepath += ".gif"
            target_edit.setText(filepath)

    def _select_output_path_2d(self):
        """Select explicit output for 2D movie animation."""
        self._select_output_path_for(self.movie_output_edit)

    def _select_output_path_ts(self):
        """Select explicit output for time-series movie animation."""
        self._select_output_path_for(self.movie_ts_output_edit)


class CmeTab(QWidget):
    """Tab for creating and managing ConeCME entries for model runs."""

    def __init__(self):
        super().__init__()
        self._cmes = []
        self.current_solver = "huxt"
        self.model_start_datetime = datetime.datetime.utcnow()
        self._last_model_start_datetime = None
        self.model_inner_boundary_rs = 21.5
        self.model_run_duration_days = 5.0

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        load_box = QGroupBox("Load Cone CMEs")
        load_form = QFormLayout()

        load_row = QWidget()
        load_row_layout = QHBoxLayout()
        load_row_layout.setContentsMargins(0, 0, 0, 0)

        self.cone_file_edit = QLineEdit()
        self.cone_file_edit.setReadOnly(True)
        self.load_cone_button = QPushButton("Select cone file")
        self.load_cone_button.clicked.connect(self.load_cone_file)

        load_row_layout.addWidget(self.cone_file_edit)
        load_row_layout.addWidget(self.load_cone_button)
        load_row.setLayout(load_row_layout)

        self.load_cone_status_label = QLabel("No cone file loaded.")
        self.load_cone_status_label.setWordWrap(True)

        self.load_donki_button = QPushButton("Get cone CME list from DONKI")
        self.load_donki_button.setProperty("role", "primary")
        self.load_donki_button.clicked.connect(self.load_donki_cmes)
        self.load_donki_status_label = QLabel(
            "Uses the current model start and run duration."
        )
        self.load_donki_status_label.setWordWrap(True)

        load_form.addRow("File", load_row)
        load_form.addRow("Status", self.load_cone_status_label)
        load_form.addRow("DONKI", self.load_donki_button)
        load_form.addRow("Status", self.load_donki_status_label)
        load_box.setLayout(load_form)

        manual_box = QGroupBox("Manually add cone CME")
        manual_layout = QHBoxLayout()
        self.open_manual_cme_button = QPushButton("Open manual CME editor")
        self.open_manual_cme_button.setProperty("role", "primary")
        self.open_manual_cme_button.clicked.connect(self._open_manual_cme_dialog)
        manual_layout.addWidget(self.open_manual_cme_button)
        manual_layout.addStretch(1)
        manual_box.setLayout(manual_layout)

        self.manual_cme_dialog = QDialog(self)
        self.manual_cme_dialog.setWindowTitle("Manually add cone CME")
        self.manual_cme_dialog.setModal(True)
        self.manual_cme_dialog.resize(720, 520)
        add_form = QFormLayout()
        add_form.setContentsMargins(20, 18, 20, 18)
        add_form.setHorizontalSpacing(14)
        add_form.setVerticalSpacing(10)

        self.cme_lon_spin = QDoubleSpinBox()
        self.cme_lon_spin.setRange(-360.0, 360.0)
        self.cme_lon_spin.setSingleStep(1.0)
        self.cme_lon_spin.setValue(0.0)
        self.cme_lon_spin.setSuffix(" deg")

        self.cme_lat_spin = QDoubleSpinBox()
        self.cme_lat_spin.setRange(-90.0, 90.0)
        self.cme_lat_spin.setSingleStep(1.0)
        self.cme_lat_spin.setValue(0.0)
        self.cme_lat_spin.setSuffix(" deg")

        self.cme_speed_spin = QDoubleSpinBox()
        self.cme_speed_spin.setRange(100.0, 4000.0)
        self.cme_speed_spin.setSingleStep(10.0)
        self.cme_speed_spin.setValue(800.0)
        self.cme_speed_spin.setSuffix(" km/s")

        self.cme_width_spin = QDoubleSpinBox()
        self.cme_width_spin.setRange(1.0, 180.0)
        self.cme_width_spin.setSingleStep(1.0)
        self.cme_width_spin.setValue(60.0)
        self.cme_width_spin.setSuffix(" deg")

        self.cme_launch_spin = QDoubleSpinBox()
        self.cme_launch_spin.setRange(-100.0, 100.0)
        self.cme_launch_spin.setSingleStep(0.1)
        self.cme_launch_spin.setValue(0.5)
        self.cme_launch_spin.setSuffix(" day")

        self.cme_launch_datetime = QDateTimeEdit()
        self.cme_launch_datetime.setCalendarPopup(True)
        self.cme_launch_datetime.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.cme_launch_datetime.setTimeSpec(Qt.TimeSpec.UTC)
        self.cme_launch_datetime.setDateTime(QDateTime.currentDateTimeUtc())

        self.cme_launch_spin.valueChanged.connect(self._sync_launch_datetime_from_day)
        self.cme_launch_datetime.dateTimeChanged.connect(self._sync_launch_day_from_datetime)

        self.cme_thickness_spin = QDoubleSpinBox()
        self.cme_thickness_spin.setRange(0.0, 50.0)
        self.cme_thickness_spin.setSingleStep(0.5)
        self.cme_thickness_spin.setValue(5.0)
        self.cme_thickness_spin.setSuffix(" Rs")

        self.cme_initial_height_spin = QDoubleSpinBox()
        self.cme_initial_height_spin.setRange(1.0, 100.0)
        self.cme_initial_height_spin.setSingleStep(0.5)
        self.cme_initial_height_spin.setValue(21.5)
        self.cme_initial_height_spin.setSuffix(" Rs")
        self.cme_initial_height_spin.valueChanged.connect(self._update_initial_height_style)

        self.cme_expansion_toggle = QCheckBox("Expansion")
        self.cme_fixed_duration_toggle = QCheckBox("Fixed duration")
        self.cme_fixed_duration_toggle.setChecked(True)

        self.cme_fixed_duration_hours_spin = QDoubleSpinBox()
        self.cme_fixed_duration_hours_spin.setRange(0.1, 240.0)
        self.cme_fixed_duration_hours_spin.setSingleStep(0.5)
        self.cme_fixed_duration_hours_spin.setValue(12.0)
        self.cme_fixed_duration_hours_spin.setSuffix(" hr")

        self.profile_type_combo = QComboBox()
        self.profile_type_combo.addItems(["square", "sinusoidal"])

        self.cme_plasma_mode_combo = QComboBox()
        self.cme_plasma_mode_combo.addItems(["Fraction of ambient", "Absolute values"])
        self.cme_plasma_mode_combo.currentTextChanged.connect(self._on_cme_plasma_mode_changed)

        self.density_fraction_spin = QDoubleSpinBox()
        self.density_fraction_spin.setRange(0.01, 100.0)
        self.density_fraction_spin.setSingleStep(0.1)
        self.density_fraction_spin.setValue(1.0)

        self.temperature_fraction_spin = QDoubleSpinBox()
        self.temperature_fraction_spin.setRange(0.01, 100.0)
        self.temperature_fraction_spin.setSingleStep(0.1)
        self.temperature_fraction_spin.setValue(1.0)

        self.cme_density_spin = QDoubleSpinBox()
        self.cme_density_spin.setRange(0.0, 1.0e6)
        self.cme_density_spin.setDecimals(3)
        self.cme_density_spin.setSingleStep(1.0)
        self.cme_density_spin.setValue(100.0)
        self.cme_density_spin.setSuffix(" p+/cm^3")

        self.cme_temperature_spin = QDoubleSpinBox()
        self.cme_temperature_spin.setRange(1.0, 1.0e8)
        self.cme_temperature_spin.setSingleStep(1000.0)
        self.cme_temperature_spin.setValue(1.0e5)
        self.cme_temperature_spin.setSuffix(" K")

        self.add_cme_button = QPushButton("Add CME")
        self.add_cme_button.setProperty("role", "primary")
        self.add_cme_button.clicked.connect(self.add_cme)
        self.cancel_cme_button = QPushButton("Cancel")
        self.cancel_cme_button.clicked.connect(self.manual_cme_dialog.reject)

        lon_lat_row = QWidget()
        lon_lat_layout = QHBoxLayout()
        lon_lat_layout.setContentsMargins(0, 0, 0, 0)
        lon_lat_layout.addWidget(QLabel("Lon"))
        lon_lat_layout.addWidget(self.cme_lon_spin)
        lon_lat_layout.addWidget(QLabel("Lat"))
        lon_lat_layout.addWidget(self.cme_lat_spin)
        lon_lat_row.setLayout(lon_lat_layout)

        speed_width_row = QWidget()
        speed_width_layout = QHBoxLayout()
        speed_width_layout.setContentsMargins(0, 0, 0, 0)
        speed_width_layout.addWidget(QLabel("Speed"))
        speed_width_layout.addWidget(self.cme_speed_spin)
        speed_width_layout.addWidget(QLabel("Width"))
        speed_width_layout.addWidget(self.cme_width_spin)
        speed_width_row.setLayout(speed_width_layout)

        launch_row = QWidget()
        launch_layout = QHBoxLayout()
        launch_layout.setContentsMargins(0, 0, 0, 0)
        launch_layout.addWidget(self.cme_launch_spin, 1)
        launch_layout.addWidget(QLabel("or"))
        launch_layout.addWidget(self.cme_launch_datetime, 2)
        launch_row.setLayout(launch_layout)

        size_row = QWidget()
        size_layout = QHBoxLayout()
        size_layout.setContentsMargins(0, 0, 0, 0)
        size_layout.addWidget(QLabel("Thickness"))
        size_layout.addWidget(self.cme_thickness_spin)
        size_layout.addWidget(QLabel("Initial height"))
        size_layout.addWidget(self.cme_initial_height_spin)
        size_row.setLayout(size_layout)

        duration_row = QWidget()
        duration_layout = QHBoxLayout()
        duration_layout.setContentsMargins(0, 0, 0, 0)
        duration_layout.addWidget(self.cme_expansion_toggle)
        duration_layout.addSpacing(12)
        duration_layout.addWidget(self.cme_fixed_duration_toggle)
        duration_layout.addSpacing(12)
        duration_layout.addWidget(QLabel("Duration"))
        duration_layout.addWidget(self.cme_fixed_duration_hours_spin)
        duration_row.setLayout(duration_layout)

        self.fraction_row = QWidget()
        fraction_layout = QHBoxLayout()
        fraction_layout.setContentsMargins(0, 0, 0, 0)
        fraction_layout.addWidget(QLabel("Density"))
        fraction_layout.addWidget(self.density_fraction_spin)
        fraction_layout.addWidget(QLabel("Temperature"))
        fraction_layout.addWidget(self.temperature_fraction_spin)
        self.fraction_row.setLayout(fraction_layout)

        self.absolute_row = QWidget()
        absolute_layout = QHBoxLayout()
        absolute_layout.setContentsMargins(0, 0, 0, 0)
        absolute_layout.addWidget(QLabel("Density"))
        absolute_layout.addWidget(self.cme_density_spin)
        absolute_layout.addWidget(QLabel("Temperature"))
        absolute_layout.addWidget(self.cme_temperature_spin)
        self.absolute_row.setLayout(absolute_layout)

        add_form.addRow("HEEQ lon / lat", lon_lat_row)
        add_form.addRow("Speed / width", speed_width_row)
        add_form.addRow("Launch (day / datetime UTC)", launch_row)
        add_form.addRow("Thickness / initial height", size_row)
        add_form.addRow("CME duration", duration_row)
        add_form.addRow("Profile type", self.profile_type_combo)
        add_form.addRow("Active plasma mode", self.cme_plasma_mode_combo)
        add_form.addRow("Fraction values", self.fraction_row)
        add_form.addRow("Absolute values", self.absolute_row)

        dialog_button_row = QWidget()
        dialog_button_layout = QHBoxLayout()
        dialog_button_layout.setContentsMargins(0, 8, 0, 0)
        dialog_button_layout.addStretch(1)
        dialog_button_layout.addWidget(self.cancel_cme_button)
        dialog_button_layout.addWidget(self.add_cme_button)
        dialog_button_row.setLayout(dialog_button_layout)
        add_form.addRow(dialog_button_row)
        self.manual_cme_dialog.setLayout(add_form)

        list_box = QGroupBox("CME List")
        list_layout = QVBoxLayout()
        self.cme_list_widget = QListWidget()
        self.cme_list_widget.setAlternatingRowColors(True)
        self.cme_list_widget.setMinimumHeight(150)

        button_row = QHBoxLayout()
        self.remove_selected_button = QPushButton("Remove Selected")
        self.remove_selected_button.setProperty("role", "danger")
        self.remove_selected_button.clicked.connect(self.remove_selected_cme)
        self.clear_all_button = QPushButton("Clear All")
        self.clear_all_button.setProperty("role", "danger")
        self.clear_all_button.clicked.connect(self.clear_cmes)
        button_row.addWidget(self.remove_selected_button)
        button_row.addWidget(self.clear_all_button)

        list_layout.addWidget(self.cme_list_widget)
        list_layout.addLayout(button_row)
        list_box.setLayout(list_layout)

        layout.addWidget(load_box)
        layout.addWidget(manual_box)
        layout.addWidget(list_box)
        self.setLayout(layout)
        self._sync_launch_datetime_from_day()
        self._on_cme_plasma_mode_changed(self.cme_plasma_mode_combo.currentText())
        self._update_initial_height_style()
        self.set_solver(self.current_solver)

    def _open_manual_cme_dialog(self):
        """Open the manual cone-CME editor as a temporary modal dialog."""
        self._sync_launch_datetime_from_day()
        self.manual_cme_dialog.exec()

    def set_solver(self, solver_name: str):
        """Enable CME plasma controls only for compressible solvers."""
        solver_key = str(solver_name).strip().lower()
        self.current_solver = solver_key
        plasma_allowed = solver_key in ("hydro", "hydro-pcm")

        self.cme_plasma_mode_combo.setEnabled(plasma_allowed)
        self.fraction_row.setEnabled(plasma_allowed)
        self.absolute_row.setEnabled(plasma_allowed)

        if not plasma_allowed:
            self.cme_plasma_mode_combo.blockSignals(True)
            self.cme_plasma_mode_combo.setCurrentText("Fraction of ambient")
            self.cme_plasma_mode_combo.blockSignals(False)

        self._on_cme_plasma_mode_changed(self.cme_plasma_mode_combo.currentText())

    def load_cone_file(self):
        """Load CMEs from a cone2bc .in file and append them to the CME list."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select cone file",
            str(EXAMPLE_INPUTS_DIR),
            "Cone input (*.in)",
        )

        if not filepath:
            return

        self._apply_cone_file(filepath)

    def load_donki_cmes(self):
        """Replace previously loaded DONKI CMEs with analyses from the run interval."""
        run_start = self.model_start_datetime
        run_end = run_start + datetime.timedelta(days=self.model_run_duration_days)

        self.load_donki_button.setEnabled(False)
        self.load_donki_button.setText("Downloading DONKI CMEs...")
        self.load_donki_status_label.setText(
            f"Querying {run_start:%Y-%m-%d %H:%M} to {run_end:%Y-%m-%d %H:%M} UTC."
        )
        QApplication.processEvents()

        try:
            analyses = []
            query_start = run_start.date()
            final_date = run_end.date()
            while query_start <= final_date:
                query_end = min(query_start + datetime.timedelta(days=29), final_date)
                query = urlencode(
                    {
                        "startDate": query_start.isoformat(),
                        "endDate": query_end.isoformat(),
                        "mostAccurateOnly": "true",
                        "completeEntryOnly": "true",
                        "speed": "0",
                        "halfAngle": "0",
                        "catalog": "ALL",
                    }
                )
                with urlopen(f"{DONKI_CME_ANALYSIS_URL}?{query}", timeout=30) as response:
                    analyses.extend(json.load(response))
                query_start = query_end + datetime.timedelta(days=1)

            donki_cmes = []
            seen_analyses = set()
            for analysis in analyses:
                launch_text = analysis.get("time21_5")
                if not launch_text:
                    continue
                launch_time = datetime.datetime.fromisoformat(
                    str(launch_text).replace("Z", "+00:00")
                ).replace(tzinfo=None)
                if not run_start <= launch_time <= run_end:
                    continue

                longitude = analysis.get("longitude")
                latitude = analysis.get("latitude")
                speed = analysis.get("speed")
                half_angle = analysis.get("halfAngle")
                if None in (longitude, latitude, speed, half_angle):
                    continue

                analysis_key = (
                    analysis.get("associatedCMEID"),
                    launch_text,
                    longitude,
                    latitude,
                    speed,
                    half_angle,
                )
                if analysis_key in seen_analyses:
                    continue
                seen_analyses.add(analysis_key)

                delta_days = (launch_time - run_start).total_seconds() / 86400.0
                donki_cmes.append(
                    {
                        "longitude": float(longitude),
                        "latitude": float(latitude),
                        "speed": float(speed),
                        "width": 2.0 * float(half_angle),
                        "t_launch_day": delta_days,
                        "t_launch_datetime": launch_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "thickness_rs": 0.0,
                        "initial_height_rs": 21.5,
                        "cme_expansion": False,
                        "cme_fixed_duration": True,
                        "fixed_duration_hr": 12.0,
                        "profile_type": "square",
                        "plasma_mode": "Fraction of ambient",
                        "density_fraction": 1.0,
                        "temperature_fraction": 1.0,
                        "cme_density_pcc": np.nan,
                        "cme_temperature_k": np.nan,
                        "source": "donki",
                        "donki_id": analysis.get("associatedCMEID", ""),
                    }
                )

            self._cmes = [cme for cme in self._cmes if cme.get("source") != "donki"]
            self._cmes.extend(donki_cmes)
            self._refresh_cme_list()
            self.load_donki_status_label.setToolTip("")
            self.load_donki_status_label.setText(
                f"Loaded {len(donki_cmes)} DONKI cone CME(s) for the run interval."
            )
        except Exception as exc:
            self.load_donki_status_label.setText("Failed to load CMEs from DONKI.")
            self.load_donki_status_label.setToolTip(str(exc))
        finally:
            self.load_donki_button.setEnabled(True)
            self.load_donki_button.setText("Get cone CME list from DONKI")

    def _apply_cone_file(
        self,
        filepath: str,
        replace_existing_loaded: bool = True,
        set_status: bool = True,
    ):
        """Apply a selected cone file path and load/refresh cone-file CMEs."""
        self.cone_file_edit.setText(filepath)

        try:
            cme_params = sin.import_cone2bc_parameters(filepath)
            if not cme_params:
                if set_status:
                    self.load_cone_status_label.setText("No CMEs found in selected cone file.")
                return

            if replace_existing_loaded:
                self._cmes = [cme for cme in self._cmes if cme.get("source") != "cone_file"]

            loaded = 0
            for cme in cme_params.values():
                launch_time = Time(cme["ldates"]).to_datetime()
                if launch_time.tzinfo is not None:
                    launch_time = launch_time.replace(tzinfo=None)

                delta_days = (launch_time - self.model_start_datetime).total_seconds() / 86400.0

                self._cmes.append(
                    {
                        "longitude": float(cme.get("lon", 0.0)),
                        "latitude": float(cme.get("lat", 0.0)),
                        "speed": float(cme.get("vcld", 800.0)),
                        "width": float(2.0 * cme.get("rmajor", 30.0)),
                        "t_launch_day": delta_days,
                        "t_launch_datetime": launch_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "thickness_rs": 0.0,
                        "initial_height_rs": 21.5,
                        "cme_expansion": False,
                        "cme_fixed_duration": True,
                        "fixed_duration_hr": 12.0,
                        "profile_type": "square",
                        "plasma_mode": "Fraction of ambient",
                        "density_fraction": 1.0,
                        "temperature_fraction": 1.0,
                        "cme_density_pcc": np.nan,
                        "cme_temperature_k": np.nan,
                        "source": "cone_file",
                    }
                )
                loaded += 1

            if set_status:
                self.load_cone_status_label.setText(f"Loaded {loaded} CME(s) from cone file.")
            self._refresh_cme_list()
        except Exception as exc:
            if set_status:
                self.load_cone_status_label.setText("Failed to load cone file. See terminal output.")
                self.load_cone_status_label.setToolTip(str(exc))

    def set_model_start_datetime(self, model_start: object):
        """Update model start reference and refresh launch datetime from day offset."""
        if isinstance(model_start, datetime.datetime):
            new_start = model_start.replace(tzinfo=None)
            if self._last_model_start_datetime == new_start:
                return

            self.model_start_datetime = new_start
            self._last_model_start_datetime = new_start
            self._sync_launch_datetime_from_day()
            self._reload_cone_file_for_updated_start()
            self._clear_stale_donki_cmes()

    def _reload_cone_file_for_updated_start(self):
        """Re-load cone-file CMEs so their launch offsets follow the current model start."""
        filepath = self.cone_file_edit.text().strip()
        if not filepath:
            return

        self._apply_cone_file(filepath, replace_existing_loaded=True, set_status=False)
        self.load_cone_status_label.setText("Cone file reloaded for updated model start time.")

    def set_model_inner_boundary(self, rmin_rs: float):
        """Update reference inner boundary and refresh CME initial-height highlighting."""
        self.model_inner_boundary_rs = float(rmin_rs)
        self._update_initial_height_style()

    def set_model_run_duration_days(self, run_days: float):
        """Update run duration used to flag CMEs outside the simulation window."""
        new_duration = float(run_days)
        if self.model_run_duration_days != new_duration:
            self.model_run_duration_days = new_duration
            self._clear_stale_donki_cmes()
        self._refresh_cme_list()

    def _clear_stale_donki_cmes(self):
        """Discard DONKI results when their defining run interval changes."""
        original_count = len(self._cmes)
        self._cmes = [cme for cme in self._cmes if cme.get("source") != "donki"]
        if len(self._cmes) != original_count:
            self.load_donki_status_label.setText(
                "Run interval changed; reload the cone CME list from DONKI."
            )
            self._refresh_cme_list()

    def _sync_launch_datetime_from_day(self):
        """Sync CME launch datetime display from relative launch day input."""
        launch_dt = self.model_start_datetime + datetime.timedelta(days=self.cme_launch_spin.value())
        self.cme_launch_datetime.blockSignals(True)
        self.cme_launch_datetime.setDateTime(QDateTime(launch_dt))
        self.cme_launch_datetime.blockSignals(False)

    def _sync_launch_day_from_datetime(self):
        """Sync relative launch day from CME launch datetime input."""
        launch_dt = self.cme_launch_datetime.dateTime().toPyDateTime()
        if launch_dt.tzinfo is not None:
            launch_dt = launch_dt.replace(tzinfo=None)
        delta_days = (launch_dt - self.model_start_datetime).total_seconds() / 86400.0
        self.cme_launch_spin.blockSignals(True)
        self.cme_launch_spin.setValue(delta_days)
        self.cme_launch_spin.blockSignals(False)

    def _on_cme_plasma_mode_changed(self, mode_text: str):
        """Keep all fields selectable, but visually mark which plasma mode is active."""
        use_absolute = mode_text == "Absolute values"
        active_style = "QWidget { background-color: rgba(20, 120, 20, 28); border-radius: 4px; }"
        inactive_style = "QWidget { background-color: rgba(120, 120, 120, 14); border-radius: 4px; }"
        self.fraction_row.setStyleSheet(inactive_style if use_absolute else active_style)
        self.absolute_row.setStyleSheet(active_style if use_absolute else inactive_style)

    def _update_initial_height_style(self):
        """Highlight CME initial height in red when it differs from model inner boundary."""
        differs = abs(self.cme_initial_height_spin.value() - self.model_inner_boundary_rs) > 1.0e-6
        if differs:
            self.cme_initial_height_spin.setStyleSheet(
                "QDoubleSpinBox { color: #b22222; font-weight: 600; }"
            )
            self.cme_initial_height_spin.setToolTip(
                "CME initial height differs from model inner boundary."
            )
        else:
            self.cme_initial_height_spin.setStyleSheet("")
            self.cme_initial_height_spin.setToolTip("")

    def _refresh_cme_list(self):
        """Refresh visible CME list from internal state."""
        self.cme_list_widget.clear()
        for idx, cme in enumerate(self._cmes, start=1):
            line = (
                f"{idx:02d}: t={cme['t_launch_day']} day, lon={cme['longitude']} deg, "
                f"lat={cme['latitude']} deg, v={cme['speed']} km/s, width={cme['width']} deg, "
                f"dt={cme['t_launch_datetime']} UTC"
            )
            item = QListWidgetItem(line)
            t_launch_day = float(cme.get("t_launch_day", 0.0))
            if t_launch_day < 0.0 or t_launch_day > self.model_run_duration_days:
                item.setForeground(QColor("#b22222"))
            self.cme_list_widget.addItem(item)

    def add_cme(self):
        """Add a CME entry with current control values."""
        self._cmes.append(
            {
                "longitude": self.cme_lon_spin.value(),
                "latitude": self.cme_lat_spin.value(),
                "speed": self.cme_speed_spin.value(),
                "width": self.cme_width_spin.value(),
                "t_launch_day": self.cme_launch_spin.value(),
                "t_launch_datetime": self.cme_launch_datetime.dateTime().toString("yyyy-MM-dd HH:mm:ss"),
                "thickness_rs": self.cme_thickness_spin.value(),
                "initial_height_rs": self.cme_initial_height_spin.value(),
                "cme_expansion": self.cme_expansion_toggle.isChecked(),
                "cme_fixed_duration": self.cme_fixed_duration_toggle.isChecked(),
                "fixed_duration_hr": self.cme_fixed_duration_hours_spin.value(),
                "profile_type": self.profile_type_combo.currentText(),
                "plasma_mode": self.cme_plasma_mode_combo.currentText(),
                "density_fraction": self.density_fraction_spin.value(),
                "temperature_fraction": self.temperature_fraction_spin.value(),
                "cme_density_pcc": self.cme_density_spin.value(),
                "cme_temperature_k": self.cme_temperature_spin.value(),
                "source": "manual",
            }
        )
        self._refresh_cme_list()
        self.manual_cme_dialog.accept()

    def remove_selected_cme(self):
        """Remove the currently selected CME entry, if any."""
        row = self.cme_list_widget.currentRow()
        if row >= 0:
            self._cmes.pop(row)
            self._refresh_cme_list()

    def clear_cmes(self):
        """Remove all CME entries."""
        self._cmes.clear()
        self._refresh_cme_list()

    def get_cmes(self):
        """Return CME entries as plain dictionaries for code generation."""
        return list(self._cmes)


class CodeDialog(QDialog):
    """Window to display generated SURF code from current GUI state."""

    def __init__(self, code_text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generated SURF Code")
        self.resize(900, 650)

        layout = QVBoxLayout()
        code_box = QTextEdit()
        code_box.setProperty("role", "console")
        code_box.setReadOnly(True)
        code_box.setPlainText(code_text)
        layout.addWidget(code_box)
        self.setLayout(layout)


class TerminalOutputDialog(QDialog):
    """Window to display captured run output and tracebacks."""

    def __init__(self, output_text: str, parent=None, title: str = "SURF Terminal Output"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 650)

        layout = QVBoxLayout()
        output_box = QTextEdit()
        output_box.setProperty("role", "console")
        output_box.setReadOnly(True)
        output_box.setPlainText(output_text)
        layout.addWidget(output_box)
        self.setLayout(layout)


class SurfRunWorker(QObject):
    """Background worker to execute generated code without freezing the UI."""

    finished = pyqtSignal(bool, str, str, object)
    solve_started = pyqtSignal()

    def __init__(self, code_text: str):
        super().__init__()
        self.code_text = code_text

    def run(self):
        """Execute generated code and emit success/error status."""
        result = run_generated_code(self.code_text, before_solve=self.solve_started.emit)
        self.finished.emit(
            result.success,
            result.message,
            result.output,
            result.model,
        )


class SurfMainWindow(QMainWindow):
    """Main SURF GUI window with tabbed workflow sections."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SURF GUI")
        self.resize(1100, 760)
        self.setMinimumSize(900, 620)

        self.run_thread = None
        self.run_worker = None
        self.last_terminal_output = "No run output available yet."
        self.last_model = None
        self.post_run_tabs_visible = False
        self.plot_code_history = []

        central = QWidget()
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(12)

        header_bar = QWidget()
        header_bar.setObjectName("headerBar")
        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(18, 11, 18, 11)

        header_text = QVBoxLayout()
        header_text.setSpacing(1)
        app_title = QLabel("SURFs UP")
        app_title.setObjectName("appTitle")
        app_subtitle = QLabel(
            "Configure, run, and inspect reduced-physics solar-wind simulations"
        )
        app_subtitle.setObjectName("appSubtitle")
        header_text.addWidget(app_title)
        header_text.addWidget(app_subtitle)
        header_layout.addLayout(header_text)
        header_layout.addStretch(1)
        root_layout.addWidget(header_bar)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.model_tab = ModelParametersTab()
        self.ambient_tab = AmbientSolarWindTab()
        self.cme_tab = CmeTab()
        self.visualisation_tab = VisualisationTab()
        self.movies_tab = MoviesTab()

        self.tabs.addTab(self.model_tab, "Model Parameters")
        self.tabs.addTab(self.ambient_tab, "Ambient Solar Wind")
        self.tabs.addTab(self.cme_tab, "CMEs")

        self.visualisation_tab.plot_map_button.clicked.connect(self.plot_map)
        self.visualisation_tab.plot_radial_button.clicked.connect(self.plot_radial)
        self.visualisation_tab.plot_timeseries_button.clicked.connect(self.plot_timeseries)
        self.visualisation_tab.export_timeseries_csv_button.clicked.connect(
            self.export_timeseries_csv
        )
        self.movies_tab.generate_movie_button.clicked.connect(self.generate_movie)
        self.movies_tab.generate_movie_with_ts_button.clicked.connect(
            self.generate_movie_with_ts
        )
        self.model_tab.one_d_toggle.toggled.connect(self._on_1d_mode_changed)
        self.model_tab.start_datetime.dateTimeChanged.connect(
            self._on_model_start_datetime_input_changed
        )
        self.model_tab.include_bpol_toggle.toggled.connect(self.ambient_tab.set_include_bpol)
        self.model_tab.solver_combo.currentTextChanged.connect(self.ambient_tab.set_model_solver)
        self.model_tab.solver_combo.currentTextChanged.connect(self.cme_tab.set_solver)
        self.model_tab.solver_combo.currentTextChanged.connect(self.movies_tab.set_solver)
        self.model_tab.start_datetime_updated.connect(self.cme_tab.set_model_start_datetime)
        self.model_tab.start_datetime_updated.connect(self.ambient_tab.set_model_start_datetime)
        self.model_tab.rmin_spin.valueChanged.connect(self.ambient_tab.set_model_inner_boundary)
        self.model_tab.latitude_spin.valueChanged.connect(self.ambient_tab.set_model_latitude)
        self.model_tab.rmin_spin.valueChanged.connect(self.cme_tab.set_model_inner_boundary)
        self.model_tab.simtime_spin.valueChanged.connect(self.cme_tab.set_model_run_duration_days)
        self._on_1d_mode_changed(self.model_tab.one_d_toggle.isChecked())
        self.ambient_tab.set_model_inner_boundary(self.model_tab.rmin_spin.value())
        self.ambient_tab.set_model_latitude(self.model_tab.latitude_spin.value())
        self.ambient_tab.set_model_solver(self.model_tab.solver_combo.currentText())
        self.ambient_tab.set_model_start_datetime(self.model_tab.start_datetime.dateTime().toPyDateTime())
        self.ambient_tab.set_include_bpol(self.model_tab.include_bpol_toggle.isChecked())
        self.cme_tab.set_model_start_datetime(self.model_tab.start_datetime.dateTime().toPyDateTime())
        self.cme_tab.set_model_inner_boundary(self.model_tab.rmin_spin.value())
        self.cme_tab.set_model_run_duration_days(self.model_tab.simtime_spin.value())
        self.cme_tab.set_solver(self.model_tab.solver_combo.currentText())
        self.movies_tab.set_solver(self.model_tab.solver_combo.currentText())
        root_layout.addWidget(self.tabs, 1)

        footer_bar = QWidget()
        footer_bar.setObjectName("footerBar")
        footer = QHBoxLayout(footer_bar)
        footer.setContentsMargins(10, 8, 10, 8)
        footer.setSpacing(8)

        self.load_config_button = QPushButton("Load Configuration")
        self.load_config_button.clicked.connect(self.load_configuration)
        footer.addWidget(self.load_config_button)

        self.save_config_button = QPushButton("Save Configuration")
        self.save_config_button.clicked.connect(self.save_configuration)
        footer.addWidget(self.save_config_button)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        footer.addWidget(self.status_label, 1)

        self.show_code_button = QPushButton("Show Code")
        self.show_code_button.clicked.connect(self.show_generated_code)
        footer.addWidget(self.show_code_button)

        self.show_output_button = QPushButton("Show Terminal Output")
        self.show_output_button.clicked.connect(self.show_terminal_output)
        self._set_output_button_idle_style()
        footer.addWidget(self.show_output_button)

        self.run_button = QPushButton("Run SURF")
        self.run_button.clicked.connect(self.run_surf)
        self._set_run_button_idle_style()
        footer.addWidget(self.run_button)

        root_layout.addWidget(footer_bar)
        root_layout.removeWidget(footer_bar)
        root_layout.insertWidget(1, footer_bar)
        central.setLayout(root_layout)
        self.setCentralWidget(central)

        self.ambient_tab.status_message.connect(self.status_label.setText)
        self.ambient_tab.error_message.connect(self._on_ambient_error)
        self.ambient_tab.start_time_selected.connect(self.model_tab.set_start_datetime)
        self.ambient_tab.emit_active_map_time_if_enabled()
        self._connect_run_invalidation_signals()

    def _on_model_start_datetime_input_changed(self, _qdt: QDateTime):
        """Immediately propagate manual model start-time edits to dependent controls."""
        model_start = self.model_tab.start_datetime.dateTime().toPyDateTime()
        self.cme_tab.set_model_start_datetime(model_start)
        self.ambient_tab.set_model_start_datetime(model_start)

    def _build_generated_code(self):
        """Create a runnable Python script from current GUI state."""
        request = self._simulation_request()
        return build_generated_code(request)

    def _simulation_request(self) -> SimulationRequest:
        """Translate Qt widget values into the shared application request."""
        return SimulationRequest.from_mappings(
            self.model_tab.get_state(),
            self.ambient_tab.get_state(),
            self.cme_tab.get_cmes(),
        )

    def _connect_run_invalidation_signals(self):
        """Mark a completed run stale when any pre-run input changes."""
        for widget in self._configuration_widgets().values():
            if isinstance(widget, (QCheckBox, QRadioButton)):
                widget.toggled.connect(self._invalidate_completed_run)
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._invalidate_completed_run)
            elif isinstance(widget, QDateTimeEdit):
                widget.dateTimeChanged.connect(self._invalidate_completed_run)
            elif isinstance(widget, QDateEdit):
                widget.dateChanged.connect(self._invalidate_completed_run)
            elif isinstance(widget, (QDoubleSpinBox, QSpinBox, QSlider)):
                widget.valueChanged.connect(self._invalidate_completed_run)
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(self._invalidate_completed_run)
            elif isinstance(widget, QTextEdit):
                widget.textChanged.connect(self._invalidate_completed_run)

        cme_model = self.cme_tab.cme_list_widget.model()
        cme_model.rowsInserted.connect(self._invalidate_completed_run)
        cme_model.rowsRemoved.connect(self._invalidate_completed_run)
        cme_model.modelReset.connect(self._invalidate_completed_run)
        self.ambient_tab.source_tabs.currentChanged.connect(
            self._invalidate_completed_run
        )

    def _invalidate_completed_run(self, *_args):
        """Discard stale post-run products after the run configuration changes."""
        if self.last_model is None:
            return
        self.last_model = None
        self.plot_code_history.clear()
        self._set_run_button_idle_style()
        self.status_label.setText("Configuration changed. Run SURF again.")

    def _configuration_widgets(self):
        """Return stable attribute paths for editable pre-run widgets."""
        supported = (
            QCheckBox,
            QComboBox,
            QDateEdit,
            QDateTimeEdit,
            QDoubleSpinBox,
            QLineEdit,
            QRadioButton,
            QSlider,
            QSpinBox,
            QTextEdit,
        )
        widgets = {}
        visited = set()

        def visit(value, path):
            if id(value) in visited:
                return
            if isinstance(value, supported):
                widgets[path] = value
                return
            if isinstance(value, dict):
                visited.add(id(value))
                for key, child in value.items():
                    visit(child, f"{path}.{key}")
                return
            if isinstance(value, (list, tuple)):
                visited.add(id(value))
                for index, child in enumerate(value):
                    visit(child, f"{path}.{index}")
                return
            if not isinstance(value, QWidget):
                return
            visited.add(id(value))
            for name, child in vars(value).items():
                if not name.startswith("_"):
                    visit(child, f"{path}.{name}")

        for name, tab in (
            ("model", self.model_tab),
            ("ambient", self.ambient_tab),
            ("cmes", self.cme_tab),
        ):
            visit(tab, name)
        return widgets

    @staticmethod
    def _widget_configuration_value(widget):
        """Convert a supported input widget value to JSON-compatible data."""
        if isinstance(widget, (QCheckBox, QRadioButton)):
            return widget.isChecked()
        if isinstance(widget, QComboBox):
            return {
                "text": widget.currentText(),
                "data": widget.currentData(),
            }
        if isinstance(widget, QDateTimeEdit):
            return widget.dateTime().toString(Qt.DateFormat.ISODate)
        if isinstance(widget, QDateEdit):
            return widget.date().toString(Qt.DateFormat.ISODate)
        if isinstance(widget, (QDoubleSpinBox, QSpinBox, QSlider)):
            return widget.value()
        if isinstance(widget, QLineEdit):
            return widget.text()
        if isinstance(widget, QTextEdit):
            return widget.toPlainText()
        raise TypeError(f"Unsupported configuration widget: {type(widget).__name__}")

    @staticmethod
    def _apply_widget_configuration_value(widget, value):
        """Apply one value loaded from a configuration file."""
        if isinstance(widget, (QCheckBox, QRadioButton)):
            widget.setChecked(bool(value))
        elif isinstance(widget, QComboBox):
            index = widget.findData(value.get("data"))
            if index < 0:
                index = widget.findText(str(value.get("text", "")))
            if index >= 0:
                widget.setCurrentIndex(index)
        elif isinstance(widget, QDateTimeEdit):
            date_time = QDateTime.fromString(str(value), Qt.DateFormat.ISODate)
            if date_time.isValid():
                widget.setDateTime(date_time)
        elif isinstance(widget, QDateEdit):
            date = QDateTime.fromString(str(value), Qt.DateFormat.ISODate).date()
            if date.isValid():
                widget.setDate(date)
        elif isinstance(widget, (QDoubleSpinBox, QSpinBox, QSlider)):
            widget.setValue(value)
        elif isinstance(widget, QLineEdit):
            widget.setText(str(value))
        elif isinstance(widget, QTextEdit):
            widget.setPlainText(str(value))

    def save_configuration(self):
        """Save pre-run GUI input values and CMEs to a JSON configuration file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save SURF GUI configuration",
            "surf_configuration.json",
            "JSON configuration (*.json)",
        )
        if not filepath:
            return
        if not filepath.lower().endswith(".json"):
            filepath += ".json"

        try:
            configuration = {
                "format": "surfs-up-gui-configuration",
                "version": 1,
                "widgets": {
                    path: self._widget_configuration_value(widget)
                    for path, widget in self._configuration_widgets().items()
                },
                "ambient_source": self.ambient_tab._selected_source(),
                "cmes": self.cme_tab.get_cmes(),
            }
            Path(filepath).write_text(
                json.dumps(configuration, indent=2, allow_nan=True),
                encoding="utf-8",
            )
            self.status_label.setText(f"Configuration saved to {filepath}")
        except Exception as exc:
            self.status_label.setText(f"Could not save configuration: {exc}")

    def load_configuration(self):
        """Load pre-run inputs and discard any previous run result."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load SURF GUI configuration",
            "",
            "JSON configuration (*.json);;All files (*)",
        )
        if not filepath:
            return

        try:
            configuration = json.loads(Path(filepath).read_text(encoding="utf-8"))
            if configuration.get("format") != "surfs-up-gui-configuration":
                raise ValueError("this is not a SURFs UP GUI configuration file")
            if configuration.get("version") != 1:
                raise ValueError(
                    f"unsupported configuration version {configuration.get('version')!r}"
                )

            widgets = self._configuration_widgets()
            for path, value in configuration.get("widgets", {}).items():
                widget = widgets.get(path)
                if widget is not None:
                    self._apply_widget_configuration_value(widget, value)
            self.ambient_tab.set_selected_source(
                configuration.get("ambient_source", "wsa_iswa")
            )

            cmes = configuration.get("cmes", [])
            if not isinstance(cmes, list):
                raise ValueError("the saved CME configuration is invalid")
            self.cme_tab._cmes = cmes
            self.cme_tab._refresh_cme_list()
            self.ambient_tab._update_source_tab_highlight()
            self.last_model = None
            self.last_terminal_output = "No run output available yet."
            self.plot_code_history.clear()
            self._set_run_button_idle_style()
            self._set_output_button_idle_style()
            self.status_label.setText(f"Configuration loaded from {filepath}")
        except Exception as exc:
            self.status_label.setText(f"Could not load configuration: {exc}")

    def _set_run_button_idle_style(self):
        """Apply neutral style used when idle or before first run."""
        self.run_button.setText("Run SURF")
        self.run_button.setStyleSheet("")

    def _set_run_button_running_style(self):
        """Apply running style (red) while SURF job executes."""
        self.run_button.setText("Running SURF...")
        self.run_button.setStyleSheet(
            "QPushButton { background-color: #b22222; color: white; font-weight: 600; }"
        )

    def _set_run_button_success_style(self):
        """Apply completion style (green) after successful run."""
        self.run_button.setText("Run Complete")
        self.run_button.setStyleSheet(
            "QPushButton { background-color: #228b22; color: white; font-weight: 600; }"
        )

    def _set_output_button_idle_style(self):
        """Apply the normal terminal-output button style."""
        self.show_output_button.setStyleSheet("")

    def _set_output_button_failed_style(self):
        """Highlight terminal output when it contains details of a failed run."""
        self.show_output_button.setStyleSheet(
            "QPushButton { background-color: #b22222; color: white; font-weight: 600; }"
        )

    def _on_1d_mode_changed(self, enabled: bool):
        """Propagate 1D mode UI state to visualisation controls."""
        self.visualisation_tab.set_1d_mode(enabled)

    def _show_post_run_tabs(self):
        """Add post-run tabs with a spacer once a model result is available."""
        if self.post_run_tabs_visible:
            return

        spacer_tab = QWidget()
        spacer_index = self.tabs.addTab(spacer_tab, " ")
        self.tabs.setTabEnabled(spacer_index, False)
        self.tabs.addTab(self.visualisation_tab, "Plots")
        self.tabs.addTab(self.movies_tab, "Movies")
        self.post_run_tabs_visible = True

    def _hide_post_run_tabs(self):
        """Restore the pre-run tab set after loading a configuration."""
        current_index = self.tabs.currentIndex()
        while self.tabs.count() > 3:
            self.tabs.removeTab(self.tabs.count() - 1)
        if 0 <= current_index < self.tabs.count():
            self.tabs.setCurrentIndex(current_index)
        self.post_run_tabs_visible = False

    def show_generated_code(self):
        """Show a text window with generated script based on current GUI state."""
        self._sync_model_inner_boundary_for_omni()
        code_text = self._build_generated_code()
        if self.plot_code_history:
            code_text += (
                "\n\n# Plotting code run from the GUI\n"
                "import matplotlib.dates as mdates\n"
                "import matplotlib.pyplot as plt\n"
                "import surf.surf_analysis as sa\n\n"
                + "\n\n".join(self.plot_code_history)
                + "\n"
            )
        dialog = CodeDialog(code_text, self)
        dialog.exec()

    def show_terminal_output(self):
        """Show captured output from the most recent SURF run in a separate window."""
        dialog = TerminalOutputDialog(self.last_terminal_output, self)
        dialog.exec()

    def run_surf(self):
        """Execute generated SURF code and update UI state accordingly."""
        self._sync_model_inner_boundary_for_omni()
        code_text = self._build_generated_code()
        self.plot_code_history.clear()
        self.status_label.setText("Grabbing and processing input data")
        self.run_button.setEnabled(False)
        self.show_code_button.setEnabled(False)
        self.show_output_button.setEnabled(False)
        self._set_run_button_running_style()
        self._set_output_button_idle_style()

        self.run_thread = QThread(self)
        self.run_worker = SurfRunWorker(code_text)
        self.run_worker.moveToThread(self.run_thread)

        self.run_thread.started.connect(self.run_worker.run)
        self.run_worker.solve_started.connect(
            lambda: self.status_label.setText("Running SURF")
        )
        self.run_worker.finished.connect(self._on_run_finished)
        self.run_worker.finished.connect(self.run_thread.quit)
        self.run_worker.finished.connect(self.run_worker.deleteLater)
        self.run_thread.finished.connect(self.run_thread.deleteLater)
        self.run_thread.start()

    def _sync_model_inner_boundary_for_omni(self):
        """Force model inner boundary to 215 Rs when OMNI source is configured to use it."""
        ambient = self.ambient_tab.get_state()
        if ambient.get("source") != "omni":
            return
        if not ambient.get("use_215_inner_boundary", True):
            return

        if abs(self.model_tab.rmin_spin.value() - 215.0) > 1.0e-6:
            self.model_tab.rmin_spin.setValue(215.0)

    def _append_terminal_output(self, text: str):
        """Append text to the captured terminal output buffer."""
        if not text:
            return
        if self.last_terminal_output and self.last_terminal_output != "No run output available yet.":
            self.last_terminal_output += "\n\n" + text
        else:
            self.last_terminal_output = text

    def _on_plot_failed(self, action: str):
        """Record plotting traceback and notify user to inspect output dialog."""
        error = traceback.format_exc()
        self._append_terminal_output(error)
        self.status_label.setText(f"{action} failed. Open 'Show Terminal Output' for details.")

    def _on_plot_succeeded(self, action: str):
        """Update status after a successful plotting call."""
        self.status_label.setText(f"{action} generated.")

    def _record_plot_code(self, code_text: str):
        """Add code executed by a plot action to the Show Code summary."""
        self.plot_code_history.append(code_text.strip())

    def _movie_output_path(self, tag: str, explicit_path: str = "") -> Path:
        """Return output path for the movie, using explicit path when provided."""
        explicit_path = str(explicit_path).strip()
        if explicit_path:
            return Path(explicit_path)

        cr_num = np.int32(self.last_model.cr_num.value)
        filename = f"SURF_CR{cr_num:03d}_{tag}_movie.gif"
        return sa.get_figure_dir().joinpath(filename)

    def _play_movie_file(self, filepath: Path):
        """Open the generated movie file in the system default media player."""
        if not filepath.exists():
            self.status_label.setText(f"Movie generated, but file not found at {filepath}.")
            return

        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(filepath))):
            self.status_label.setText(f"Movie generated at {filepath}, but auto-play failed.")

    def _call_movie_animation(self, animate_fn, model_obj, **kwargs):
        """Call a SURF animation function using only supported keyword arguments."""
        try:
            signature = inspect.signature(animate_fn)
            accepts_any_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if accepts_any_kwargs:
                return animate_fn(model_obj, **kwargs)
            filtered_kwargs = {
                key: value for key, value in kwargs.items() if key in signature.parameters
            }
            return animate_fn(model_obj, **filtered_kwargs)
        except (TypeError, ValueError):
            return animate_fn(model_obj, **kwargs)

    def _on_ambient_error(self, error_text: str):
        """Capture ambient-tab tracebacks and surface a concise status message."""
        self._append_terminal_output(error_text)
        self.status_label.setText(
            "Ambient solar wind action failed. Open 'Show Terminal Output' for details."
        )

    def plot_map(self):
        """Generate a 2D map plot matching notebook sa.plot usage."""
        if self.last_model is None:
            self.status_label.setText("Run SURF first to enable plotting.")
            return

        try:
            plot_time = self.visualisation_tab.map_time_spin.value() * u.day
            plot_rmax = (
                self.visualisation_tab.map_rmax_spin.value()
                if self.visualisation_tab.map_limit_rmax_toggle.isChecked()
                else None
            )
            selected_solver = self.model_tab.solver_combo.currentText().strip().lower()
            if selected_solver == "huxt":
                self._record_plot_code(
                    "\n".join(
                        [
                            "# Plot a two-dimensional map at the selected model time.",
                            "sa.plot(",
                            "    model,",
                            f"    {plot_time.to_value(u.day)!r} * u.day,",
                            f"    minimalplot={self.visualisation_tab.map_minimalplot_toggle.isChecked()!r},",
                            f"    plotHCS={self.visualisation_tab.map_plot_hcs_toggle.isChecked()!r},",
                            f"    annotateplot={self.visualisation_tab.map_annotate_toggle.isChecked()!r},",
                            f"    trace_earth_connection={self.visualisation_tab.map_trace_earth_toggle.isChecked()!r},",
                            f"    plot_rmax={plot_rmax!r},",
                            ")",
                            "plt.show()",
                        ]
                    )
                )
                sa.plot(
                    self.last_model,
                    plot_time,
                    minimalplot=self.visualisation_tab.map_minimalplot_toggle.isChecked(),
                    plotHCS=self.visualisation_tab.map_plot_hcs_toggle.isChecked(),
                    annotateplot=self.visualisation_tab.map_annotate_toggle.isChecked(),
                    trace_earth_connection=self.visualisation_tab.map_trace_earth_toggle.isChecked(),
                    plot_rmax=plot_rmax,
                )
            else:
                self._record_plot_code(
                    "\n".join(
                        [
                            "# Plot a two-dimensional compressible-model map.",
                            "sa.plot_compressible(",
                            "    model,",
                            f"    {plot_time.to_value(u.day)!r} * u.day,",
                            f"    minimalplot={self.visualisation_tab.map_minimalplot_toggle.isChecked()!r},",
                            f"    annotateplot={self.visualisation_tab.map_annotate_toggle.isChecked()!r},",
                            f"    plot_rmax={plot_rmax!r},",
                            f"    plotHCS={self.visualisation_tab.map_plot_hcs_toggle.isChecked()!r},",
                            ")",
                            "plt.show()",
                        ]
                    )
                )
                sa.plot_compressible(
                    self.last_model,
                    plot_time,
                    minimalplot=self.visualisation_tab.map_minimalplot_toggle.isChecked(),
                    annotateplot=self.visualisation_tab.map_annotate_toggle.isChecked(),
                    plot_rmax=plot_rmax,
                    plotHCS=self.visualisation_tab.map_plot_hcs_toggle.isChecked(),
                )
            plt.show()
            self._on_plot_succeeded("2D map")
        except Exception:
            self._on_plot_failed("2D map")

    def plot_radial(self):
        """Generate a radial profile plot matching notebook sa.plot_radial usage."""
        if self.last_model is None:
            self.status_label.setText("Run SURF first to enable plotting.")
            return

        try:
            plot_time = self.visualisation_tab.radial_time_spin.value() * u.day
            lon = self.visualisation_tab.radial_lon_spin.value() * u.deg
            self._record_plot_code(
                "\n".join(
                    [
                        "# Plot variables along a radial line at fixed longitude.",
                        "sa.plot_radial(",
                        "    model,",
                        f"    {plot_time.to_value(u.day)!r} * u.day,",
                        f"    lon={lon.to_value(u.deg)!r} * u.deg,",
                        ")",
                        "plt.show()",
                    ]
                )
            )
            plot_radial_profile(self.last_model, plot_time, lon=lon)
            plt.show()
            self._on_plot_succeeded("Radial profile")
        except Exception:
            self._on_plot_failed("Radial profile")

    def plot_timeseries(self):
        """Plot at a fixed custom coordinate or along a standard observer ephemeris."""
        if self.last_model is None:
            self.status_label.setText("Run SURF first to enable plotting.")
            return

        try:
            observer = self.visualisation_tab.ts_location_combo.currentData()
            if observer == "custom":
                radius = self.visualisation_tab.ts_radius_spin.value() * u.AU
                lon = self.visualisation_tab.ts_lon_spin.value() * u.deg
                self._record_plot_code(
                    "\n".join(
                        [
                            "# Plot a time series at a fixed radius and SURF model longitude.",
                            "# In sidereal runs this is HEEQ/model longitude at run start;",
                            "# in synodic runs this is the corotating model longitude.",
                            "from surfs_up.core import plot_custom_timeseries",
                            "plot_custom_timeseries(",
                            "    model,",
                            f"    {radius.to_value(u.AU)!r} * u.AU,",
                            f"    lon={lon.to_value(u.deg)!r} * u.deg,",
                            ")",
                            "plt.show()",
                        ]
                    )
                )
                plot_custom_timeseries(self.last_model, radius, lon=lon)
            else:
                if str(observer).upper() not in SUPPORTED_OBSERVERS:
                    self.status_label.setText(
                        f"Observer '{observer}' is not supported; no plot was generated."
                    )
                    return
                time_series = sa.get_observer_timeseries(
                    self.last_model,
                    observer=observer,
                )
                selected_solver = self.model_tab.solver_combo.currentText().strip().lower()
                speed = np.asarray(time_series.get("vsw", np.nan), dtype=float)
                if not np.isfinite(speed).any():
                    self.status_label.setText(
                        "No observer data fall within the model domain; no plot was generated."
                    )
                    return

                is_compressible = selected_solver != "huxt"
                solver_label = (
                    f"SURF-{self.model_tab.solver_combo.currentText()}"
                    if is_compressible
                    else "SURF-HUXt"
                )
                bpol = np.asarray(time_series.get("bpol", np.nan), dtype=float)
                has_bpol = np.isfinite(bpol).any()

                observer_plot_lines = [
                    "# Sample the moving observer and plot the available model variables.",
                    f"time_series = sa.get_observer_timeseries(model, observer={observer!r})",
                    "times = time_series['time']",
                    "speed = np.asarray(time_series.get('vsw', np.nan), dtype=float)",
                    f"fig, axes = plt.subplots({1 + int(has_bpol) + (2 if is_compressible else 0)}, 1, figsize=(10, 6.25), sharex=True)",
                    "axes = np.atleast_1d(axes)",
                    f"axes[0].plot(times, speed, 'r', label={solver_label!r})",
                    "axes[0].set_ylim(300, 900)",
                    "axes[0].set_ylabel('V [km/s]')",
                ]
                panel_code_index = 0
                if has_bpol:
                    panel_code_index += 1
                    observer_plot_lines.extend(
                        [
                            "bpol = np.asarray(time_series.get('bpol', np.nan), dtype=float)",
                            f"axes[{panel_code_index}].plot(times, np.sign(bpol), 'r.', label={solver_label!r})",
                            f"axes[{panel_code_index}].set_ylabel(r\"B$_{{\\text{{POL}}}}$\")",
                        ]
                    )
                if is_compressible:
                    panel_code_index += 1
                    observer_plot_lines.extend(
                        [
                            "density = np.asarray(time_series.get('n', time_series.get('density', np.nan)), dtype=float)",
                            f"axes[{panel_code_index}].semilogy(times, density, 'r-', label={solver_label!r})",
                            f"axes[{panel_code_index}].set_ylabel(r\"n$_\\text{{P}}$ [cm$^{{-3}}$]\")",
                        ]
                    )
                    panel_code_index += 1
                    observer_plot_lines.extend(
                        [
                            "temperature = np.asarray(time_series.get('T', time_series.get('temperature', np.nan)), dtype=float)",
                            f"axes[{panel_code_index}].semilogy(times, temperature, 'r-', label={solver_label!r})",
                            f"axes[{panel_code_index}].set_ylabel('T [K]')",
                        ]
                    )
                observer_plot_lines.extend(
                    [
                        "for axis in axes:",
                        "    axis.legend()",
                        "date_locator = mdates.DayLocator() if (times.iloc[-1] - times.iloc[0]).total_seconds() / 86400 <= 7 else mdates.AutoDateLocator()",
                        "axes[-1].xaxis.set_major_locator(date_locator)",
                        "axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))",
                        "fig.autofmt_xdate(rotation=0, ha='center')",
                        "axes[-1].set_xlabel(f\"DD-MM of {times.iloc[0].year}\", fontsize=12, fontweight='bold')",
                        "plt.show()",
                    ]
                )
                self._record_plot_code("\n".join(observer_plot_lines))

                n_panels = 1
                if has_bpol:
                    n_panels += 1
                if is_compressible:
                    n_panels += 2
                fig, axes = plt.subplots(n_panels, 1, figsize=timeseries_figsize(), sharex=True)
                if n_panels == 1:
                    axes = np.array([axes])

                times = time_series["time"]
                starttime = times.iloc[0] if hasattr(times, "iloc") else times[0]
                endtime = times.iloc[-1] if hasattr(times, "iloc") else times[-1]

                panel_idx = 0
                axes[panel_idx].plot(times, speed, "r", label=solver_label)
                axes[panel_idx].set_ylim(300, 900)
                axes[panel_idx].set_ylabel("V [km/s]")

                if has_bpol:
                    panel_idx += 1
                    axes[panel_idx].plot(times, np.sign(bpol), "r.", label=solver_label)
                    axes[panel_idx].set_ylabel(r"B$_{\text{POL}}$")
                    axes[panel_idx].set_ylim(-1.1, 1.1)

                if is_compressible:
                    density = np.asarray(
                        time_series.get("n", time_series.get("density", np.nan)),
                        dtype=float,
                    )
                    temperature = np.asarray(
                        time_series.get("T", time_series.get("temperature", np.nan)),
                        dtype=float,
                    )

                    panel_idx += 1
                    axes[panel_idx].semilogy(times, density, "r-", label=solver_label)
                    axes[panel_idx].set_ylabel(r"n$_\text{P}$ [cm$^{-3}$]")
                    axes[panel_idx].set_ylim(0.101, 999)
                    axes[panel_idx].grid(True, alpha=0.3)

                    panel_idx += 1
                    axes[panel_idx].semilogy(times, temperature, "r-", label=solver_label)
                    axes[panel_idx].set_ylabel(r"T [K]")
                    axes[panel_idx].set_ylim(1e4, 9.9e6)
                    axes[panel_idx].grid(True, alpha=0.3)

                for axis in axes:
                    axis.set_xlim(starttime, endtime)
                    axis.legend()

                for i in range(len(axes) - 1):
                    axes[i].set_xticklabels([])
                format_datetime_axis_like_surf(fig, axes, times)
                fig.subplots_adjust(left=0.10, bottom=0.14, right=0.98, top=0.95, hspace=0.05)
            plt.show()
            self._on_plot_succeeded("Time series")
        except Exception:
            self._on_plot_failed("Time series")

    def _default_timeseries_csv_path(self, location_tag: str) -> Path:
        """Build a default CSV filename for time-series export."""
        safe_tag = re.sub(r"[^A-Za-z0-9_-]+", "_", str(location_tag)).strip("_")
        if not safe_tag:
            safe_tag = "timeseries"
        cr_num = np.int32(self.last_model.cr_num.value)
        filename = f"SURF_CR{cr_num:03d}_{safe_tag}_timeseries.csv"
        return sa.get_figure_dir().joinpath(filename)

    def _custom_location_timeseries_data(self, radius, lon):
        """Sample model grids at nearest custom radius/longitude for CSV export."""
        model = self.last_model
        id_r = int(np.argmin(np.abs(model.r - radius)))
        if model.lon.size == 1:
            id_lon = 0
            lon_out = float(np.asarray(model.lon.value).reshape(-1)[0])
        else:
            id_lon = int(np.argmin(np.abs(model.lon - lon)))
            lon_out = float(model.lon[id_lon].value)

        r_out = float(model.r[id_r].value)
        m_p = 1.6726e-27
        data = {
            "time_days": np.asarray(model.time_out.to(u.day).value, dtype=float),
            "vsw": np.asarray(model.v_grid[:, id_r, id_lon], dtype=float),
        }
        if hasattr(model, "b_grid"):
            data["bpol"] = np.asarray(model.b_grid[:, id_r, id_lon], dtype=float)
        if hasattr(model, "compressible") and model.compressible:
            data["n"] = np.asarray(model.rho_grid[:, id_r, id_lon].value / m_p / 1e6, dtype=float)
            data["T"] = np.asarray(model.temp_grid[:, id_r, id_lon].value, dtype=float)

        location_tag = f"custom_r{r_out:.1f}Rs_lon{lon_out:.1f}deg"
        return data, location_tag

    def _write_timeseries_csv(self, filepath: Path, data):
        """Write pandas dataframe-like or dict-of-arrays time-series data to CSV."""
        if hasattr(data, "to_csv"):
            data.to_csv(filepath, index=False)
            return

        if not isinstance(data, dict) or not data:
            raise ValueError("No time-series data available for CSV export.")

        columns = list(data.keys())
        arrays = [np.asarray(data[column]) for column in columns]
        row_count = len(arrays[0]) if arrays else 0
        if any(len(arr) != row_count for arr in arrays):
            raise ValueError("Time-series data columns have inconsistent lengths.")

        with open(filepath, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for idx in range(row_count):
                writer.writerow([arr[idx] for arr in arrays])

    def _open_exported_csv(self, filepath: Path):
        """Open exported CSV in the app text viewer, or fallback to system default."""
        try:
            csv_text = filepath.read_text(encoding="utf-8")
            dialog = TerminalOutputDialog(
                csv_text,
                self,
                title=f"Time Series CSV - {filepath.name}",
            )
            dialog.exec()
            return
        except Exception:
            pass

        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(filepath))):
            self.status_label.setText(
                f"CSV exported to {filepath}, but it could not be opened automatically."
            )

    def export_timeseries_csv(self):
        """Export the currently selected time-series data to CSV."""
        if self.last_model is None:
            self.status_label.setText("Run SURF first to enable time-series export.")
            return

        try:
            observer = self.visualisation_tab.ts_location_combo.currentData()
            if observer == "custom":
                radius = self.visualisation_tab.ts_radius_spin.value() * u.AU
                lon = self.visualisation_tab.ts_lon_spin.value() * u.deg
                data, location_tag = self._custom_location_timeseries_data(radius, lon)
            else:
                if str(observer).upper() not in SUPPORTED_OBSERVERS:
                    self.status_label.setText(
                        f"Observer '{observer}' is not supported; no CSV was exported."
                    )
                    return
                data = sa.get_observer_timeseries(self.last_model, observer=observer)
                location_tag = str(observer).strip().lower()

            explicit_output = self.visualisation_tab.ts_csv_output_edit.text().strip()
            output_path = (
                Path(explicit_output)
                if explicit_output
                else self._default_timeseries_csv_path(location_tag)
            )
            if output_path.suffix.lower() != ".csv":
                output_path = output_path.with_suffix(".csv")

            self._write_timeseries_csv(output_path, data)
            self.status_label.setText(f"Time-series CSV exported to {output_path}.")
            self._open_exported_csv(output_path)
        except Exception:
            self._on_plot_failed("Time-series CSV export")

    def _run_movie_generation(
        self,
        button: QPushButton,
        animation_fn,
        animation_kwargs: dict,
        output_path: Path,
        play_on_complete: bool,
        status_label: str,
    ):
        """Run the selected movie renderer and handle button/status transitions."""
        if self.last_model is None:
            self.status_label.setText("Run SURF first to enable movie generation.")
            return

        original_text = button.text()
        original_style = button.styleSheet()
        button.setEnabled(False)
        button.setText("Rendering movie frames")
        button.setStyleSheet(
            "QPushButton { background-color: #b22222; color: white; font-weight: 600; }"
        )
        QApplication.processEvents()

        try:
            saved_path = self._call_movie_animation(
                animation_fn,
                self.last_model,
                **animation_kwargs,
            )
            self._on_plot_succeeded(status_label)

            if play_on_complete:
                self._play_movie_file(Path(saved_path) if saved_path else output_path)
        except Exception:
            self._on_plot_failed("Movie generation")
        finally:
            button.setEnabled(True)
            button.setText(original_text)
            button.setStyleSheet(original_style)

    def generate_movie(self):
        """Generate a standard 2D SURF movie using sa.animate."""
        tag = self.movies_tab.movie_tag_edit.text().strip() or "gui"
        plot_rmax = (
            self.movies_tab.movie_rmax_spin.value()
            if self.movies_tab.movie_limit_rmax_toggle.isChecked()
            else None
        )
        output_path = self._movie_output_path(
            tag,
            self.movies_tab.movie_output_edit.text(),
        )
        animation_kwargs = {
            "tag": tag,
            "duration": self.movies_tab.movie_duration_spin.value(),
            "fps": self.movies_tab.movie_fps_spin.value(),
            "plotHCS": self.movies_tab.movie_plot_hcs_toggle.isChecked(),
            "trace_earth_connection": self.movies_tab.movie_trace_earth_toggle.isChecked(),
            "outputfilepath": str(output_path),
            "plot_rmax": plot_rmax,
        }
        self._run_movie_generation(
            button=self.movies_tab.generate_movie_button,
            animation_fn=sa.animate,
            animation_kwargs=animation_kwargs,
            output_path=output_path,
            play_on_complete=self.movies_tab.movie_play_on_complete_toggle.isChecked(),
            status_label="2D movie",
        )

    def generate_movie_with_ts(self):
        """Generate a movie with time-series overlay using sa.animate_with_ts."""
        if not hasattr(sa, "animate_with_ts"):
            self.status_label.setText("animate_with_ts is not available in this SURF version.")
            return

        tag = self.movies_tab.movie_ts_tag_edit.text().strip() or "gui"
        selected_solver = self.model_tab.solver_combo.currentText().strip().lower()
        selected_field = self.movies_tab.movie_ts_field_combo.currentText()
        polar_var = "V" if selected_solver == "huxt" else selected_field
        plot_rmax = (
            self.movies_tab.movie_ts_rmax_spin.value()
            if self.movies_tab.movie_ts_limit_rmax_toggle.isChecked()
            else None
        )
        output_path = self._movie_output_path(
            tag,
            self.movies_tab.movie_ts_output_edit.text(),
        )
        animation_kwargs = {
            "tag": tag,
            "duration": self.movies_tab.movie_ts_duration_spin.value(),
            "fps": self.movies_tab.movie_ts_fps_spin.value(),
            "plotHCS": self.movies_tab.movie_ts_plot_hcs_toggle.isChecked(),
            "outputfilepath": str(output_path),
            "plot_rmax": plot_rmax,
            "polar_var": polar_var,
        }
        self._run_movie_generation(
            button=self.movies_tab.generate_movie_with_ts_button,
            animation_fn=sa.animate_with_ts,
            animation_kwargs=animation_kwargs,
            output_path=output_path,
            play_on_complete=True,
            status_label="Movie with time series",
        )

    def _on_run_finished(self, success: bool, message: str, terminal_output: str, model_obj):
        """Handle SURF completion state and update UI styling/availability."""
        self.run_button.setEnabled(True)
        self.show_code_button.setEnabled(True)
        self.show_output_button.setEnabled(True)
        self.last_terminal_output = terminal_output or "(No terminal output captured.)"

        if success:
            self._set_run_button_success_style()
            self._set_output_button_idle_style()
            self.status_label.setText(message)
            self.last_model = model_obj
            self._show_post_run_tabs()
        else:
            self._set_run_button_idle_style()
            self._set_output_button_failed_style()
            self.status_label.setText(message + " Open 'Show Terminal Output' for details.")


def main():
    """Launch the SURF GUI application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)
    window = SurfMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
