import sys
import os
import json
import tempfile
import threading
import urllib.request
import numpy as np
import sounddevice as sd
import soundfile as sf
import yt_dlp
from ytmusicapi import YTMusic
from scipy.signal import iirpeak
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize, QEvent
from PyQt6.QtGui import QShortcut, QKeySequence, QIcon, QPixmap, QAction, QColor, QPainter
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFileDialog, QListWidget,
    QLineEdit, QTabWidget, QListWidgetItem, QSplitter, QDialog, QMenu, QCheckBox
)

EQ_FREQS = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
STATE_FILE = "lumen_state.json"

class AudioEngine:
    def __init__(self):
        self.data = None
        self.sr = 44100
        self.position = 0
        self.is_playing = False
        self.track_finished = False
        self.preamp_gain = 1.0
        self.eq_gains = [0.0] * len(EQ_FREQS)
        self.speed = 1.0
        self.is_muted = False
        self.stream = None
        self.lock = threading.Lock()

    def load_source(self, source_path_or_url):
        with self.lock:
            try:
                data, self.sr = sf.read(source_path_or_url, dtype='float32')
                if data.ndim == 1: data = np.column_stack((data, data))
                self.data = data
                self.position = 0
                self.track_finished = False
            except Exception as e: print(f"Error loading audio: {e}")

    def play(self):
        if self.data is None: return
        self.is_playing = True
        self.track_finished = False
        if self.stream is None or not self.stream.active:
            self.stream = sd.OutputStream(samplerate=self.sr, channels=2, callback=self._audio_callback, blocksize=2048)
            self.stream.start()

    def pause(self): self.is_playing = False
    def seek(self, target_sample):
        with self.lock:
            if self.data is not None: self.position = max(0, min(target_sample, len(self.data)))

    def _audio_callback(self, outdata, frames, time_info, status):
        with self.lock:
            if not self.is_playing or self.data is None:
                outdata.fill(0)
                return
            req_samples = int(frames * self.speed)
            start_idx = int(self.position)
            end_idx = start_idx + req_samples
            chunk_orig = self.data[start_idx:end_idx]

            if len(chunk_orig) < req_samples:
                pad = np.zeros((req_samples - len(chunk_orig), 2), dtype='float32')
                chunk_orig = np.vstack((chunk_orig, pad)) if len(chunk_orig) > 0 else pad
                self.is_playing = False
                self.position = 0
                self.track_finished = True
            else: self.position += req_samples

            if self.speed == 1.0: chunk = chunk_orig
            else:
                x_orig = np.linspace(0, 1, req_samples)
                x_new = np.linspace(0, 1, frames)
                chunk = np.zeros((frames, 2), dtype='float32')
                for ch in range(2): chunk[:, ch] = np.interp(x_new, x_orig, chunk_orig[:, ch])

        processed = chunk * self.preamp_gain
        if self.is_muted: processed = processed * 0.0

        for freq, gain_db in zip(EQ_FREQS, self.eq_gains):
            if gain_db != 0.0:
                q = 1.414; gain_linear = 10.0 ** (gain_db / 40.0)
                b, a = iirpeak(freq, q, fs=self.sr)
                filtered = np.zeros_like(processed)
                for ch in range(processed.shape[1]): filtered[:, ch] = np.convolve(processed[:, ch], b, mode='same')
                processed = processed + (filtered * (gain_linear - 1.0))
        np.clip(processed, -1.0, 1.0, out=outdata)


# --- Workers ---
class SearchWorker(QThread):
    results_ready = pyqtSignal(list)
    def __init__(self, query): super().__init__(); self.query = query
    def run(self):
        try: self.results_ready.emit(YTMusic().search(self.query, filter="songs", limit=20))
        except: self.results_ready.emit([])

class RecommendWorker(QThread):
    recommendations_ready = pyqtSignal(list)
    def __init__(self, video_id): super().__init__(); self.video_id = video_id
    def run(self):
        try: self.recommendations_ready.emit(YTMusic().get_watch_playlist(videoId=self.video_id).get('tracks', []))
        except: self.recommendations_ready.emit([])

class ImageFetcher(QThread):
    image_ready = pyqtSignal(int, bytes, QListWidget)
    def __init__(self, index, url, target_list):
        super().__init__()
        self.index = index; self.url = url; self.target_list = target_list
    def run(self):
        try:
            req = urllib.request.Request(self.url, headers={'User-Agent': 'Mozilla/5.0'})
            self.image_ready.emit(self.index, urllib.request.urlopen(req).read(), self.target_list)
        except: pass

class StreamFetchWorker(QThread):
    stream_ready = pyqtSignal(str, str, str)
    def __init__(self, video_id, title): super().__init__(); self.video_id = video_id; self.title = title
    def run(self):
        temp_dir = tempfile.gettempdir()
        out_tmpl = os.path.join(temp_dir, f"{self.video_id}.%(ext)s")
        url = f"https://music.youtube.com/watch?v={self.video_id}"
        ydl_opts = {'format': 'bestaudio/best', 'outtmpl': out_tmpl, 'quiet': True, 'noplaylist': True, 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}]}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)
                wav_filepath = os.path.join(temp_dir, f"{self.video_id}.wav")
                if os.path.exists(wav_filepath): self.stream_ready.emit(wav_filepath, self.title, self.video_id)
        except Exception: pass


# --- Preferences & EQ ---
class EQDialog(QDialog):
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.setWindowTitle("Control Panel")
        self.setFixedSize(500, 250)
        self.setStyleSheet("""
            QDialog { background-color: #1B1C20; color: #D0D0D0; font-family: 'Segoe UI', sans-serif; }
            QLabel { font-size: 11px; }
            QSlider::groove:vertical { width: 4px; background: #0E0F12; border-radius: 2px; }
            QSlider::handle:vertical { background: #00B4FF; height: 12px; margin: 0 -4px; border-radius: 6px; }
        """)
        layout = QHBoxLayout(self)
        
        preamp_box = QVBoxLayout()
        lbl_p = QLabel("Gain")
        self.preamp_slider = QSlider(Qt.Orientation.Vertical)
        self.preamp_slider.setRange(0, 300)
        self.preamp_slider.setValue(int(self.engine.preamp_gain * 100))
        self.preamp_slider.valueChanged.connect(lambda v: setattr(self.engine, 'preamp_gain', v / 100.0))
        preamp_box.addWidget(self.preamp_slider, alignment=Qt.AlignmentFlag.AlignCenter)
        preamp_box.addWidget(lbl_p, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addLayout(preamp_box)

        for i, f in enumerate(EQ_FREQS):
            vbox = QVBoxLayout()
            lbl = QLabel(f"{f if f < 1000 else str(f//1000)+'k'}")
            slider = QSlider(Qt.Orientation.Vertical)
            slider.setRange(-12, 12)
            slider.setValue(int(self.engine.eq_gains[i]))
            slider.valueChanged.connect(lambda val, idx=i: self.update_eq(idx, val))
            vbox.addWidget(slider, alignment=Qt.AlignmentFlag.AlignCenter)
            vbox.addWidget(lbl, alignment=Qt.AlignmentFlag.AlignCenter)
            layout.addLayout(vbox)

    def update_eq(self, index, val): self.engine.eq_gains[index] = float(val)


class PrefsDialog(QDialog):
    def __init__(self, prefs, parent=None):
        super().__init__(parent)
        self.prefs = prefs
        self.setWindowTitle("Preferences")
        self.setFixedSize(300, 150)
        self.setStyleSheet("background-color: #1B1C20; color: #FFF;")
        layout = QVBoxLayout(self)
        self.chk_auto_mini = QCheckBox("Switch to audio skin when playing")
        self.chk_auto_mini.setChecked(self.prefs.get('auto_mini', False))
        self.chk_pause_min = QCheckBox("Pause when minimized")
        self.chk_pause_min.setChecked(self.prefs.get('pause_on_minimize', False))
        layout.addWidget(self.chk_auto_mini)
        layout.addWidget(self.chk_pause_min)
        save_btn = QPushButton("OK")
        save_btn.setStyleSheet("background: #00B4FF; color: black; padding: 4px;")
        save_btn.clicked.connect(self.accept)
        layout.addStretch()
        layout.addWidget(save_btn)

    def accept(self):
        self.prefs['auto_mini'] = self.chk_auto_mini.isChecked()
        self.prefs['pause_on_minimize'] = self.chk_pause_min.isChecked()
        super().accept()


# --- Main UI ---
class AdvancedPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = AudioEngine()
        self.image_workers = []
        self.eq_window = None
        self.current_playlist = None
        self.current_index = -1
        self.custom_playlists = {}
        self.hide_extensions = False
        self.prefs = {'auto_mini': False, 'pause_on_minimize': False}
        self.is_mini_mode = False
        self.is_pinned = False
        self.normal_geometry = None

        self.setAcceptDrops(True)
        self.init_ui()
        self.setup_shortcuts()
        self.setup_context_menus()
        self.load_state()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui_state)
        self.timer.start(50)

    def init_ui(self):
        self.setWindowTitle("Lumen Player")
        self.setMinimumSize(850, 550)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #141419; }
            QWidget { color: #DDDDDD; font-family: 'Segoe UI', Tahoma, sans-serif; font-size: 12px; }
            QPushButton { background: transparent; border: none; padding: 4px; border-radius: 2px; }
            QPushButton:hover { background: #2A2A35; }
            QPushButton:pressed { background: #00B4FF; color: black; }
            QLineEdit { background-color: #0E0F12; border: 1px solid #282830; padding: 4px; color: #FFF; }
            QListWidget { background-color: #1B1C20; border: none; outline: none; }
            QListWidget::item { padding: 4px; border-bottom: 1px solid #222228; }
            QListWidget::item:selected { background: #282A33; color: #00B4FF; }
            QSlider::groove:horizontal { height: 3px; background: #333333; }
            QSlider::sub-page:horizontal { background: #00B4FF; }
            QSlider::handle:horizontal { background: #FFF; width: 10px; margin: -4px 0; border-radius: 5px; }
            QTabWidget::pane { border: none; border-top: 1px solid #282830; }
            QTabBar::tab { background: #141419; padding: 6px 12px; border: 1px solid #282830; border-bottom: none; }
            QTabBar::tab:selected { background: #1B1C20; color: #00B4FF; border-top: 2px solid #00B4FF; }
            QMenu { background-color: #1B1C20; border: 1px solid #333; }
            QMenu::item:selected { background-color: #00B4FF; color: black; }
        """)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- LEFT PANE (Video/Audio Area) ---
        left_widget = QWidget()
        left_widget.setStyleSheet("background-color: #0E0F12;")
        self.left_layout = QVBoxLayout(left_widget)
        self.left_layout.setContentsMargins(0,0,0,0)
        self.left_layout.setSpacing(0)
        
        # Header Controls
        header_row = QHBoxLayout()
        header_row.setContentsMargins(8,4,8,4)
        title_lbl = QLabel("Lumen Player")
        title_lbl.setStyleSheet("color: #888; font-weight: bold; font-size: 11px;")
        header_row.addWidget(title_lbl)
        header_row.addStretch()
        self.pin_btn = QPushButton("📌"); self.pin_btn.clicked.connect(self.toggle_pin)
        self.mini_btn = QPushButton("🗕"); self.mini_btn.clicked.connect(self.toggle_mini_mode)
        header_row.addWidget(self.pin_btn); header_row.addWidget(self.mini_btn)
        self.left_layout.addLayout(header_row)

        # Main Logo / Album Art Area
        self.np_image = QLabel("LUMEN PLAYER")
        self.np_image.setStyleSheet("font-size: 36px; font-weight: bold; color: #1B1C20; background: #0E0F12;")
        self.np_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_layout.addWidget(self.np_image, 1)

        # Bottom Control Bar
        control_bar = QWidget()
        control_bar.setStyleSheet("background-color: #141419; border-top: 1px solid #222;")
        control_layout = QVBoxLayout(control_bar)
        control_layout.setContentsMargins(10,5,10,10)

        # Seek & Time
        seek_row = QHBoxLayout()
        self.stats_label = QLabel("00:00")
        self.stats_label.setStyleSheet("color: #00B4FF; font-family: monospace;")
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.sliderMoved.connect(self.seek_audio)
        self.total_label = QLabel("00:00")
        self.total_label.setStyleSheet("color: #666; font-family: monospace;")
        seek_row.addWidget(self.stats_label); seek_row.addWidget(self.seek_slider); seek_row.addWidget(self.total_label)
        control_layout.addLayout(seek_row)

        # Playback Buttons
        btn_row = QHBoxLayout()
        self.play_btn = QPushButton("►"); self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setStyleSheet("font-size: 16px; padding: 0 10px;")
        self.prev_btn = QPushButton("⏮"); self.prev_btn.clicked.connect(self.play_previous)
        self.next_btn = QPushButton("⏭"); self.next_btn.clicked.connect(self.play_next)
        stop_btn = QPushButton("■"); stop_btn.clicked.connect(self.pause_and_rewind)
        
        self.track_label = QLabel("Ready")
        self.track_label.setStyleSheet("padding-left: 10px; color: #AAA;")
        
        self.eq_btn = QPushButton("⚙"); self.eq_btn.clicked.connect(self.open_eq)
        self.list_toggle_btn = QPushButton("≣"); self.list_toggle_btn.clicked.connect(self.toggle_playlist)
        
        btn_row.addWidget(self.play_btn); btn_row.addWidget(self.prev_btn)
        btn_row.addWidget(stop_btn); btn_row.addWidget(self.next_btn)
        btn_row.addWidget(self.track_label, 1)
        btn_row.addWidget(self.eq_btn); btn_row.addWidget(self.list_toggle_btn)
        control_layout.addLayout(btn_row)
        
        self.left_layout.addWidget(control_bar)
        self.main_splitter.addWidget(left_widget)

        # --- RIGHT PANE (Sidebar Playlist) ---
        self.sidebar_widget = QWidget()
        self.sidebar_widget.setStyleSheet("background: #1B1C20; border-left: 1px solid #111;")
        sidebar_layout = QVBoxLayout(self.sidebar_widget)
        sidebar_layout.setContentsMargins(0,0,0,0)
        
        self.sidebar_tabs = QTabWidget()
        
        # Local Tab
        local_widget = QWidget(); local_layout = QVBoxLayout(local_widget); local_layout.setContentsMargins(0,0,0,0)
        self.local_playlist = QListWidget(); self.local_playlist.setIconSize(QSize(40, 40))
        self.local_playlist.itemDoubleClicked.connect(self.handle_list_click)
        add_btn = QPushButton("+ ADD"); add_btn.clicked.connect(self.browse_file)
        add_btn.setStyleSheet("background: #282A33; border-radius:0; border-top: 1px solid #111;")
        local_layout.addWidget(self.local_playlist); local_layout.addWidget(add_btn)
        self.sidebar_tabs.addTab(local_widget, "Default")

        # Network Tab
        net_widget = QWidget(); net_layout = QVBoxLayout(net_widget); net_layout.setContentsMargins(0,0,0,0)
        search_row = QHBoxLayout(); search_row.setContentsMargins(4,4,4,4)
        self.search_input = QLineEdit(); self.search_input.setPlaceholderText("Search...")
        self.search_input.returnPressed.connect(self.perform_online_search)
        search_row.addWidget(self.search_input)
        self.online_results = QListWidget(); self.online_results.setIconSize(QSize(40, 40))
        self.online_results.itemDoubleClicked.connect(self.handle_list_click)
        net_layout.addLayout(search_row); net_layout.addWidget(self.online_results)
        self.sidebar_tabs.addTab(net_widget, "Network")

        # Recommendations Tab
        rec_widget = QWidget(); rec_layout = QVBoxLayout(rec_widget); rec_layout.setContentsMargins(0,0,0,0)
        self.rec_list = QListWidget(); self.rec_list.setIconSize(QSize(40, 40))
        self.rec_list.itemDoubleClicked.connect(self.handle_list_click)
        rec_layout.addWidget(self.rec_list)
        self.sidebar_tabs.addTab(rec_widget, "Related")

        # Custom Playlists Tab
        pl_widget = QWidget(); pl_layout = QVBoxLayout(pl_widget); pl_layout.setContentsMargins(0,0,0,0)
        pl_splitter = QSplitter(Qt.Orientation.Vertical)
        self.pl_names_list = QListWidget(); self.pl_names_list.itemClicked.connect(self.load_playlist_view)
        self.pl_names_list.setStyleSheet("background: #141419;")
        self.pl_songs_list = QListWidget(); self.pl_songs_list.setIconSize(QSize(40, 40))
        self.pl_songs_list.itemDoubleClicked.connect(self.handle_list_click)
        pl_splitter.addWidget(self.pl_names_list); pl_splitter.addWidget(self.pl_songs_list)
        pl_layout.addWidget(pl_splitter)
        
        pl_create_row = QHBoxLayout(); pl_create_row.setContentsMargins(4,4,4,4)
        self.pl_input = QLineEdit(); self.pl_input.setPlaceholderText("New Playlist")
        pl_btn = QPushButton("+"); pl_btn.clicked.connect(self.create_playlist)
        pl_create_row.addWidget(self.pl_input); pl_create_row.addWidget(pl_btn)
        pl_layout.addLayout(pl_create_row)
        
        self.sidebar_tabs.addTab(pl_widget, "Albums")

        sidebar_layout.addWidget(self.sidebar_tabs)
        self.main_splitter.addWidget(self.sidebar_widget)
        
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 0)
        self.main_splitter.setSizes([600, 250])
        self.setCentralWidget(self.main_splitter)

    # --- UI Generators & Modals ---
    def generate_local_icon(self):
        pix = QPixmap(40, 40)
        pix.fill(QColor("#222"))
        painter = QPainter(pix)
        painter.setPen(QColor("#555"))
        painter.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "MP3")
        painter.end()
        return QIcon(pix)

    def pause_and_rewind(self):
        self.engine.pause()
        self.engine.seek(0)
        self.play_btn.setText("►")

    def open_prefs(self):
        dialog = PrefsDialog(self.prefs, self)
        dialog.exec()

    def toggle_pin(self):
        self.is_pinned = not self.is_pinned
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, self.is_pinned)
        self.pin_btn.setStyleSheet("color: #00B4FF;" if self.is_pinned else "color: #FFF;")
        self.show()

    def toggle_mini_mode(self):
        self.is_mini_mode = not self.is_mini_mode
        if self.is_mini_mode:
            self.normal_geometry = self.geometry()
            self.sidebar_widget.hide()
            self.np_image.hide()
            self.mini_btn.setStyleSheet("color: #00B4FF;")
            self.setMinimumSize(300, 100)
            self.resize(300, 100)
        else:
            self.sidebar_widget.show()
            self.np_image.show()
            self.mini_btn.setStyleSheet("color: #FFF;")
            self.setMinimumSize(850, 550)
            if self.normal_geometry: self.setGeometry(self.normal_geometry)

    def changeEvent(self, event):
        if event.type() == QEvent.Type.WindowStateChange:
            if self.isMinimized() and self.prefs.get('pause_on_minimize', False) and self.engine.is_playing:
                self.engine.pause()
                self.play_btn.setText("►")
        super().changeEvent(event)

    # --- Key Binds ---
    def setup_shortcuts(self):
        QShortcut(QKeySequence("Space"), self).activated.connect(self.toggle_play)
        QShortcut(QKeySequence("Right"), self).activated.connect(lambda: self.relative_seek(5))
        QShortcut(QKeySequence("Left"), self).activated.connect(lambda: self.relative_seek(-5))
        QShortcut(QKeySequence("Page Down"), self).activated.connect(self.play_next)
        QShortcut(QKeySequence("Page Up"), self).activated.connect(self.play_previous)
        QShortcut(QKeySequence("Ctrl+R"), self).activated.connect(self.pause_and_rewind)
        QShortcut(QKeySequence("F5"), self).activated.connect(self.open_prefs)
        QShortcut(QKeySequence("F6"), self).activated.connect(self.toggle_playlist)
        QShortcut(QKeySequence("F7"), self).activated.connect(self.open_eq)
        QShortcut(QKeySequence("A"), self).activated.connect(self.open_eq)
        QShortcut(QKeySequence("Up"), self).activated.connect(self.volume_up)
        QShortcut(QKeySequence("Down"), self).activated.connect(self.volume_down)
        QShortcut(QKeySequence("M"), self).activated.connect(self.toggle_mute)
        QShortcut(QKeySequence("C"), self).activated.connect(self.speed_up)
        QShortcut(QKeySequence("X"), self).activated.connect(self.speed_down)
        QShortcut(QKeySequence("Z"), self).activated.connect(self.speed_reset)

    def toggle_playlist(self):
        sizes = self.main_splitter.sizes()
        if sizes[1] > 0:
            self.last_sidebar_size = sizes[1]
            self.main_splitter.setSizes([sizes[0] + sizes[1], 0])
        else:
            restore = getattr(self, 'last_sidebar_size', 250)
            self.main_splitter.setSizes([max(1, sizes[0] - restore), restore])

    def volume_up(self):
        self.engine.preamp_gain = min(3.0, self.engine.preamp_gain + 0.1)
        if self.eq_window and self.eq_window.isVisible(): self.eq_window.preamp_slider.setValue(int(self.engine.preamp_gain * 100))
    def volume_down(self):
        self.engine.preamp_gain = max(0.0, self.engine.preamp_gain - 0.1)
        if self.eq_window and self.eq_window.isVisible(): self.eq_window.preamp_slider.setValue(int(self.engine.preamp_gain * 100))
    def toggle_mute(self): self.engine.is_muted = not self.engine.is_muted
    def speed_up(self): self.engine.speed = min(3.0, round(self.engine.speed + 0.1, 1))
    def speed_down(self): self.engine.speed = max(0.2, round(self.engine.speed - 0.1, 1))
    def speed_reset(self): self.engine.speed = 1.0

    def update_ui_state(self):
        mute_txt = "[MUTED] " if self.engine.is_muted else ""
        title = f"{mute_txt}Lumen Player ({self.engine.speed}x)"
        if self.windowTitle() != title: self.setWindowTitle(title)

        if self.engine.track_finished:
            self.engine.track_finished = False
            self.play_next()

        if self.engine.data is not None:
            curr = self.engine.position
            total = len(self.engine.data)
            curr_s = int(curr / self.engine.sr)
            tot_s = int(total / self.engine.sr)
            self.stats_label.setText(f"{curr_s//60:02}:{curr_s%60:02}")
            self.total_label.setText(f"{tot_s//60:02}:{tot_s%60:02}")
            
            if not self.seek_slider.isSliderDown():
                self.seek_slider.setValue(curr)

    # --- Universal Data Handling & Drag-Drop ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()
    
    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path): self.load_folder(path)
            elif path.endswith(('.mp3', '.flac', '.wav', '.ogg', '.m4a')): self.add_local_item(path)
        if not self.engine.is_playing and self.local_playlist.count() > 0:
            self.handle_list_click(self.local_playlist.item(self.local_playlist.count()-1))

    def load_folder(self, folder_path):
        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.endswith(('.mp3', '.flac', '.wav', '.ogg', '.m4a')): self.add_local_item(os.path.join(root, f))

    def add_local_item(self, path):
        display = os.path.basename(path)
        if self.hide_extensions: display = os.path.splitext(display)[0]
        item = QListWidgetItem(display)
        item.setData(Qt.ItemDataRole.UserRole, {'type': 'local', 'path': path, 'thumb': None})
        item.setIcon(self.generate_local_icon())
        self.local_playlist.addItem(item)

    # --- Menus ---
    def setup_context_menus(self):
        for lw in [self.online_results, self.rec_list]:
            lw.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            lw.customContextMenuRequested.connect(lambda pos, w=lw: self.show_online_context(pos, w))
        for lw in [self.local_playlist, self.pl_songs_list]:
            lw.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            lw.customContextMenuRequested.connect(lambda pos, w=lw: self.show_sidebar_context(pos, w))

    def show_online_context(self, pos, list_widget):
        item = list_widget.itemAt(pos)
        if not item: return
        menu = QMenu(self)
        add_menu = menu.addMenu("Add to Album")
        if not self.custom_playlists:
            add_menu.addAction(QAction("(Create an album first)", self)).setEnabled(False)
        else:
            for pl_name in self.custom_playlists.keys():
                action = QAction(pl_name, self)
                action.triggered.connect(lambda checked, p=pl_name, i=item: self.add_to_playlist(p, i))
                add_menu.addAction(action)
        menu.exec(list_widget.mapToGlobal(pos))

    def show_sidebar_context(self, pos, list_widget):
        menu = QMenu(self)
        if list_widget == self.local_playlist:
            ext_action = QAction("Show File Extensions", self)
            ext_action.setCheckable(True)
            ext_action.setChecked(not self.hide_extensions)
            ext_action.triggered.connect(self.toggle_extensions)
            menu.addAction(ext_action)
        menu.exec(list_widget.mapToGlobal(pos))

    def toggle_extensions(self, checked):
        self.hide_extensions = not checked
        for i in range(self.local_playlist.count()):
            item = self.local_playlist.item(i)
            data = item.data(Qt.ItemDataRole.UserRole)
            item.setText(os.path.basename(data['path']) if not self.hide_extensions else os.path.splitext(os.path.basename(data['path']))[0])

    def closeEvent(self, event):
        state = {
            'custom_playlists': self.custom_playlists, 
            'local_queue': [self.local_playlist.item(i).data(Qt.ItemDataRole.UserRole)['path'] for i in range(self.local_playlist.count())], 
            'hide_extensions': self.hide_extensions,
            'prefs': self.prefs
        }
        with open(STATE_FILE, 'w') as f: json.dump(state, f)
        event.accept()

    def load_state(self):
        if not os.path.exists(STATE_FILE): return
        try:
            with open(STATE_FILE, 'r') as f: state = json.load(f)
            self.hide_extensions = state.get('hide_extensions', False)
            self.prefs = state.get('prefs', {'auto_mini': False, 'pause_on_minimize': False})
            for path in state.get('local_queue', []):
                if os.path.exists(path): self.add_local_item(path)
            self.custom_playlists = state.get('custom_playlists', {})
            for name in self.custom_playlists.keys():
                self.pl_names_list.addItem(name)
        except Exception: pass

    # --- Custom Library & Playlists ---
    def create_playlist(self):
        name = self.pl_input.text().strip()
        if name and name not in self.custom_playlists:
            self.custom_playlists[name] = []
            self.pl_names_list.addItem(name)
            self.pl_input.clear()

    def add_to_playlist(self, pl_name, item):
        data = item.data(Qt.ItemDataRole.UserRole)
        self.custom_playlists[pl_name].append({
            'text': item.text(), 
            'type': data.get('type', 'online'),
            'vid': data.get('vid'), 
            'path': data.get('path'),
            'thumb': data.get('thumb')
        })
        current_pl = self.pl_names_list.currentItem()
        if current_pl and current_pl.text() == pl_name: self.load_playlist_view(current_pl)

    def load_playlist_view(self, item):
        self.pl_songs_list.clear(); self.image_workers.clear()
        for idx, track in enumerate(self.custom_playlists.get(item.text(), [])):
            list_item = QListWidgetItem(track['text'])
            list_item.setData(Qt.ItemDataRole.UserRole, track)
            
            if track.get('type') == 'local':
                list_item.setIcon(self.generate_local_icon())
            elif track.get('thumb'):
                worker = ImageFetcher(idx, track['thumb'], self.pl_songs_list)
                worker.image_ready.connect(self.apply_image_to_list)
                self.image_workers.append(worker)
                worker.start()
                
            self.pl_songs_list.addItem(list_item)

    # --- Playback Execution ---
    def relative_seek(self, seconds):
        if self.engine.data is not None: self.engine.seek(self.engine.position + (seconds * self.engine.sr))

    def open_eq(self):
        if self.eq_window is None or not self.eq_window.isVisible():
            self.eq_window = EQDialog(self.engine, self); self.eq_window.show()

    def handle_list_click(self, item):
        self.current_playlist = item.listWidget()
        self.current_index = self.current_playlist.row(item)
        self.play_item(item)

    def play_item(self, item):
        self.current_playlist.setCurrentItem(item)
        data = item.data(Qt.ItemDataRole.UserRole)
        
        icon = item.icon()
        if not icon.isNull() and data.get('type') != 'local': 
            self.np_image.setPixmap(icon.pixmap(200, 200))
            self.np_image.setStyleSheet("background: #0E0F12; border:none;")
        else: 
            self.np_image.clear()
            self.np_image.setText("LUMEN PLAYER")
            self.np_image.setStyleSheet("font-size: 36px; font-weight: bold; color: #1B1C20; background: #0E0F12;")

        if self.prefs.get('auto_mini', False) and not self.is_mini_mode: self.toggle_mini_mode()

        if data.get('type') == 'local':
            self.track_label.setText(item.text())
            self.engine.load_source(data['path'])
            self.seek_slider.setRange(0, len(self.engine.data))
            self.engine.play()
            self.play_btn.setText("ǁ")
        else:
            lines = item.text().split(chr(10))
            self.track_label.setText(lines[0])
            self.stream_worker = StreamFetchWorker(data['vid'], lines[0])
            self.stream_worker.stream_ready.connect(self.start_stream_playback)
            self.stream_worker.start()
            
            self.rec_worker = RecommendWorker(data['vid'])
            self.rec_worker.recommendations_ready.connect(self.render_recommendations)
            self.rec_worker.start()

    def play_next(self):
        if self.current_playlist and self.current_index + 1 < self.current_playlist.count():
            self.current_index += 1
            self.play_item(self.current_playlist.item(self.current_index))

    def play_previous(self):
        if self.current_playlist and self.current_index - 1 >= 0:
            self.current_index -= 1
            self.play_item(self.current_playlist.item(self.current_index))

    def perform_online_search(self):
        query = self.search_input.text().strip()
        if not query: return
        self.track_label.setText("Searching...")
        self.online_results.clear()
        self.search_worker = SearchWorker(query)
        self.search_worker.results_ready.connect(self.render_online_results)
        self.search_worker.start()

    def render_online_results(self, items):
        self.image_workers.clear()
        for idx, item in enumerate(items):
            title = item.get('title', 'Unknown')
            artists = ", ".join([a['name'] for a in item.get('artists', []) if 'name' in a])
            thumb_url = item.get('thumbnails', [{}])[-1].get('url', None)
            
            list_item = QListWidgetItem(f"{title}\n{artists}")
            list_item.setData(Qt.ItemDataRole.UserRole, {'type': 'online', 'vid': item.get('videoId'), 'thumb': thumb_url})
            self.online_results.addItem(list_item)
            
            if thumb_url:
                worker = ImageFetcher(idx, thumb_url, self.online_results)
                worker.image_ready.connect(self.apply_image_to_list)
                self.image_workers.append(worker)
                worker.start()

    def render_recommendations(self, tracks):
        self.rec_list.clear()
        for idx, item in enumerate(tracks):
            title = item.get('title', 'Unknown')
            vid = item.get('videoId')
            if not vid: continue
            artists = ", ".join([a['name'] for a in item.get('artists', []) if 'name' in a])
            thumb_url = item.get('thumbnails', [{}])[0].get('url', None)
            
            list_item = QListWidgetItem(f"{title}\n{artists}")
            list_item.setData(Qt.ItemDataRole.UserRole, {'type': 'online', 'vid': vid, 'thumb': thumb_url})
            self.rec_list.addItem(list_item)
            
            if thumb_url:
                worker = ImageFetcher(idx, thumb_url, self.rec_list)
                worker.image_ready.connect(self.apply_image_to_list)
                self.image_workers.append(worker)
                worker.start()

    def start_stream_playback(self, direct_url, title, vid):
        self.engine.load_source(direct_url)
        self.seek_slider.setRange(0, len(self.engine.data))
        self.engine.play()
        self.play_btn.setText("ǁ")

    def apply_image_to_list(self, index, image_data, target_list):
        if index < target_list.count():
            pixmap = QPixmap()
            pixmap.loadFromData(image_data)
            target_list.item(index).setIcon(QIcon(pixmap))
            if self.current_playlist == target_list and self.current_index == index:
                self.np_image.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio))
                self.np_image.setStyleSheet("background: #0E0F12; border:none;")

    def browse_file(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open Files", "", "Audio (*.mp3 *.flac *.wav *.ogg *.m4a)")
        for f in files: self.add_local_item(f)

    def toggle_play(self):
        if self.engine.is_playing:
            self.engine.pause()
            self.play_btn.setText("►")
        else:
            self.engine.play()
            self.play_btn.setText("ǁ")

    def seek_audio(self, val): self.engine.seek(val)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = AdvancedPlayer()
    player.show()
    sys.exit(app.exec())