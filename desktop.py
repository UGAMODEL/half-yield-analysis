"""File-picker entry point for processing staff. Run with python desktop.py."""
from datetime import datetime
from hmi_style import UGAStyle
from pathlib import Path
import subprocess
import sys
import tempfile
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk, font as tkfont
except ImportError:
    raise SystemExit(
        "The file-picker launcher needs Python with Tk support. On Homebrew Python 3.11, "
        "install python-tk@3.11. You can use the dashboard now with: "
        "python main.py --source video.mp4 --show"
    )


class ReviewLauncher:
    def __init__(self, root):
        self.root = root
        self.process = None
        self.log = None
        self.video = tk.StringVar()
        self.report = tk.StringVar()
        self.window = tk.StringVar(value='60')
        self.status = tk.StringVar(value='Choose a recorded line video to begin.')
        root.title('HalfYield | Start a review')
        root.geometry('800x720')
        root.minsize(760, 720)
        root.configure(background=UGAStyle.WHITE)
        families = set(tkfont.families(root))
        body_font = 'Merriweather Sans' if 'Merriweather Sans' in families else 'Georgia'
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background=UGAStyle.WHITE)
        style.configure('TLabel', background=UGAStyle.WHITE, foreground=UGAStyle.BLACK, font=(body_font, 13))
        style.configure('Title.TLabel', font=('Georgia', 25, 'bold'))
        style.configure('TButton', font=(body_font, 12), padding=(16, 10),
                        background=UGAStyle.CONTROL, foreground=UGAStyle.BLACK)
        style.map('TButton', background=[('active', UGAStyle.SURFACE)],
                  foreground=[('disabled', UGAStyle.SANFORD)])
        style.configure('Primary.TButton', background=UGAStyle.RED, foreground=UGAStyle.WHITE)
        style.map('Primary.TButton', background=[('disabled', UGAStyle.CONTROL), ('active', UGAStyle.RED)],
                  foreground=[('disabled', UGAStyle.SANFORD), ('active', UGAStyle.WHITE)])
        style.configure('TEntry', padding=8, font=(body_font, 13), foreground=UGAStyle.BLACK)
        style.configure('TCombobox', padding=8, font=(body_font, 13))
        header = tk.Frame(root, background=UGAStyle.RED)
        header.pack(fill='x')
        tk.Label(header, text='HALFYIELD  /  QUALITY CONTROL', background=UGAStyle.RED,
                 foreground=UGAStyle.WHITE, font=(body_font, 14), padx=32, pady=18).pack(anchor='w')
        body = ttk.Frame(root, padding=32)
        body.pack(fill='both', expand=True)
        body.columnconfigure(0, weight=1)
        self._label(body, 'Review your processing line', 1, 'Title.TLabel')
        self._label(body, 'Measure the share of detected kernel area classified as halves.', 2)
        self._file_row(body, '1. Recorded video', self.video, self._choose_video, 3)
        self._file_row(body, '2. CSV report location', self.report, self._choose_report, 5)
        self._label(body, '3. Rolling measurement window (seconds)', 7)
        ttk.Combobox(body, textvariable=self.window, values=('10', '30', '60', '120'),
                     state='readonly', width=12).grid(row=8, column=0, sticky='w', pady=(0, 12))
        self.start = ttk.Button(body, text='Start review', style='Primary.TButton', command=self._start)
        self.start.grid(row=9, column=0, sticky='w', pady=(4, 12))
        ttk.Label(body, textvariable=self.status, wraplength=650).grid(row=10, column=0, sticky='w', pady=(0, 12))
        ttk.Label(body, text='Area-based estimate, not weight yield. Shell and obscured material are excluded.\n'
                  'Check camera alignment: the measurement region is fixed.\n'
                  'Finish in the review window to save the CSV report.',
                  wraplength=650).grid(row=11, column=0, sticky='w')
        root.protocol('WM_DELETE_WINDOW', self._close)

    def _choose_video(self):
        path = filedialog.askopenfilename(title='Select recorded processing video',
                                         filetypes=[('Video files', '*.mp4 *.mov *.avi *.mkv'), ('All files', '*')])
        if path:
            self.video.set(path)
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.report.set(str(Path(path).with_name(f'{Path(path).stem}_review_{stamp}.csv')))

    def _choose_report(self):
        path = filedialog.asksaveasfilename(title='Save analysis report', defaultextension='.csv',
                                           filetypes=[('CSV report', '*.csv')])
        if path:
            self.report.set(path)

    def _start(self):
        source = Path(self.video.get()).expanduser()
        report = Path(self.report.get()).expanduser()
        if not source.is_file():
            messagebox.showerror('Select a video', 'Choose an existing video file before starting.')
            return
        if not self.report.get() or not report.parent.is_dir() or report.suffix.lower() != '.csv':
            messagebox.showerror('Choose a report location', 'Choose a CSV file in an existing folder.')
            return
        if source.resolve() == report.resolve():
            messagebox.showerror('Choose another location', 'The report cannot replace the source video.')
            return
        if report.exists() and not messagebox.askyesno('Replace report?', f'Replace the existing report?\n{report}'):
            return
        self.log = tempfile.TemporaryFile(mode='w+')
        command = [sys.executable, str(Path(__file__).with_name('main.py')), '--source', str(source.resolve()),
                   '--show', '--ratio-window-sec', self.window.get(), '--yield-output', str(report.resolve())]
        try:
            self.process = subprocess.Popen(command, cwd=Path(__file__).parent, stdout=self.log, stderr=self.log)
        except OSError as error:
            self.log.close()
            messagebox.showerror('Could not start review', str(error))
            return
        self.active_report = report
        self.start.state(['disabled'])
        self.status.set('Loading model and video. The review window will open shortly...')
        self.root.after(500, self._poll)

    def _poll(self):
        if self.process.poll() is None:
            self.root.after(500, self._poll)
            return
        self.log.seek(0)
        details = self.log.read()
        self.log.close()
        self.start.state(['!disabled'])
        if self.process.returncode:
            self.status.set('Review could not finish. Check the error details and try again.')
            messagebox.showerror('Review failed', details[-2400:] or 'The processing application exited unexpectedly.')
        elif 'Insufficient data' in details:
            self.status.set('No report saved: play at least two frames before finishing.')
        elif 'Could not generate yield report' in details:
            self.status.set('Video review finished, but the CSV report could not be saved.')
            messagebox.showerror('Report not saved', details[-1600:])
        else:
            self.status.set(f'Review finished. Report saved to:\n{self.active_report}')
        self.process = None

    def _close(self):
        if self.process is not None and self.process.poll() is None:
            messagebox.showinfo('Review is running', 'Use Finish / save CSV in the review window first.')
            return
        self.root.destroy()

    @staticmethod
    def _label(parent, text, row, style='TLabel'):
        ttk.Label(parent, text=text, style=style).grid(row=row, column=0, sticky='w', pady=(0, 12))

    def _file_row(self, parent, label, variable, command, row):
        self._label(parent, label, row)
        line = ttk.Frame(parent)
        line.grid(row=row + 1, column=0, sticky='ew', pady=(0, 16))
        line.columnconfigure(0, weight=1)
        ttk.Entry(line, textvariable=variable).grid(row=0, column=0, sticky='ew', padx=(0, 10))
        ttk.Button(line, text='Browse...', command=command).grid(row=0, column=1)


if __name__ == '__main__':
    app = tk.Tk()
    ReviewLauncher(app)
    app.mainloop()
