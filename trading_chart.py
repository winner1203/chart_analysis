import os
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates

# --- 데이터 처리 함수들 ---
def load_data(directory):
    dataset = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            data = pd.read_csv(file_path)
            dataset.append(data)
    if dataset:
        return pd.concat(dataset, ignore_index=True)
    else:
        raise ValueError("No CSV files found in the directory.")

def clean_data(data):
    data['Gmt time'] = pd.to_datetime(data['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')
    data = data.drop_duplicates(subset='Gmt time')
    data = data[data['Volume'] > 0]
    return data.sort_values(by='Gmt time').reset_index(drop=True)

def adjust_trading_dates(data, summer_start, summer_end):
    is_summer = (data['Gmt time'] > summer_start) & (data['Gmt time'] < summer_end)
    data['Open time'] = np.where(is_summer, time(22, 0), time(23, 0))
    time_of_day = data['Gmt time'].dt.time
    is_after_open = time_of_day >= data['Open time']
    data['Trading Date'] = data['Gmt time'].dt.normalize()
    data.loc[is_after_open, 'Trading Date'] += pd.Timedelta(days=1)
    data.drop(columns=['Open time'], inplace=True)
    return data

def resample_data(df, freq):
    df_copy = df.copy()
    df_copy['Gmt time'] = pd.to_datetime(df_copy['Gmt time'])
    df_copy.set_index('Gmt time', inplace=True)
    df_copy = df_copy.groupby('Trading Date').resample(freq).agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna().reset_index()
    df_copy['Volume'] = df_copy['Volume'].round().astype(np.int64)
    return df_copy

def add_moving_averages(df, periods, price_column='Close'):
    for period in periods:
        ma_column = f'MA{period}'
        df[ma_column] = df[price_column].rolling(window=period).mean()
    return df

def generate_daily_data(df):
    df.sort_values(['Trading Date', 'Gmt time'], inplace=True)
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df = df.dropna(subset=['Volume'])
    daily_data = df.groupby('Trading Date').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).reset_index()
    daily_data.dropna(subset=['Volume'], inplace=True)
    daily_data['Volume'] = daily_data['Volume'].round().astype(np.int64)
    return daily_data

# --- 서머타임 정의 ---
summer_times = {
    2014: (datetime(2014, 3, 9), datetime(2014, 11, 2)),
    2015: (datetime(2015, 3, 8), datetime(2015, 11, 1)),
    2016: (datetime(2016, 3, 13), datetime(2016, 11, 6)),
    2017: (datetime(2017, 3, 12), datetime(2017, 11, 5)),
    2018: (datetime(2018, 3, 11), datetime(2018, 11, 4)),
    2019: (datetime(2019, 3, 10), datetime(2019, 11, 3)),
    2020: (datetime(2020, 3, 8), datetime(2020, 11, 1)),
    2021: (datetime(2021, 3, 14), datetime(2021, 11, 7)),
    2022: (datetime(2022, 3, 13), datetime(2022, 11, 6)),
    2023: (datetime(2023, 3, 12), datetime(2023, 11, 5)),
    2024: (datetime(2024, 3, 10), datetime(2024, 11, 3)),
    2025: (datetime(2025, 3, 9), datetime(2024, 11, 2)),
}

# --- 클래스 기반 GUI ---
class TradingChartApp:
    def __init__(self, master):
        self.master = master
        master.title("HTS 스타일 시세 차트")
        master.geometry("1200x800")

        self.data = None
        self.df_resampled = None
        self.filtered_data = None
        self.start_index = 0
        self.window_size = 250
        self.scroll_step = 1
        self.container = None

        self.setup_ui()

    def setup_ui(self):
        control_frame = tk.Frame(self.master)
        control_frame.pack(padx=10, pady=10, fill=tk.X)

        tk.Button(control_frame, text="CSV 폴더 선택", command=self.select_folder).pack(side=tk.LEFT, padx=5)

        self.time_frame_combo = ttk.Combobox(control_frame, values=['1min', '5min', '30min', '60min', 'daily'])
        self.time_frame_combo.set('60min')
        self.time_frame_combo.pack(side=tk.LEFT, padx=5)

        tk.Button(control_frame, text="차트 표시", command=self.plot_chart).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="데이터 저장", command=self.save_data).pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="시작일 (YYYY-MM-DD):").pack(side=tk.LEFT, padx=(20, 5))
        self.start_entry = tk.Entry(control_frame, width=12)
        self.start_entry.pack(side=tk.LEFT)

        tk.Label(control_frame, text="종료일 (YYYY-MM-DD):").pack(side=tk.LEFT, padx=(10, 5))
        self.end_entry = tk.Entry(control_frame, width=12)
        self.end_entry.pack(side=tk.LEFT)

        self.canvas = None

    def select_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        try:
            raw_data = load_data(folder)
            clean = clean_data(raw_data)
            df_list = []
            for year, (start, end) in summer_times.items():
                df_year = clean[clean['Gmt time'].dt.year == year].copy()
                if not df_year.empty:
                    df_list.append(adjust_trading_dates(df_year, start, end))
            self.data = pd.concat(df_list, ignore_index=True)
            messagebox.showinfo("완료", f"총 {len(self.data)}개 분봉이 로드되었습니다.")
        except Exception as e:
            messagebox.showerror("오류", str(e))

    def plot_chart(self):
        if self.data is None:
            messagebox.showwarning("경고", "먼저 데이터를 로드하세요.")
            return

        tf = self.time_frame_combo.get()
        if tf == 'daily':
            df = generate_daily_data(self.data)
            df.set_index('Trading Date', inplace=True)
        else:
            df = resample_data(self.data, tf)
            df.set_index('Gmt time', inplace=True)

        start_date = self.start_entry.get()
        end_date = self.end_entry.get()
        try:
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
        except Exception as e:
            messagebox.showerror("오류", f"기간 필터링 오류: {e}")
            return

        self.filtered_data = df
        self.df_resampled = add_moving_averages(df, [5, 10, 20, 60])
        self.start_index = 0
        self.show_scroll_chart()

    def save_data(self):
        if self.filtered_data is None:
            messagebox.showwarning("경고", "저장할 데이터가 없습니다. 먼저 차트를 표시하세요.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="저장할 파일 경로를 선택하세요"
        )

        if not file_path:
            return

        try:
            self.filtered_data.to_csv(file_path, index=True)
            messagebox.showinfo("성공", f"데이터가 저장되었습니다:\n{file_path}")
        except Exception as e:
            messagebox.showerror("오류", f"저장 중 오류 발생:\n{str(e)}")

    def update_chart(self):
        if self.df_resampled is None or len(self.df_resampled) <= self.window_size:
            return
        data_slice = self.df_resampled.iloc[self.start_index:self.start_index + self.window_size]
        self.ax.clear()
        mc = mpf.make_marketcolors(up='red', down='blue', edge='black', wick='black')
        style = mpf.make_mpf_style(marketcolors=mc)
        mpf.plot(data_slice, type='candle', ax=self.ax, mav=(5, 10, 20, 60), style=style)

        last_time = data_slice.index[-1]
        self.ax.annotate(f"last time: {last_time.strftime('%Y-%m-%d %H:%M')}",
                         xy=(1, 0), xycoords='axes fraction', textcoords='offset points',
                         xytext=(-10, 10), ha='right', va='bottom', fontsize=9, color='gray')

        self.canvas.draw()

    def show_scroll_chart(self):
        if self.container:
            self.container.destroy()

        self.container = tk.Frame(self.master)
        self.container.pack(fill=tk.BOTH, expand=True)

        btn_left = tk.Button(self.container, text="<<", width=3, height=1, command=lambda: self.scroll_by(-1))
        btn_left.pack(side=tk.LEFT, padx=5, pady=5)

        fig, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(fig, master=self.container)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        btn_right = tk.Button(self.container, text=">>", width=3, height=1, command=lambda: self.scroll_by(1))
        btn_right.pack(side=tk.RIGHT, padx=5, pady=5)

        self.canvas.get_tk_widget().bind("<Button-1>", lambda e: self.scroll_by(-1))
        self.canvas.get_tk_widget().bind("<Button-3>", lambda e: self.scroll_by(1))

        self.master.after(200, self.update_chart)

    def scroll_by(self, step):
        self.start_index = max(0, min(self.start_index + step * self.scroll_step, len(self.df_resampled) - self.window_size))
        self.update_chart()

if __name__ == '__main__':
    root = tk.Tk()
    app = TradingChartApp(root)
    root.mainloop()
