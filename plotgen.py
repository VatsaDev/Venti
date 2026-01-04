# plot everything to images, wandb style legends, local

import sys
import os
import duckdb
import pandas as pd
import zlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

matplotlib.use('Agg')

def get_consistent_color(run_name, pool, offset=0):
    """Generates a stable color index based on a CRC32 hash of the run name."""
    seed = zlib.crc32(run_name.encode('utf-8'))
    idx = (seed + offset) % len(pool)
    return pool[idx]

def plot_runs(run_names, db_path="log.db"):
    os.makedirs('plots', exist_ok=True)

    try:
        con = duckdb.connect(db_path, read_only=True)
    except Exception as e:
        print(f"CRITICAL: Connection failed: {e}")
        return

    COLOR_POOL = [
        "#00FFC2", "#FF007F", "#7FFF00", "#FF9D00", "#00D4FF", 
        "#BD00FF", "#FF3B30", "#FFFF00", "#5856D6", "#FF2D55", 
        "#00E5FF", "#AF52DE", "#007AFF", "#70FF00", "#5E5CE6",
        "#FF5E00", "#4CD964", "#FFCC00", "#FF9500", "#5AC8FA"
    ]

    DARK_BG, CARD_BG, GRID_COLOR = '#080808', '#121212', '#282828'
    AXIS_WHITE = '#ffffff'
    TICK_GREY = '#bbbbbb'

    plt.rcParams.update({
        'axes.facecolor': DARK_BG, 'figure.facecolor': DARK_BG,
        'axes.edgecolor': '#444444', 'grid.color': GRID_COLOR,
        'axes.titlecolor': 'white', 'savefig.facecolor': DARK_BG,
        'text.color': 'white', 'axes.labelcolor': AXIS_WHITE,
        'xtick.color': TICK_GREY, 'ytick.color': TICK_GREY,
    })

    groups = [
        {"title": "loss", "cols": ["train_loss", "val_loss"]},
        {"title": "pplx", "cols": ["train_pplx", "val_pplx"]},
        {"title": "lr", "cols": ["lr"]},
        {"title": "bpb", "cols": ["val_bpb"]},
        {"title": "throughput", "cols": ["throughput"]}
    ]

    for g_idx, group in enumerate(groups):
        fig, ax = plt.subplots(figsize=(14, 8))
        
        is_dual = any(m in group["title"] for m in ["loss", "pplx"])
        title_prefix = "train & val / " if is_dual else "train / "
        ax.set_title(f"{title_prefix}{group['title']}", loc='left', pad=45, fontweight='bold', fontsize=24)
        
        ax.grid(True, axis='both', alpha=0.25, linestyle=':')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        has_data = False
        
        for run in run_names:
            try:
                query = f'SELECT step, {", ".join(group["cols"])} FROM "{run}" WHERE step IS NOT NULL ORDER BY step ASC'
                df = con.execute(query).df()
            except Exception: continue

            if df.empty: continue
            has_data = True

            for col_idx, col in enumerate(group["cols"]):
                if col not in df.columns: continue
                
                df_col = df[['step', col]].dropna()
                if df_col.empty: continue
                
                color_offset = (g_idx * 17 + col_idx * 11) 
                line_color = get_consistent_color(run, COLOR_POOL, offset=color_offset)
                
                is_val = "val" in col
                label_suffix = " (val)" if is_val and len(group["cols"]) > 1 else ""
                z_order = 3 if is_val else 4

                # 1. RAW DATA (Fuzz)
                ax.plot(df_col['step'], df_col[col], color=line_color, alpha=0.22, linewidth=1, zorder=z_order-1)

                # 2. MAIN LINE (Smoothing removed entirely)
                y_data = df_col[col]

                # 3. PLOT
                line, = ax.plot(df_col['step'], y_data, color=line_color, linewidth=3.2, zorder=z_order)
                
                last_v, last_s = y_data.iloc[-1], df_col['step'].iloc[-1]
                
                # 4. DOT
                dot_size = 55 if is_val else 90
                ax.scatter(last_s, last_v, color=line_color, s=dot_size, zorder=z_order + 10, edgecolors=DARK_BG, linewidth=1.5)
                
                line.set_label(f"{last_v:.4f} — {run}{label_suffix}")

        if has_data:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x/1000)}k' if x >= 1000 else f'{int(x)}'))
            ax.tick_params(axis='both', which='major', labelsize=12, pad=18)
            
            ax.relim()
            ax.autoscale_view()
            
            leg = ax.legend(
                loc='upper right', frameon=True, facecolor=CARD_BG, edgecolor='#333333', 
                fontsize=11, labelcolor='white', labelspacing=1.8, borderpad=2.2, handletextpad=1.2
            )
            
            safe_run = run_names[0].replace('/', '_').replace('-', '_')
            plt.tight_layout(pad=6.0)
            plt.savefig(f"plots/{safe_run}_{group['title']}.png", dpi=130)
        
        plt.close(fig)
    con.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_runs(sys.argv[1].split(','))
