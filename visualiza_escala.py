
import argparse
import os
from datetime import datetime
from collections import defaultdict

import pandas as pd

HAS_MATPLOTLIB = True
try:
    import matplotlib.pyplot as plt
    import numpy as np
except Exception:
    HAS_MATPLOTLIB = False

def parse_args():
    p = argparse.ArgumentParser(description="Visualizar escala gerada pelo main.py")
    p.add_argument("--csv", default="escala_mes.csv", help="Caminho para o CSV de escala")
    p.add_argument("--outdir", default="visuals", help="Pasta de saída para imagens/relatórios")
    return p.parse_args()

def read_schedule(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "Data" not in df.columns or "Assigned" not in df.columns:
        raise SystemExit("CSV inválido: precisa conter colunas 'Data' e 'Assigned'")
    df["Data"] = pd.to_datetime(df["Data"], format="%Y-%m-%d")
    df["Assigned_list"] = df["Assigned"].astype(str).apply(lambda s: [x.strip() for x in s.split(",") if x.strip()])
    return df

def build_summary(df):
    plantao_dates = defaultdict(list)
    for _, row in df.iterrows():
        for c in row["Assigned_list"]:
            plantao_dates[c].append(row["Data"].date())
    summary = []
    for c, dates in plantao_dates.items():
        summary.append({"Corretor": c, "Plantões": len(dates),
                        "Dias": ";".join(sorted({d.isoformat() for d in dates}))})
    summary_df = pd.DataFrame(summary).sort_values(by="Plantões", ascending=False)
    return summary_df, plantao_dates

def plot_bars(summary_df, outdir):
    if not HAS_MATPLOTLIB:
        print("[skip] matplotlib não disponível — pulando gráfico de barras.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    x = summary_df["Corretor"].tolist()
    y = summary_df["Plantões"].tolist()
    ax.bar(range(len(x)), y)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=45, ha="right")
    ax.set_ylabel("Número de plantões no mês")
    ax.set_title("Plantões por corretor")
    plt.tight_layout()
    out = os.path.join(outdir, "plantoes_por_corretor.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Gráfico barras salvo: {out}")

def plot_gantt(plantao_dates, outdir):
    if not HAS_MATPLOTLIB:
        print("[skip] matplotlib não disponível — pulando Gantt.")
        return
    all_dates = sorted({d for dates in plantao_dates.values() for d in dates})
    if not all_dates:
        print("Sem dados de plantões para Gantt.")
        return
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    corretores = sorted(plantao_dates.keys(), key=lambda c: -len(plantao_dates[c]))
    fig, ax = plt.subplots(figsize=(12, max(6, len(corretores)*0.35)))
    for yi, c in enumerate(corretores):
        xs = [date_to_idx[d] for d in plantao_dates[c]]
        ys = [yi] * len(xs)
        ax.scatter(xs, ys, marker='s', s=50)
    ax.set_yticks(range(len(corretores)))
    ax.set_yticklabels(corretores)
    # reduzir labels no eixo x para não poluir
    if len(all_dates) > 20:
        step = max(1, len(all_dates)//20)
        xticks = list(range(0, len(all_dates), step))
    else:
        xticks = list(range(len(all_dates)))
    ax.set_xticks(xticks)
    ax.set_xticklabels([all_dates[i].isoformat() for i in xticks], rotation=45, ha='right')
    ax.set_xlabel("Data")
    ax.set_title("Mapa de plantões por corretor (pontos = dia com plantão)")
    plt.tight_layout()
    out = os.path.join(outdir, "gantt_plantoes.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Gantt salvo: {out}")

def plot_schedule_table(df, outdir):
    if not HAS_MATPLOTLIB:
        print("[skip] matplotlib não disponível — pulando tabela de escala.")
        return

    df_p = df.copy()
    df_p["Data_str"] = df_p["Data"].dt.date.astype(str)
    pivot = df_p.pivot(index="Data_str", columns="Turno", values="Assigned")
    
    cols = []
    if "Manha" in pivot.columns:
        cols.append("Manha")
    if "Tarde" in pivot.columns:
        cols.append("Tarde")
    
    table_data = pivot[cols].fillna("").reset_index()

    nrows, ncols = table_data.shape
    fig_w = min(20, 0.6 * ncols + 8)
    fig_h = min(20, 0.25 * nrows + 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    col_labels = table_data.columns.tolist()
    cell_text = table_data.values.tolist()

    fontsize = max(6, int(12 - (nrows / 10)))
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='left')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.2)

    plt.tight_layout()
    out = os.path.join(outdir, "tabela_escala.png")
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Tabela da escala salva: {out}")

def main_visualize(csv_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    df = read_schedule(csv_path)
    summary_df, plantao_dates = build_summary(df)
    summary_csv = os.path.join(outdir, "summary_plantao.csv")
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    print(f"Resumo salvo: {summary_csv}")

    plot_bars(summary_df, outdir)
    plot_gantt(plantao_dates, outdir)
    plot_schedule_table(df, outdir)

    if not HAS_MATPLOTLIB:
        print("\nmatplotlib NÃO está instalado — para gerar imagens execute:")
        print("  source .venv/bin/activate   # ou ative seu venv")
        print("  python -m pip install matplotlib numpy pandas")
        print("Depois rode novamente: python visualiza_escala.py --csv escala_mes.csv")
    else:
        print("Visualizações geradas em:", os.path.abspath(outdir))

if __name__ == '__main__':
    args = parse_args()
    main_visualize(args.csv, args.outdir)
