#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — Escalonador mensal usando GLPK (pulp).
Coloque PO_corretores.csv na mesma pasta (colunas: Nome, comissao, imoveis_captados, postagens, novos_leads, vendas_realizadas).
Edite as configurações no topo conforme necessário.
"""

import os
import re
import unicodedata
import calendar
import datetime as dt
import pandas as pd
import pulp

# ------------------ CONFIGURAÇÕES (EDITE AQUI) ------------------
LIMIAR_SENIOR_K = 100          # threshold para senior (em milhares conforme seu CSV)
EXIGIR_IMOVEIS = True         # exigir por turno corretores com imoveis_captados >= LIMIAR_IMOVEIS
LIMIAR_IMOVEIS = 6
EXIGIR_POSTAGENS = True       # exigir por turno corretores com postagens >= LIMIAR_POSTAGENS
LIMIAR_POSTAGENS = 10
EXIGIR_VENDAS = True          # exigir por turno corretores com vendas > LIMIAR_VENDAS
LIMIAR_VENDAS = 4
# Nova regra: leads por turno (obrigatória se EXIGIR_LEADS True)
EXIGIR_LEADS = True
LIMIAR_LEADS = 40             # exige por turno: novos_leads > LIMIAR_LEADS

MIN_PLANTAO_POR_MES = 4
MAX_PLANTAO_POR_MES = 12

GLPSOL_PATH = "glpsol"   # caminho para glpsol (se instalado)
CSV_IN = "PO_corretores.csv"
CSV_OUT = "escala_mes.csv"
# -----------------------------------------------------------------

def sanitize_name(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_only = "".join([c for c in nfkd if not unicodedata.combining(c)])
    safe = re.sub(r'[^A-Za-z0-9]', '_', ascii_only)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe or "anon"

def ler_corretores(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[info] CSV lido com encoding: {enc}")
            break
        except Exception:
            df = None
    if df is None:
        raise SystemExit(f"Erro ao ler {path}")

    if "Nome" not in df.columns or "comissao" not in df.columns:
        raise SystemExit("CSV precisa ter colunas 'Nome' e 'comissao'.")

    df["Nome"] = df["Nome"].astype(str).str.strip()
    df["comissao"] = pd.to_numeric(df["comissao"], errors="coerce").fillna(0)

    for col in ("imoveis_captados", "postagens", "novos_leads", "vendas_realizadas"):
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df

def build_turnos_do_mes(ano: int, mes: int):
    _, ndays = calendar.monthrange(ano, mes)
    turnos = []
    req = {}
    dates_com_turno = []
    for d in range(1, ndays + 1):
        date = dt.date(ano, mes, d)
        wd = date.weekday()  # 0=Seg ... 5=Sab ... 6=Dom
        if wd <= 4:  # Segunda - Sexta -> Manhã e Tarde
            for part in ("Manha", "Tarde"):
                t = (date.isoformat(), part)
                turnos.append(t)
                req[t] = 3
            dates_com_turno.append(date.isoformat())
        elif wd == 5:  # Sábado - apenas Manhã
            t = (date.isoformat(), "Manha")
            turnos.append(t)
            req[t] = 3
            dates_com_turno.append(date.isoformat())
    return turnos, req, sorted(list(set(dates_com_turno)))

def weekday_pt(weekday: int) -> str:
    return ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"][weekday]

def main():
    exigir_imoveis = EXIGIR_IMOVEIS
    exigir_postagens = EXIGIR_POSTAGENS
    exigir_vendas = EXIGIR_VENDAS
    exigir_leads = EXIGIR_LEADS

    # entrada ano/mês
    try:
        ano = int(input("Digite o ano (ex: 2025): ").strip())
        mes = int(input("Digite o mês (1-12): ").strip())
        if mes < 1 or mes > 12:
            raise ValueError
    except Exception:
        raise SystemExit("Entrada inválida. Exemplo: ano=2025, mês=8")

    if not os.path.exists(CSV_IN):
        raise SystemExit(f"Arquivo '{CSV_IN}' não encontrado.")

    df = ler_corretores(CSV_IN)
    corretores = df["Nome"].tolist()
    comissao = dict(zip(df["Nome"], df["comissao"]))
    imoveis = dict(zip(df["Nome"], df["imoveis_captados"]))
    postagens = dict(zip(df["Nome"], df["postagens"]))
    leads = dict(zip(df["Nome"], df["novos_leads"]))
    vendas = dict(zip(df["Nome"], df["vendas_realizadas"]))

    # seniors
    seniors = [n for n, c in comissao.items() if c >= LIMIAR_SENIOR_K]
    if len(seniors) == 0:
        top = max(comissao.items(), key=lambda x: x[1])[0]
        seniors = [top]
        print(f"[aviso] nenhum corretor com comissão >= {LIMIAR_SENIOR_K}k. Marcando '{top}' como senior provisório.")

    # validar requisitos por turno (se for impossível, desligamos apenas as exigências opcionais EXIGIR_*)
    if exigir_vendas:
        tem_vendas = [n for n in corretores if vendas.get(n, 0) > LIMIAR_VENDAS]
        if len(tem_vendas) == 0:
            print(f"[aviso] NÃO existe corretor com vendas_realizadas > {LIMIAR_VENDAS}. Desligando EXIGIR_VENDAS.")
            exigir_vendas = False

    if exigir_postagens:
        tem_post = [n for n in corretores if postagens.get(n, 0) >= LIMIAR_POSTAGENS]
        if len(tem_post) == 0:
            print(f"[aviso] NÃO existe corretor com postagens >= {LIMIAR_POSTAGENS}. Desligando EXIGIR_POSTAGENS.")
            exigir_postagens = False

    if exigir_imoveis:
        tem_imoveis = [n for n in corretores if imoveis.get(n, 0) >= LIMIAR_IMOVEIS]
        if len(tem_imoveis) == 0:
            print(f"[aviso] NÃO existe corretor com imoveis_captados >= {LIMIAR_IMOVEIS}. Desligando EXIGIR_IMOVEIS.")
            exigir_imoveis = False

    # LEADS obrigatório
    if exigir_leads:
        tem_leads = [n for n in corretores if leads.get(n, 0) > LIMIAR_LEADS]
        if len(tem_leads) == 0:
            raise SystemExit(f"[erro] Requisito rígido: NÃO existe nenhum corretor com novos_leads > {LIMIAR_LEADS} no CSV. Ajuste CSV ou LIMIAR_LEADS.")

    # sanitizar nomes
    safe_by_real = {r: sanitize_name(r) for r in corretores}
    real_by_safe = {s: r for r, s in safe_by_real.items()}
    corretores_safe = list(real_by_safe.keys())
    seniors_safe = [safe_by_real[r] for r in seniors if r in safe_by_real]

    # gerar turnos
    turnos, req, datas_sorted = build_turnos_do_mes(ano, mes)
    n_turnos = len(turnos)
    date_to_idxs = {}
    for ti, (date_iso, part) in enumerate(turnos):
        date_to_idxs.setdefault(date_iso, []).append(ti)

    total_needed = sum(req.values())
    max_possible_assignments = len(corretores_safe) * MAX_PLANTAO_POR_MES
    if max_possible_assignments < total_needed:
        raise SystemExit("[erro] Impossível atender todos os plantões com o MAX_PLANTAO_POR_MES atual e número de corretores.")

    # CRIAR MODELO (minimizando o máximo de plantões por corretor)
    prob = pulp.LpProblem("Escalonamento_com_leads_por_turno", pulp.LpMinimize)

    # variáveis binárias
    x = pulp.LpVariable.dicts("x", (corretores_safe, range(n_turnos)), lowBound=0, upBound=1, cat="Binary")

    # variável que representa o máximo de plantões por corretor (nome seguro)
    MaxPlant = pulp.LpVariable("MaxPlant", lowBound=0, upBound=MAX_PLANTAO_POR_MES, cat="Integer")

    # objetivo: minimizar o MaxPlant (minimax)
    prob += MaxPlant

    # demanda por turno
    for ti in range(n_turnos):
        prob += pulp.lpSum([x[b][ti] for b in corretores_safe]) == req[turnos[ti]]

    # pelo menos 1 senior por turno
    for ti in range(n_turnos):
        if seniors_safe:
            prob += pulp.lpSum([x[b][ti] for b in seniors_safe]) >= 1

    # restrições por atributo por turno (se ativadas)
    if exigir_vendas:
        brokers_vendas_safe = [safe_by_real[r] for r in corretores if vendas.get(r, 0) > LIMIAR_VENDAS]
        for ti in range(n_turnos):
            prob += pulp.lpSum([x[b][ti] for b in brokers_vendas_safe]) >= 1

    if exigir_postagens:
        brokers_post_safe = [safe_by_real[r] for r in corretores if postagens.get(r, 0) >= LIMIAR_POSTAGENS]
        for ti in range(n_turnos):
            prob += pulp.lpSum([x[b][ti] for b in brokers_post_safe]) >= 1

    if exigir_imoveis:
        brokers_imoveis_safe = [safe_by_real[r] for r in corretores if imoveis.get(r, 0) >= LIMIAR_IMOVEIS]
        for ti in range(n_turnos):
            prob += pulp.lpSum([x[b][ti] for b in brokers_imoveis_safe]) >= 1

    # LEADS por turno (obrigatório)
    if exigir_leads:
        brokers_leads_safe = [safe_by_real[r] for r in corretores if leads.get(r, 0) > LIMIAR_LEADS]
        for ti in range(n_turnos):
            prob += pulp.lpSum([x[b][ti] for b in brokers_leads_safe]) >= 1

    # min/max por corretor no mês e vínculo com MaxPlant
    for b in corretores_safe:
        prob += pulp.lpSum([x[b][ti] for ti in range(n_turnos)]) >= MIN_PLANTAO_POR_MES
        prob += pulp.lpSum([x[b][ti] for ti in range(n_turnos)]) <= MAX_PLANTAO_POR_MES
        prob += pulp.lpSum([x[b][ti] for ti in range(n_turnos)]) <= MaxPlant

    # não trabalhar manhã+tarde no mesmo dia
    for b in corretores_safe:
        for date_iso, idxs in date_to_idxs.items():
            if len(idxs) > 1:
                prob += pulp.lpSum([x[b][i] for i in idxs]) <= 1

    # não trabalhar dias consecutivos
    for b in corretores_safe:
        for i in range(len(datas_sorted) - 1):
            d0 = datas_sorted[i]
            d1 = datas_sorted[i + 1]
            idxs0 = date_to_idxs.get(d0, [])
            idxs1 = date_to_idxs.get(d1, [])
            if idxs0 and idxs1:
                prob += pulp.lpSum([x[b][ii] for ii in idxs0] + [x[b][ii] for ii in idxs1]) <= 1

    # tentar resolver com GLPK; se falhar, fallback para CBC
    status_str = None
    try:
        solver = pulp.GLPK_CMD(path=GLPSOL_PATH, msg=True)
        status = prob.solve(solver)
        status_str = pulp.LpStatus[status]
    except Exception as e_glpk:
        print("[aviso] GLPK falhou: tentarei CBC (fallback). Mensagem:", str(e_glpk))
        try:
            solver2 = pulp.PULP_CBC_CMD(msg=True)
            status = prob.solve(solver2)
            status_str = pulp.LpStatus[status]
        except Exception as e_cbc:
            print("[erro] Ambos GLPK e CBC falharam:", str(e_cbc))
            return

    print("Status do solver:", status_str)
    if status_str not in ("Optimal", "Feasible"):
        print("[erro] O solver não encontrou solução viável com as restrições atuais.")
        print("Verifique limites MIN/MAX plantões, requisitos rígidos (postagens/vendas/imoveis/leads) e quantidade de corretores.")
        return

    # imprimir valor mínimo de MaxPlant encontrado
    try:
        valM = int(round(pulp.value(MaxPlant)))
        print(f"Máximo de plantões por corretor minimizado (MaxPlant): {valM}")
    except Exception:
        print("Não foi possível recuperar o valor de MaxPlant.")

    # construir saída
    rows = []
    for ti, (date_iso, part) in enumerate(turnos):
        alloc_safe = [b for b in corretores_safe if pulp.value(x[b][ti]) is not None and round(pulp.value(x[b][ti])) == 1]
        alloc_real = [real_by_safe[b] for b in alloc_safe]

        senior_name = None
        juniors = []
        for r in alloc_real:
            if r in seniors and senior_name is None:
                senior_name = r
            else:
                juniors.append(r)
        if senior_name is None and alloc_real:
            senior_name = max(alloc_real, key=lambda n: comissao.get(n, 0))
            juniors = [n for n in alloc_real if n != senior_name]

        d = dt.date.fromisoformat(date_iso)
        rows.append({
            "Data": date_iso,
            "DiaSemana": weekday_pt(d.weekday()),
            "Turno": part,
            "Senior": senior_name or "",
            "Juniores": ", ".join(juniors),
            "Assigned": ", ".join(alloc_real)
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print(f"Escala salva em '{CSV_OUT}'.")

    # resumo por corretor (inclui leads agora)
    summary = []
    for b in corretores_safe:
        assigned_val = int(round(sum([pulp.value(x[b][ti]) or 0 for ti in range(n_turnos)])))
        summary.append({
            "Corretor": real_by_safe[b],
            "Plantões": assigned_val,
            "Comissao": comissao.get(real_by_safe[b], 0),
            "Imoveis": imoveis.get(real_by_safe[b], 0),
            "Postagens": postagens.get(real_by_safe[b], 0),
            "Leads": leads.get(real_by_safe[b], 0),
            "Vendas": vendas.get(real_by_safe[b], 0)
        })
    summary_df = pd.DataFrame(summary).sort_values(by="Plantões", ascending=False)
    print("\nResumo por corretor (Plantões no mês):")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
