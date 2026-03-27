"""
sweep/orchestrator.py
=====================
Orquestrador de varredura de parâmetros (sweep).

Lê sweep_config.json (ou qualquer JSON no formato base+runs),
executa run_simulation() para cada run e salva um resumo consolidado.

Formato do JSON de entrada
--------------------------
{
  "sweep_name": "meu_sweep",
  "base": {
      "Re": 100, "N_int": 2000, ...   <- parâmetros comuns a todos os runs
  },
  "runs": [
      {"run_id": "run_01", "Re": 100, ...},  <- overrides por run
      {"run_id": "run_02", "Re": 400, ...}
  ]
}

Cada run sobrescreve apenas os campos que define; o restante vem de "base".

Funções públicas
----------------
    carregar_config(caminho) -> dict
        Lê e retorna o JSON de configuração.

    montar_params(base, run) -> dict
        Mescla base com overrides do run.

    run_sweep(config, run_simulation_fn) -> list[dict]
        Executa todos os runs e retorna a lista de resultados.
        run_simulation_fn é passado externamente para evitar import circular.
"""

import json
import time
from pathlib import Path


# ------------------------------------------------------------------------------
# Leitura do arquivo de configuração
# ------------------------------------------------------------------------------
def carregar_config(caminho: str) -> dict:
    """Lê e retorna o dicionário do JSON de configuração."""
    with open(caminho, "r") as f:
        return json.load(f)


# ------------------------------------------------------------------------------
# Mescla parâmetros base + overrides do run
# ------------------------------------------------------------------------------
def montar_params(base: dict, run: dict) -> dict:
    """
    Retorna um novo dicionário com os parâmetros de `base` sobrescritos
    pelos campos definidos em `run`.
    """
    params = base.copy()
    params.update(run)
    return params


# ------------------------------------------------------------------------------
# Execução do sweep completo
# ------------------------------------------------------------------------------
def run_sweep(config: dict, run_simulation_fn) -> list:
    """
    Executa todos os runs definidos em config["runs"].

    Parâmetros
    ----------
    config             : dicionário carregado do JSON (com "base" e "runs")
    run_simulation_fn  : função run_simulation(params) -> dict
                         passada pelo main.py para evitar import circular

    Retorna
    -------
    Lista de dicts com resultado de cada run (run_id, status, métricas, tempo).
    """
    sweep_name = config.get("sweep_name", "sweep")
    base       = config["base"]
    runs       = config["runs"]
    n_total    = len(runs)

    raiz = Path(sweep_name)
    raiz.mkdir(exist_ok=True)

    # Salva cópia do JSON usado (rastreabilidade)
    with open(raiz / "sweep_config_usado.json", "w") as f:
        json.dump(config, f, indent=4)

    print(f"\n{'='*60}")
    print(f"  Varredura : {sweep_name}")
    print(f"  Total runs: {n_total}")
    print(f"  Saida em  : {raiz}/")
    print(f"{'='*60}\n")

    resultados      = []
    tempo_varredura = time.time()

    for i, run in enumerate(runs, start=1):
        run_id = run.get("run_id", f"run_{i:02d}")
        params = montar_params(base, run)
        params["output_dir"] = str(raiz / run_id)

        # Linha de status no terminal
        mag_info = (
            f"H0={params.get('H0')} | chi={params.get('chi')} | "
            if params.get("magnetic_case") else ""
        )
        print(
            f"[{i}/{n_total}] {run_id} | "
            f"{mag_info}"
            f"w_f={params.get('w_f')} | w_u_top={params.get('w_u_top')} | "
            f"w_u_rest={params.get('w_u_rest')} | w_v={params.get('w_v')}"
        )

        t0 = time.time()
        try:
            resultado               = run_simulation_fn(params)
            resultado["run_id"]     = run_id
            resultado["status"]     = "ok"
            resultado["tempo_run_s"] = round(time.time() - t0, 2)
            print(f"    ok — {resultado['tempo_run_s']:.1f}s\n")
        except Exception as e:
            resultado = {
                "run_id":      run_id,
                "status":      "erro",
                "mensagem":    str(e),
                "tempo_run_s": round(time.time() - t0, 2),
            }
            print(f"    erro em {run_id}: {e}\n")

        resultados.append(resultado)

    tempo_total = time.time() - tempo_varredura

    # Resumo consolidado
    with open(raiz / "resumo_varredura.json", "w") as f:
        json.dump({
            "sweep_name":    sweep_name,
            "n_runs":        n_total,
            "tempo_total_s": round(tempo_total, 2),
            "runs":          resultados,
        }, f, indent=4)

    print(f"{'='*60}")
    print(f"  Varredura concluida em {tempo_total:.1f}s")
    print(f"  Resumo: {raiz}/resumo_varredura.json")
    print(f"{'='*60}\n")

    return resultados
