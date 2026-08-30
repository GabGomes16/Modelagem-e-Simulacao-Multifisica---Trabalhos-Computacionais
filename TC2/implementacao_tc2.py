"""Funções usadas nas atividades do Trabalho Computacional 02.

O notebook fornecido pelo professor é mantido como material de referência.
Este arquivo reúne apenas rotinas pequenas que são chamadas nas células do
trabalho, evitando repetir leitura de áudio, integração e construção de
gráficos.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.signal import resample_poly


PASTA_FIGURAS = Path("figuras_relatorio")
PASTA_FIGURAS.mkdir(exist_ok=True)


def carregar_audio(nome_arquivo="TC02-in.wav", tensao_pico=1.99, taxa_alvo=12000):
    """Lê o WAV, converte para mono, reamostra e limita o pico de tensão."""
    caminho = Path(nome_arquivo)
    if not caminho.exists() and caminho.name == "TC02-in.wav":
        caminho = Path("TC2_in.wav")

    taxa_original, dados = wavfile.read(caminho)
    dados = dados.astype(float)

    if dados.ndim == 2:
        dados = dados.mean(axis=1)

    maior_amplitude = np.max(np.abs(dados))
    if maior_amplitude == 0:
        raise ValueError("O arquivo de áudio não possui sinal.")
    entrada = tensao_pico * dados / maior_amplitude

    if taxa_alvo is not None and taxa_alvo != taxa_original:
        divisor = np.gcd(taxa_original, taxa_alvo)
        entrada = resample_poly(
            entrada,
            taxa_alvo // divisor,
            taxa_original // divisor,
        )
        taxa = taxa_alvo
    else:
        taxa = taxa_original

    tempo = np.arange(len(entrada)) / taxa
    return taxa, tempo, entrada


def simular_modelo_linear(tempo, entrada, taxa, m, b, k, Bl, L, R):
    """Resolve o modelo linear e devolve corrente, posição, velocidade e aceleração."""
    entrada_t = interp1d(
        tempo,
        entrada,
        bounds_error=False,
        fill_value=(entrada[0], entrada[-1]),
    )

    def equacoes(t, estado):
        corrente, posicao, velocidade = estado
        tensao = float(entrada_t(t))
        return [
            (-R * corrente - Bl * velocidade + tensao) / L,
            velocidade,
            (Bl * corrente - k * posicao - b * velocidade) / m,
        ]

    solucao = solve_ivp(
        equacoes,
        (tempo[0], tempo[-1]),
        (0.0, 0.0, 0.0),
        t_eval=tempo,
        max_step=1 / taxa,
        rtol=1e-6,
        atol=1e-9,
    )
    if not solucao.success:
        raise RuntimeError(solucao.message)

    corrente, posicao, velocidade = solucao.y
    aceleracao = (Bl * corrente - k * posicao - b * velocidade) / m
    return {
        "tempo": solucao.t,
        "corrente": corrente,
        "posicao": posicao,
        "velocidade": velocidade,
        "aceleracao": aceleracao,
    }


def criar_fator_forca(Bl, x1, x2):
    """Cria Bl(x) com o decaimento quadrático indicado na figura da Aula 07."""
    if not 0 < x1 < x2:
        raise ValueError("É necessário ter 0 < x1 < x2.")

    def fator_forca(posicao):
        modulo = np.abs(posicao)
        transicao = Bl * ((x2 - modulo) / (x2 - x1)) ** 2
        return np.where(modulo <= x1, Bl, np.where(modulo >= x2, 0.0, transicao))

    return fator_forca


def simular_modelo_nao_linear(tempo, entrada, taxa, m, b, k, fator_forca, L, R):
    """Resolve o modelo no qual o fator de força depende da posição do cone."""
    entrada_t = interp1d(
        tempo,
        entrada,
        bounds_error=False,
        fill_value=(entrada[0], entrada[-1]),
    )

    def equacoes(t, estado):
        corrente, posicao, velocidade = estado
        Bl_x = float(fator_forca(posicao))
        tensao = float(entrada_t(t))
        return [
            (-R * corrente - Bl_x * velocidade + tensao) / L,
            velocidade,
            (Bl_x * corrente - k * posicao - b * velocidade) / m,
        ]

    solucao = solve_ivp(
        equacoes,
        (tempo[0], tempo[-1]),
        (0.0, 0.0, 0.0),
        t_eval=tempo,
        max_step=1 / taxa,
        rtol=1e-6,
        atol=1e-9,
    )
    if not solucao.success:
        raise RuntimeError(solucao.message)

    corrente, posicao, velocidade = solucao.y
    Bl_t = fator_forca(posicao)
    aceleracao = (Bl_t * corrente - k * posicao - b * velocidade) / m
    return {
        "tempo": solucao.t,
        "corrente": corrente,
        "posicao": posicao,
        "velocidade": velocidade,
        "aceleracao": aceleracao,
        "Bl": Bl_t,
    }


def espectro(sinal, taxa, janela=True):
    """Calcula o espectro unilateral normalizado em decibéis."""
    sinal = np.asarray(sinal) - np.mean(sinal)
    pesos = np.hanning(len(sinal)) if janela else np.ones(len(sinal))
    frequencias = np.fft.rfftfreq(len(sinal), d=1 / taxa)
    magnitude = np.abs(np.fft.rfft(sinal * pesos))
    if np.max(magnitude) > 0:
        magnitude = magnitude / np.max(magnitude)
    magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-8))
    return frequencias, magnitude_db


def salvar_wav(nome_arquivo, taxa, sinal):
    """Normaliza um sinal para a faixa [-1, 1] e o salva como WAV mono."""
    sinal = np.asarray(sinal, dtype=float)
    pico = np.max(np.abs(sinal))
    if pico == 0:
        raise ValueError("Não é possível salvar um sinal nulo.")
    wavfile.write(nome_arquivo, taxa, (sinal / pico).astype(np.float32))


def _amostrar_para_grafico(*sinais, maximo=6000):
    passo = max(1, len(sinais[0]) // maximo)
    return [sinal[::passo] for sinal in sinais]


def figura_entrada(tempo, entrada, taxa):
    t, u = _amostrar_para_grafico(tempo, entrada)
    frequencias, magnitude = espectro(entrada, taxa)

    fig, eixos = plt.subplots(2, 1, figsize=(10, 7))
    eixos[0].plot(t, u, linewidth=0.8)
    eixos[0].set(xlabel="Tempo [s]", ylabel="$V_{in}$ [V]", title="Sinal de entrada no tempo")
    eixos[0].grid(True, alpha=0.3)
    eixos[1].semilogx(frequencias[1:], magnitude[1:], linewidth=0.9)
    eixos[1].set(xlabel="Frequência [Hz]", ylabel="Magnitude normalizada [dB]", title="Espectro do sinal de entrada", xlim=(20, taxa / 2), ylim=(-100, 5))
    eixos[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / "fig01_entrada.png", dpi=180, bbox_inches="tight")
    plt.show()


def figura_estados(resposta, nome, titulo):
    t, i, x, a = _amostrar_para_grafico(
        resposta["tempo"], resposta["corrente"], resposta["posicao"], resposta["aceleracao"]
    )
    fig, eixos = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    eixos[0].plot(t, i, linewidth=0.8)
    eixos[0].set_ylabel("Corrente [A]")
    eixos[1].plot(t, x, linewidth=0.8)
    eixos[1].set_ylabel("Posição [m]")
    eixos[2].plot(t, a, linewidth=0.8)
    eixos[2].set(xlabel="Tempo [s]", ylabel="Aceleração [m/s²]")
    for eixo in eixos:
        eixo.grid(True, alpha=0.3)
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / nome, dpi=180, bbox_inches="tight")
    plt.show()


def figura_comparacao_entrada(tempo, entrada, aceleracao, taxa):
    t, u, a = _amostrar_para_grafico(tempo, entrada, aceleracao)
    u = u / np.max(np.abs(u))
    a = a / np.max(np.abs(a))
    f_u, U = espectro(entrada, taxa)
    f_a, A = espectro(aceleracao, taxa)

    fig, eixos = plt.subplots(2, 1, figsize=(10, 7))
    eixos[0].plot(t, u, label="$V_{in}(t)$", linewidth=0.8)
    eixos[0].plot(t, a, label="$\\ddot{x}(t)$", linewidth=0.8, alpha=0.8)
    eixos[0].set(xlabel="Tempo [s]", ylabel="Amplitude normalizada", title="Comparação no domínio do tempo")
    eixos[0].legend()
    eixos[1].semilogx(f_u[1:], U[1:], label="$V_{in}(j\\omega)$", linewidth=0.9)
    eixos[1].semilogx(f_a[1:], A[1:], label="$\\ddot{x}(j\\omega)$", linewidth=0.9)
    eixos[1].set(xlabel="Frequência [Hz]", ylabel="Magnitude normalizada [dB]", title="Comparação no domínio da frequência", xlim=(20, taxa / 2), ylim=(-100, 5))
    eixos[1].legend()
    for eixo in eixos:
        eixo.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / "fig03_comparacao_linear.png", dpi=180, bbox_inches="tight")
    plt.show()


def figura_fator_forca(fator_forca, Bl, x1, x2):
    posicao = np.linspace(-1.15 * x2, 1.15 * x2, 1200)
    fig, eixo = plt.subplots(figsize=(9, 4.5))
    eixo.plot(posicao * 1e3, fator_forca(posicao), linewidth=2)
    for limite in (-x2, -x1, x1, x2):
        eixo.axvline(limite * 1e3, color="gray", linestyle="--", linewidth=0.8)
    eixo.set(xlabel="Posição do cone [mm]", ylabel="$Bl(x)$ [N/A]", title="Fator de força não linear")
    eixo.set_ylim(-0.05 * Bl, 1.1 * Bl)
    eixo.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / "fig04_fator_forca.png", dpi=180, bbox_inches="tight")
    plt.show()


def figura_comparacao_audio(linear, nao_linear, taxa):
    t, a_l, a_n, bl = _amostrar_para_grafico(
        linear["tempo"], linear["aceleracao"], nao_linear["aceleracao"], nao_linear["Bl"]
    )
    f_l, A_l = espectro(linear["aceleracao"], taxa)
    f_n, A_n = espectro(nao_linear["aceleracao"], taxa)

    fig, eixos = plt.subplots(3, 1, figsize=(10, 9))
    eixos[0].plot(t, a_l, label="Linear", linewidth=0.8)
    eixos[0].plot(t, a_n, label="Não linear", linewidth=0.8, alpha=0.8)
    eixos[0].set(xlabel="Tempo [s]", ylabel="Aceleração [m/s²]", title="Resposta ao áudio")
    eixos[0].legend()
    eixos[1].plot(t, bl, linewidth=0.8)
    eixos[1].set(xlabel="Tempo [s]", ylabel="$Bl(x)$ [N/A]", title="Fator de força durante a resposta")
    eixos[2].semilogx(f_l[1:], A_l[1:], label="Linear", linewidth=0.9)
    eixos[2].semilogx(f_n[1:], A_n[1:], label="Não linear", linewidth=0.9)
    eixos[2].set(xlabel="Frequência [Hz]", ylabel="Magnitude normalizada [dB]", title="Espectros das acelerações", xlim=(20, taxa / 2), ylim=(-100, 5))
    eixos[2].legend()
    for eixo in eixos:
        eixo.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / "fig06_comparacao_audio.png", dpi=180, bbox_inches="tight")
    plt.show()


def erro_periodicidade_por_ciclo(sinal, amostras_periodo, atraso_periodos=1):
    """Calcula o erro RMS relativo entre ciclos separados por um dado atraso."""
    numero_ciclos = len(sinal) // amostras_periodo
    erros = []
    for ciclo in range(atraso_periodos, numero_ciclos):
        inicio_atual = ciclo * amostras_periodo
        inicio_anterior = (ciclo - atraso_periodos) * amostras_periodo
        atual = sinal[inicio_atual:inicio_atual + amostras_periodo]
        anterior = sinal[inicio_anterior:inicio_anterior + amostras_periodo]
        referencia = np.sqrt(np.mean(atual ** 2))
        erro = np.sqrt(np.mean((atual - anterior) ** 2)) / referencia
        erros.append(100 * erro)
    return np.asarray(erros)


def figura_senoide(entrada, linear, nao_linear, taxa, frequencia, frequencia_natural=None):
    tempo = linear["tempo"]
    periodo = 1 / frequencia
    amostras_periodo = int(round(taxa / frequencia))
    inicio_transitorio = tempo <= 5 * periodo
    trecho_final = tempo >= tempo[-1] - 5 * periodo

    fig, eixos = plt.subplots(3, 2, figsize=(12, 9), sharex="col")
    grandezas = (("corrente", "Corrente [A]"), ("posicao", "Posição [m]"), ("aceleracao", "Aceleração [m/s²]"))
    for linha, (chave, rotulo) in enumerate(grandezas):
        eixos[linha, 0].plot(tempo[inicio_transitorio], linear[chave][inicio_transitorio], label="Linear")
        eixos[linha, 0].plot(tempo[inicio_transitorio], nao_linear[chave][inicio_transitorio], "--", label="Não linear")
        eixos[linha, 1].plot(tempo[trecho_final], linear[chave][trecho_final], label="Linear")
        eixos[linha, 1].plot(tempo[trecho_final], nao_linear[chave][trecho_final], "--", label="Não linear")
        eixos[linha, 0].set_ylabel(rotulo)
        eixos[linha, 0].grid(True, alpha=0.3)
        eixos[linha, 1].grid(True, alpha=0.3)
    eixos[0, 0].set_title("Primeiros cinco períodos")
    eixos[0, 1].set_title("Últimos cinco períodos da simulação")
    eixos[0, 0].legend()
    eixos[0, 1].legend()
    eixos[2, 0].set_xlabel("Tempo [s]")
    eixos[2, 1].set_xlabel("Tempo [s]")
    fig.suptitle(f"Comparação para a entrada senoidal de {frequencia:.0f} Hz")
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / "fig07_senoide_tempo.png", dpi=180, bbox_inches="tight")
    plt.show()

    erro_linear_1T = erro_periodicidade_por_ciclo(
        linear["aceleracao"], amostras_periodo, atraso_periodos=1
    )
    erro_nao_linear_1T = erro_periodicidade_por_ciclo(
        nao_linear["aceleracao"], amostras_periodo, atraso_periodos=1
    )
    erro_nao_linear_3T = erro_periodicidade_por_ciclo(
        nao_linear["aceleracao"], amostras_periodo, atraso_periodos=3
    )

    fig, eixo = plt.subplots(figsize=(9, 4.8))
    eixo.semilogy(
        np.arange(2, len(erro_linear_1T) + 2),
        np.maximum(erro_linear_1T, 1e-6),
        label="Linear: comparação com 1T",
    )
    eixo.semilogy(
        np.arange(2, len(erro_nao_linear_1T) + 2),
        np.maximum(erro_nao_linear_1T, 1e-6),
        label="Não linear: comparação com 1T",
    )
    eixo.semilogy(
        np.arange(4, len(erro_nao_linear_3T) + 4),
        np.maximum(erro_nao_linear_3T, 1e-6),
        label="Não linear: comparação com 3T",
    )
    eixo.set(
        xlabel="Número do ciclo da entrada",
        ylabel="Erro RMS relativo [%]",
        title="Convergência da resposta periódica",
        xlim=(1, len(tempo) // amostras_periodo),
    )
    eixo.grid(True, which="both", alpha=0.3)
    eixo.legend()
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / "fig08_convergencia_senoide.png", dpi=180, bbox_inches="tight")
    plt.show()

    periodos_fft = min(60, len(tempo) // amostras_periodo)
    trecho_fft = slice(-periodos_fft * amostras_periodo, None)
    f_l, A_l = espectro(linear["aceleracao"][trecho_fft], taxa, janela=False)
    f_n, A_n = espectro(nao_linear["aceleracao"][trecho_fft], taxa, janela=False)
    fig, eixo = plt.subplots(figsize=(9, 4.5))
    eixo.plot(f_l, A_l, label="Linear")
    eixo.plot(f_n, A_n, label="Não linear", alpha=0.8)
    eixo.axvline(frequencia / 3, color="gray", linestyle=":", linewidth=1, label="$f_0/3$")
    eixo.axvline(frequencia, color="black", linestyle=":", linewidth=1, label="$f_0$")
    if frequencia_natural is not None:
        eixo.axvline(
            frequencia_natural,
            color="tab:green",
            linestyle="--",
            linewidth=1,
            label="$f_n$ mecânica",
        )
    eixo.set(
        xlabel="Frequência [Hz]",
        ylabel="Magnitude normalizada [dB]",
        title=f"Espectro da aceleração nos últimos {periodos_fft} períodos",
        xlim=(0, 8 * frequencia),
        ylim=(-100, 5),
    )
    eixo.grid(True, alpha=0.3)
    eixo.legend()
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / "fig09_senoide_espectro.png", dpi=180, bbox_inches="tight")
    plt.show()

    indice_dominante = np.argmax(A_n[1:]) + 1
    erros_finais_nao_lineares = erro_nao_linear_1T[-20:]
    return {
        "erro_linear_1T_percent": float(np.mean(erro_linear_1T[-20:])),
        "erro_nao_linear_1T_percent": float(np.mean(erros_finais_nao_lineares)),
        "erro_nao_linear_1T_min_percent": float(np.min(erros_finais_nao_lineares)),
        "erro_nao_linear_1T_max_percent": float(np.max(erros_finais_nao_lineares)),
        "erro_nao_linear_3T_percent": float(np.mean(erro_nao_linear_3T[-20:])),
        "frequencia_dominante_nao_linear_hz": float(f_n[indice_dominante]),
        "periodos_fft": int(periodos_fft),
    }


def amplitude_em_frequencia(sinal, taxa, frequencia):
    """Obtém a amplitude da raia mais próxima de uma frequência."""
    frequencias = np.fft.rfftfreq(len(sinal), d=1 / taxa)
    amplitudes = np.abs(np.fft.rfft(sinal - np.mean(sinal)))
    indice = np.argmin(np.abs(frequencias - frequencia))
    return amplitudes[indice]


def salvar_resumo(dados):
    Path("resultados_tc2.json").write_text(
        json.dumps(dados, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
