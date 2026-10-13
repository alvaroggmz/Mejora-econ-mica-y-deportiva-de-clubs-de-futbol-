import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CARGA DE DATOS
# =============================================================================

try:
    df_clubes    = pd.read_csv('finanzas.csv',       encoding='latin-1', sep=';', decimal=',', on_bad_lines='skip')
    df_jugadores = pd.read_csv('jugadores24-25.csv', encoding='utf-8-sig')
except FileNotFoundError:
    print("Error: no se encuentran los archivos CSV. Ponlos en la misma carpeta que este script.")
    exit()

# Eliminar filas vacías al final del CSV de clubes
df_clubes = df_clubes.dropna(subset=['EQUIPO'])

# -----------------------------------------------------------------------------
# Limpieza y renombrado del CSV de clubes
# IMPORTANTE: todos los valores económicos del nuevo CSV ya vienen en millones de euros
# -----------------------------------------------------------------------------

# El nombre de la columna 'SALARIOS TOTAL' tiene un espacio al final, lo quitamos
df_clubes.columns = df_clubes.columns.str.strip()

df_clubes = df_clubes.rename(columns={
    'EQUIPO':                     'Nombre del club',
    'INGRESOS':                   'ingresos totales',
    'COMPETICIONES':              'ingresos competiciones',
    'RETRANSMISIÓN':              'ingresos TV',
    'PUBLICIDAD':                 'ingresos comerciales',
    'ABONADOS':                   'ingresos matchday',
    'I.FICHAJES':                 'ingresos fichajes',
    'GASTOS':                     'gastos totales',
    'G.FICHAJES':                 'gasto en fichajes',
    'SALARIOS PD':                'salarios plantilla deportiva',
    'SALARIOS PND':               'salarios plantilla no deportiva',
    'SALARIOS TOTAL':             'masa salarial',
    'INVERSION EN CANTERA':       'inversion en cantera',
    'VALOR DE MERCADO PLANTILLA': 'valor de mercado de la plantilla',
})

# Columnas derivadas que usa el diagnóstico y el módulo financiero
df_clubes['balance traspasos'] = (
    df_clubes['ingresos fichajes'].fillna(0) - df_clubes['gasto en fichajes'].fillna(0)
)
df_clubes['balance operativo'] = df_clubes['ingresos totales'] - df_clubes['gastos totales']

# Posición real en la tabla de LaLiga 24/25 (dato directo del CSV)
df_clubes['posicion_liga'] = df_clubes['POSICION LIGA'].fillna(20).round(0).astype(int)

# -----------------------------------------------------------------------------
# Limpieza y renombrado del CSV de jugadores
# -----------------------------------------------------------------------------

df_jugadores = df_jugadores.rename(columns={
    'Jugador':                      'Nombre',
    'Minutos':                      'Minutos jugados',
    'Goles':                        'Goles marcados',
    'Asistencias':                  'Asistencias de gol',
    'Entradas ganadas':             'Entradas con exito',
    'Pases completados':            'Pases',
    'Goles esperados':              'xG',
    'Asistencias esperadas':        'xA',
    'Pases progresivos':            'Pases progresivos',
    'Acciones de creación de tiro': 'Acciones tiro',
    'Intercepciones':               'Intercepciones',
    'Valor de Mercado (M€)':        'VM',
    'Coste de Traspaso (M€)':       'Coste traspaso',
    '% paradas':                    'Pct paradas',
    'Goles encajados por 90':       'Goles encajados por 90',
})

for col in ['Minutos jugados', 'Goles marcados', 'Asistencias de gol',
            'Entradas con exito', 'Recuperaciones', 'Pases', 'Edad',
            'xG', 'xA', 'Pases progresivos', 'Acciones tiro', 'Intercepciones',
            'VM', 'Coste traspaso',
            'Paradas', 'Pct paradas', 'Goles encajados', 'Goles encajados por 90',
            'Porterías a cero', 'PSxG +/-']:
    df_jugadores[col] = pd.to_numeric(df_jugadores[col], errors='coerce').fillna(0)

# El nuevo CSV usa códigos ingleses (GK, DF, MF, FW) -> mapeamos a esquema interno
def mapear_posicion(pos):
    if pd.isna(pos):
        return np.nan
    primera = str(pos).strip().upper().split(',')[0]
    return {'GK': 'POR', 'DF': 'DEF', 'MF': 'MED', 'FW': 'DEL'}.get(primera, np.nan)

df_jugadores['Pos'] = df_jugadores['Posición'].apply(mapear_posicion)


# =============================================================================
# NIVEL DE CLUB (para que los fichajes sean coherentes con el tamaño del equipo)
# Umbrales ajustados a los valores reales del nuevo CSV (en M€)
# =============================================================================

NIVEL_CLUB = {}
for _, row in df_clubes.iterrows():
    ing = row['ingresos totales']   # en M€
    if ing >= 300:
        nivel = 3   # Grande: Real Madrid, Barça, Atlético
    elif ing >= 100:
        nivel = 2   # Medio-alto: Athletic, Villarreal, Sevilla...
    else:
        nivel = 1   # Pequeño: Leganés, Alavés, Valladolid...
    NIVEL_CLUB[row['Nombre del club']] = nivel


# =============================================================================
# PREPARACION DE DATOS
# =============================================================================

def preparar_datos():
    resultado = pd.DataFrame()

    # -------------------------------------------------------------------------
    # FÓRMULA DE RENDIMIENTO POR POSICIÓN
    # Cada posición valora lo que realmente importa en ese rol.
    # La edad actúa como factor corrector igual para todos (±0 a los 24 años).
    # Rango final: 0.5 – 10
    # -------------------------------------------------------------------------

    # Normalizadores globales (evitan que un outlier infle el score)
    max_min   = df_jugadores['Minutos jugados'].max() or 1
    max_rec   = df_jugadores['Recuperaciones'].max() or 1
    max_pas   = df_jugadores['Pases'].max() or 1
    max_prog  = df_jugadores['Pases progresivos'].max() or 1
    max_int   = df_jugadores['Intercepciones'].max() or 1
    max_par   = df_jugadores['Paradas'].max() or 1
    max_xg    = df_jugadores['xG'].max() or 1
    max_xa    = df_jugadores['xA'].max() or 1
    max_acc   = df_jugadores['Acciones tiro'].max() or 1

    def rend_por_posicion(row):
        pos     = row['Pos']
        minutos = row['Minutos jugados'] / max_min

        # Factor de edad: peak a los 24 años, penaliza +/- alejamiento
        # Rango aproximado: -0.5 (muy joven o muy mayor) a 0 (peak)
        factor_edad = -abs(row['Edad'] - 24) * 0.04

        if pos == 'POR':
            # Porteros: lo que importa es parar, encajar poco y aportar con el balón
            # PSxG+/- mide si para más o menos de lo esperado (clave)
            score = (
                minutos                                            * 2.0 +   # participación
                (row['Paradas'] / max_par)                         * 2.5 +   # volumen de paradas
                (row['Pct paradas'] / 100)                         * 1.5 +   # efectividad (0-1)
                max(0, -row['Goles encajados por 90'] + 2.0) / 2.0 * 1.0 +  # pocos goles encajados
                (row['PSxG +/-'] + 5) / 10                        * 1.5 +   # rendimiento vs xG
                (row['Porterías a cero'] / max(df_jugadores['Porterías a cero'].max(), 1)) * 1.0
            )

        elif pos == 'DEF':
            # Defensas: prioridad defensiva, pero también juego con balón
            score = (
                minutos                                            * 2.0 +   # participación
                (row['Intercepciones'] / max_int)                  * 1.8 +   # lectura defensiva
                (row['Recuperaciones'] / max_rec)                  * 1.5 +   # recuperación de balón
                (row['Entradas con exito'] /
                 (df_jugadores['Entradas con exito'].max() or 1))  * 1.2 +   # entradas exitosas
                (row['Pases'] / max_pas)                           * 1.0 +   # juego con balón
                (row['Pases progresivos'] / max_prog)              * 0.8 +   # salida de balón
                (row['xA'] / max_xa if max_xa else 0)              * 0.5 +   # aporte ofensivo (bonus)
                (row['% duelos aéreos ganados'] / 100)             * 0.7     # dominio aéreo
            )

        elif pos == 'MED':
            # Mediocampistas: equilibrio entre defensa, pase y creación
            score = (
                minutos                                            * 2.0 +   # participación
                (row['Pases'] / max_pas)                           * 1.5 +   # volumen de pase
                (row['Pases progresivos'] / max_prog)              * 1.2 +   # progresión
                (row['Acciones tiro'] / max_acc)                   * 1.0 +   # creación de tiro
                (row['xA'] / max_xa if max_xa else 0)              * 1.0 +   # creatividad
                (row['Recuperaciones'] / max_rec)                  * 0.8 +   # trabajo defensivo
                (row['Intercepciones'] / max_int)                  * 0.8 +   # lectura del juego
                (row['xG'] / max_xg if max_xg else 0)              * 0.5     # llegada al gol
            )

        elif pos == 'DEL':
            # Delanteros: gol y creación ofensiva son lo primero
            score = (
                minutos                                            * 1.5 +   # participación
                (row['xG'] / max_xg if max_xg else 0)              * 2.5 +   # peligro de gol
                (row['xA'] / max_xa if max_xa else 0)              * 1.5 +   # creación
                (row['Acciones tiro'] / max_acc)                   * 1.2 +   # generación de tiros
                (row['Pases progresivos'] / max_prog)              * 0.8 +   # progresión
                (row['Recuperaciones'] / max_rec)                  * 0.5     # presión alta
            )

        else:
            # Posición desconocida: fórmula genérica
            score = (
                minutos                              * 2.5 +
                (row['xG'] / max_xg if max_xg else 0) * 1.0 +
                (row['xA'] / max_xa if max_xa else 0) * 1.0 +
                (row['Recuperaciones'] / max_rec)    * 0.8 +
                (row['Pases'] / max_pas)             * 0.8
            )

        return max(0.5, min(10.0, round(score + factor_edad, 2)))

    df_jugadores['Rendimiento'] = df_jugadores.apply(rend_por_posicion, axis=1)

    for equipo in df_clubes['Nombre del club']:
        jugadores_equipo = df_jugadores[df_jugadores['Equipo'] == equipo].copy()
        if jugadores_equipo.empty:
            continue

        datos_club = df_clubes[df_clubes['Nombre del club'] == equipo].iloc[0]

        # Salario estimado: proporcional al rendimiento dentro de la masa salarial
        # La masa salarial en el CSV está en M€, la pasamos a euros para el salario individual
        masa_euros  = datos_club['masa salarial'] * 1_000_000
        suma_rend   = jugadores_equipo['Rendimiento'].sum() or 1
        jugadores_equipo['Salario'] = (
            jugadores_equipo['Rendimiento'] / suma_rend
        ) * masa_euros

        # Valor de mercado: usamos el VM real del CSV (ya en M€), lo convertimos a euros
        jugadores_equipo['Valor de mercado'] = jugadores_equipo['VM'] * 1_000_000

        # Score de venta (0-100%): penaliza edad, bajo rendimiento y salario elevado
        salario_medio = jugadores_equipo['Salario'].mean() or 1
        jugadores_equipo['Score_venta'] = (
            (jugadores_equipo['Edad'] - 24).clip(0) * 4 +
            (9 - jugadores_equipo['Rendimiento']) * 8 +
            ((jugadores_equipo['Salario'] / salario_medio) - 1).clip(0) * 15
        )
        score_max = jugadores_equipo['Score_venta'].max()
        if score_max > 0:
            jugadores_equipo['Pct_venta'] = (
                jugadores_equipo['Score_venta'] / score_max * 100
            ).clip(0, 100).round(1)
        else:
            jugadores_equipo['Pct_venta'] = 0

        resultado = pd.concat([resultado, jugadores_equipo], ignore_index=True)

    return resultado


df_jugadores = preparar_datos()


# =============================================================================
# DISTRIBUCIÓN ESTADÍSTICA DE MÉTRICAS  (base matemática de las puntuaciones)
# Se calcula UNA SOLA VEZ con los datos reales de todos los clubes de LaLiga.
#
# En lugar de umbrales arbitrarios, cada puntuación 0-5 se deriva de la
# función de distribución acumulada normal Φ(z) calculada con la media (μ)
# y la desviación típica (σ) de toda la liga:
#
#   • Métricas "mayor = mejor"  →  puntos = 5 × Φ(+z)
#   • Métricas "menor = mejor"  →  puntos = 5 × Φ(−z)
#
# Interpretación directa:
#   Club en la media (z = 0)   → 2.50 pts  (percentil 50)
#   Club a +1σ                 → 4.21 pts  (percentil 84)
#   Club a +2σ                 → 4.77 pts  (percentil 98)
#   Club a −1σ                 → 0.79 pts  (percentil 16)
# =============================================================================

def _compute_distribucion_metricas():
    """Calcula μ y σ de las 3 métricas sobre todos los clubes de LaLiga."""
    fpf_vals = (df_clubes['masa salarial'] /
                df_clubes['ingresos totales'].clip(lower=0.1)) * 100
    ef_vals  = (df_clubes['ingresos totales'] /
                df_clubes['gastos totales'].clip(lower=0.1))
    vm_ratios = []
    for _, row in df_clubes.iterrows():
        jug   = df_jugadores[df_jugadores['Equipo'] == row['Nombre del club']]
        coste = jug['Coste traspaso'].fillna(0).sum()
        vm_ratios.append(row['valor de mercado de la plantilla'] / max(coste, 1))
    vm_vals = pd.Series(vm_ratios)

    dist = {
        'fpf': {'mean': fpf_vals.mean(), 'std': max(fpf_vals.std(), 0.01),
                'min': fpf_vals.min(),   'max': fpf_vals.max()},
        'ef':  {'mean': ef_vals.mean(),  'std': max(ef_vals.std(),  0.01),
                'min': ef_vals.min(),    'max': ef_vals.max()},
        'vm':  {'mean': vm_vals.mean(),  'std': max(vm_vals.std(),  0.01),
                'min': vm_vals.min(),    'max': vm_vals.max()},
    }
    print("\n  ── Distribución estadística LaLiga 24/25 ────────────────────────")
    print(f"  FPF  (Sal/Ing):   μ={dist['fpf']['mean']:.1f}%  σ={dist['fpf']['std']:.1f}%"
          f"  [{dist['fpf']['min']:.1f}%–{dist['fpf']['max']:.1f}%]")
    print(f"  Ef.Op(Ing/Gas):   μ={dist['ef']['mean']:.3f}x  σ={dist['ef']['std']:.3f}x"
          f"  [{dist['ef']['min']:.3f}x–{dist['ef']['max']:.3f}x]")
    print(f"  VM/Coste hist.:   μ={dist['vm']['mean']:.2f}x  σ={dist['vm']['std']:.2f}x"
          f"  [{dist['vm']['min']:.2f}x–{dist['vm']['max']:.2f}x]")
    print("  ─────────────────────────────────────────────────────────────────\n")
    return dist


DIST_METRICAS = _compute_distribucion_metricas()


# =============================================================================
# DIAGNOSTICO FINANCIERO
# 3 indicadores normalizados con distribución normal (LaLiga 24/25)
# =============================================================================

# =============================================================================
# TABLA DE RENDIMIENTO MEDIO POR CLUB → POSICIÓN EN LIGA
# Lógica: tomamos el rendimiento medio real de los 20 clubes en 24/25.
# Para proyectar la posición futura de un club, sustituimos su rendimiento
# por el proyectado y re-ordenamos los 20 clubes de mayor a menor rendimiento.
# El nuevo puesto del club en ese ranking es su posición estimada.
# Así el resultado siempre está acotado entre 1 y 20 y es relativo al resto.
# =============================================================================

# Rendimiento medio actual de cada club (se construye una sola vez)
_REND_MEDIO_CLUBES = (
    df_jugadores.groupby('Equipo')['Rendimiento']
    .mean()
    .rename('rend_medio')
)


def proyectar_posicion(nombre_club, rend_nuevo):
    """
    Estima la posición en liga del club si su plantilla tuviese un rendimiento
    medio de `rend_nuevo`.

    Método: re-ranking directo.
      1. Parte del rendimiento medio actual de los 20 clubes.
      2. Sustituye el valor del club analizado por `rend_nuevo`.
      3. Ordena los 20 clubes de mayor a menor rendimiento.
      4. Devuelve el rango (1 = mejor) del club en ese nuevo orden.
    """
    rend = _REND_MEDIO_CLUBES.copy()
    # Actualizar el rendimiento del club con el valor proyectado
    if nombre_club in rend.index:
        rend[nombre_club] = rend_nuevo
    else:
        # Si el club no está en la tabla de jugadores, añadirlo
        rend[nombre_club] = rend_nuevo
    # Ordenar de mayor rendimiento (mejor) a menor; el rango es la posición
    ranking = rend.sort_values(ascending=False).reset_index()
    ranking.columns = ['Equipo', 'rend_medio']
    fila = ranking[ranking['Equipo'] == nombre_club]
    if fila.empty:
        return 20
    return int(fila.index[0]) + 1   # índice 0-based → posición 1-based
# 5 indicadores basados en las columnas reales del nuevo CSV
# =============================================================================

def diagnostico_club(nombre_club):
    """
    Evalua la SOSTENIBILIDAD del club con 3 indicadores normalizados
    mediante distribución normal sobre los datos reales de LaLiga 24/25.

    Cada puntuación (0-5) se calcula como 5 × Φ(±z), donde z es el z-score
    del club dentro de la distribución real de la liga. Esto da un fundamento
    estadístico a las puntuaciones: la media de la liga vale siempre 2.5 pts,
    un club a +1σ vale ~4.2 pts, y uno a −1σ vale ~0.8 pts.
    """
    club = df_clubes[df_clubes['Nombre del club'] == nombre_club]
    if club.empty:
        print(f"Club '{nombre_club}' no encontrado.")
        return None

    c = club.iloc[0]

    ingresos     = c['ingresos totales']                   # M€
    gastos       = c['gastos totales']                     # M€
    salarios     = c['masa salarial']
    if salarios <= 0:
        salarios = c.get('salarios plantilla deportiva', 0) + c.get('salarios plantilla no deportiva', 0)
    vm_plantilla = c['valor de mercado de la plantilla']   # M€

    jugadores_club       = df_jugadores[df_jugadores['Equipo'] == nombre_club]
    coste_traspasos_hist = jugadores_club['Coste traspaso'].fillna(0).sum()  # M€

    # Parámetros de distribución calculados sobre toda la liga
    μ = DIST_METRICAS
    # Dep. TV se mantiene como ratio informativo pero NO puntúa


    # ------------------------------------------------------------------
    # 1. FPF - RATIO SALARIAL (menor = mejor)
    #    μ y σ calculados sobre los 20 clubes de LaLiga 24/25.
    #    Puntuación = 5 × Φ(−z)  → club con FPF bajo (bueno) puntuación alta.
    # ------------------------------------------------------------------
    fpf    = salarios / max(ingresos, 0.1) * 100
    z_fpf  = (fpf - μ['fpf']['mean']) / μ['fpf']['std']
    puntos_fpf = round(float(np.clip(5 * norm.cdf(-z_fpf), 0, 5)), 2)

    # ------------------------------------------------------------------
    # 2. EFICIENCIA OPERATIVA = ingresos / gastos totales (mayor = mejor)
    #    >1.0 significa que el club no gasta más de lo que ingresa.
    #    Puntuación = 5 × Φ(+z)  → mayor eficiencia = mayor puntuación.
    # ------------------------------------------------------------------
    ef_operativa = ingresos / max(gastos, 0.1)
    z_ef         = (ef_operativa - μ['ef']['mean']) / μ['ef']['std']
    puntos_ef    = round(float(np.clip(5 * norm.cdf(z_ef), 0, 5)), 2)

    # ------------------------------------------------------------------
    # 3. REVALORIZACION DE PLANTILLA = VM / coste historico traspasos
    #    Mide si los jugadores valen más ahora de lo que costaron.
    #    Puntuación = 5 × Φ(+z)  → mayor revalorización = mayor puntuación.
    # ------------------------------------------------------------------
    vm_ratio  = vm_plantilla / max(coste_traspasos_hist, 1)
    z_vm      = (vm_ratio - μ['vm']['mean']) / μ['vm']['std']
    puntos_vm = round(float(np.clip(5 * norm.cdf(z_vm), 0, 5)), 2)

    scores = {
        'FPF - Ratio salarial (Sal/Ing)':            puntos_fpf,
        'Eficiencia operativa (Ing/Gas)':            puntos_ef,
        'Revalorizacion plantilla (VM/Coste hist.)': puntos_vm,
    }

    total   = sum(scores.values())
    max_tot = len(scores) * 5        # 3 métricas × 5 pts = 15
    salud   = total / max_tot * 100

    if salud >= 65:   semaforo = "SOSTENIBLE"
    elif salud >= 40: semaforo = "EN RIESGO"
    else:             semaforo = "NO SOSTENIBLE"

    return {
        'nombre':   nombre_club,
        'club':     c,
        'scores':   scores,
        'salud':    round(salud, 1),
        'semaforo': semaforo,
        'ratios': {
            'fpf':             fpf,
            'ef_operativa':    ef_operativa,
            'vm_ratio':        vm_ratio,
            'coste_traspasos': coste_traspasos_hist,
            'ingresos':        ingresos,
            'gastos':          gastos,
            'salarios':        salarios,
            'vm_plantilla':    vm_plantilla,
            # z-scores para transparencia metodológica
            'z_fpf':  round(float(z_fpf), 3),
            'z_ef':   round(float(z_ef),  3),
            'z_vm':   round(float(z_vm),  3),
        },
    }


# =============================================================================
# MODULO FINANCIERO  (versión mejorada)
# Valores en M€ (coherente con el CSV)
# =============================================================================

def analisis_financiero(diag, plazo):
    """
    Genera el plan financiero a corto (1 temporada) o largo plazo (3 temporadas).

    PALANCAS:
      P1 — Reducción de masa salarial  (según FPF real del club)
      P2 — Diversificación de ingresos (comercial + matchday vs mediana LaLiga)
      P3 — Reducción del gasto en fichajes (hacia el 8% de ingresos como referencia sana)
      P4 — Inversión en cantera (solo largo plazo; empuja hacia ≥8% de masa salarial)

    NUEVO en largo plazo:
      · Proyección año a año (Y0, Y1, Y2, Y3) de ingresos, gastos y balance
      · Modelo de ROI de cantera: la inversión acumulada genera valor de mercado
        de jugadores propios que reduce el gasto en fichajes en los años siguientes.
      · La cantera se muestra como palanca clave de sostenibilidad a largo plazo.
    """
    c        = diag['club']
    ingresos = c['ingresos totales']        # M€
    # Si masa salarial viene a 0 en el CSV (error de datos), reconstruirla
    # sumando los dos conceptos de salario disponibles
    masa     = c['masa salarial']
    if masa <= 0:
        masa = c.get('salarios plantilla deportiva', 0) + c.get('salarios plantilla no deportiva', 0)
    gastos   = c['gastos totales']          # M€
    cantera  = c['inversion en cantera']    # M€
    comercial = c['ingresos comerciales']   # M€
    matchday  = c['ingresos matchday']      # M€
    g_fich    = c['gasto en fichajes']      # M€

    fpf        = masa / max(ingresos, 0.1) * 100
    balance_op = ingresos - gastos

    # Medianas reales de LaLiga (del CSV cargado)
    med_comercial_pct = (df_clubes['ingresos comerciales'] / df_clubes['ingresos totales']).median() * 100
    med_matchday_pct  = (df_clubes['ingresos matchday']    / df_clubes['ingresos totales']).median() * 100
    pct_cantera       = cantera / max(masa, 0.1) * 100
    pct_comercial     = comercial / max(ingresos, 0.1) * 100
    pct_matchday      = matchday  / max(ingresos, 0.1) * 100

    recortes    = {}
    inversiones = {}
    notas       = {}

    # ------------------------------------------------------------------
    # PALANCA 1: Reducción de masa salarial según FPF del club
    # ------------------------------------------------------------------
    if fpf >= 85:
        pct_reduccion_cp = 0.18
        pct_reduccion_lp = 0.28
        notas['fpf_nivel'] = 'CRÍTICO'
    elif fpf >= 73:
        pct_reduccion_cp = 0.12
        pct_reduccion_lp = 0.20
        notas['fpf_nivel'] = 'EN RIESGO'
    elif fpf >= 60:
        pct_reduccion_cp = 0.06
        pct_reduccion_lp = 0.10
        notas['fpf_nivel'] = 'ELEVADO'
    else:
        pct_reduccion_cp = 0.03
        pct_reduccion_lp = 0.05
        notas['fpf_nivel'] = 'SANEADO'

    ahorro_masa_cp = masa * pct_reduccion_cp
    ahorro_masa_lp = masa * pct_reduccion_lp

    # ------------------------------------------------------------------
    # PALANCA 2: Diversificación de ingresos
    # ------------------------------------------------------------------
    margen_comercial = max(0, med_comercial_pct - pct_comercial)
    margen_matchday  = max(0, med_matchday_pct  - pct_matchday)

    crecim_comercial_cp = min(0.12, margen_comercial / 100 * 0.5) * comercial
    crecim_matchday_cp  = min(0.08, margen_matchday  / 100 * 0.5) * matchday
    crecim_comercial_lp = min(0.25, margen_comercial / 100 * 1.2) * comercial
    crecim_matchday_lp  = min(0.18, margen_matchday  / 100 * 1.2) * matchday

    # ------------------------------------------------------------------
    # PALANCA 3: Reducción del gasto en fichajes
    # ------------------------------------------------------------------
    objetivo_fich     = ingresos * 0.08
    reduccion_fich_cp = max(0, g_fich - objetivo_fich) * 0.7
    reduccion_fich_lp = max(0, g_fich - objetivo_fich)

    # ------------------------------------------------------------------
    # PALANCA 4: Inversión en cantera (LP)
    # Objetivo: llevar la inversión en cantera al ≥8% de la masa salarial.
    # ROI de cantera: por cada M€ invertido en cantera durante 3 años se
    # estima un retorno de 3.5M€ en valor de mercado de jugadores propios
    # (referencia conservadora basada en clubes como Athletic o Osasuna).
    # Ese VM generado se traduce en menor necesidad de fichajes (ahorro).
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # PALANCA 4: Inversión en cantera (LP)
    #
    # Objetivo basado en la MEDIA DE LA LIGA de (cantera / ingresos):
    #   · Se calcula la media real de todos los clubes y se añade +1 pp.
    #   · El objetivo de cada club es llegar a ese umbral en 3 años.
    #   · Si ya supera el objetivo, se le pide crecer un 10% más.
    #   · Mínimo 1M€/año para que siempre haya una recomendación visible.
    #
    # Usar ingresos (no masa salarial) evita distorsiones en clubes con
    # datos de masa salarial incorrectos o atípicos.
    # ------------------------------------------------------------------
    media_pct_cantera  = (df_clubes['inversion en cantera'] /
                          df_clubes['ingresos totales'].clip(lower=1)).mean() * 100
    median_pct_cantera = (df_clubes['inversion en cantera'] /
                          df_clubes['ingresos totales'].clip(lower=1)).median() * 100
    std_pct_cantera    = (df_clubes['inversion en cantera'] /
                          df_clubes['ingresos totales'].clip(lower=1)).std() * 100
    objetivo_pct_ing   = median_pct_cantera + std_pct_cantera   # mediana + 1σ

    objetivo_cantera_abs = objetivo_pct_ing / 100 * ingresos
    if cantera >= objetivo_cantera_abs:                   # ya supera la media
        objetivo_cantera_abs = cantera * 1.10            # crecer solo un 10%

    incremento_anual = max(1.0, (objetivo_cantera_abs - cantera) / 3)
    incremento_anual = min(incremento_anual, ahorro_masa_lp * 0.25)

    # Inversión acumulativa: cada año crece sobre el anterior
    cantera_y1 = cantera    + incremento_anual
    cantera_y2 = cantera_y1 + incremento_anual
    cantera_y3 = cantera_y2 + incremento_anual

    # Objetivo en % sobre ingresos (informativo para gráfico)
    objetivo_cantera_pct = cantera_y3 / max(ingresos, 1) * 100

    # ROI: VM proyectado de jugadores propios
    roi_cantera_y1 = cantera_y1 * 1.0    # año 1: jugadores jóvenes, sin valor aún
    roi_cantera_y2 = cantera_y2 * 1.8    # año 2: primeros talentos asoman
    roi_cantera_y3 = cantera_y3 * 3.5    # año 3: jugadores vendibles

    # Ahorro en fichajes gracias a la cantera (el club usa jugadores propios)
    ahorro_fich_cantera_y3 = min(roi_cantera_y3 * 0.15, g_fich * 0.30)

    # ------------------------------------------------------------------
    # CORTO PLAZO
    # ------------------------------------------------------------------
    if plazo == 'corto':
        if ahorro_masa_cp > 0:
            recortes[f'Renegociación salarial\n({pct_reduccion_cp*100:.0f}% masa · FPF {notas["fpf_nivel"]})'] = ahorro_masa_cp
        if reduccion_fich_cp > 0:
            recortes['Reducción gasto fichajes\n(hacia 8% de ingresos)'] = reduccion_fich_cp

        if crecim_comercial_cp > 0:
            inversiones[f'Crecimiento comercial\n(+{margen_comercial:.1f}pp vs mediana LaLiga)'] = crecim_comercial_cp
        if crecim_matchday_cp > 0:
            inversiones[f'Crecimiento matchday/abonados\n(+{margen_matchday:.1f}pp)'] = crecim_matchday_cp

        nueva_masa   = masa - ahorro_masa_cp
        ing_extra_cp = crecim_comercial_cp + crecim_matchday_cp
        ahorro_total = ahorro_masa_cp + reduccion_fich_cp

        nuevo_balance = balance_op + ahorro_total + ing_extra_cp

        proyeccion = {
            'nueva_masa':      nueva_masa,
            'nuevo_fpf':       nueva_masa / max(ingresos + ing_extra_cp, 0.1) * 100,
            'nuevo_balance':   nuevo_balance,
            'ingresos_extra':  ing_extra_cp,
            'ahorro_total':    ahorro_total,
            'balance_inicial': balance_op,
        }

    # ------------------------------------------------------------------
    # LARGO PLAZO (3 temporadas)
    # ------------------------------------------------------------------
    else:
        if ahorro_masa_lp > 0:
            recortes[f'Reestructuración salarial 3 años\n({pct_reduccion_lp*100:.0f}% de reducción progresiva)'] = ahorro_masa_lp
        if reduccion_fich_lp > 0:
            recortes['Normalización gasto fichajes\n(objetivo: 8% de ingresos)'] = reduccion_fich_lp

        if crecim_comercial_lp > 0:
            inversiones[f'Expansión comercial\n(objetivo mediana LaLiga {med_comercial_pct:.1f}%)'] = crecim_comercial_lp
        if crecim_matchday_lp > 0:
            inversiones[f'Crecimiento abonados/matchday\n(objetivo {med_matchday_pct:.1f}%)'] = crecim_matchday_lp
        if incremento_anual > 0:
            inversiones[f'Inversión cantera\n(+{incremento_anual:.1f}M€/año · objetivo {cantera_y3:.1f}M€ en año 3)'] = incremento_anual * 3

        # Crecimiento orgánico de ingresos: 7% anual (histórico LaLiga)
        # más los crecimientos específicos del club en comercial y matchday
        ing_crecim_pct = 0.07
        ingresos_y0 = ingresos
        ingresos_y1 = ingresos * (1 + ing_crecim_pct) + crecim_comercial_lp * 0.33
        ingresos_y2 = ingresos * (1 + ing_crecim_pct)**2 + crecim_comercial_lp * 0.66 + crecim_matchday_lp * 0.5
        ingresos_y3 = ingresos * (1 + ing_crecim_pct)**3 + crecim_comercial_lp + crecim_matchday_lp

        # Gastos: bajan gradualmente por reestructuración salarial y fichajes
        gastos_y0 = gastos
        gastos_y1 = gastos - ahorro_masa_lp * 0.33 - reduccion_fich_lp * 0.50
        gastos_y2 = gastos - ahorro_masa_lp * 0.66 - reduccion_fich_lp * 0.75
        gastos_y3 = gastos - ahorro_masa_lp        - reduccion_fich_lp - ahorro_fich_cantera_y3

        balance_y0 = ingresos_y0 - gastos_y0
        balance_y1 = ingresos_y1 - gastos_y1
        balance_y2 = ingresos_y2 - gastos_y2
        balance_y3 = ingresos_y3 - gastos_y3

        masa_y3 = masa - ahorro_masa_lp

        proyeccion = {
            'nueva_masa':      masa_y3,
            'nuevo_fpf':       masa_y3 / max(ingresos_y3, 0.1) * 100,
            'nuevo_balance':   balance_y3,
            'ingresos_extra':  ingresos_y3 - ingresos_y0,
            'ahorro_total':    ahorro_masa_lp + reduccion_fich_lp,
            'balance_inicial': balance_y0,
            # Datos anuales para los gráficos de evolución
            'anual': {
                'ingresos': [ingresos_y0, ingresos_y1, ingresos_y2, ingresos_y3],
                'gastos':   [gastos_y0,  gastos_y1,  gastos_y2,  gastos_y3],
                'balance':  [balance_y0, balance_y1, balance_y2, balance_y3],
            },
            # Datos de cantera para los gráficos
            'cantera': {
                'inversion':  [cantera, cantera_y1, cantera_y2, cantera_y3],
                'roi_vm':     [cantera * 1.0, roi_cantera_y1, roi_cantera_y2, roi_cantera_y3],
                'ahorro_fich_y3': ahorro_fich_cantera_y3,
                'pct_actual': round(pct_cantera, 1),
                'pct_obj':    objetivo_cantera_pct,
            },
        }

    notas.update({
        'fpf_actual':         round(fpf, 1),
        'pct_comercial':      round(pct_comercial, 1),
        'pct_matchday':       round(pct_matchday, 1),
        'med_comercial_pct':  round(med_comercial_pct, 1),
        'med_matchday_pct':   round(med_matchday_pct, 1),
        'pct_cantera':        round(pct_cantera, 1),
        'g_fich_actual':      round(g_fich, 1),
        'objetivo_fich':      round(objetivo_fich, 1),
    })

    return {
        'plazo':       plazo,
        'club':        c,
        'recortes':    recortes,
        'inversiones': inversiones,
        'proyeccion':  proyeccion,
        'notas':       notas,
    }


# =============================================================================
# MODULO DEPORTIVO
# VM y Coste traspaso ya están en M€ en el nuevo CSV
# =============================================================================

def analisis_deportivo(nombre_club, modo='deportivo'):
    """
    Recomienda UN sustituto por cada jugador candidato a salir.

    modo='deportivo' -> prioriza rendimiento (objetivo: mejorar resultados)
    modo='mixto'     -> prioriza eficiencia coste/rendimiento (objetivo: mejorar finanzas Y plantilla)
    """
    club_data = df_clubes[df_clubes['Nombre del club'] == nombre_club].iloc[0]
    plantilla = df_jugadores[df_jugadores['Equipo'] == nombre_club].copy()
    mercado   = df_jugadores[df_jugadores['Equipo'] != nombre_club].copy()

    if plantilla.empty:
        return None

    nivel_mi_club = NIVEL_CLUB.get(nombre_club, 1)
    mercado['nivel_club_origen'] = mercado['Equipo'].map(NIVEL_CLUB).fillna(1)

    def etiqueta(pct):
        if pct >= 80:   return "VENDER YA"
        elif pct >= 60: return "Vender"
        elif pct >= 40: return "Analizar"
        elif pct >= 20: return "Conservar"
        else:           return "Intocable"

    plantilla['Decision'] = plantilla['Pct_venta'].apply(etiqueta)

    candidatos_venta_todos = plantilla[
        plantilla['Decision'].isin(['VENDER YA', 'Vender'])
    ].sort_values('Pct_venta', ascending=False)

    # Solo se ejecutan los 7 movimientos con mayor prioridad de venta
    candidatos_venta = candidatos_venta_todos.head(7)

    fichajes_recomendados = []
    ya_recomendados = set()

    for _, jugador_sale in candidatos_venta.iterrows():
        pos = jugador_sale['Pos']
        opciones = mercado[
            (mercado['Pos'] == pos) & (~mercado.index.isin(ya_recomendados))
        ].copy()
        if opciones.empty:
            continue

        # Filtro de nivel de club: coherencia en el mercado
        if nivel_mi_club == 3:
            opciones_filtradas = opciones[opciones['nivel_club_origen'] >= 2]
        elif nivel_mi_club == 2:
            opciones_filtradas = opciones.copy()
        else:
            opciones_filtradas = opciones[opciones['nivel_club_origen'] <= 2]

        if opciones_filtradas.empty:
            opciones_filtradas = opciones

        opciones_filtradas = opciones_filtradas.copy()

        # Eficiencia = rendimiento / coste total en M€
        # VM y Coste traspaso ya están en M€ en el CSV
        opciones_filtradas['Eficiencia'] = opciones_filtradas['Rendimiento'] / (
            (opciones_filtradas['VM'] + opciones_filtradas['Coste traspaso']) + 0.1
        )

        if modo == 'deportivo':
            if nivel_mi_club == 3:
                opciones_filtradas['Score_fichaje'] = (
                    opciones_filtradas['Rendimiento'] * 0.80 +
                    (opciones_filtradas['Edad'] < 27).astype(int) * 0.20 * 10
                )
            else:
                opciones_filtradas['Score_fichaje'] = (
                    opciones_filtradas['Rendimiento'] * 0.65 +
                    (opciones_filtradas['Edad'] < 26).astype(int) * 0.35 * 10
                )
        else:  # mixto
            coste_venta = jugador_sale['VM']   # M€ del jugador que sale
            opciones_filtradas['Score_fichaje'] = (
                opciones_filtradas['Eficiencia'] * 0.55 +
                (opciones_filtradas['VM'] <= coste_venta).astype(int) * 0.25 * 10 +
                (opciones_filtradas['Edad'] < 26).astype(int) * 0.20 * 10
            )

        mejor = opciones_filtradas.nlargest(1, 'Score_fichaje').copy()
        ya_recomendados.update(mejor.index.tolist())
        mejor['Nombre_sale']   = jugador_sale['Nombre']
        mejor['Pos_necesaria'] = pos
        fichajes_recomendados.append(mejor)

    df_fichajes = pd.concat(fichajes_recomendados, ignore_index=True) if fichajes_recomendados else pd.DataFrame()

    # Proyección de posición en liga mediante re-ranking sobre rendimiento medio
    # ─────────────────────────────────────────────────────────────────────────────
    # El rendimiento proyectado NO usa la media bruta de la plantilla nueva, porque
    # los fichajes traen su rendimiento de su club actual (donde el contexto táctico,
    # la calidad de los compañeros y el nivel del equipo es diferente). Usar la media
    # bruta infla el resultado: Leganés ficharía jugadores de nivel medio-alto y su
    # media saltaría por encima de la de clubes mucho mejores.
    #
    # Solución: aplicar el DELTA de rendimiento ponderado por la fracción de
    # plantilla que cambia (n_cambios / tamaño_plantilla). Así 7 cambios sobre 25
    # jugadores (~28%) solo trasladan el 28% de la mejora bruta al equipo completo.
    # Adicionalmente, se limita la mejora máxima de posición a un techo realista
    # calculado en función de la fracción de plantilla renovada: con el 28% del
    # equipo cambiado, el máximo histórico razonable en LaLiga es ~5 puestos.
    posicion_ant  = int(club_data['posicion_liga'])
    plantilla_base = plantilla.copy()
    tam_plantilla = max(len(plantilla_base), 1)

    n_cambios = min(len(candidatos_venta), len(df_fichajes)) if not df_fichajes.empty else 0

    if n_cambios > 0:
        rend_actual    = plantilla_base['Rendimiento'].mean()
        rend_salen     = candidatos_venta.head(n_cambios)['Rendimiento'].mean()
        rend_llegan    = df_fichajes.head(n_cambios)['Rendimiento'].mean()
        # Delta bruto de los 7 movimientos, ponderado por la fracción del equipo
        fraccion       = n_cambios / tam_plantilla
        rend_nuevo     = rend_actual + (rend_llegan - rend_salen) * fraccion
    else:
        rend_nuevo = plantilla_base['Rendimiento'].mean()

    posicion_calculada = proyectar_posicion(nombre_club, rend_nuevo)

    # Techo de mejora realista: con una fracción f de plantilla renovada,
    # el máximo de puestos que se puede subir en una temporada es f * 18
    # (si se renueva el 100% del equipo se podría subir hasta 18 puestos,
    # pero con el 28% el techo es ~5 puestos, coherente con LaLiga histórica).
    mejora_max     = max(1, round(fraccion * 18))
    posicion_proy  = max(posicion_calculada, posicion_ant - mejora_max)

    # Balance económico — solo los 7 movimientos ejecutados (valores en M€)
    # Ingresos: VM de mercado de los jugadores que salen (lo que pagará el comprador)
    # Gastos:   VM de los jugadores que entran (coste de adquisición)
    ingreso_ventas  = candidatos_venta['VM'].sum()           # M€ que ingresa el club
    gasto_fichajes  = df_fichajes['VM'].sum() if not df_fichajes.empty else 0  # M€ que paga
    balance_mercado = ingreso_ventas - gasto_fichajes        # M€ (positivo = superávit)

    return {
        'nombre_club':          nombre_club,
        'club':                 club_data,
        'modo':                 modo,
        'plantilla':            plantilla.sort_values('Pct_venta', ascending=False),
        'candidatos_venta':     candidatos_venta,       # los 7 que se ejecutan
        'candidatos_venta_todos': candidatos_venta_todos,  # lista completa para el grafico
        'fichajes':             df_fichajes,
        'posicion_ant':         posicion_ant,
        'posicion_proy':        posicion_proy,
        'balance_mercado':      balance_mercado,   # M€
        'ingreso_ventas':       ingreso_ventas,
        'gasto_fichajes':       gasto_fichajes,
    }


# =============================================================================
# GRAFICOS
# =============================================================================

def grafico_diagnostico(diag):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Sostenibilidad del club: {diag['nombre']}", fontsize=14, fontweight='bold')

    metricas = list(diag['scores'].keys())
    valores  = list(diag['scores'].values())
    # Verde ≥4, Naranja ≥2, Rojo <2
    colores  = ['#27ae60' if v >= 4 else ('#f39c12' if v >= 2 else '#e74c3c') for v in valores]

    # -----------------------------------------------------------------------
    # Panel izquierdo: barras horizontales por indicador de sostenibilidad
    # -----------------------------------------------------------------------
    bars = ax1.barh(metricas, valores, color=colores, height=0.45,
                    edgecolor='white', linewidth=0.5)
    ax1.set_xlim(0, 5.8)
    ax1.axvline(3, color='#7f8c8d', linestyle='--', linewidth=0.8, alpha=0.7)
    ax1.set_xlabel('Puntuación (0 a 5)', fontsize=9)
    ax1.set_title('Indicadores de sostenibilidad', fontsize=11, fontweight='bold')
    for i, (v, bar) in enumerate(zip(valores, bars)):
        etiqueta = '✓ BIEN' if v >= 4 else ('↗ REGULAR' if v >= 2 else '✗ BAJO')
        ax1.text(v + 0.08, i, f'{v}/5  {etiqueta}', va='center', fontsize=8.5,
                 color=colores[i], fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='y', labelsize=8.5)

    # -----------------------------------------------------------------------
    # Panel derecho: velocímetro semicircular (gauge) con zonas de color
    # -----------------------------------------------------------------------
    ax2.axis('off')
    salud = diag['salud']
    color_salud = '#27ae60' if salud >= 65 else ('#f39c12' if salud >= 40 else '#e74c3c')

    # Fondo gris (arco completo)
    angulos_bg = np.linspace(np.pi, 0, 300)
    ax2.plot(np.cos(angulos_bg), np.sin(angulos_bg), color='#ecf0f1', linewidth=22, solid_capstyle='round')

    # Zona roja (0-40%)
    ang_rojo = np.linspace(np.pi, np.pi - np.pi * 0.40, 150)
    ax2.plot(np.cos(ang_rojo), np.sin(ang_rojo), color='#fadbd8', linewidth=22, solid_capstyle='butt')

    # Zona naranja (40-65%)
    ang_naranja = np.linspace(np.pi - np.pi * 0.40, np.pi - np.pi * 0.65, 100)
    ax2.plot(np.cos(ang_naranja), np.sin(ang_naranja), color='#fdebd0', linewidth=22, solid_capstyle='butt')

    # Zona verde (65-100%)
    ang_verde = np.linspace(np.pi - np.pi * 0.65, 0, 150)
    ax2.plot(np.cos(ang_verde), np.sin(ang_verde), color='#d5f5e3', linewidth=22, solid_capstyle='butt')

    # Relleno activo con el color correspondiente
    angulos_relleno = np.linspace(np.pi, np.pi - np.pi * salud / 100, 300)
    ax2.plot(np.cos(angulos_relleno), np.sin(angulos_relleno),
             color=color_salud, linewidth=22, solid_capstyle='round', alpha=0.92)

    # Aguja indicadora
    angulo_aguja = np.pi - np.pi * salud / 100
    ax2.annotate('', xy=(0.72 * np.cos(angulo_aguja), 0.72 * np.sin(angulo_aguja)),
                 xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5))
    ax2.plot(0, 0, 'o', color='#2c3e50', markersize=7, zorder=5)

    # Etiquetas de zona
    ax2.text(-1.15,  0.05, 'NO\nSOS.',  ha='center', fontsize=6.5, color='#e74c3c', fontweight='bold')
    ax2.text( 0,     1.08, 'EN\nRIESGO', ha='center', fontsize=6.5, color='#f39c12', fontweight='bold')
    ax2.text( 1.15,  0.05, 'SOS.',       ha='center', fontsize=6.5, color='#27ae60', fontweight='bold')

    # Porcentaje central y veredicto
    ax2.text(0,  0.22, f"{salud}%",       ha='center', fontsize=38, fontweight='bold', color=color_salud)
    ax2.text(0, -0.15, diag['semaforo'],  ha='center', fontsize=13, fontweight='bold', color=color_salud)

    # Ratios detallados bajo el gauge
    r = diag['ratios']
    ax2.text(0, -0.50,
             f"FPF: {r['fpf']:.1f}%  |  Ef.Operativa: {r['ef_operativa']:.2f}x  |  VM/Coste hist: {r['vm_ratio']:.2f}x\n"
             f"Ingresos: {r['ingresos']:.0f}M€  |  Gastos: {r['gastos']:.0f}M€  |  Salarios: {r['salarios']:.0f}M€",
             ha='center', fontsize=8, color='#7f8c8d',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6, edgecolor='none'))

    ax2.set_xlim(-1.45, 1.45)
    ax2.set_ylim(-0.85, 1.35)
    ax2.set_title('Índice global de sostenibilidad', fontsize=11, fontweight='bold')

    plt.tight_layout()
    nombre_archivo = f"sostenibilidad_{diag['nombre'].replace(' ', '_')}.png"
    plt.savefig(nombre_archivo, dpi=130, bbox_inches='tight')
    plt.show()
    print(f"  Gráfico guardado: {nombre_archivo}")


def _color_fpf(fpf):
    """Devuelve el color semáforo del FPF."""
    return '#e74c3c' if fpf > 73 else ('#f39c12' if fpf > 60 else '#27ae60')


def _safe_max(lst, default=1.0):
    """max() seguro ante listas vacías."""
    return max(lst) if lst else default


def grafico_financiero(res):
    """
    Genera entre 4 y 5 gráficos claros y funcionales del plan financiero.

    CORTO PLAZO — 4 paneles:
      1. Puente (waterfall) de balance: cómo pasamos del balance actual al proyectado
      2. Recortes vs inversiones: qué tocamos y cuánto
      3. FPF antes / después con zonas de color
      4. Mix de ingresos actual (dona) con proyección de nuevas fuentes

    LARGO PLAZO — 5 paneles:
      1. Evolución anual (ingresos, gastos y balance) en 3 años
      2. Cantera: inversión acumulada y VM generado (clave de sostenibilidad)
      3. Recortes vs inversiones: qué tocamos y cuánto
      4. FPF año a año con zona objetivo
      5. KPIs resumen antes / después
    """
    c     = res['club']
    plazo = res['plazo']
    p     = res['proyeccion']
    nombre_club = c['Nombre del club']

    COLOR_CORTE  = '#e74c3c'   # rojo     → recorte
    COLOR_INVEST = '#27ae60'   # verde    → inversión / ganancia
    COLOR_BASE   = '#2980b9'   # azul     → base / actual
    COLOR_CANT   = '#8e44ad'   # morado   → cantera
    COLOR_NEG    = '#c0392b'   # rojo osc.→ negativo

    # =========================================================================
    # CORTO PLAZO
    # =========================================================================
    if plazo == 'corto':
        fig = plt.figure(figsize=(20, 10))
        fig.patch.set_facecolor('#f8f9fa')
        fig.suptitle(
            f"Plan de mejora económica · CORTO PLAZO · {nombre_club}",
            fontsize=15, fontweight='bold', y=0.99, color='#1a1a2e'
        )
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.42,
                               left=0.06, right=0.97, top=0.92, bottom=0.08)

        # -----------------------------------------------------------------
        # PANEL 1 — WATERFALL: puente de balance
        # Muestra visualmente cómo se pasa del balance actual al proyectado
        # -----------------------------------------------------------------
        ax1 = fig.add_subplot(gs[:, 0:2])   # ocupa las 2 primeras columnas, ambas filas
        ax1.set_facecolor('#ffffff')

        bal_ini = p['balance_inicial']
        ahorro  = p['ahorro_total']
        ing_ext = p['ingresos_extra']
        bal_fin = p['nuevo_balance']

        # Acumulados para posicionar cada barra
        etiquetas = ['Balance\nactual', 'Ahorro en\ncostes', 'Nuevos\ningresos', 'Balance\nproyectado']
        valores   = [bal_ini, ahorro, ing_ext, bal_fin]
        bottoms   = [0, bal_ini, bal_ini + ahorro, 0]
        colores_wf = [
            COLOR_BASE if bal_ini >= 0 else COLOR_CORTE,
            COLOR_INVEST,
            COLOR_INVEST,
            COLOR_INVEST if bal_fin >= 0 else COLOR_CORTE,
        ]

        x_pos = np.arange(len(etiquetas))
        for i, (v, bot, col) in enumerate(zip(valores, bottoms, colores_wf)):
            ax1.bar(x_pos[i], abs(v) if i not in [0, 3] else v,
                    bottom=bot if i not in [0, 3] else 0,
                    color=col, width=0.55, alpha=0.88, edgecolor='white', linewidth=1.2)
            # Línea conectora entre barras (excepto la última)
            if i < len(etiquetas) - 1 and i != 0:
                ax1.plot([x_pos[i] - 0.28, x_pos[i] + 0.28],
                         [bot + abs(v), bot + abs(v)], '--', color='#aaaaaa', lw=0.8)

        # Etiquetas de valor
        for i, v in enumerate(valores):
            y_label = (bottoms[i] + abs(v) / 2) if i not in [0, 3] else v / 2
            signo = '+' if v > 0 else ''
            ax1.text(x_pos[i], y_label, f'{signo}{v:.1f}M€',
                     ha='center', va='center', fontsize=9.5, fontweight='bold', color='white')

        ax1.axhline(0, color='#333333', linewidth=0.8, linestyle='-')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(etiquetas, fontsize=9.5, fontweight='bold')
        ax1.set_ylabel('Millones de euros (M€)', fontsize=9)
        ax1.set_title('① ¿Cuánto mejora el balance?', fontsize=11, fontweight='bold', pad=10)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.tick_params(axis='y', labelsize=8.5)

        # -----------------------------------------------------------------
        # PANEL 2 — Recortes e inversiones (barra horizontal doble)
        # -----------------------------------------------------------------
        ax2 = fig.add_subplot(gs[0, 2:4])
        ax2.set_facecolor('#ffffff')

        keys_r = list(res['recortes'].keys())
        vals_r = list(res['recortes'].values())
        keys_i = list(res['inversiones'].keys())
        vals_i = list(res['inversiones'].values())

        all_keys = keys_r + keys_i
        all_vals = vals_r + vals_i
        all_cols = [COLOR_CORTE] * len(keys_r) + [COLOR_INVEST] * len(keys_i)
        all_keys_short = [k.replace('\n', ' ')[:45] + ('…' if len(k.replace('\n', ' ')) > 45 else '') for k in all_keys]

        if all_vals:
            max_v = _safe_max(all_vals)
            y_pos = np.arange(len(all_keys))
            ax2.barh(y_pos, all_vals, color=all_cols, height=0.55, alpha=0.88, edgecolor='white')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(all_keys_short, fontsize=8.2)
            ax2.set_xlabel('Millones de euros (M€)', fontsize=8.5)
            ax2.set_title('② ¿Dónde recortar y dónde invertir?', fontsize=11, fontweight='bold', pad=10)
            ax2.axvline(0, color='#333333', linewidth=0.8)
            for i, v in enumerate(all_vals):
                ax2.text(v + max_v * 0.02, i, f'{v:.1f}M', va='center', fontsize=8.5,
                         fontweight='bold', color=COLOR_NEG if all_cols[i] == COLOR_CORTE else '#1a6b34')
            # Leyenda manual
            from matplotlib.patches import Patch
            ax2.legend(handles=[Patch(facecolor=COLOR_CORTE, label='Recorte / Ahorro'),
                                 Patch(facecolor=COLOR_INVEST, label='Nueva inversión')],
                       fontsize=8, loc='lower right', frameon=False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.invert_yaxis()

        # -----------------------------------------------------------------
        # PANEL 3 — FPF antes / después con zonas de peligro
        # -----------------------------------------------------------------
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.set_facecolor('#ffffff')

        fpf_a = c['masa salarial'] / c['ingresos totales'] * 100
        fpf_d = p['nuevo_fpf']

        # Zonas de fondo
        ax3.axhspan(0,  60, alpha=0.08, color='green',  zorder=0)
        ax3.axhspan(60, 73, alpha=0.08, color='orange', zorder=0)
        ax3.axhspan(73, max(fpf_a, fpf_d, 85) * 1.15, alpha=0.08, color='red', zorder=0)

        bars = ax3.bar(['Actual', 'Proyectado'], [fpf_a, fpf_d],
                       color=[_color_fpf(fpf_a), _color_fpf(fpf_d)],
                       width=0.45, alpha=0.88, edgecolor='white', linewidth=1.2)
        ymax_fpf = max(fpf_a, fpf_d, 85) * 1.15
        ax3.set_ylim(0, ymax_fpf)
        ax3.axhline(73, color='#e74c3c', linestyle='--', linewidth=1.2, label='Riesgo FPF (73%)')
        ax3.axhline(60, color='#f39c12', linestyle='--', linewidth=1.2, label='Zona segura (60%)')
        for i, v in enumerate([fpf_a, fpf_d]):
            ax3.text(i, v + ymax_fpf * 0.025, f'{v:.1f}%',
                     ha='center', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Masa salarial / Ingresos (%)', fontsize=8.5)
        ax3.set_title('③ FPF: antes y después', fontsize=11, fontweight='bold', pad=10)
        ax3.legend(fontsize=7.5, loc='upper right', frameon=False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        # -----------------------------------------------------------------
        # PANEL 4 — Mix de ingresos: actual vs proyectado (dona doble)
        # -----------------------------------------------------------------
        ax4 = fig.add_subplot(gs[1, 3])
        ax4.set_facecolor('#ffffff')

        fuentes_raw = [
            (c['ingresos TV'],              'TV'),
            (c['ingresos comerciales'],     'Comercial'),
            (c['ingresos matchday'],        'Matchday'),
            (c['ingresos competiciones'],   'Competiciones'),
        ]
        fuentes = [(v, l) for v, l in fuentes_raw if v > 0]
        sizes   = [f[0] for f in fuentes]
        labels  = [f[1] for f in fuentes]
        wcols   = ['#2980b9', '#27ae60', '#e67e22', '#8e44ad'][:len(fuentes)]

        wedges, _, autotexts = ax4.pie(
            sizes, labels=None, autopct='%1.1f%%',
            colors=wcols, startangle=90, pctdistance=0.78,
            wedgeprops=dict(width=0.55, linewidth=1.5, edgecolor='white')
        )
        for at in autotexts:
            at.set_fontsize(8); at.set_fontweight('bold')
        ax4.legend(wedges, labels, loc='lower center', bbox_to_anchor=(0.5, -0.18),
                   ncol=2, fontsize=7.5, frameon=False)
        ing_tot = c['ingresos totales']
        ax4.text(0, 0.08, f'{ing_tot:.0f}M€', ha='center', fontsize=11,
                 fontweight='bold', color='#1a1a2e')
        ax4.text(0, -0.15, 'Ingresos\nanuales', ha='center', fontsize=8, color='gray')
        ax4.set_title('④ Mix de ingresos actual', fontsize=11, fontweight='bold', pad=10)

    # =========================================================================
    # LARGO PLAZO
    # =========================================================================
    else:
        fig = plt.figure(figsize=(22, 12))
        fig.patch.set_facecolor('#f8f9fa')
        fig.suptitle(
            f"Plan de mejora económica · LARGO PLAZO (3 temporadas) · {nombre_club}",
            fontsize=15, fontweight='bold', y=0.99, color='#1a1a2e'
        )
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.38,
                               left=0.06, right=0.97, top=0.92, bottom=0.08)

        anual   = p['anual']
        cant    = p['cantera']
        anios   = ['Actual', 'Año 1', 'Año 2', 'Año 3']
        x_anios = np.arange(4)

        # -----------------------------------------------------------------
        # PANEL 1 — Evolución ingresos, gastos y balance año a año (GRANDE)
        # -----------------------------------------------------------------
        ax1 = fig.add_subplot(gs[0, 0:2])   # 2 columnas, fila superior
        ax1.set_facecolor('#ffffff')

        ing_vals = anual['ingresos']
        gas_vals = anual['gastos']
        bal_vals = anual['balance']

        # Áreas sombreadas
        ax1.fill_between(x_anios, ing_vals, alpha=0.12, color=COLOR_INVEST, zorder=1)
        ax1.fill_between(x_anios, gas_vals, alpha=0.12, color=COLOR_CORTE,  zorder=1)

        # Líneas principales
        ax1.plot(x_anios, ing_vals, 'o-', color=COLOR_INVEST, linewidth=2.5, markersize=8,
                 label='Ingresos', zorder=3)
        ax1.plot(x_anios, gas_vals, 's-', color=COLOR_CORTE,  linewidth=2.5, markersize=8,
                 label='Gastos',   zorder=3)

        # Barras de balance en el eje secundario
        ax1b = ax1.twinx()
        bal_colors = [COLOR_INVEST if b >= 0 else COLOR_CORTE for b in bal_vals]
        ax1b.bar(x_anios, bal_vals, color=bal_colors, alpha=0.35, width=0.4, label='Balance', zorder=2)
        ax1b.axhline(0, color='#555555', linewidth=0.8, linestyle='--')
        ax1b.set_ylabel('Balance (M€)', fontsize=9, color='#555555')
        ax1b.tick_params(axis='y', labelsize=8, colors='#555555')

        # Etiquetas de valor sobre los puntos
        for i in range(4):
            ax1.text(i, ing_vals[i] + _safe_max(ing_vals) * 0.025,
                     f'{ing_vals[i]:.0f}M', ha='center', fontsize=8.5, color='#1a6b34', fontweight='bold')
            ax1.text(i, gas_vals[i] - _safe_max(gas_vals) * 0.045,
                     f'{gas_vals[i]:.0f}M', ha='center', fontsize=8.5, color=COLOR_NEG, fontweight='bold')
            signo_b = '+' if bal_vals[i] >= 0 else ''
            ax1b.text(i, bal_vals[i] + (_safe_max([abs(b) for b in bal_vals]) * 0.06 * (1 if bal_vals[i] >= 0 else -1.5)),
                      f'{signo_b}{bal_vals[i]:.0f}M', ha='center', fontsize=8,
                      color=COLOR_INVEST if bal_vals[i] >= 0 else COLOR_CORTE, fontweight='bold')

        ax1.set_xticks(x_anios)
        ax1.set_xticklabels(anios, fontsize=10, fontweight='bold')
        ax1.set_ylabel('Millones de euros (M€)', fontsize=9)
        ax1.set_title('① ¿Cuánto dinero ganará el club cada año?', fontsize=12, fontweight='bold', pad=12)
        lines1, labels1 = ax1.get_legend_handles_labels()
        from matplotlib.patches import Patch
        lines1.append(Patch(facecolor=COLOR_INVEST, alpha=0.35, label='Balance +'))
        ax1.legend(handles=lines1, loc='upper left', fontsize=9, frameon=False)
        ax1.spines['top'].set_visible(False)
        ax1.tick_params(axis='y', labelsize=8.5)

        # -----------------------------------------------------------------
        # PANEL 2 — Cantera: inversión vs valor de mercado generado
        # -----------------------------------------------------------------
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_facecolor('#ffffff')

        inv_cant = cant['inversion']
        roi_cant = cant['roi_vm']

        bar_w = 0.35
        ax2.bar(x_anios - bar_w / 2, inv_cant, bar_w,
                color=COLOR_CANT, alpha=0.85, label='Inversión cantera', edgecolor='white')
        ax2.bar(x_anios + bar_w / 2, roi_cant, bar_w,
                color='#d35400', alpha=0.85, label='VM generado (jugadores propios)', edgecolor='white')

        # Etiquetas
        max_all = _safe_max(inv_cant + roi_cant)
        for i in range(4):
            ax2.text(i - bar_w / 2, inv_cant[i] + max_all * 0.02,
                     f'{inv_cant[i]:.1f}M', ha='center', fontsize=7.5, fontweight='bold', color='#5b2c6f')
            ax2.text(i + bar_w / 2, roi_cant[i] + max_all * 0.02,
                     f'{roi_cant[i]:.1f}M', ha='center', fontsize=7.5, fontweight='bold', color='#784212')

        # Anotación de ahorro en fichajes por cantera
        if cant['ahorro_fich_y3'] > 0:
            ax2.annotate(
                f"Ahorro en fichajes\npor cantera en Año 3:\n-{cant['ahorro_fich_y3']:.1f}M€",
                xy=(3 + bar_w / 2, roi_cant[3]),
                xytext=(2.3, roi_cant[3] * 0.7),
                fontsize=7.5, color='#c0392b', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2)
            )

        ax2.set_xticks(x_anios)
        ax2.set_xticklabels(anios, fontsize=9.5, fontweight='bold')
        ax2.set_ylabel('Millones de euros (M€)', fontsize=9)
        ax2.set_title(f'② Cantera: {cant["inversion"][0]:.1f}M€ → {cant["inversion"][3]:.1f}M€ en año 3\n(+{(cant["inversion"][1]-cant["inversion"][0]):.1f}M€/año)',
                      fontsize=11, fontweight='bold', pad=10)
        ax2.legend(fontsize=7.5, loc='upper left', frameon=False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.tick_params(axis='y', labelsize=8.5)

        # -----------------------------------------------------------------
        # PANEL 3 — Recortes e inversiones acumuladas en 3 años
        # -----------------------------------------------------------------
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor('#ffffff')

        keys_r = list(res['recortes'].keys())
        vals_r = list(res['recortes'].values())
        keys_i = list(res['inversiones'].keys())
        vals_i = list(res['inversiones'].values())

        all_keys = keys_r + keys_i
        all_vals = vals_r + vals_i
        all_cols = [COLOR_CORTE] * len(keys_r) + [COLOR_INVEST] * len(keys_i)

        if all_vals:
            max_v = _safe_max(all_vals)
            y_pos = np.arange(len(all_keys))
            bars_rc = ax3.barh(y_pos, all_vals, color=all_cols, height=0.55, alpha=0.88, edgecolor='white')
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(
                [k.replace('\n', ' ')[:38] + ('…' if len(k.replace('\n', ' ')) > 38 else '') for k in all_keys],
                fontsize=7.8
            )
            for i, v in enumerate(all_vals):
                ax3.text(v + max_v * 0.02, i, f'{v:.1f}M',
                         va='center', fontsize=8.5, fontweight='bold',
                         color=COLOR_NEG if all_cols[i] == COLOR_CORTE else '#1a6b34')
            ax3.legend(
                handles=[Patch(facecolor=COLOR_CORTE, label='Recorte / Ahorro'),
                          Patch(facecolor=COLOR_INVEST, label='Nueva inversión')],
                fontsize=8, loc='lower right', frameon=False
            )
        ax3.set_xlabel('Millones de euros acumulados (M€)', fontsize=8.5)
        ax3.set_title('③ ¿Dónde recortar y dónde reinvertir?', fontsize=11, fontweight='bold', pad=10)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.invert_yaxis()

        # -----------------------------------------------------------------
        # PANEL 4 — FPF año a año con zona objetivo
        # -----------------------------------------------------------------
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor('#ffffff')

        fpf_vals = [
            c['masa salarial'] / c['ingresos totales'] * 100,
        ]
        # FPF estimado año a año a partir de la masa y los ingresos proyectados
        masa_lp   = c['masa salarial']
        ahorro_lp = p['ahorro_total']
        for yr in range(1, 4):
            masa_yr    = masa_lp - ahorro_lp * (yr / 3)
            ingresos_yr = anual['ingresos'][yr]
            fpf_vals.append(masa_yr / max(ingresos_yr, 0.1) * 100)

        fpf_colors = [_color_fpf(v) for v in fpf_vals]

        # Zonas de fondo
        ax4.axhspan(0,  60, alpha=0.07, color='green',  zorder=0)
        ax4.axhspan(60, 73, alpha=0.07, color='orange', zorder=0)
        ax4.axhspan(73, max(max(fpf_vals), 85) * 1.15, alpha=0.07, color='red', zorder=0)

        bars_fpf = ax4.bar(x_anios, fpf_vals, color=fpf_colors, width=0.55, alpha=0.88, edgecolor='white')
        ax4.plot(x_anios, fpf_vals, 'o--', color='#333333', linewidth=1.2, markersize=6, zorder=5)
        ymax_f = max(max(fpf_vals), 85) * 1.15
        ax4.set_ylim(0, ymax_f)
        ax4.axhline(73, color='#e74c3c', linestyle='--', linewidth=1.2, label='Riesgo FPF (73%)')
        ax4.axhline(60, color='#f39c12', linestyle='--', linewidth=1.2, label='Zona segura (60%)')

        for i, v in enumerate(fpf_vals):
            ax4.text(i, v + ymax_f * 0.025, f'{v:.1f}%',
                     ha='center', fontsize=9, fontweight='bold')

        ax4.set_xticks(x_anios)
        ax4.set_xticklabels(anios, fontsize=9.5, fontweight='bold')
        ax4.set_ylabel('Masa salarial / Ingresos (%)', fontsize=9)
        ax4.set_title('④ FPF: evolución año a año', fontsize=11, fontweight='bold', pad=10)
        ax4.legend(fontsize=8, loc='upper right', frameon=False)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.tick_params(axis='y', labelsize=8.5)

        # -----------------------------------------------------------------
        # PANEL 5 — KPIs resumen: antes vs después
        # -----------------------------------------------------------------
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.set_facecolor('#ffffff')
        ax5.axis('off')

        bal_ini = p['balance_inicial']
        bal_fin = p['nuevo_balance']
        ing_ini = c['ingresos totales']
        ing_fin = anual['ingresos'][3]
        fpf_ini = fpf_vals[0]
        fpf_fin = fpf_vals[3]
        masa_ini = c['masa salarial']
        masa_fin = p['nueva_masa']
        cant_ini = c['inversion en cantera']
        cant_fin = cant['inversion'][3]

        ax5.text(0.5, 0.97, '⑤ Resumen · Antes vs Después (Año 3)',
                 ha='center', fontsize=11, fontweight='bold', transform=ax5.transAxes, color='#1a1a2e')

        kpis = [
            ('Balance operativo',  f'{bal_ini:+.0f}M€',  f'{bal_fin:+.0f}M€',  bal_fin > bal_ini),
            ('Ingresos totales',   f'{ing_ini:.0f}M€',   f'{ing_fin:.0f}M€',   True),
            ('Masa salarial',      f'{masa_ini:.0f}M€',  f'{masa_fin:.0f}M€',  masa_fin < masa_ini),
            ('Ratio FPF',          f'{fpf_ini:.1f}%',    f'{fpf_fin:.1f}%',    fpf_fin < fpf_ini),
            ('Inversión cantera',  f'{cant_ini:.1f}M€',  f'{cant_fin:.1f}M€',  cant_fin > cant_ini),
        ]
        headers = ['Indicador', 'Actual', 'Año 3', '']
        col_x   = [0.02, 0.38, 0.65, 0.88]
        y_start = 0.87

        # Cabecera
        for hx, h in zip(col_x, headers):
            ax5.text(hx, y_start, h, fontsize=8.5, fontweight='bold',
                     color='#444444', transform=ax5.transAxes)
        ax5.plot([0.02, 0.98], [y_start - 0.04, y_start - 0.04],
                 color='#cccccc', lw=1.0, transform=ax5.transAxes)

        y = y_start - 0.11
        for label, v_ini, v_fin, es_mejora in kpis:
            tick  = '▲' if es_mejora else '▼'
            color_tick = COLOR_INVEST if es_mejora else COLOR_CORTE
            ax5.text(col_x[0], y, label,  fontsize=8.5, transform=ax5.transAxes, color='#555555')
            ax5.text(col_x[1], y, v_ini,  fontsize=9,   transform=ax5.transAxes, color='#888888', fontweight='bold')
            ax5.text(col_x[2], y, v_fin,  fontsize=9.5, transform=ax5.transAxes, color='#1a1a2e', fontweight='bold')
            ax5.text(col_x[3], y, tick,   fontsize=11,  transform=ax5.transAxes, color=color_tick, fontweight='bold')
            ax5.plot([0.02, 0.98], [y - 0.045, y - 0.045],
                     color='#eeeeee', lw=0.7, transform=ax5.transAxes)
            y -= 0.13

        # Nota sobre cantera
        ax5.text(0.5, 0.06,
                 f"La cantera generará ~{cant['roi_vm'][3]:.0f}M€ en VM de\njugadores propios en Año 3,\n"
                 f"ahorrando {cant['ahorro_fich_y3']:.0f}M€ en fichajes.",
                 ha='center', fontsize=8, color=COLOR_CANT, style='italic',
                 transform=ax5.transAxes,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f3e5f5', edgecolor=COLOR_CANT, alpha=0.7))

    # =========================================================================
    # Guardado
    # =========================================================================
    nombre_archivo = f"financiero_{plazo}_{nombre_club.replace(' ', '_')}.png"
    plt.savefig(nombre_archivo, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()
    print(f"  Gráfico guardado: {nombre_archivo}")


def grafico_deportivo(res):
    if res is None:
        print("No hay datos deportivos disponibles.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle(f"Análisis deportivo: {res['nombre_club']}", fontsize=13, fontweight='bold')

    # Panel 1: % de venta por jugador
    ax = axes[0]
    plantilla = res['candidatos_venta_todos'].dropna(subset=['Pct_venta']).head(18)
    nombres = [f"{row['Nombre']}  [{row['Pos']}] {int(row['Edad'])}a" for _, row in plantilla.iterrows()]
    pcts    = plantilla['Pct_venta'].tolist()
    colores = []
    for p in pcts:
        if p >= 80:   colores.append('red')
        elif p >= 60: colores.append('orange')
        elif p >= 40: colores.append('gold')
        elif p >= 20: colores.append('lightgray')
        else:         colores.append('mediumseagreen')

    ax.barh(range(len(nombres)), pcts, color=colores, height=0.65)
    ax.set_yticks(range(len(nombres)))
    ax.set_yticklabels(nombres, fontsize=8.5)
    ax.set_xlabel('% recomendación de venta')
    ax.set_xlim(0, 115)
    ax.axvline(80, color='red',    linestyle='--', linewidth=1, alpha=0.6, label='Vender ya (>80%)')
    ax.axvline(60, color='orange', linestyle='--', linewidth=1, alpha=0.6, label='Vender (>60%)')
    ax.axvline(40, color='gold',   linestyle='--', linewidth=1, alpha=0.6, label='Analizar (>40%)')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_title('Recomendación de venta por jugador')
    ax.invert_yaxis()
    for i, p in enumerate(pcts):
        ax.text(p + 1, i, f'{p:.0f}%', va='center', fontsize=8, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: fichajes recomendados
    ax = axes[1]
    ax.axis('off')
    ax.text(0.5, 0.97, 'Fichajes recomendados', ha='center', fontsize=11,
            fontweight='bold', transform=ax.transAxes)
    modo = res.get('modo', 'deportivo')
    subtitulo = '(prioridad: mejor rendimiento)' if modo == 'deportivo' else '(prioridad: mejor coste/rendimiento)'
    ax.text(0.5, 0.92, subtitulo, ha='center', fontsize=8, color='gray',
            transform=ax.transAxes, style='italic')
    if not res['fichajes'].empty:
        y = 0.84
        for _, j in res['fichajes'].iterrows():
            ax.text(0.02, y, f"Por {j['Nombre_sale']}  ->  {j['Nombre']} ({j['Equipo']})",
                    fontsize=9, fontweight='bold', transform=ax.transAxes)
            ax.text(0.02, y - 0.045,
                    f"   {j['Pos_necesaria']} | {int(j['Edad'])}a | Rend: {j['Rendimiento']:.1f} "
                    f"| VM: {j['VM']:.1f}M€ | Coste: {j['Coste traspaso']:.1f}M€",
                    fontsize=8, color='gray', transform=ax.transAxes)
            ax.plot([0.02, 0.98], [y - 0.07, y - 0.07], color='lightgray',
                    linewidth=0.5, transform=ax.transAxes)
            y -= 0.13
            if y < 0.05:
                break
    else:
        ax.text(0.5, 0.5, 'Sin candidatos disponibles', ha='center',
                va='center', fontsize=10, color='gray', transform=ax.transAxes)

    # Panel 3: proyección de posición en liga
    ax = axes[2]
    p_ant  = res['posicion_ant']
    p_proy = res['posicion_proy']
    delta  = p_ant - p_proy   # positivo = sube puestos (posición numérica baja)
    # Sube puestos → verde; baja o igual → naranja/rojo
    color_barra = 'mediumseagreen' if delta > 0 else ('orange' if delta == 0 else 'tomato')

    bars = ax.bar(['Posición\nactual', 'Posición\nproyectada'],
                  [p_ant, p_proy], color=['steelblue', color_barra], width=0.45)
    ax.set_ylabel('Posición en LaLiga (1 = 1.º)')
    ax.set_title('Proyección de posición en liga\n(basada en rendimiento medio interpolado)')
    ax.invert_yaxis()   # 1.º arriba, 20.º abajo
    ax.set_ylim(21, 0)
    ax.set_yticks(range(1, 21))
    for bar, v in zip(bars, [p_ant, p_proy]):
        ax.text(bar.get_x() + bar.get_width() / 2, v - 0.4,
                f'{v}º', ha='center', fontsize=16, fontweight='bold', color='white')
    if delta > 0:
        msg = f'▲ Sube {delta} puesto{"s" if delta > 1 else ""}'
    elif delta == 0:
        msg = '= Mantiene posición'
    else:
        msg = f'▼ Baja {abs(delta)} puesto{"s" if abs(delta) > 1 else ""}'
    ax.text(0.5, 0.10, msg,
            ha='center', transform=ax.transAxes, fontsize=11,
            fontweight='bold', color=color_barra)
    n_mov = len(res['candidatos_venta'])
    ax.text(0.5, 0.04, f'({n_mov} movimientos ejecutados)',
            ha='center', transform=ax.transAxes, fontsize=8, color='gray')
    ax.text(0.5, 0.96, f"Ingresos ventas:  +{res['ingreso_ventas']:.1f}M€",
            ha='center', transform=ax.transAxes, fontsize=8.5, color='mediumseagreen',
            verticalalignment='top')
    ax.text(0.5, 0.90, f"Gasto fichajes:   -{res['gasto_fichajes']:.1f}M€",
            ha='center', transform=ax.transAxes, fontsize=8.5, color='tomato',
            verticalalignment='top')
    color_bal = 'mediumseagreen' if res['balance_mercado'] >= 0 else 'tomato'
    ax.text(0.5, 0.83, f"Balance neto:  {res['balance_mercado']:+.1f}M€",
            ha='center', transform=ax.transAxes, fontsize=9.5,
            fontweight='bold', color=color_bal, verticalalignment='top')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    nombre_archivo = f"deportivo_{res['nombre_club'].replace(' ', '_')}.png"
    plt.savefig(nombre_archivo, dpi=130, bbox_inches='tight')
    plt.show()
    print(f"  Gráfico guardado: {nombre_archivo}")


# =============================================================================
# IMPRIMIR RESULTADOS POR CONSOLA
# =============================================================================

def mostrar_diagnostico(diag):
    print()
    print("=" * 60)
    print(f"  SOSTENIBILIDAD: {diag['nombre'].upper()}")
    print("=" * 60)
    print(f"\n  Resultado: {diag['semaforo']}  ({diag['salud']}% de sostenibilidad global)\n")
    print("  Indicadores de sostenibilidad:")
    for k, v in diag['scores'].items():
        barra = '|' * int(v) + '-' * (5 - int(v))
        icono = 'OK' if v >= 4 else ('??' if v >= 2 else 'MAL')
        print(f"    [{icono}] {k:<45} [{barra}] {v}/5")
    r = diag['ratios']
    print(f"\n  Ratios clave:")
    print(f"    FPF - Ratio salarial (Sal/Ing):       {r['fpf']:.1f}%")
    print(f"    Eficiencia operativa (Ing/Gas):       {r['ef_operativa']:.3f}x")
    print(f"    Revalorizacion plantilla (VM/Coste):  {r['vm_ratio']:.2f}x")
    print(f"    Ingresos totales:                     {r['ingresos']:.1f}M€")
    print(f"    Gastos totales:                       {r['gastos']:.1f}M€")
    print(f"    Coste historico traspasos:            {r['coste_traspasos']:.1f}M€")
    print()


def mostrar_financiero(res):
    c     = res['club']
    plazo = res['plazo']
    p     = res['proyeccion']
    n     = res.get('notas', {})
    print()
    print("=" * 68)
    print(f"  PLAN FINANCIERO A {plazo.upper()} PLAZO: {c['Nombre del club'].upper()}")
    print("=" * 68)
    print(f"\n  Situación de partida:")
    print(f"    FPF actual (masa/ingresos):   {n.get('fpf_actual','?')}%  [{n.get('fpf_nivel','?')}]")
    print(f"    Ingresos comerciales:         {n.get('pct_comercial','?')}%  (mediana LaLiga: {n.get('med_comercial_pct','?')}%)")
    print(f"    Ingresos matchday/abonados:   {n.get('pct_matchday','?')}%  (mediana LaLiga: {n.get('med_matchday_pct','?')}%)")
    if plazo == 'largo':
        print(f"    Inversión cantera actual:     {n.get('pct_cantera','?')}%  sobre masa salarial")
    print(f"    Gasto actual en fichajes:     {n.get('g_fich_actual','?')}M€  (objetivo sano: {n.get('objetivo_fich','?')}M€)")
    print()
    if res['recortes']:
        print("  Palancas de ahorro:")
        for k, v in res['recortes'].items():
            print(f"    - {k}: {v:.1f}M€")
    else:
        print("  Palancas de ahorro: ninguna necesaria (club saneado)")
    print()
    if res['inversiones']:
        print("  Palancas de crecimiento:")
        for k, v in res['inversiones'].items():
            print(f"    + {k}: {v:.1f}M€")
    print()
    print("  Proyección:")
    print(f"    Nueva masa salarial:          {p['nueva_masa']:.1f}M€  (actual: {c['masa salarial']:.1f}M€)")
    print(f"    Nuevo ratio FPF:              {p['nuevo_fpf']:.1f}%")
    print(f"    Ahorro total estimado:        {p.get('ahorro_total', 0):.1f}M€")
    print(f"    Ingresos extra estimados:     {p['ingresos_extra']:.1f}M€")
    balance_label = p['nuevo_balance']
    signo = '+' if balance_label >= 0 else ''
    print(f"    Nuevo balance operativo:      {signo}{balance_label:.1f}M€")
    print()


def mostrar_deportivo(res):
    if res is None:
        return
    print()
    print("=" * 60)
    print(f"  ANÁLISIS DEPORTIVO: {res['nombre_club'].upper()}")
    print("=" * 60)
    print(f"\n  {'Jugador':<22} {'Pos':<5} {'Edad':<5} {'Rend':<7} {'Salario (M€)':<14} {'VM (M€)':<10} {'Venta%':<8} Decisión")
    print("  " + "-" * 88)
    for _, j in res['plantilla'].iterrows():
        if pd.isna(j.get('Nombre')):
            continue
        print(f"  {str(j['Nombre']):<22} {str(j.get('Pos','?')):<5} {int(j['Edad']):<5} "
              f"{j['Rendimiento']:<7.1f} {j['Salario']/1_000_000:<14.2f} {j['VM']:<10.1f} "
              f"{j['Pct_venta']:<8.0f} {j['Decision']}")

    print(f"\n  Candidatos a salir:")
    for _, j in res['candidatos_venta'].iterrows():
        print(f"    -> {j['Nombre']} ({j['Pos']}) | {int(j['Edad'])}a | "
              f"Rend: {j['Rendimiento']:.1f} | VM: {j['VM']:.1f}M€ | {j['Pct_venta']:.0f}%")

    modo = res.get('modo', 'deportivo')
    if modo == 'deportivo':
        print(f"\n  Fichajes recomendados — MODO DEPORTIVO (mejor jugador disponible):")
    else:
        print(f"\n  Fichajes recomendados — MODO MIXTO (mejor relación calidad/precio):")
    if not res['fichajes'].empty:
        for _, j in res['fichajes'].iterrows():
            extra = f" | Efic: {j['Eficiencia']:.2f}" if 'Eficiencia' in j and modo == 'mixto' else ""
            print(f"    -> Sale {j['Nombre_sale']:<20}  Entra {j['Nombre']} ({j['Equipo']}) | "
                  f"{j['Pos_necesaria']} | {int(j['Edad'])}a | Rend: {j['Rendimiento']:.1f} | "
                  f"VM: {j['VM']:.1f}M€ | Coste: {j['Coste traspaso']:.1f}M€{extra}")
    else:
        print("    Sin candidatos disponibles en el mercado.")

    n_mov = len(res['candidatos_venta'])
    print(f"\n  Movimientos ejecutados: {n_mov} (los {n_mov} de mayor prioridad)")
    print(f"  Ingresos por ventas:               +{res['ingreso_ventas']:.1f}M€")
    print(f"  Gasto en fichajes:                 -{res['gasto_fichajes']:.1f}M€")
    print(f"  Balance neto de mercado:            {res['balance_mercado']:+.1f}M€")
    print(f"  Posición actual en liga:            {res['posicion_ant']}º")
    print(f"  Posición proyectada (con fichajes): {res['posicion_proy']}º", end='')
    delta_pos = res['posicion_ant'] - res['posicion_proy']
    if delta_pos > 0:
        print(f"  (▲ sube {delta_pos} puesto{'s' if delta_pos > 1 else ''})")
    elif delta_pos == 0:
        print(f"  (= mantiene posición)")
    else:
        print(f"  (▼ baja {abs(delta_pos)} puesto{'s' if abs(delta_pos) > 1 else ''})")
    print()


# =============================================================================
# GRAFICO ANTES / DESPUES
# =============================================================================

def grafico_antes_despues(diag, res_fin=None, res_dep=None):
    tiene_fin = res_fin is not None
    tiene_dep = res_dep is not None

    if not tiene_fin and not tiene_dep:
        print("  No hay recomendaciones que comparar.")
        return

    nombre_club = diag['nombre']
    c = diag['club']

    if tiene_fin:
        p = res_fin['proyeccion']
        r = diag['ratios']
        fin_antes = {
            'FPF (%)':               round(r['fpf'], 1),
            'Ef. Operativa (x)':     round(r['ef_operativa'], 2),
            'Sostenibilidad (%)':    diag['salud'],
        }

        μ = DIST_METRICAS
        nueva_masa = p['nueva_masa']
        nueva_ing  = c['ingresos totales'] * (1.07**3 if res_fin['plazo'] == 'largo' else 1) + p.get('ingresos_extra', 0)
        nuevo_gas  = max(c['gastos totales'] - sum(res_fin['recortes'].values()), nueva_ing * 0.5)
        nuevo_fpf  = nueva_masa / max(nueva_ing, 0.1) * 100
        nuevo_ef   = nueva_ing  / max(nuevo_gas, 0.1)
        z_fpf_n = (nuevo_fpf - μ['fpf']['mean']) / μ['fpf']['std']
        z_ef_n  = (nuevo_ef  - μ['ef']['mean'])  / μ['ef']['std']
        pts_fpf_n = float(np.clip(5 * norm.cdf(-z_fpf_n), 0, 5))
        pts_ef_n  = float(np.clip(5 * norm.cdf( z_ef_n),  0, 5))

        irs_nuevo = (
            pts_fpf_n +
            pts_ef_n  +
            diag['scores'].get('Revalorizacion plantilla (VM/Coste hist.)', 2.5)
        )
        salud_nueva = round(irs_nuevo / 15 * 100, 1)

        fin_despues = {
            'FPF (%)':               round(nuevo_fpf, 1),
            'Ef. Operativa (x)':     round(nuevo_ef, 2),
            'Sostenibilidad (%)':    salud_nueva,
        }

    n_cols = sum([tiene_fin, tiene_dep])
    fig = plt.figure(figsize=(7 * n_cols + 2, 9))
    fig.suptitle(f'Impacto de las recomendaciones: {nombre_club}', fontsize=14, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, n_cols, figure=fig, hspace=0.55, wspace=0.35)
    col = 0

    if tiene_fin:
        plazo_label = 'corto plazo' if res_fin['plazo'] == 'corto' else 'largo plazo (3 años)'

        ax1 = fig.add_subplot(gs[0, col])
        metricas = list(fin_antes.keys())
        vals_a   = list(fin_antes.values())
        vals_d   = list(fin_despues.values())
        x = np.arange(len(metricas)); w = 0.32
        bars_a = ax1.bar(x - w/2, vals_a, w, label='Antes',   color='#E24B4A', alpha=0.85)
        bars_d = ax1.bar(x + w/2, vals_d, w, label='Después', color='#639922', alpha=0.85)
        ax1.set_xticks(x); ax1.set_xticklabels(metricas, fontsize=8)
        ax1.set_title(f'Indicadores financieros\n({plazo_label})', fontsize=9, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
        for b in bars_a:
            ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{b.get_height():.1f}', ha='center', fontsize=7.5, color='#A32D2D')
        for b in bars_d:
            ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{b.get_height():.1f}', ha='center', fontsize=7.5, color='#27500A')

        ax2 = fig.add_subplot(gs[1, col])
        nueva_masa = res_fin['proyeccion']['nueva_masa']
        brs = ax2.bar(['Masa\nactual', 'Masa\nproyectada', 'Ingresos\ntotales'],
                      [c['masa salarial'], nueva_masa, c['ingresos totales']],
                      color=['#E24B4A', '#639922', '#378ADD'], width=0.5, alpha=0.85)
        ax2.axhline(c['ingresos totales'] * 0.70, color='gray', linestyle='--', linewidth=1, label='Límite FPF 70%')
        ax2.set_ylabel('Millones €', fontsize=8)
        ax2.set_title('Masa salarial vs ingresos', fontsize=9, fontweight='bold')
        ax2.legend(fontsize=7)
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
        for b in brs:
            ax2.text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{b.get_height():.0f}M', ha='center', fontsize=8, fontweight='bold')

        ax3 = fig.add_subplot(gs[2, col])
        ax3.axis('off')
        salud_a = diag['salud']; salud_d = fin_despues['Sostenibilidad (%)']
        sem_a   = diag['semaforo']
        sem_d   = 'SOSTENIBLE' if salud_d >= 65 else ('EN RIESGO' if salud_d >= 40 else 'NO SOSTENIBLE')
        col_a   = '#E24B4A' if sem_a == 'NO SOSTENIBLE' else ('#EF9F27' if sem_a == 'EN RIESGO' else '#639922')
        col_d   = '#E24B4A' if sem_d == 'NO SOSTENIBLE' else ('#EF9F27' if sem_d == 'EN RIESGO' else '#639922')
        delta_s = salud_d - salud_a; signo = '+' if delta_s >= 0 else ''
        ax3.text(0.5,  0.92, 'Índice de salud financiera', ha='center', fontsize=9, fontweight='bold', transform=ax3.transAxes)
        ax3.text(0.22, 0.60, f'{salud_a}%', ha='center', fontsize=28, fontweight='bold', color=col_a, transform=ax3.transAxes)
        ax3.text(0.22, 0.35, sem_a, ha='center', fontsize=9, color=col_a, transform=ax3.transAxes)
        ax3.text(0.50, 0.55, '->', ha='center', fontsize=22, color='gray', transform=ax3.transAxes)
        ax3.text(0.78, 0.60, f'{salud_d}%', ha='center', fontsize=28, fontweight='bold', color=col_d, transform=ax3.transAxes)
        ax3.text(0.78, 0.35, sem_d, ha='center', fontsize=9, color=col_d, transform=ax3.transAxes)
        col_delta = '#639922' if delta_s >= 0 else '#E24B4A'
        ax3.text(0.5, 0.12, f'Mejora: {signo}{delta_s:.1f} puntos porcentuales', ha='center', fontsize=9,
                 color=col_delta, fontweight='bold', transform=ax3.transAxes)
        col += 1

    if tiene_dep:
        pos_a = res_dep['posicion_ant']; pos_d = res_dep['posicion_proy']
        delta_pos = pos_a - pos_d   # positivo = sube puestos
        plantilla_actual = res_dep['plantilla']
        candidatos = res_dep['candidatos_venta']
        fichajes   = res_dep['fichajes']

        rend_antes = plantilla_actual['Rendimiento'].mean()
        if not fichajes.empty and not candidatos.empty:
            rend_sin = plantilla_actual[~plantilla_actual['Nombre'].isin(candidatos['Nombre'])]['Rendimiento'].mean()
            rend_fic = fichajes['Rendimiento'].mean()
            n_nv = len(fichajes); n_tot = len(plantilla_actual) - len(candidatos) + n_nv
            rend_despues = (rend_sin * (n_tot - n_nv) + rend_fic * n_nv) / n_tot
        else:
            rend_despues = rend_antes

        vm_antes   = plantilla_actual['VM'].sum()   # M€
        vm_despues = vm_antes - candidatos['VM'].sum() + (fichajes['VM'].sum() if not fichajes.empty else 0)

        ax4 = fig.add_subplot(gs[0, col])
        col_d2 = '#639922' if delta_pos > 0 else ('#EF9F27' if delta_pos == 0 else '#E24B4A')
        brs = ax4.bar(['Posición\nactual', 'Posición\nproyectada'], [pos_a, pos_d],
                       color=['#378ADD', col_d2], width=0.45, alpha=0.9)
        ax4.set_title('Posición proyectada en liga', fontsize=9, fontweight='bold')
        ax4.set_ylabel('Posición (1 = 1.º)')
        ax4.invert_yaxis()
        ax4.set_ylim(21, 0)
        ax4.set_yticks(range(1, 21, 2))
        ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)
        for b in brs:
            ax4.text(b.get_x()+b.get_width()/2, b.get_height()-0.3,
                     f'{int(b.get_height())}º', ha='center', fontsize=12,
                     fontweight='bold', color='white', va='top')
        if delta_pos > 0:
            msg_pos = f'▲ Sube {delta_pos} puesto{"s" if delta_pos > 1 else ""}'
        elif delta_pos == 0:
            msg_pos = '= Mantiene posición'
        else:
            msg_pos = f'▼ Baja {abs(delta_pos)} puesto{"s" if abs(delta_pos) > 1 else ""}'
        ax4.text(0.5, 0.08, msg_pos, ha='center', transform=ax4.transAxes,
                 fontsize=9, fontweight='bold', color=col_d2)

        ax5 = fig.add_subplot(gs[1, col])
        cats = ['Rend. medio\nplantilla', 'Valor mercado\nplantilla (M€)']
        x2 = np.arange(len(cats))
        ba = ax5.bar(x2 - 0.18, [round(rend_antes,2), round(vm_antes,1)], 0.32, label='Antes', color='#E24B4A', alpha=0.85)
        bd = ax5.bar(x2 + 0.18, [round(rend_despues,2), round(vm_despues,1)], 0.32, label='Después', color='#639922', alpha=0.85)
        ax5.set_xticks(x2); ax5.set_xticklabels(cats, fontsize=8)
        ax5.set_title('Calidad de la plantilla', fontsize=9, fontweight='bold')
        ax5.legend(fontsize=8)
        ax5.spines['top'].set_visible(False); ax5.spines['right'].set_visible(False)
        for b in ba:
            ax5.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{b.get_height():.1f}', ha='center', fontsize=7.5, color='#A32D2D')
        for b in bd:
            ax5.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{b.get_height():.1f}', ha='center', fontsize=7.5, color='#27500A')

        ax6 = fig.add_subplot(gs[2, col])
        ax6.axis('off')
        bm = res_dep['balance_mercado']   # M€
        col_bm = '#639922' if bm >= 0 else '#E24B4A'
        signo_bm = '+' if bm >= 0 else ''
        ax6.text(0.5, 0.93, 'Resumen de movimientos', ha='center', fontsize=9, fontweight='bold', transform=ax6.transAxes)
        items = [
            ('Jugadores que salen:',     f'{len(candidatos)}'),
            ('Jugadores que entran:',    f'{len(fichajes) if not fichajes.empty else 0}'),
            ('Ingresos por ventas:',     f'+{candidatos["VM"].sum():.1f}M€'),
            ('Gasto en fichajes:',       f'-{fichajes["VM"].sum():.1f}M€' if not fichajes.empty else '0M€'),
            ('Balance neto de mercado:', f'{signo_bm}{bm:.1f}M€'),
        ]
        y = 0.76
        for label, val in items:
            ax6.text(0.05, y, label, fontsize=8.5, color='gray', transform=ax6.transAxes)
            ax6.text(0.95, y, val, fontsize=8.5, fontweight='bold', ha='right',
                     color=col_bm if 'Balance' in label else '#333333', transform=ax6.transAxes)
            y -= 0.14
        ax6.plot([0.05, 0.95], [0.10, 0.10], color='lightgray', linewidth=0.5, transform=ax6.transAxes)
        modo_label = 'Deportivo (calidad máxima)' if res_dep['modo'] == 'deportivo' else 'Mixto (mejor coste/rend.)'
        ax6.text(0.5, 0.04, f'Modo: {modo_label}', ha='center', fontsize=7.5, color='gray',
                 style='italic', transform=ax6.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    nombre_archivo = f"antes_despues_{nombre_club.replace(' ', '_')}.png"
    plt.savefig(nombre_archivo, dpi=130, bbox_inches='tight')
    plt.show()
    print(f"  Gráfico guardado: {nombre_archivo}")


# =============================================================================
# REEVALUACIÓN POST-MODELO FINANCIERO
# Recalcula el diagnóstico con las cifras proyectadas tras aplicar el plan
# =============================================================================

def diagnostico_proyectado(diag, res_fin):
    """
    Recalcula los 3 indicadores de sostenibilidad con las cifras proyectadas
    tras aplicar el modelo financiero (corto o largo plazo).
    Las puntuaciones se calculan con distribución normal (igual que diagnostico_club).

    Qué cambia vs el diagnóstico original:
      · FPF           → nueva masa salarial + ingresos mejorados
      · Ef. Operativa → ingresos y gastos proyectados
      · VM plantilla  → sin cambio (el modelo financiero no toca traspasos)
    """
    p      = res_fin['proyeccion']
    c      = diag['club']
    plazo  = res_fin['plazo']

    # ── Ingresos proyectados ────────────────────────────────────────────────
    if plazo == 'largo':
        # Crecimiento orgánico 3 años (7%/año) + mejoras comerciales del plan
        nueva_ing = c['ingresos totales'] * (1.07 ** 3) + p.get('ingresos_extra', 0)
    else:
        nueva_ing = c['ingresos totales'] + p.get('ingresos_extra', 0)

    # ── Gastos proyectados ──────────────────────────────────────────────────
    ahorro_total = p.get('ahorro_total', 0)
    nuevo_gas    = max(c['gastos totales'] - ahorro_total, nueva_ing * 0.5)

    nueva_masa = p['nueva_masa']

    # ── Funciones de puntuación con distribución normal (igual que diagnostico_club) ─
    μ = DIST_METRICAS

    nuevo_fpf = nueva_masa / max(nueva_ing, 0.1) * 100
    nuevo_ef  = nueva_ing  / max(nuevo_gas,  0.1)

    z_fpf_p   = (nuevo_fpf - μ['fpf']['mean']) / μ['fpf']['std']
    z_ef_p    = (nuevo_ef  - μ['ef']['mean'])  / μ['ef']['std']

    puntos_fpf_p = round(float(np.clip(5 * norm.cdf(-z_fpf_p), 0, 5)), 2)
    puntos_ef_p  = round(float(np.clip(5 * norm.cdf( z_ef_p),  0, 5)), 2)

    scores_proy = {
        'FPF - Ratio salarial (Sal/Ing)':            puntos_fpf_p,
        'Eficiencia operativa (Ing/Gas)':            puntos_ef_p,
        'Revalorizacion plantilla (VM/Coste hist.)': diag['scores']['Revalorizacion plantilla (VM/Coste hist.)'],
    }

    total_proy  = sum(scores_proy.values())
    salud_proy  = round(total_proy / 15 * 100, 1)   # 3 métricas × 5 pts = 15
    semaforo_proy = ('SOSTENIBLE'    if salud_proy >= 65 else
                     'EN RIESGO'     if salud_proy >= 40 else
                     'NO SOSTENIBLE')

    return {
        'nombre':   diag['nombre'],
        'scores':   scores_proy,
        'salud':    salud_proy,
        'semaforo': semaforo_proy,
        'plazo':    plazo,
        'ratios': {
            'fpf':          round(nuevo_fpf, 1),
            'ef_operativa': round(nuevo_ef, 3),
            'vm_ratio':     diag['ratios']['vm_ratio'],
            'ingresos':     round(nueva_ing, 1),
            'gastos':       round(nuevo_gas, 1),
            'salarios':     round(nueva_masa, 1),
        },
    }


def mostrar_reevaluacion(diag_orig, diag_proy):
    """
    Imprime por consola el diagnóstico antes / después del modelo financiero.
    """
    plazo_label = 'CORTO PLAZO' if diag_proy['plazo'] == 'corto' else 'LARGO PLAZO (3 años)'
    delta       = diag_proy['salud'] - diag_orig['salud']
    signo       = '+' if delta >= 0 else ''
    col_sem     = {'SOSTENIBLE': '✅', 'EN RIESGO': '⚠️ ', 'NO SOSTENIBLE': '❌'}

    print()
    print("=" * 60)
    print(f"  REEVALUACIÓN TRAS APLICAR EL MODELO · {plazo_label}")
    print("=" * 60)

    # ── Cabecera antes / después ────────────────────────────────────────────
    print(f"\n  {'':30} {'ANTES':>10}   {'DESPUÉS':>10}   DELTA")
    print(f"  {'-'*58}")
    print(f"  {'Sostenibilidad global':30} {diag_orig['salud']:>9.1f}%  {diag_proy['salud']:>9.1f}%  {signo}{delta:.1f} pp")
    print(f"  {'Estado':30} {col_sem.get(diag_orig['semaforo'], '') + diag_orig['semaforo']:>10}   "
          f"{col_sem.get(diag_proy['semaforo'], '') + diag_proy['semaforo']:>10}")

    # ── Tabla de indicadores ────────────────────────────────────────────────
    print(f"\n  Indicadores  (0-5 pts):")
    print(f"  {'':45} {'Antes':>5}  {'Desp.':>5}  {'Δ':>4}")
    print(f"  {'-'*58}")
    for k in diag_orig['scores']:
        va = diag_orig['scores'][k]
        vd = diag_proy['scores'][k]
        dd = vd - va
        sd = ('+' if dd > 0 else '') + str(dd) if dd != 0 else '  ='
        icono = '▲' if dd > 0 else ('▼' if dd < 0 else '─')
        print(f"  {k:<45} {va:>5}  {vd:>5}  {icono} {sd}")

    # ── Ratios clave proyectados ────────────────────────────────────────────
    ro = diag_orig['ratios'];  rp = diag_proy['ratios']
    print(f"\n  Ratios clave:")
    print(f"    FPF - Ratio salarial:       {ro['fpf']:.1f}%  →  {rp['fpf']:.1f}%")
    print(f"    Eficiencia operativa:       {ro['ef_operativa']:.3f}x  →  {rp['ef_operativa']:.3f}x")
    print(f"    Ingresos proyectados:       {ro['ingresos']:.1f}M€  →  {rp['ingresos']:.1f}M€")
    print(f"    Gastos proyectados:         {ro['gastos']:.1f}M€  →  {rp['gastos']:.1f}M€")
    print(f"    Masa salarial proyectada:   {ro['salarios']:.1f}M€  →  {rp['salarios']:.1f}M€")

    # ── Veredicto final ─────────────────────────────────────────────────────
    print()
    if diag_proy['semaforo'] == 'SOSTENIBLE' and diag_orig['semaforo'] != 'SOSTENIBLE':
        print(f"  ✅  ¡El club PASA a ser SOSTENIBLE tras aplicar el modelo!")
    elif diag_proy['semaforo'] == diag_orig['semaforo']:
        print(f"  ℹ️   El estado se mantiene en {diag_proy['semaforo']} pero mejora {signo}{delta:.1f} pp.")
    else:
        print(f"  ⚠️   Mejora de {diag_orig['semaforo']} a {diag_proy['semaforo']} ({signo}{delta:.1f} pp).")
    print()


def grafico_reevaluacion(diag_orig, diag_proy):
    """
    Genera un gráfico de reevaluación con:
      · Panel izquierdo : dos gauges semicirculares antes / después
      · Panel central   : barras comparativas por indicador
      · Panel derecho   : resumen de ratios clave (tabla visual)
    """
    salud_a  = diag_orig['salud'];  sem_a = diag_orig['semaforo']
    salud_d  = diag_proy['salud'];  sem_d = diag_proy['semaforo']
    delta    = salud_d - salud_a
    plazo_label = 'Corto plazo' if diag_proy['plazo'] == 'corto' else 'Largo plazo (3 años)'

    col_sem  = lambda s: ('#27ae60' if s == 'SOSTENIBLE' else ('#f39c12' if s == 'EN RIESGO' else '#e74c3c'))
    col_a    = col_sem(sem_a)
    col_d    = col_sem(sem_d)
    col_delta = '#27ae60' if delta >= 0 else '#e74c3c'

    fig = plt.figure(figsize=(18, 7))
    fig.patch.set_facecolor('#f8f9fa')
    fig.suptitle(
        f"Reevaluación de sostenibilidad tras el modelo financiero · {diag_orig['nombre']} · {plazo_label}",
        fontsize=13, fontweight='bold', y=0.99, color='#1a1a2e'
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38, left=0.04, right=0.97, top=0.88, bottom=0.06)

    # =========================================================================
    # PANEL IZQUIERDO — Dos gauges semicirculares: ANTES y DESPUÉS
    # =========================================================================
    ax_g = fig.add_subplot(gs[0, 0])
    ax_g.axis('off')
    ax_g.set_xlim(-0.1, 2.1)
    ax_g.set_ylim(-0.6, 1.5)

    def _gauge(cx, cy, salud, color, label, ax, r=0.82):
        """Dibuja un gauge semicircular centrado en (cx, cy)."""
        angulos_bg = np.linspace(np.pi, 0, 200)
        xs = cx + r * np.cos(angulos_bg)
        ys = cy + r * np.sin(angulos_bg)
        ax.plot(xs, ys, color='#ecf0f1', linewidth=16, solid_capstyle='round')

        # Zonas de color
        for pct_ini, pct_fin, col_zona in [(0, 0.40, '#fadbd8'), (0.40, 0.65, '#fdebd0'), (0.65, 1.0, '#d5f5e3')]:
            angs = np.linspace(np.pi - np.pi * pct_ini, np.pi - np.pi * pct_fin, 80)
            ax.plot(cx + r * np.cos(angs), cy + r * np.sin(angs), color=col_zona, linewidth=16, solid_capstyle='butt')

        # Relleno activo
        angs_act = np.linspace(np.pi, np.pi - np.pi * salud / 100, 200)
        ax.plot(cx + r * np.cos(angs_act), cy + r * np.sin(angs_act),
                color=color, linewidth=16, solid_capstyle='round', alpha=0.92)

        # Aguja
        ang_ag = np.pi - np.pi * salud / 100
        ax.annotate('', xy=(cx + 0.60 * np.cos(ang_ag), cy + 0.60 * np.sin(ang_ag)),
                    xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.0))
        ax.plot(cx, cy, 'o', color='#2c3e50', markersize=6, zorder=5)

        # Textos
        ax.text(cx, cy + 0.22, f'{salud:.1f}%', ha='center', va='center',
                fontsize=20, fontweight='bold', color=color)
        ax.text(cx, cy - 0.10, label, ha='center', fontsize=9, fontweight='bold', color='#555555')
        estado = ('SOSTENIBLE' if salud >= 65 else 'EN RIESGO' if salud >= 40 else 'NO SOSTENIBLE')
        ax.text(cx, cy - 0.28, estado, ha='center', fontsize=8, color=color, fontweight='bold')

    _gauge(0.5, 0.45, salud_a, col_a,  'ANTES',  ax_g)
    _gauge(1.6, 0.45, salud_d, col_d,  'DESPUÉS', ax_g)

    # Flecha central + delta
    ax_g.annotate('', xy=(1.25, 0.45), xytext=(0.85, 0.45),
                  arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=2.5))
    signo = '+' if delta >= 0 else ''
    ax_g.text(1.05, 0.55, f'{signo}{delta:.1f} pp', ha='center', fontsize=11,
              fontweight='bold', color=col_delta)

    ax_g.set_title('Sostenibilidad global', fontsize=10, fontweight='bold', pad=6)

    # =========================================================================
    # PANEL CENTRAL — Barras por indicador (0-5) antes vs después
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor('#ffffff')

    metricas  = [k.split('(')[0].strip() for k in diag_orig['scores'].keys()]
    vals_a    = list(diag_orig['scores'].values())
    vals_d    = list(diag_proy['scores'].values())
    x         = np.arange(len(metricas))
    w         = 0.33

    bars_a = ax_b.bar(x - w / 2, vals_a, w, label='Antes',   color='#E24B4A', alpha=0.82)
    bars_d = ax_b.bar(x + w / 2, vals_d, w, label='Después', color='#27ae60', alpha=0.82)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(metricas, fontsize=7.5, wrap=True)
    ax_b.set_ylim(0, 6.2)
    ax_b.set_ylabel('Puntuación (0 – 5)', fontsize=8)
    ax_b.axhline(3, color='#bdc3c7', linestyle='--', linewidth=0.8)
    ax_b.set_title('Indicadores por métrica', fontsize=10, fontweight='bold')
    ax_b.legend(fontsize=8)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    for bar in bars_a:
        v = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width() / 2, v + 0.08, str(int(v)),
                  ha='center', fontsize=8, color='#A32D2D', fontweight='bold')
    for bar in bars_d:
        v = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width() / 2, v + 0.08, str(int(v)),
                  ha='center', fontsize=8, color='#1a6e3f', fontweight='bold')

    # =========================================================================
    # PANEL DERECHO — Resumen visual de ratios clave
    # =========================================================================
    ax_r = fig.add_subplot(gs[0, 2])
    ax_r.axis('off')
    ax_r.set_facecolor('#ffffff')
    ax_r.set_title('Ratios clave proyectados', fontsize=10, fontweight='bold', pad=6)

    ro = diag_orig['ratios']
    rp = diag_proy['ratios']

    filas = [
        ('FPF (Sal/Ing)',        f"{ro['fpf']:.1f}%",         f"{rp['fpf']:.1f}%",        rp['fpf'] < ro['fpf']),
        ('Ef. Operativa',        f"{ro['ef_operativa']:.3f}x", f"{rp['ef_operativa']:.3f}x", rp['ef_operativa'] > ro['ef_operativa']),
        ('Ingresos totales',     f"{ro['ingresos']:.1f}M€",   f"{rp['ingresos']:.1f}M€",  True),
        ('Gastos totales',       f"{ro['gastos']:.1f}M€",     f"{rp['gastos']:.1f}M€",    rp['gastos'] < ro['gastos']),
        ('Masa salarial',        f"{ro['salarios']:.1f}M€",   f"{rp['salarios']:.1f}M€",  rp['salarios'] < ro['salarios']),
    ]

    # Cabecera
    y0 = 0.88
    ax_r.text(0.02, y0, 'Ratio',    fontsize=8.5, fontweight='bold', color='#333', transform=ax_r.transAxes)
    ax_r.text(0.50, y0, 'Antes',    fontsize=8.5, fontweight='bold', color='#E24B4A', ha='center', transform=ax_r.transAxes)
    ax_r.text(0.78, y0, 'Después',  fontsize=8.5, fontweight='bold', color='#27ae60', ha='center', transform=ax_r.transAxes)
    ax_r.text(0.96, y0, 'Mejora',   fontsize=8.5, fontweight='bold', color='#333', ha='right', transform=ax_r.transAxes)
    ax_r.plot([0.01, 0.99], [y0 - 0.02, y0 - 0.02], color='#ddd', linewidth=0.8, transform=ax_r.transAxes)

    for i, (nombre, va, vd, mejora) in enumerate(filas):
        y = y0 - 0.12 * (i + 1)
        fondo = '#f0faf4' if mejora else '#fff5f5'
        ax_r.add_patch(plt.Rectangle((0.01, y - 0.04), 0.97, 0.10,
                                     transform=ax_r.transAxes, color=fondo, zorder=0))
        ax_r.text(0.03, y, nombre, fontsize=8, color='#444', transform=ax_r.transAxes)
        ax_r.text(0.50, y, va,     fontsize=8, color='#E24B4A', ha='center', transform=ax_r.transAxes)
        ax_r.text(0.78, y, vd,     fontsize=8, color='#27ae60', ha='center', transform=ax_r.transAxes)
        icono = '✓' if mejora else '─'
        ax_r.text(0.96, y, icono,  fontsize=9, color='#27ae60' if mejora else '#aaa',
                  ha='right', fontweight='bold', transform=ax_r.transAxes)

    # Veredicto final
    y_ver = y0 - 0.12 * (len(filas) + 1.5)
    if sem_d == 'SOSTENIBLE' and sem_a != 'SOSTENIBLE':
        msg = '✅  ¡El club PASA a ser SOSTENIBLE!'
        mc  = '#27ae60'
    elif sem_d == sem_a:
        msg = f'ℹ  Mantiene estado: {sem_d} (+{delta:.1f} pp)'
        mc  = '#f39c12'
    else:
        msg = f'⬆  Mejora: {sem_a} → {sem_d}'
        mc  = '#27ae60'

    ax_r.text(0.50, y_ver, msg, ha='center', fontsize=9, fontweight='bold',
              color=mc, transform=ax_r.transAxes,
              bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=mc, linewidth=1.2))

    plt.tight_layout()
    nombre_archivo = f"reevaluacion_{diag_orig['nombre'].replace(' ', '_')}.png"
    plt.savefig(nombre_archivo, dpi=130, bbox_inches='tight')
    plt.show()
    print(f"  Gráfico guardado: {nombre_archivo}")


# =============================================================================
# MENU PRINCIPAL
# =============================================================================

def menu():
    clubes = sorted(df_clubes['Nombre del club'].tolist())

    print()
    print("=" * 55)
    print("  SPORTCONSULT PRO - Consultoría Deportivo-Financiera")
    print("=" * 55)
    print("\n  Clubes disponibles:")
    for i, club in enumerate(clubes, 1):
        print(f"    {i:>2}. {club}")

    print()
    while True:
        entrada = input("  Escribe el nombre del club o su número: ").strip()
        if entrada.isdigit() and 1 <= int(entrada) <= len(clubes):
            nombre_club = clubes[int(entrada) - 1]
            break
        elif entrada in clubes:
            nombre_club = entrada
            break
        else:
            print("  No se ha encontrado ese club, inténtalo de nuevo.")

    print(f"\n  Calculando diagnóstico para {nombre_club}...")
    diag = diagnostico_club(nombre_club)
    mostrar_diagnostico(diag)

    input("  [Pulsa ENTER para ver el gráfico de diagnóstico]")
    grafico_diagnostico(diag)

    if diag['semaforo'] == 'SOSTENIBLE':
        print(f"\n  {nombre_club} ya es financieramente sostenible. Puedes continuar igualmente.")
    else:
        print(f"\n  {nombre_club} necesita mejorar. Apliquemos el modelo.")

    seguir = input("  ¿Quieres continuar con el análisis completo? (s/n): ").strip().lower()
    if seguir != 's':
        print("\n  Análisis finalizado.")
        return

    print()
    print("  ESCENARIOS DISPONIBLES:")
    print("    1. Mejora deportiva: Optimizar la plantilla sin límites presupuestarios")
    print("    2. Mejora económica a corto plazo: Acciones inmediatas en finanzas")
    print("    3. Mejora económica a largo plazo: Estabilidad financiera sostenida")
    while True:
        obj = input("  Elige una opción (1/2/3): ").strip()
        if obj in ['1', '2', '3']:
            break

    res_fin = None
    res_dep = None
    
    if obj == '1':
        # Escenario 1: Solo mejora deportiva
        print(f"\n  ESCENARIO 1: MEJORA DEPORTIVA")
        print(f"  Analizando la plantilla de {nombre_club}...")
        res_dep = analisis_deportivo(nombre_club, modo='deportivo')
        mostrar_deportivo(res_dep)
        input("  [Pulsa ENTER para ver el gráfico deportivo]")
        grafico_deportivo(res_dep)
    
    elif obj == '2':
        # Escenario 2: Mejora económica a corto plazo
        print()
        print("  ESCENARIO 2: MEJORA ECONÓMICA A CORTO PLAZO")
        print("  Generando plan financiero a corto plazo (esta temporada)...")
        res_fin = analisis_financiero(diag, 'corto')
        mostrar_financiero(res_fin)
        input("  [Pulsa ENTER para ver el gráfico financiero]")
        grafico_financiero(res_fin)

        # ── Reevaluación post-modelo ────────────────────────────────────────
        diag_proy = diagnostico_proyectado(diag, res_fin)
        mostrar_reevaluacion(diag, diag_proy)
        input("  [Pulsa ENTER para ver el gráfico de reevaluación]")
        grafico_reevaluacion(diag, diag_proy)

    elif obj == '3':
        # Escenario 3: Mejora económica a largo plazo
        print()
        print("  ESCENARIO 3: MEJORA ECONÓMICA A LARGO PLAZO")
        print("  Generando plan financiero a largo plazo (3 temporadas)...")
        res_fin = analisis_financiero(diag, 'largo')
        mostrar_financiero(res_fin)
        input("  [Pulsa ENTER para ver el gráfico financiero]")
        grafico_financiero(res_fin)

        # ── Reevaluación post-modelo ────────────────────────────────────────
        diag_proy = diagnostico_proyectado(diag, res_fin)
        mostrar_reevaluacion(diag, diag_proy)
        input("  [Pulsa ENTER para ver el gráfico de reevaluación]")
        grafico_reevaluacion(diag, diag_proy)

    if res_dep is not None:
        input("  [Pulsa ENTER para ver el impacto global ANTES / DESPUÉS]")
        grafico_antes_despues(diag, res_fin=None, res_dep=res_dep)

    print("=" * 55)
    print(f"  Análisis completado para {nombre_club}.")
    print("  Los gráficos se han guardado en la carpeta del script.")
    print("=" * 55)


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--ranking':
        print("\nRANKING DE SALUD FINANCIERA - TODOS LOS CLUBES")
        print("-" * 70)
        resultados = []
        for club in df_clubes['Nombre del club']:
            d = diagnostico_club(club)
            if d:
                resultados.append({
                    'Club':               club,
                    'Sostenibilidad (%)': d['salud'],
                    'Estado':             d['semaforo'],
                    'FPF (%)':            round(d['ratios']['fpf'], 1),
                    'Ef.Op (x)':          round(d['ratios']['ef_operativa'], 2),
                    'VM/Coste (x)':       round(d['ratios']['vm_ratio'], 2),
                })
        ranking = pd.DataFrame(resultados).sort_values('Sostenibilidad (%)', ascending=False)
        print(ranking.to_string(index=False))
        print()
    else:
        menu()