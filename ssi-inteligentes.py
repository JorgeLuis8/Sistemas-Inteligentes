import random
import math
import matplotlib.pyplot as plt


# ======================= Configurações do mapa e obstáculos =======================

# Limites do mapa (min_x, max_x, min_y, max_y)
BOUNDS = (0, 100, 0, 100)

# Obstáculos
NUM_OBS = 100
RADIUS  = 3

# Distância mínima dos pontos Início/Fim em relação à borda do obstáculo
MARGIN_FROM_POINTS = 6.0

# Configurações de espaçamento

BORDER_GAP          = 1.0   # folga extra entre obstáculo e borda do mapa (além de RADIUS)

# Tentativas máximas de geração
MAX_TRIES = 50_000

# Pontos inicial e final, puxados para dentro do mapa
MARGEM_PONTOS = 2.0  # ajuste a distância da borda (em unidades do mapa)
INICIO = (BOUNDS[0] + MARGEM_PONTOS, BOUNDS[3] - MARGEM_PONTOS)  
FIM    = (BOUNDS[1] - MARGEM_PONTOS, BOUNDS[2] + MARGEM_PONTOS)  




def dist(a, b):
    
    return math.hypot(a[0] - b[0], a[1] - b[1])


def inside_bounds(center, r, bounds, border_gap=0.0):

    x, y = center
    min_x, max_x, min_y, max_y = bounds
    return (x - r - border_gap) >= min_x and (x + r + border_gap) <= max_x \
       and (y - r - border_gap) >= min_y and (y + r + border_gap) <= max_y


def valid_center(candidate, centers, r, bounds, inicio, fim,
                 clearance_pts, min_gap_between_obs=0.0, border_gap=0.0):
 
    if not inside_bounds(candidate, r, bounds, border_gap):
        return False

  
    min_center_dist = 2*r + min_gap_between_obs
    for c in centers:
        if dist(candidate, c) < min_center_dist:
            return False

    if dist(candidate, inicio) < (r + clearance_pts):
        return False
    if dist(candidate, fim) < (r + clearance_pts):
        return False

    return True


def generate_random_centers(n, r, bounds, start, goal,
                            clearance_pts, min_gap_between_obs=0.0,
                            border_gap=0.0, max_tries=10000, seed=42):
 
    if seed is not None:
        random.seed(seed)

    centers, tries = [], 0
    min_x, max_x, min_y, max_y = bounds

    while len(centers) < n and tries < max_tries:
        tries += 1
        cand = (random.uniform(min_x, max_x), random.uniform(min_y, max_y))

        if valid_center(cand, centers, r, bounds, start, goal,
                        clearance_pts, min_gap_between_obs, border_gap):
            centers.append(cand)

    if len(centers) < n:
        raise RuntimeError(
            f"Não foi possível posicionar {n} círculos após {max_tries} tentativas. "
            f"Tente diminuir NUM_OBS, reduzir folgas, ou aumentar MAX_TRIES."
        )
    return centers




def pontos_cardeais(center, r):

    cx, cy = center
    cima     = (cx, cy + r)
    direita  = (cx + r, cy)
    baixo    = (cx, cy - r)
    esquerda = (cx - r, cy)
    return {"cima": cima, "direita": direita, "baixo": baixo, "esquerda": esquerda}


# ======================= Visualização =======================

def plot_map_and_points(bounds, inicio, fim, title):
    """Plota o mapa com os pontos de início e fim."""
    min_x, max_x, min_y, max_y = bounds
    fig, ax = plt.subplots()
    ax.set_title(title)

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.scatter([inicio[0]], [inicio[1]], marker='o', s=80, label='Início',
               color='tab:blue', zorder=3)
    ax.scatter([fim[0]],  [fim[1]],  marker='o', s=80, label='Fim',
               color='tab:orange', zorder=3)

    # legenda fora do gráfico, à direita
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()  # evita corte da legenda


def plot_obstacles(centers, r):
    """Desenha os obstáculos circulares."""
    ax = plt.gca()
    for c in centers:
        ax.add_patch(plt.Circle(c, r, fill=False))


def plot_pontos_cardeais(centers, r):
    """
    Plota os pontos cardeais de cada círculo como bolinhas pequenas.
    Usa 'proxy artists' para legenda única.
    """
    ax = plt.gca()

    # Proxy artists para legenda
    cima_proxy    = plt.Line2D([0], [0], marker='o', color='tab:blue',
                               linestyle='', markersize=4, label='Cima')
    direita_proxy = plt.Line2D([0], [0], marker='o', color='tab:blue',
                               linestyle='', markersize=4, label='Direita')
    baixo_proxy   = plt.Line2D([0], [0], marker='o', color='tab:blue',
                               linestyle='', markersize=4, label='Baixo')
    esquerda_proxy= plt.Line2D([0], [0], marker='o', color='tab:blue',
                               linestyle='', markersize=4, label='blue')

    for c in centers:
        pts = pontos_cardeais(c, r)
        ax.scatter(*pts["cima"],     marker='o', s=20, color='tab:blue',   zorder=4)
        ax.scatter(*pts["direita"],  marker='o', s=20, color='tab:blue', zorder=4)
        ax.scatter(*pts["baixo"],    marker='o', s=20, color='tab:blue',  zorder=4)
        ax.scatter(*pts["esquerda"], marker='o', s=20, color='tab:blue',    zorder=4)

    # legenda única
    handles = [cima_proxy, direita_proxy, baixo_proxy, esquerda_proxy]
    leg = ax.get_legend()
    if leg is None:
        ax.legend(handles=handles, loc='upper right')

# ======================= Execução principal =======================

def main():
 
    total_clearance = MARGIN_FROM_POINTS
    

    
    plot_map_and_points(BOUNDS, INICIO, FIM, "Mapa: apenas pontos")
    plt.show()

    centers = generate_random_centers(
        NUM_OBS, RADIUS, BOUNDS, INICIO, FIM,
        total_clearance,
     
        border_gap=BORDER_GAP,
        max_tries=MAX_TRIES,
        seed=None,
    )

    print(f"{len(centers)} centros gerados.")

    plot_map_and_points(BOUNDS, INICIO, FIM, "Mapa: pontos, obstáculos e cardinais")
    plot_obstacles(centers, RADIUS)
    plot_pontos_cardeais(centers, RADIUS)
    plt.show()


if __name__ == "__main__":
    main()