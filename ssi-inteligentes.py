import random
import math
import matplotlib.pyplot as plt


# Limites do mapa (min_x, max_x, min_y, max_y)
BOUNDS = (0, 100, 0, 100)

# Obstáculos
NUM_OBS = 100
RADIUS  = 3

# Distância mínima dos pontos Início/Fim em relação à borda do obstáculo
MARGIN_FROM_POINTS = 6.0

# Configurações de espaçamento
BORDER_GAP = 1.0   # folga extra entre obstáculo e borda do mapa (além de RADIUS)

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

    lo_x = min_x + r + border_gap
    hi_x = max_x - r - border_gap
    lo_y = min_y + r + border_gap
    hi_y = max_y - r - border_gap

    if lo_x > hi_x or lo_y > hi_y:
        raise ValueError("Mapa pequeno demais para o raio/gaps escolhidos.")

    while len(centers) < n and tries < max_tries:
        tries += 1
        cand = (random.uniform(lo_x, hi_x), random.uniform(lo_y, hi_y))

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

def generate_cardinal_points(centers, r):
    """
    Gera e ARMAZENA os pontos cardeais de cada círculo.
    Retorna:
      - cardinais_por_circulo: [{'idx': i, 'center': (x,y), 'pts': {'cima':(...), ...}}, ...]
      - cardinais_flat: [(x,y), (x,y), ...]  # lista “achatada” útil para grafos
    """
    cardinais_por_circulo = []
    cardinais_flat = []
    for i, c in enumerate(centers):
        pts = pontos_cardeais(c, r)
        cardinais_por_circulo.append({'idx': i, 'center': c, 'pts': pts})
        cardinais_flat.extend(pts.values())
    return cardinais_por_circulo, cardinais_flat


def plot_map_and_points(bounds, inicio, fim, title):
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

    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()  

def plot_obstacles(centers, r):
    """Desenha os obstáculos circulares."""
    ax = plt.gca()
    for c in centers:
        ax.add_patch(plt.Circle(c, r, fill=False))

def plot_pontos_cardeais_from_list(cardinais_flat):
    """
    Plota os pontos cardeais ARMAZENADOS (lista achatada).
    """
    ax = plt.gca()

    proxy = plt.Line2D([0], [0], marker='o', color='tab:blue',
                       linestyle='', markersize=4, label='Cardinais')

    for p in cardinais_flat:
        ax.scatter(p[0], p[1], marker='o', s=20, color='tab:blue', zorder=4)

    leg = ax.get_legend()
    if leg is None:
        ax.legend(handles=[proxy], loc='upper right')

def plot_pontos_cardeais(centers, r):
    ax = plt.gca()
    cima_proxy    = plt.Line2D([0], [0], marker='o', color='tab:blue',
                               linestyle='', markersize=4, label='Cima')
    direita_proxy = plt.Line2D([0], [0], marker='o', color='tab:blue',
                               linestyle='', markersize=4, label='Direita')
    baixo_proxy   = plt.Line2D([0], [0], marker='o', color='tab:blue',
                               linestyle='', markersize=4, label='Baixo')
    esquerda_proxy= plt.Line2D([0], [0], marker='o', color='tab:blue',
                               linestyle='', markersize=4, label='Esquerda')

    for c in centers:
        pts = pontos_cardeais(c, r)
        ax.scatter(*pts["cima"],     marker='o', s=20, color='tab:blue', zorder=4)
        ax.scatter(*pts["direita"],  marker='o', s=20, color='tab:blue', zorder=4)
        ax.scatter(*pts["baixo"],    marker='o', s=20, color='tab:blue', zorder=4)
        ax.scatter(*pts["esquerda"], marker='o', s=20, color='tab:blue', zorder=4)

    handles = [cima_proxy, direita_proxy, baixo_proxy, esquerda_proxy]
    leg = ax.get_legend()
    if leg is None:
        ax.legend(handles=handles, loc='upper right')


def main():
    total_clearance = MARGIN_FROM_POINTS

    plot_map_and_points(BOUNDS, INICIO, FIM, "Mapa: apenas pontos")
    plt.show()

    centers = generate_random_centers(
        NUM_OBS, RADIUS, BOUNDS, INICIO, FIM,
        total_clearance,
        border_gap=BORDER_GAP,
        max_tries=MAX_TRIES,
        seed=42,  
    )
    print(f"{len(centers)} centros gerados.")

    cardinais_por_circulo, cardinais_flat = generate_cardinal_points(centers, RADIUS)
    print(f"{len(cardinais_flat)} pontos cardeais armazenados "
          f"({len(cardinais_por_circulo)} círculos × 4 pontos cada).")

    plot_map_and_points(BOUNDS, INICIO, FIM, "Mapa: pontos, obstáculos e cardinais (armazenados)")
    plot_obstacles(centers, RADIUS)
    plot_pontos_cardeais_from_list(cardinais_flat)
    plt.show()

if __name__ == "__main__":
    main()
