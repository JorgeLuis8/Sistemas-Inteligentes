import random
import math
import matplotlib.pyplot as plt


# ======================= Configurações do mapa e obstáculos =======================

# Limites do mapa (min_x, max_x, min_y, max_y)
BOUNDS = (0, 100, 0, 60)

# Obstáculos
NUM_OBS = 10
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


# ======================= Utilidades geométricas e geração =======================

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

def generate_random_centers(n, r, bounds, inicio, fim,
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

        if valid_center(cand, centers, r, bounds, inicio, fim,
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
      - cardinais_flat: [(x,y), (x,y), ...]  # lista “achatada”
    """
    cardinais_por_circulo = []
    cardinais_flat = []
    for i, c in enumerate(centers):
        pts = pontos_cardeais(c, r)
        cardinais_por_circulo.append({'idx': i, 'center': c, 'pts': pts})
        cardinais_flat.extend(pts.values())
    return cardinais_por_circulo, cardinais_flat


# ======================= Plot helpers =======================

def plot_map_and_points(bounds, inicio, fim, title):
    min_x, max_x, min_y, max_y = bounds
    fig, ax = plt.subplots()
    ax.set_title(title)

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
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
    """Plota os pontos cardeais armazenados (lista achatada)."""
    ax = plt.gca()

    proxy = plt.Line2D([0], [0], marker='o', color='tab:blue',
                       linestyle='', markersize=4, label='Cardinais')

    for p in cardinais_flat:
        ax.scatter(p[0], p[1], marker='o', s=20, color='tab:blue', zorder=4)

    leg = ax.get_legend()
    if leg is None:
        ax.legend(handles=[proxy], loc='upper right')


# ======================= Geometria: segmento x círculo =======================

EPS = 1e-9

def closest_param_t_on_segment(p, a, b):
    """Retorna t em [0,1] tal que a + t*(b-a) é o ponto do segmento AB mais próximo de P."""
    ax, ay = a; bx, by = b; px, py = p
    vx, vy = (bx - ax), (by - ay)
    wx, wy = (px - ax), (py - ay)
    vv = vx*vx + vy*vy
    if vv <= EPS:
        return 0.0  # A e B coincidem
    t = (wx*vx + wy*vy)/vv
    if t < 0.0: return 0.0
    if t > 1.0: return 1.0
    return t

def dist_point_to_segment(p, a, b):
    """Distância do ponto P ao segmento AB e o parâmetro t do ponto mais próximo."""
    t = closest_param_t_on_segment(p, a, b)
    cx = a[0] + t*(b[0]-a[0])
    cy = a[1] + t*(b[1]-a[1])
    return math.hypot(p[0]-cx, p[1]-cy), t

def segment_intersects_circle(a, b, center, r, *,
                              allow_touch_at_endpoint_for=None,
                              tol=1e-7):
    """
    True se o segmento AB invade a área do círculo (dist < r) ou encosta (dist == r)
    em ponto INTERNO do segmento (0<t<1). Encostar no endpoint é permitido SOMENTE
    quando esse endpoint pertence ao círculo 'allow_touch_at_endpoint_for' (índice).
    """
    d, t = dist_point_to_segment(center, a, b)

    # Interseção 'dura' (atravessa interior)
    if d < r - tol:
        return True  # corta o obstáculo

    # Toque (d ≈ r)
    if abs(d - r) <= tol:
        # Toque em ponto interno do segmento é proibido
        if EPS < t < 1.0 - EPS:
            return True
        # Toque no endpoint: só permitir se o endpoint tocar for daquele círculo
        if t <= EPS:
            # a é o endpoint tocando
            if allow_touch_at_endpoint_for is None or allow_touch_at_endpoint_for["a"] is None:
                return True
            if allow_touch_at_endpoint_for["a"] != allow_touch_at_endpoint_for["circle_k"]:
                return True
            return False  # toque permitido
        if t >= 1.0 - EPS:
            # b é o endpoint tocando
            if allow_touch_at_endpoint_for is None or allow_touch_at_endpoint_for["b"] is None:
                return True
            if allow_touch_at_endpoint_for["b"] != allow_touch_at_endpoint_for["circle_k"]:
                return True
            return False  # toque permitido

    return False  # não intersecta


# ======================= Construção do grafo de visibilidade =======================

def build_visibility_graph(bounds, inicio, fim, centers, r, cardinais_por_circulo, cardinais_flat):
    """
    Retorna:
      - vertices: [(x,y), ...]  (start, goal e cardinais)
      - owners:   [None|-1|idx_circulo, ...]  (dono de cada vértice; None para start/goal)
      - edges:    [(i,j), ...] pares de índices de vertices com aresta válida

    Regras:
      - Proíbe ligar dois pontos do MESMO círculo (o segmento seria um acorde que atravessa o interior).
      - Proíbe atravessar qualquer círculo.
      - Permite tocar o círculo apenas no endpoint que pertence a ele (o próprio ponto cardeal).
    """
    # Monta lista de vértices e "donos"
    vertices = [inicio, fim] + list(cardinais_flat)
    owners = [None, None]    # start e goal não pertencem a círculo

    # Mapa rápido de ponto -> índice do círculo dono
    owner_map = {}
    for item in cardinais_por_circulo:
        ci = item['idx']
        for _, pt in item['pts'].items():
            owner_map[pt] = ci

    for pt in cardinais_flat:
        owners.append(owner_map.get(pt, None))

    n = len(vertices)
    edges = []

    for i in range(n):
        for j in range(i+1, n):
            a = vertices[i]; b = vertices[j]
            owner_i = owners[i]; owner_j = owners[j]

            # 1) Evita ligar dois pontos do MESMO círculo
            if (owner_i is not None) and (owner_i == owner_j):
                continue

            # 2) Checa interseção com todos os círculos
            ok = True
            for k, c in enumerate(centers):
                allow = {
                    "a": owner_i,
                    "b": owner_j,
                    "circle_k": k
                }
                if segment_intersects_circle(a, b, c, RADIUS, allow_touch_at_endpoint_for=allow):
                    ok = False
                    break

            if ok:
                edges.append((i, j))

    return vertices, owners, edges


# ======================= Plot das arestas =======================

def plot_edges(vertices, edges):
    ax = plt.gca()
    for (i, j) in edges:
        x1, y1 = vertices[i]
        x2, y2 = vertices[j]
        ax.plot([x1, x2], [y1, y2], linewidth=0.8, alpha=0.6)


# ======================= Main =======================

def main():
    total_clearance = MARGIN_FROM_POINTS

    # Apenas para ver start/goal primeiro
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

    # ===== Grafo de visibilidade =====
    vertices, owners, edges = build_visibility_graph(
        BOUNDS, INICIO, FIM, centers, RADIUS, cardinais_por_circulo, cardinais_flat
    )
    print(f"Arestas válidas: {len(edges)}")

    # Plot final
    plot_map_and_points(BOUNDS, INICIO, FIM, "Mapa: obstáculos, cardinais e arestas válidas")
    plot_obstacles(centers, RADIUS)
    plot_pontos_cardeais_from_list(cardinais_flat)
    plot_edges(vertices, edges)
    plt.show()


if __name__ == "__main__":
    main()
