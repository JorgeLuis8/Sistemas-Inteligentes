import random
import math
import matplotlib.pyplot as plt
from collections import deque


# ======================= Configurações do mapa e obstáculos =======================

BOUNDS = (0, 100, 0, 60)   # Limites do mapa
NUM_OBS = 10               # Quantidade de obstáculos
RADIUS  = 3                # Raio dos obstáculos
MARGIN_FROM_POINTS = 6.0   # Distância mínima de início/fim até obstáculo
MAX_TRIES = 50_000         # Tentativas máximas de geração
MARGEM_PONTOS = 2.0        # Margem para início e fim
INICIO = (BOUNDS[0] + MARGEM_PONTOS, BOUNDS[3] - MARGEM_PONTOS)
FIM    = (BOUNDS[1] - MARGEM_PONTOS, BOUNDS[2] + MARGEM_PONTOS)


# ======================= Utilidades =======================

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def inside_bounds(center, r, bounds):
    x, y = center
    min_x, max_x, min_y, max_y = bounds
    return (x - r) >= min_x and (x + r) <= max_x \
       and (y - r) >= min_y and (y + r) <= max_y

def valid_center(candidate, centers, r, bounds, inicio, fim, clearance_pts):
    if not inside_bounds(candidate, r, bounds):
        return False
    for c in centers:
        if dist(candidate, c) < 2*r:
            return False
    if dist(candidate, inicio) < (r + clearance_pts):
        return False
    if dist(candidate, fim) < (r + clearance_pts):
        return False
    return True

def generate_random_centers(n, r, bounds, inicio, fim,
                            clearance_pts, max_tries=10000, seed=None):
    if seed is not None:
        random.seed(seed)

    centers, tries = [], 0
    min_x, max_x, min_y, max_y = bounds
    lo_x, hi_x = min_x + r, max_x - r
    lo_y, hi_y = min_y + r, max_y - r

    if lo_x > hi_x or lo_y > hi_y:
        raise ValueError("Mapa pequeno demais para o raio escolhido.")

    while len(centers) < n and tries < max_tries:
        tries += 1
        cand = (random.uniform(lo_x, hi_x), random.uniform(lo_y, hi_y))
        if valid_center(cand, centers, r, bounds, inicio, fim, clearance_pts):
            centers.append(cand)

    if len(centers) < n:
        raise RuntimeError(f"Não foi possível posicionar {n} círculos após {max_tries} tentativas.")
    return centers

def pontos_cardeais(center, r):
    cx, cy = center
    return {
        "cima":     (cx, cy + r),
        "direita":  (cx + r, cy),
        "baixo":    (cx, cy - r),
        "esquerda": (cx - r, cy)
    }

def generate_cardinal_points(centers, r):
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
    ax.scatter([fim[0]], [fim[1]], marker='o', s=80, label='Fim',
               color='tab:orange', zorder=3)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()

def plot_obstacles(centers, r):
    ax = plt.gca()
    for c in centers:
        ax.add_patch(plt.Circle(c, r, fill=False))

def plot_pontos_cardeais_from_list(cardinais_flat):
    ax = plt.gca()
    proxy = plt.Line2D([0], [0], marker='o', color='tab:blue',
                       linestyle='', markersize=4, label='Cardinais')
    for p in cardinais_flat:
        ax.scatter(p[0], p[1], marker='o', s=20, color='tab:blue', zorder=4)
    if ax.get_legend() is None:
        ax.legend(handles=[proxy], loc='upper right')


# ======================= Geometria =======================

EPS = 1e-9

def closest_param_t_on_segment(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    vx, vy = (bx - ax), (by - ay)
    wx, wy = (px - ax), (py - ay)
    vv = vx*vx + vy*vy
    if vv <= EPS:
        return 0.0
    t = (wx*vx + wy*vy)/vv
    return max(0.0, min(1.0, t))

def dist_point_to_segment(p, a, b):
    t = closest_param_t_on_segment(p, a, b)
    cx = a[0] + t*(b[0]-a[0])
    cy = a[1] + t*(b[1]-a[1])
    return math.hypot(p[0]-cx, p[1]-cy), t

def segment_intersects_circle(a, b, center, r, *,
                              allow_touch_at_endpoint_for=None,
                              tol=1e-7):
    d, t = dist_point_to_segment(center, a, b)
    if d < r - tol:
        return True
    if abs(d - r) <= tol:
        if EPS < t < 1.0 - EPS:
            return True
        if t <= EPS:
            if allow_touch_at_endpoint_for is None or allow_touch_at_endpoint_for["a"] != allow_touch_at_endpoint_for["circle_k"]:
                return True
            return False
        if t >= 1.0 - EPS:
            if allow_touch_at_endpoint_for is None or allow_touch_at_endpoint_for["b"] != allow_touch_at_endpoint_for["circle_k"]:
                return True
            return False
    return False


# ======================= Grafo de Visibilidade =======================

def build_visibility_graph(bounds, inicio, fim, centers, r, cardinais_por_circulo, cardinais_flat):
    vertices = [inicio, fim] + list(cardinais_flat)
    owners = [None, None]
    owner_map = {pt: item['idx'] for item in cardinais_por_circulo for pt in item['pts'].values()}
    for pt in cardinais_flat:
        owners.append(owner_map.get(pt, None))
    n = len(vertices)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            a, b = vertices[i], vertices[j]
            owner_i, owner_j = owners[i], owners[j]
            if (owner_i is not None) and (owner_i == owner_j):
                continue
            ok = True
            for k, c in enumerate(centers):
                allow = {"a": owner_i, "b": owner_j, "circle_k": k}
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
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=0.8)


# ======================= DFS =======================

def edges_to_adj_list(n, edges):
    adj = {i: [] for i in range(n)}
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    return adj

def dfs_find_path(start_node, end_node, adj):
    if start_node == end_node:
        return [start_node]
    stack = [(start_node, [start_node])]
    visited = {start_node}
    while stack:
        current_node, path = stack.pop()
        if current_node == end_node:
            return path
        for neighbor in reversed(adj.get(current_node, [])):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, path + [neighbor]))
    return None


# ======================= Plot do caminho =======================

def plot_path(vertices, path):
    if not path:
        return
    ax = plt.gca()
    for i in range(len(path) - 1):
        x1, y1 = vertices[path[i]]
        x2, y2 = vertices[path[i+1]]
        ax.plot([x1, x2], [y1, y2], color='red', linewidth=2.5, zorder=5)
    xs = [vertices[i][0] for i in path]
    ys = [vertices[i][1] for i in path]
    ax.scatter(xs, ys, c='red', s=40, zorder=6, label='Caminho')
    if 'Caminho' not in ax.get_legend_handles_labels()[1]:
        proxy = plt.Line2D([0], [0], color='red', lw=2.5, label='Caminho')
        ax.legend(handles=ax.get_legend_handles_labels()[0] + [proxy],
                  loc='center left', bbox_to_anchor=(1.02, 0.5))


# ======================= Main =======================

def main():
    # Passo 1
    plot_map_and_points(BOUNDS, INICIO, FIM, "Passo 1: Pontos de Início e Fim")
    plt.show()

    # Geração
    centers = generate_random_centers(NUM_OBS, RADIUS, BOUNDS, INICIO, FIM,
                                      MARGIN_FROM_POINTS,
                                      max_tries=MAX_TRIES)
    cardinais_por_circulo, cardinais_flat = generate_cardinal_points(centers, RADIUS)
    vertices, owners, edges = build_visibility_graph(BOUNDS, INICIO, FIM,
                                                     centers, RADIUS,
                                                     cardinais_por_circulo,
                                                     cardinais_flat)

    # Passo 2
    plot_map_and_points(BOUNDS, INICIO, FIM, "Passo 2: Obstáculos e Grafo de Visibilidade")
    plot_obstacles(centers, RADIUS)
    plot_pontos_cardeais_from_list(cardinais_flat)
    plot_edges(vertices, edges)
    plt.show()

    # DFS
    adj_list = edges_to_adj_list(len(vertices), edges)
    path = dfs_find_path(0, 1, adj_list)

    # Passo 3
    plot_map_and_points(BOUNDS, INICIO, FIM, "Passo 3: Rota Final Encontrada (DFS)")
    plot_obstacles(centers, RADIUS)
    plot_pontos_cardeais_from_list(cardinais_flat)   
    plot_edges(vertices, edges)
    if path:
        plot_path(vertices, path)
    plt.show()


if __name__ == "__main__":
    main()
