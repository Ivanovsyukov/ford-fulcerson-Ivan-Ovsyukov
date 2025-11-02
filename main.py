import random
import time
import copy
import matplotlib.pyplot as plt
import numpy as np

# ==========================
# ПАРАМЕТРЫ ЭКСПЕРИМЕНТОВ
# ==========================
REPEATS_TABLES = 50      # число повторений для усреднения тестов из таблиц
REPEATS_HISTOGRAM = 500  # число повторов для гистограммы


# ==========================
# ГЕНЕРАТОР МАТРИЦЫ ГРАФА
# ==========================
def generate_matrix_graph(n, graph_type="sparse", p=0.5, low_fraction=0.1, min_cap=1, max_cap=100):
    """Создает граф в виде матрицы смежности (n x n)."""
    mat = [[0] * n for _ in range(n)]

    vertices = list(range(n))
    random.shuffle(vertices)
    
    for i in range(n - 1):
        u = vertices[i]
        v = vertices[i + 1]
        mat[u][v] = random.randint(min_cap, max_cap)

    if graph_type == "sparse":
        for u in range(n):
            for v in range(n):
                if u != v and random.random() < p:
                    mat[u][v] = random.randint(min_cap, max_cap)

    elif graph_type == "dense":
        for u in range(n):
            for v in range(n):
                if u != v and random.random() < 0.8:
                    mat[u][v] = random.randint(min_cap, max_cap)

    elif graph_type == "chain":
        for u in range(n - 1):
            mat[u][u + 1] = random.randint(min_cap, max_cap)

    elif graph_type == "bottleneck":
        for u in range(n):
            for v in range(n):
                if u != v and random.random() < p:
                    if random.random() < low_fraction:
                        mat[u][v] = random.randint(min_cap, min_cap + 3)
                    else:
                        mat[u][v] = random.randint(max_cap // 2, max_cap)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    return mat


# ==========================
# РЕАЛИЗАЦИЯ АЛГОРИТМА
# ==========================
class Graph:
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.size = len(adj_matrix)

    def dfs(self, s, t, visited=None, path=None):
        if visited is None:
            visited = [False] * self.size
        if path is None:
            path = []
        visited[s] = True
        path.append(s)
        if s == t:
            return path
        for ind, val in enumerate(self.adj_matrix[s]):
            if not visited[ind] and val > 0:
                result_path = self.dfs(ind, t, visited, path.copy())
                if result_path:
                    return result_path
        return None

    def fordFulkerson(self, source, sink):
        max_flow = 0
        path = self.dfs(source, sink)
        while path:
            path_flow = float("inf")
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                path_flow = min(path_flow, self.adj_matrix[u][v])
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                self.adj_matrix[u][v] -= path_flow
                self.adj_matrix[v][u] += path_flow
            max_flow += path_flow
            path = self.dfs(source, sink)
        return max_flow


# ==========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================
def count_edges(mat):
    """Подсчёт числа рёбер в матрице смежности."""
    n = len(mat)
    count = 0
    for i in range(n):
        for j in range(n):
            if mat[i][j] > 0:
                count += 1
    return count

def measure_average_time(n, **kwargs):
    """Измеряет среднее время работы алгоритма при фиксированных параметрах."""
    times = []
    flows = []
    edges_list = []

    for _ in range(REPEATS_TABLES):
        mat = generate_matrix_graph(n, **kwargs)
        E = count_edges(mat)
        edges_list.append(E)

        g = Graph(copy.deepcopy(mat))
        start = time.perf_counter()
        flow = g.fordFulkerson(0, n - 1)
        end = time.perf_counter()

        times.append((end - start) * 1000)
        flows.append(flow)

    return np.mean(times), np.mean(flows), np.mean(edges_list)



# ==========================
# 1. Тест S — Зависимость от размера графа
# ==========================
def test_size_dependence():
    print("\n=== ТЕСТ S: Влияние числа вершин ===")
    n_values = [10, 20, 50, 100, 200, 400, 600, 800]

    avg_times = []
    avg_flows = []
    avg_edges = []
    R_values = []

    for n in n_values:
        t, f, e = measure_average_time(
            n,
            graph_type="sparse",
            p=0.2,
            min_cap=1,
            max_cap=100
        )

        # Нормированное отношение R = T / (E * |f*|)
        R = t / (e * f) if e > 0 and f > 0 else float("nan")

        avg_times.append(t)
        avg_flows.append(f)
        avg_edges.append(e)
        R_values.append(R)

        print(f"n={n:4d} | E≈{e:8.1f} | f*≈{f:10.2f} | T={t:10.3f} ms | R={R:.6e}")

    # --- Табличный вид для отчёта ---
    print("\nТаблица результатов:")
    print(f"{'n':>5} {'E':>10} {'|f*|':>10} {'T(ms)':>12} {'R':>15}")
    for n, e, f, t, R in zip(n_values, avg_edges, avg_flows, avg_times, R_values):
        print(f"{n:5d} {e:10.1f} {f:10.2f} {t:12.3f} {R:15.6e}")

    # --- Графики ---
    plt.figure(figsize=(8,5))
    plt.plot(n_values, avg_times, marker='o', label='T(n), мс')
    plt.xlabel("Число вершин n")
    plt.ylabel("Среднее время (мс)")
    plt.title("Зависимость времени выполнения от размера графа")
    plt.grid(True)
    plt.legend()
    plt.show()



# ==========================
# 2. Тест F — Влияние диапазона пропускных способностей
# ==========================
def test_flow_dependence():
    print("\n=== ТЕСТ F: Влияние величины потока графа ===")

    # Фиксируем параметры
    n = 100
    graph_type = "sparse"
    p = 0.7

    # Меняем диапазоны ёмкостей рёбер
    cap_configs = [
        (1, 10),
        (5, 15),
        (10, 25),
        (20, 50),
        (30, 70),
        (40, 80),
        (50, 100),
        (60, 120),
        (80, 160),
        (100, 200)
    ]

    avg_flows = []
    avg_times = []
    labels = []

    for min_cap, max_cap in cap_configs:
        t, f, _ = measure_average_time(
            n,
            graph_type=graph_type,
            p=p,
            min_cap=min_cap,
            max_cap=max_cap
        )
        avg_flows.append(f)
        avg_times.append(t)
        labels.append(f"{min_cap}-{max_cap}")
        print(f"cap={min_cap}-{max_cap}: flow≈{f:.2f}, time={t:.3f} ms")

    plt.figure(figsize=(10, 6))
    plt.bar(labels, avg_times, color="gold", edgecolor="black", alpha=0.8)
    plt.xlabel("Диапазон пропускных способностей (min–max)", fontsize=12)
    plt.ylabel("Среднее время (мс)", fontsize=12)
    plt.title("Влияние величины потока графа на время работы алгоритма", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()



# ==========================
# 3. Тест T — Влияние структуры графа
# ==========================
def test_structure_dependence():
    print("\n=== ТЕСТ T: Влияние структуры графа ===")
    types = [
        ("chain", 100, None, None),
        ("sparse", 100, 0.01, None),
        ("sparse", 100, 0.1, None),
        ("sparse", 100, 0.3, None),
        ("sparse", 100, 0.5, None),
        ("dense", 100, 0.8, None),
        ("bottleneck", 100, 0.3, 0.05),
        ("bottleneck", 100, 0.3, 0.2),
    ]

    labels = []
    times = []

    for t in types:
        gtype, n, p, lf = t
        kwargs = dict(graph_type=gtype, min_cap=1, max_cap=100)
        if p is not None:
            kwargs["p"] = p
        if lf is not None:
            kwargs["low_fraction"] = lf
        avg_time, _, _ = measure_average_time(n, **kwargs)
        # формируем аккуратную подпись (пустой p не добавляем)
        label = gtype + (f" p={p}" if p is not None else "")
        labels.append(label)
        times.append(avg_time)
        print(f"{labels[-1]}: {avg_time:.3f} ms")

    # --- Надёжная отрисовка ---
    fig, ax = plt.subplots(figsize=(10, 6))             # больше места
    x = list(range(len(labels)))
    ax.bar(x, times, color="skyblue", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Среднее время (мс)")
    ax.set_title("Влияние структуры графа на время работы алгоритма")

    # расширяем границы по X чуть-чуть, чтобы крайний столбец точно был виден
    ax.set_xlim(-0.5, len(labels) - 0.5 + 0.02)

    plt.subplots_adjust(bottom=0.28)   # отступ снизу для повернутых меток
    plt.tight_layout()
    plt.show()


# ==========================
# 4. Гистограмма распределения времени
# ==========================
def test_histogram():
    print("\n=== ТЕСТ ГИСТОГРАММЫ ===")
    n = 100
    p = 0.5
    times = []

    for _ in range(REPEATS_HISTOGRAM):
        mat = generate_matrix_graph(n, graph_type="sparse", p=p, min_cap=1, max_cap=20)
        g = Graph(copy.deepcopy(mat))
        start = time.perf_counter()
        g.fordFulkerson(0, n - 1)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)

    # Создаём гистограмму и получаем интервалы (bins)
    plt.figure(figsize=(11, 6))
    counts, bins, patches = plt.hist(
        times,
        bins=15,
        color="#90ee90",
        edgecolor="black",
        alpha=0.85,
        rwidth=0.95
    )

    # Линия среднего
    plt.axvline(mean_time, color='red', linestyle='--', linewidth=2,
                label=f"Среднее: {mean_time:.2f} мс")

    # Формируем подписи диапазонов (например: 25–30, 30–35)
    bin_labels = [f"{bins[i]:.0f}–{bins[i+1]:.0f}" for i in range(len(bins) - 1)]
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]

    plt.xticks(bin_centers, bin_labels, rotation=45, ha="right")

    plt.xlabel("Диапазон времени выполнения (мс)", fontsize=12)
    plt.ylabel("Частота", fontsize=12)
    plt.title(f"Гистограмма времени выполнения (n={n}, p={p})", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Среднее: {mean_time:.2f} мс, σ={std_time:.2f}")

# ==========================
# ГЛАВНАЯ ТОЧКА ВХОДА
# ==========================
if __name__ == "__main__":
    random.seed(42)
    #test_size_dependence()
    #test_flow_dependence()
    #test_structure_dependence()
    test_histogram()
