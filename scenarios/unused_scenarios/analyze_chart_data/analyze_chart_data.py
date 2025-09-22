import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils.functions import load_json

def draw_rectangles(items, ax=None, padding=0.05, alpha=0.8, edgecolor=None):
    """
    items: list of dicts, каждый dict должен содержать keys:
           'x1','x2','y1_real','y2_real','color'
    ax: matplotlib Axes (если None — создаётся новый)
    padding: доля для добавления отступа по осям
    alpha: прозрачность прямоугольников
    edgecolor: цвет границы (если None — используем тот же, что и заливку)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    xs = []
    ys = []
    for it in items:
        try:
            x1 = float(it['x1'])
            x2 = float(it['x2'])
            y1 = float(it['y1_real'])
            y2 = float(it['y2_real'])
            color = it.get('color', 'gray')
        except KeyError:
            # пропускаем элементы с некорректными данными
            continue

        xmin = min(x1, x2)
        width = abs(x2 - x1)
        ymin = min(y1, y2)
        height = abs(y2 - y1)

        face = color if color in ('red', 'green', 'gray') else 'gray'
        ec = edgecolor if edgecolor is not None else face

        rect = Rectangle((xmin, ymin), width, height,
                         facecolor=face, edgecolor=ec, alpha=alpha)
        ax.add_patch(rect)

        xs.extend([xmin, xmin + width])
        ys.extend([ymin, ymin + height])

    if xs and ys:
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        dx = (x_max - x_min) * padding if (x_max - x_min) > 0 else 1
        dy = (y_max - y_min) * padding if (y_max - y_min) > 0 else 1
        ax.set_xlim(x_min - dx, x_max + dx)
        ax.set_ylim(y_min - dy, y_max + dy)

    ax.set_xlabel('x')
    ax.set_ylabel('y (real)')
    ax.set_title('Rectangles from input data')
    ax.grid(True)
    ax.set_aspect('auto')
    plt.show()

    return ax

def draw_order_book_by_file(file_path):
    items = load_json(file_path)
    print(items)
    draw_rectangles(items)