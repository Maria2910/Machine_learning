import pygame


def dist(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

def region_query(data, point, eps):
    neighbors = []
    for neighbor in data:
        if dist(point, neighbor) < eps:
            neighbors.append(neighbor)
    return neighbors

def expand_cluster(data, point, neighbors, cluster_id, eps, min_pts, labels):
    labels[data.index(point)] = cluster_id
    for neighbor in neighbors:
        if labels[data.index(neighbor)] == -1 or labels[data.index(neighbor)] == 0:
            labels[data.index(neighbor)] = cluster_id
            new_neighbors = region_query(data, neighbor, eps)
            if len(new_neighbors) >= min_pts:
                neighbors += new_neighbors


def dbscan(data, eps, min_pts):
    cluster_id = 0
    labels = [0] * len(data)
    for point in data:
        if labels[data.index(point)] == 0:
            neighbors = region_query(data, point, eps)
            # print(data)
            if len(neighbors) < min_pts:
                labels[data.index(point)] = -1
            else:
                # print(neighbors)
                cluster_id += 1
                # print(cluster_id)
                expand_cluster(data, point, neighbors, cluster_id, eps, min_pts, labels)
    return labels


def add_color(color):
    coloring = [0,0,0]
    if color == -1:
        return coloring == [255,0,0]
    else:
        coloring[color % 3] = color * 30
        # print(coloring, color)
        return coloring



WIDTH = 360
HEIGHT = 480
FPS = 30
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
screen.fill(WHITE)
pygame.display.set_caption("My Game")
clock = pygame.time.Clock()
points = []
eps, min_pts = 30, 3

running = True




while running:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                point = event.pos
                points.append(point)
                pygame.draw.circle(screen, BLACK, point, 5)
                # print(points)

        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_e:
                labels = dbscan(points, eps, min_pts)
                print(labels, points)
                # Отрисовка точек на экране с учетом кластеров
                for point, label in zip(points, labels):
                    color = add_color(label)
                    pygame.draw.circle(screen, color, point, 5)

        pygame.display.flip()
pygame.quit()
