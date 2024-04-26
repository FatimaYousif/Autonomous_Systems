import sys
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

class path_planning:
    def __init__(self, map, start, goal, probability, step_q):
        self.map = map
        self.start = start
        self.goal = goal
        self.step = step_q
        self.V = np.array([start])
        self.E = {}
        self.t = 5
        self.Path = []
        self.probability = probability

    def build(self, q0, n):
        for i in range(0, n):
            qrand = self.rand_conf(self.map.shape[0], self.probability, self.goal)
            self.extend(qrand)
            if self.V[-1][0] == self.goal[0] and self.V[-1][1] == self.goal[1]:
                print("Solution found")
                return self.V, self.E, i
        print("No solution found, iterate again!")
        return [], [], 0

    def rand_conf(self, size_map, p, goal):
        if np.random.random_sample() < p:
            qrand = goal
        else:
            qrand = np.random.randint(size_map, size=(1, 2)).flatten()
        return qrand

    def extend(self, qrand):
        distances = np.sqrt(np.power(self.V[:, 0] - qrand[0], 2) + np.power(self.V[:, 1] - qrand[1], 2))
        qnear = self.V[np.argmin(distances)]
        if np.min(distances) == 0.:
            return
        qnew = self.step * (qrand - qnear) / np.sqrt(np.sum(np.power(qrand - qnear, [2, 2])))
        qnew = qnew.astype(int) + qnear
        if np.sqrt((qnew[0] - self.goal[0]) ** 2 + (qnew[1] - self.goal[1]) ** 2) < self.t:
            qnew = self.goal
        qnew = np.clip(qnew, 0, self.map.shape[0] - 1)
        if self.is_collision_free(qnear, qnew) and self.map[qnew[1], qnew[0]] != 0.5:
            self.V = np.append(self.V, [qnew], axis=0)
            self.E[tuple(qnew)] = tuple(qnear)
            self.map[qnew[1], qnew[0]] = 0.5

    def is_collision_free(self, qnear, qnew, divisor=7):
        magnitude = np.sqrt(np.sum(np.power(qnew - qnear, [2, 2])))
        direction = (qnew - qnear) / magnitude
        check = np.array([i * (magnitude / divisor) * direction for i in range(1, divisor + 1)])
        check = check.astype(int) + qnear
        check = np.clip(check, 0, self.map.shape[0] - 1)
        if 1. in self.map[check[:, 1], check[:, 0]]:
            return False
        else:
            return True

    def fill_path(self):
        self.Path = [tuple(self.goal)]
        current = self.Path[-1]
        i = 0
        while not (current[0] == self.start[0] and current[1] == self.start[1]):
            self.Path.append(self.E[current])
            current = tuple(self.E[current])
            i += 1
        return self.Path[::-1]

    def smooth_path(self):
        smooth = [self.start]
        while not (smooth[-1][0] == self.goal[0] and smooth[-1][1] == self.goal[1]):
            for i in range(0, len(self.Path)):
                if self.is_collision_free(np.array(smooth[-1]), self.Path[i], divisor=200):
                    smooth.append(list(self.Path[i]))
                    break
        return smooth

    def total_distance(self, path):
        total = 0
        for i in range(0, len(path) - 1):
            total = total + np.sqrt((path[i][0] - path[i + 1][0]) ** 2 + (path[i][1] - path[i + 1][1]) ** 2)
        return total

# Function to draw the tree
def Draw_Tree(V, E, goal, start):
    plt.scatter(V[:,0], V[:,1], linewidths=0.1, c="black", marker='*')
    for e in E.keys():
        plt.plot([e[0], E[e][0]], [e[1], E[e][1]], linewidth=0.5, color="white")
        plt.draw()
    plt.scatter(start[0], start[1], linewidths=0.1, c="red", zorder=10, label='Start node', marker='*')
    plt.scatter(goal[0], goal[1], linewidths=0.1, c="green", zorder=10, label='Goal node', marker='*')

# Function to draw the path
def Draw_Path(path, is_smooth=False):
    if is_smooth:
        for x in range(0, len(path)-1):
            plt.plot([path[x][0], path[x+1][0]], [path[x][1], path[x+1][1]], color="red", linewidth=2)
    else:
        for x in range(0, len(path)-1):
            plt.plot([path[x][0], path[x+1][0]], [path[x][1], path[x+1][1]], color="red")

def main():

    if len(sys.argv) != 9:
        print("Usage: ./rrt_star_FATIMA_SAZID.py path_to_grid_map_image K Δq p qstart_x qstart_y qgoal_x qgoal_y")
        sys.exit(1)

    path_env = sys.argv[1]
    K = int(sys.argv[2])
    step = int(sys.argv[3])
    p = float(sys.argv[4])
    qstart_x = int(sys.argv[5])
    qstart_y = int(sys.argv[6])
    qgoal_x = int(sys.argv[7])
    qgoal_y = int(sys.argv[8])

    # Load grid map
    image = Image.open(path_env).convert('L')
    grid_map = np.array(image.getdata()).reshape(image.size[0], image.size[1]) / 255
    grid_map[grid_map > 0.5] = 1
    grid_map[grid_map <= 0.5] = 0
    grid_map = (grid_map * -1) + 1

    # Show grid map
    plt.matshow(grid_map)
    plt.colorbar()

    start = [qstart_y, qstart_x]
    goal = [qgoal_y, qgoal_x]

    path_planner = path_planning(grid_map, start, goal, probability=p, step_q=step)
    V, E, iterations = path_planner.build(start, K)

    if len(V) != 0:
        print("Path Found in: ", iterations, " Iterations")
        Draw_Tree(V, E, goal, start)
        path = path_planner.fill_path()
        print("Path: ", path, "\nTotal Distance: ", path_planner.total_distance(path))
        Draw_Path(path)
        plt.title('RRT Path Found')

        plt.matshow(grid_map)
        plt.colorbar()
        Draw_Tree(V, E, goal, start)
        smooth = path_planner.smooth_path()
        print("Smooth Path: ", smooth, "\nTotal distance: ", path_planner.total_distance(smooth))
        Draw_Path(smooth, is_smooth=True)
        plt.title('RRT Smooth Path Found')

    plt.show()

if __name__ == "__main__":
    main()
