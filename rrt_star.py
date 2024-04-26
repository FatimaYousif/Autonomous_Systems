import sys
import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image
from matplotlib import pyplot as plt

class RRTStar:
    def __init__(self, map, start, goal, probability, step_q, gamma):
        self.map = map
        self.start = start
        self.goal = goal
        self.step = step_q
        self.V = np.array([start])
        self.E = {}
        self.t = 5
        self.Path = []
        self.probability = probability
        self.gamma = gamma

    def build(self, q0, n):
        for _ in range(n):
            qrand = self.rand_conf(self.map.shape[0], self.probability, self.goal)
            qnear, qnew = self.nearest_and_new(qrand)

            if self.is_collision_free(qnear, qnew) and self.map[qnew[1], qnew[0]] != 0.5:
                self.V = np.append(self.V, [qnew], axis=0)
                self.E[tuple(qnew)] = tuple(qnear)
                self.map[qnew[1], qnew[0]] = 0.5

                # rewire 
                self.rewire(qnew)

                if np.sqrt((qnew[0] - self.goal[0]) ** 2 + (qnew[1] - self.goal[1]) ** 2) < self.t:
                    print("Solution found")
                    return self.V, self.E, _

        print("No solution found, iterate again!")
        return [], [], 0

    def nearest_and_new(self, qrand):
        distances = cdist(self.V, [qrand])
        qnear = self.V[np.argmin(distances)]

        qnew = self.step * (qrand - qnear) / np.sqrt(np.sum(np.power(qrand - qnear, [2, 2])))
        qnew = qnew.astype(int) + qnear

        if np.sqrt((qnew[0] - self.goal[0]) ** 2 + (qnew[1] - self.goal[1]) ** 2) < self.t:
            qnew = self.goal

        qnew = np.clip(qnew, 0, self.map.shape[0] - 1)

        return qnear, qnew

    def rewire(self, qnew):
        near_indices = np.where(cdist(self.V, [qnew]) < self.gamma)[0]

        for i in near_indices:
            if i < len(self.Path):
                if self.is_collision_free(self.V[i], qnew):
                    potential_cost = self.total_distance(self.Path[i:]) + self.total_distance([qnew, self.goal])
              
                    if potential_cost < self.total_distance(self.Path[i:]) + self.total_distance([self.Path[i], self.goal]):
                        self.E[tuple(qnew)] = tuple(self.V[i])
                        self.Path[i] = qnew

    def rand_conf(self, size_map, p, goal):
        if np.random.random_sample() < p:
            qrand = goal
        else:
            qrand = np.random.randint(size_map, size=(1, 2)).flatten()
        return qrand

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

    def Draw_Tree(self, V, E, goal, start):
        for e in E.keys():
            plt.plot([e[0], E[e][0]], [e[1], E[e][1]], linewidth=0.5, color="white")
            plt.draw()

        plt.scatter(start[0], start[1], linewidths=0.1, c="red", zorder=10, label='Start node', marker='*')
        plt.scatter(goal[0], goal[1], linewidths=0.1, c="green", zorder=10, label='Goal node', marker='*')

    def Draw_Path(self, path, is_smooth=False):
        if is_smooth:
            for x in range(0, len(path) - 1):
                plt.plot([path[x][0], path[x + 1][0]], [path[x][1], path[x + 1][1]], color="red", linewidth=2)
        else:
            for x in range(0, len(path) - 1):
                plt.plot([path[x][0], path[x + 1][0]], [path[x][1], path[x + 1][1]], color="red")

def main():
    if len(sys.argv) != 10:
        print("Usage: ./rrt_star_FATIMA_SAZID.py path_to_grid_map_image K Δq p max_distance qstart_x qstart_y qgoal_x qgoal_y")
        sys.exit(1)

    path_env = sys.argv[1]
    K = int(sys.argv[2])
    step = int(sys.argv[3])
    p = float(sys.argv[4])
    max_distance= int(sys.argv[5])
    qstart_x = int(sys.argv[6])
    qstart_y = int(sys.argv[7])
    qgoal_x = int(sys.argv[8])
    qgoal_y = int(sys.argv[9])

    image = Image.open(path_env).convert('L')
    grid_map = np.array(image.getdata()).reshape(image.size[0], image.size[1]) / 255
    grid_map[grid_map > 0.5] = 1
    grid_map[grid_map <= 0.5] = 0
    grid_map = (grid_map * -1) + 1

    plt.matshow(grid_map)
    plt.colorbar()

    start = [qstart_y, qstart_x]
    goal = [qgoal_y, qgoal_x]

    rrt_star_planner = RRTStar(grid_map, start, goal, probability=p, step_q=step, gamma=max_distance)
    V_star, E_star, iterations_star = rrt_star_planner.build(start, K)

    if len(V_star) != 0:
        print("Path Found in: ", iterations_star, " Iterations")
        rrt_star_planner.Draw_Tree(V_star, E_star, goal, start)
        path_star = rrt_star_planner.fill_path()
        print("Path: ", path_star, "\nTotal Distance: ", rrt_star_planner.total_distance(path_star))
        rrt_star_planner.Draw_Path(path_star)
        plt.title('RRT* Path Found')

        plt.matshow(grid_map)
        plt.colorbar()
        print("Path Found in: ", iterations_star, " Iterations")
        rrt_star_planner.Draw_Tree(V_star, E_star, goal, start)
        path_star = rrt_star_planner.smooth_path()
        print("Path: ", path_star, "\nTotal Distance: ", rrt_star_planner.total_distance(path_star))
        rrt_star_planner.Draw_Path(path_star)
        plt.title('RRT* Path Found')

    plt.show()

if __name__ == "__main__":
    main()
