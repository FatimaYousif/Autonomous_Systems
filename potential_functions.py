import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



# isValid checks = invalid for pathfinding, as you wouldn't want to traverse through obstacles.
# FALSE
      # box=1
      # box!=0
      # x<0    -ve co-ordinate
      # y<0
      # x>rows
      # y>cols

# wavefront_planner_connect_8



# ---------------needed for psuedo code-------------
# no obstacles

def isValid(point, map):
    try:
        if map[point[0], point[1]] == 1 or map[point[0], point[1]] != 0 or point[0] < 0 or point[1] < 0 or point[0] > map.shape[0] or point[1] > map.shape[1]:
            return False
    except IndexError:
        return False
    return True

# ---------following psuedo code------------------
#----------q1: wavefront planner------------------
# 8 point connectivity
# goal=given

def wavefront_planner_connect_8(map, goal):
    motions = [[0, -1], [0, 1], [-1, 0], [1, 0],  [-1, -1], [-1, 1], [1, -1], [1, 1]]
    value = 2           #goal
    map[goal[0], goal[1]] = value
    queue = [goal]
    while queue:
      # not doing value++
        new_queue = []
        for p in queue:
            for motion in motions:
                 if isValid([p[0] + motion[0], p[1] + motion[1]], map):

                  # a little value changings
                    value = (motion[0]**2 + motion[1]**2)**0.5

                  # prev value + value
                    map[p[0] + motion[0], p[1] + motion[1]] = map[p[0] , p[1]] + value
                    new_queue.append([p[0] + motion[0], p[1] + motion[1]])
        queue = new_queue
    return map



# checks pt within boundaries (function)
# p= point = neighbour

# no -ve coordinates
# shouldnt be more than rows/cols

def isInBoundaries(p, m):
    try:
        if p[0] < 0 or p[1] < 0 or p[0] >= m.shape[0] or p[1] >= m.shape[1]:
            return False
    except IndexError:
        return False
    return True

#------------- q2: finding path -------------------------

def find_the_path(grid_map, goal, start):

  #8-connectivity
    motions = [[0, -1], [0, 1], [-1, 0], [1, 0],  [-1, -1], [-1, 1], [1, -1], [1, 1]]

  # path=start
    path = [start]

   # traverse until current pt is goal
    while path[-1]!= goal:

        #Assume the current point is along the shortest path to the goal
        shortest = path[-1]

        for motion in motions:

            #Check the neighbors of the current point
            neighbor = [path[-1][0] + motion[0], path[-1][1] + motion[1]]

            if not isInBoundaries(neighbor, grid_map):
                continue

            #If the neighbor is closer to the goal than the current point, set it as the new shortest point
            if grid_map[neighbor[0], neighbor[1]] < grid_map[shortest[0], shortest[1]] and grid_map[neighbor[0], neighbor[1]] != 1:
                shortest = neighbor

        #If the shortest point is the same as the current point, we have reached a local minima
        if(shortest == path[-1]):
            print("local minima detected at: ", shortest)
            return path
        path.append(shortest)
    return path




# ---------------------q3: brushfire algorithm-----------------
# no goal

def brush_fire(map):
    motions = [[0, -1], [0, 1], [-1, 0], [1, 0],  [-1, -1], [-1, 1], [1, -1], [1, 1]]
    value = 1

    #complete traversal
    while 0 in map:

    #++
        new_value = value + 1

       #rows,cols traverse
        for i in range(0, map.shape[0]):
            for j in range(0, map.shape[1]):

              #if new cell= obstacle -> move
                if map[i, j] == value:
                    for m in motions:

                      #new point + check boundaries and then ++
                        p = [i + m[0], j + m[1]]
                        if isInBoundaries(p, map) and (new_value < map[p[0], p[1]] or map[p[0], p[1]] == 0):
                            map[p[0], p[1]] = value + 1

        #new value changes for the above value
        value = new_value
    return map



# -------------- q4: repulsive function --------------
#Q = radius of repulsion

def repulsive_function(Q, map):

  #rows/cols traverse
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):

          #new block = obstacles -> skip
            if map[i, j] == 1:
                continue

            #new block > radius -> new block=0
            elif map[i, j] > Q:
                map[i, j] = 0

            #new block <= radius -> new block=formula
            #formula = 4 times and square for strong repulsion
            elif map[i, j] <= Q:
                map[i, j] = 4*(1/map[i, j] - 1/Q)**2
    return map



# Define the main function
def main():
    if len(sys.argv) != 7:
        print(
            "Usage: ./potential_function_YOUR_NAME.py path_to_grid_map_image start_x start_y goal_x goal_y Q"
        )
        sys.exit(1)

    path_to_grid_map_image = sys.argv[1]
    start_x = int(sys.argv[2])
    start_y = int(sys.argv[3])
    goal_x = int(sys.argv[4])
    goal_y = int(sys.argv[5])
    Q = float(sys.argv[6])

    print(start_x, start_y, goal_x, goal_y, Q)

    image = Image.open(path_to_grid_map_image).convert("L")
    grid_map = np.array(image.getdata()).reshape(image.size[0], image.size[1]) / 255
    grid_map[grid_map <= 0.5] = 0
    grid_map[grid_map > 0.5] = 1
    grid_map = (grid_map * -1) + 1

    plt.matshow(grid_map)
    plt.colorbar()
    plt.show()

    
    attraction_potential = wavefront_planner_connect_8(grid_map, [goal_x, goal_y])
    attraction_potential[attraction_potential == 1] = 1 + attraction_potential.max()
    attraction_potential /= attraction_potential.max()

    path_attraction_potential = find_the_path(attraction_potential, [goal_x, goal_y], [start_x, start_y])

    plt.matshow(attraction_potential)
    plt.colorbar()
    plt.show()

    plt.matshow(attraction_potential, interpolation="nearest")
    plt.plot([start_y], [start_x], marker="x", color="red", markersize=10)
    plt.plot([goal_y], [goal_x], marker="x", color="green", markersize=10)
    plt.plot([p[1] for p in path_attraction_potential], [p[0] for p in path_attraction_potential], color="red", linewidth=3)
    plt.colorbar()
    plt.show()

    image2 = Image.open(path_to_grid_map_image).convert("L")
    grid_map2 = np.array(image2.getdata()).reshape(image2.size[0], image2.size[1]) / 255
    grid_map2[grid_map2 <= 0.5] = 0
    grid_map2[grid_map2 > 0.5] = 1
    grid_map2 = (grid_map2 * -1) + 1

    brushfire_map = brush_fire(grid_map2)

    plt.matshow(brushfire_map)
    plt.colorbar()
    plt.show()

    repulsive_potential = repulsive_function(Q, brushfire_map)

    plt.matshow(repulsive_potential)
    plt.colorbar()
    plt.show()

    added_potential = attraction_potential + repulsive_potential
    added_potential /= added_potential.max()

    plt.matshow(added_potential)
    plt.colorbar()
    plt.show()

    path = find_the_path(added_potential, [goal_x, goal_y], [start_x, start_y])


    plt.matshow(added_potential, interpolation="nearest")
    plt.plot([start_y], [start_x], marker="x", color="red", markersize=10)
    plt.plot([goal_y], [goal_x], marker="x", color="green", markersize=10)
    plt.plot([p[1] for p in path], [p[0] for p in path], color="red", linewidth=3)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()


