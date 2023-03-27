#%%
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

class Agent():
    def __init__(self, position):
        self.position = position
        self.escaped = False
        self.panic = False
    
class Map():
    def __init__(self, map_shape, num_agents, num_obstacles, num_exits):
        self.map_shape = map_shape
        self.map = np.zeros(self.map_shape)
        self.speed = 3
        
        # Make obstacles
        self.obstacle_bool = np.zeros(map_shape, dtype=bool)
        num_obstacles_to_spawn = num_obstacles
        while num_obstacles_to_spawn != 0:
            self.obstacle_bool[np.unravel_index(np.random.choice(np.linspace(0, self.map.size, self.map.size, endpoint=False, dtype = int), num_obstacles_to_spawn), self.map.shape)] = 1
            num_obstacles_to_spawn = num_obstacles - np.sum(self.obstacle_bool.astype(int))
        
        # Make exits
        self.exit_bool = np.zeros(map_shape, dtype= bool)
        exit_loc = np.zeros((num_exits, 2), dtype=int)
        i = 0 
        while i < num_exits:
            exit_loc[i, :] = np.array([np.random.randint(-1, 1), np.random.randint(0, map_shape[0])])[:: (-1, 1)[np.random.randint(0, 2)]]
            self.exit_bool[exit_loc[i, 0], exit_loc[i, 1]] = True
            self.exit_bool[self.obstacle_bool] = False
            i = np.sum(self.exit_bool.astype(int))
        self.exit_positions = np.argwhere(self.exit_bool == True)
        
        # Spawn Agents
        self.num_agents = num_agents
        self.agent_bool = np.zeros(map_shape, dtype= bool)
        num_agents_to_spawn = num_agents
        while num_agents_to_spawn != 0:
            self.agent_bool[np.random.randint(0, map_shape[0], num_agents_to_spawn), np.random.randint(0, map_shape[1], num_agents_to_spawn)] = True
            self.agent_bool[np.logical_or(self.obstacle_bool, self.exit_bool)] = False
            num_agents_to_spawn = num_agents - np.sum(self.agent_bool.astype(int))
        self.agent_positions = np.argwhere(self.agent_bool== True)
        
        
    def update(self):
        dist_to_exit = distance_matrix(self.agent_positions, self.exit_positions)
        agent_exits = np.argmin(dist_to_exit, 1)
        agent_dir = np.subtract(self.exit_positions[agent_exits], self.agent_positions)
        agent_move = np.round(self.speed * agent_dir/np.array([np.linalg.norm(agent_dir, axis = 1)]).T).astype(int)
        self.agent_positions = np.add(self.agent_positions, agent_move)
        self.agent_positions[self.agent_positions < 0] = 0
        self.agent_positions[self.agent_positions >= map_shape[0]] = map_shape[0] - 1
        
    def draw_map(self):
        self.map = np.zeros(self.map_shape, dtype=int)
        self.agent_bool = np.zeros(map_shape, dtype= bool)
        self.agent_bool[self.agent_positions[:, 0], self.agent_positions[:, 1]] = True
        self.map[self.obstacle_bool] = 1
        self.map[self.agent_bool] = 2
        self.map[self.exit_bool] = 3
        cmap = ['w', 'k', 'r', 'b']
        sns.heatmap(self.map, cmap=cmap)
        plt.show()
        
map_shape = (20, 20)
num_agents = 4
num_obstacles = 4
num_exits = 2
if num_agents + num_exits + num_obstacles >= map_shape[0] * map_shape[1]:
    raise Exception('There are too many objects for the size of map.')

map1 = Map(map_shape, num_agents, num_obstacles, num_exits)
map1.update()
map1.draw_map()

map1.update()
map1.draw_map()


# %%
