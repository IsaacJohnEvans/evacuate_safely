#%%
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

class Agent():
    def __init__(self, position):
        self.position = position
        self.escaped = False
        self.panic = False
    
class AllAgents():
    def __init__(self, map_shape, num_agents, obstacle_bool):
        self.num_agents = num_agents
        self.agent_positions = np.zeros(map_shape, dtype= bool)
        num_agents_to_spawn = num_agents
        while num_agents_to_spawn != 0:
            self.agent_positions[np.random.randint(0, map_shape[0], num_agents_to_spawn), np.random.randint(0, map_shape[1], num_agents_to_spawn)] = True
            self.agent_positions[obstacle_bool] = False
            num_agents_to_spawn = num_agents - np.sum(self.agent_positions.astype(int))
    
    def update(self):
        pass
    
class Map():
    def __init__(self, map_shape, num_agents, num_obstacles, num_exits):
        self.map_shape = map_shape
        self.map = np.zeros(self.map_shape)
        self.obstacle_bool = np.zeros(map_shape, dtype=bool)
        num_obstacles_to_spawn = num_obstacles
        while num_obstacles_to_spawn != 0:
            self.obstacle_bool[np.unravel_index(np.random.choice(np.linspace(0, self.map.size, self.map.size, endpoint=False, dtype = int), num_obstacles_to_spawn), self.map.shape)] = 1
            num_obstacles_to_spawn = num_obstacles - np.sum(self.obstacle_bool.astype(int))
        
        self.exit_bool = np.zeros(map_shape, dtype= bool)
        exit_loc = np.zeros((num_exits, 2), dtype=int)
        i = 0 
        while i < num_exits:
            exit_loc[i, :] = np.array([np.random.randint(-1, 1), np.random.randint(0, map_shape[0])])[:: (-1, 1)[np.random.randint(0, 2)]]
            self.exit_bool[exit_loc[i, 0], exit_loc[i, 1]] = True
            self.exit_bool[self.obstacle_bool] = False
            i = np.sum(self.exit_bool.astype(int))
        print(np.where(self.exit_bool == True))
        self.agents = AllAgents(self.map_shape, num_agents, np.logical_or(self.obstacle_bool, self.exit_bool))
        
    def update(self):
        self.agents.update()
        
    def draw_map(self):
        self.map[self.obstacle_bool] = 1
        self.map[self.agents.agent_positions] = 2
        self.map[self.exit_bool] = 3
        cmap = ['w', 'k', 'r', 'b']
        sns.heatmap(self.map, cmap=cmap)
        plt.show()
        
map_shape = (10, 10)
num_agents = 7
num_obstacles = 4
num_exits = 3
if num_agents + num_exits + num_obstacles >= map_shape[0] * map_shape[1]:
    raise Exception('There are too many objects for the size of map.')

map1 = Map(map_shape, num_agents, num_obstacles, num_exits)

map1.update()
map1.draw_map()