#%%
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from scipy.spatial import distance_matrix
from scipy.ndimage import binary_dilation

#%%
class Agent():
    def __init__(self, position):
        self.position = position
        self.escaped = False
        self.panic = False
    
class Map():
    def __init__(self, map_shape, num_agents, num_obstacles, num_exits, max_runs = 10):
        self.map_shape = map_shape
        self.map = np.zeros(self.map_shape)
        self.ta = 0.5
        self.run = 0
        self.max_runs = max_runs
        
        # Make obstacles
        self.obstacle_bool = np.zeros(np.array(map_shape), dtype=bool)
        num_obstacles_to_spawn = num_obstacles
        while num_obstacles_to_spawn != 0:
            self.obstacle_bool[np.unravel_index(np.random.choice(np.linspace(0, self.map.size, self.map.size, endpoint=False, dtype = int), num_obstacles_to_spawn), self.map.shape)] = 1
            num_obstacles_to_spawn = num_obstacles - np.sum(self.obstacle_bool.astype(int))
        edges = np.zeros((2*np.sum(np.array(map_shape))-2, 2))
        edges[:map_shape[0]*2 +4, 0] = np.repeat(np.array([np.linspace(-1, map_shape[1], map_shape[1]+2)]).T, 2, axis= 1).T.flatten()
        edges[:map_shape[0]*2 +4, 1] = np.repeat(np.array([-1, map_shape[1]]), (map_shape[1] + 2))
        edges[map_shape[0]*2 +4:, 0] = np.repeat(np.array([-1, map_shape[1]]), (map_shape[1] - 3))
        edges[map_shape[0]*2 +4:, 1] = np.repeat(np.array([np.linspace(0, map_shape[1]-1, map_shape[1]-3)]).T, 2, axis= 1).T.flatten()
        self.obstacle_positions = np.concatenate((np.argwhere(self.obstacle_bool == True), edges))
        
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
        self.agent_in = np.ones(num_agents, int)
        self.agent_bool = np.zeros(map_shape, dtype= bool)
        num_agents_to_spawn = num_agents
        self.agent_v = np.zeros((num_agents, 2))
        self.agent_positions = np.zeros((num_agents, 2, self.max_runs))
        while num_agents_to_spawn != 0:
            self.agent_bool[np.random.randint(0, map_shape[0], num_agents_to_spawn), np.random.randint(0, map_shape[1], num_agents_to_spawn)] = True
            self.agent_bool[np.logical_or(self.obstacle_bool, self.exit_bool)] = False
            num_agents_to_spawn = num_agents - np.sum(self.agent_bool.astype(int))
        self.agent_positions[:, :, 0] = np.argwhere(self.agent_bool== True)
        
        
    def update(self):
        if self.run < self.max_runs - 1:
            agent_positions = self.agent_positions[:, :, self.run]
            
            dist_to_exit = distance_matrix(agent_positions, self.exit_positions)
            agent_exits = np.argmin(dist_to_exit, 1)
            exit_force = np.subtract(self.exit_positions[agent_exits], agent_positions)
            exit_force = np.divide(exit_force, np.array([np.linalg.norm(exit_force, axis =1)]).T)
            self.agent_in[np.min(dist_to_exit, axis= 1)<1] = 0
            d = distance_matrix(agent_positions, agent_positions)
            d[np.isnan(d)] = 0
            # dx and dy are the directions that the agent forces act in
            
            dx = np.subtract(agent_positions[:, 0], np.array([agent_positions[:, 0]]).T)/d
            dy = np.subtract(agent_positions[:, 1], np.array([agent_positions[:, 1]]).T)/d
            dx[np.isnan(dx)] = 0
            dy[np.isnan(dy)] = 0
            # Tune agent force
            agent_force = np.multiply(np.exp(-(d)), np.array([self.agent_in]).T)
            agent_force_x = np.sum(np.multiply(agent_force, dx), axis = 1)
            agent_force_y = np.sum(np.multiply(agent_force, dy), axis = 1)
            agent_force_dir = np.concatenate((np.array([agent_force_x]).T, np.array([agent_force_y]).T), axis = 1)
            
            print(self.obstacle_positions.shape, agent_positions.shape)
            do = distance_matrix(agent_positions, self.obstacle_positions)
            dox = np.subtract(np.array([agent_positions[:, 0]]).T, np.array([self.obstacle_positions[:, 0]]))/do
            doy = np.subtract(np.array([agent_positions[:, 1]]).T, np.array([self.obstacle_positions[:, 1]]))/do
            dox[np.isnan(dox)] = 0
            doy[np.isnan(doy)] = 0
            # Tune obstacle force
            obstacle_force = np.multiply(np.exp(-(do)), np.array([self.agent_in]).T)
            obstacle_force_x = np.sum(np.multiply(obstacle_force, dox), axis = 1)
            obstacle_force_y = np.sum(np.multiply(obstacle_force, doy), axis = 1)
            obstacle_force_dir = np.concatenate((np.array([obstacle_force_x]).T, np.array([obstacle_force_y]).T), axis = 1)
            
            total_force = exit_force + agent_force_dir + obstacle_force_dir
            agent_move = np.multiply(total_force/np.array([np.linalg.norm(total_force, axis = 1)]).T, np.array([self.agent_in]).T)
            agent_move[np.isnan(agent_move)] = 0
            agent_positions = np.add(agent_positions, agent_move)
            #agent_positions[agent_positions < 0] = 0
            #agent_positions[agent_positions >= map_shape[0]] = map_shape[0] - 1
            self.run += 1
            self.agent_exits = agent_exits
            self.agent_v = self.agent_positions[:, :, self.run] - self.agent_positions[:, :, self.run- 1]
            self.agent_positions[:, :, self.run] = agent_positions
        
    def run_map(self):
        while self.run < self.max_runs -1 and self.agent_in.any():
            self.update()
        self.map_metrics()
        self.draw_map()
        
    def map_metrics(self):
        min_dist_out = np.sum(np.linalg.norm(self.agent_positions[:, :, 0] - self.exit_positions[self.agent_exits], axis = 1))
        X = self.agent_positions[:, 0, :]
        Y = self.agent_positions[:, 1, :]
        dist_out = np.sum(((-np.diff(X))**2 + (-np.diff(Y))**2)**0.5)
        print(min_dist_out, dist_out)
        
    def draw_map(self):
        '''
        Heatmap plot
        self.agent_bool = np.zeros(self.map_shape, dtype= bool)
        self.agent_bool[np.round(self.agent_positions[:, 0]).astype(int), np.round(self.agent_positions[:, 0]).astype(int)] = True
        self.map = np.zeros(self.map_shape, dtype=int)
        self.map[self.obstacle_bool] = 1
        self.map[self.agent_bool] = 2
        self.map[self.exit_bool] = 3
        cmap = ['w', 'k', 'r', 'b']
        sns.heatmap(self.map, cmap=cmap)
        '''

        fig = plt.figure()
        self.agent_positions = self.agent_positions[:, :, :self.run]       
        plt.scatter(self.agent_positions[:, 0, 0], self.agent_positions[:, 1, 0],color = 'r', label = 'Agent')
        plt.scatter(self.exit_positions[:, 0], self.exit_positions[:, 1], color = 'b', label = 'Exit')
        plt.scatter(self.obstacle_positions[:, 0], self.obstacle_positions[:, 1], color = 'k', label = 'Obstacles')
        plt.plot(self.agent_positions[:, 0, :].T, self.agent_positions[:, 1, :].T, 'orange')
        plt.xlim(0, self.map_shape[0])
        plt.ylim(0, self.map_shape[1])
        plt.legend()
        plt.axis("square")
        plt.show()

map_shape = (20, 20)
num_agents = 4
num_obstacles = 10
num_exits = 2
max_runs = 20
if num_agents + num_exits + num_obstacles >= map_shape[0] * map_shape[1]:
    raise Exception('There are too many objects for the size of map.')

map1 = Map(map_shape, num_agents, num_obstacles, num_exits, max_runs)

map1.run_map()


# %%
