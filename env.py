import pygame
import math
import numpy as np
import random
import json
import os
from shapely.geometry import Point, Polygon

# -------------------------------
# 工具函数
# -------------------------------
def load_config(config_path):
    file_ext = config_path.split('.')[-1]
    if file_ext == "json":
        with open(config_path) as f:
            config = json.load(f)
    else:
        raise ValueError(f"Invalid config file extension: {file_ext}")
    return config

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    px1, py1 = polygon[0]
    for i in range(n + 1):
        px2, py2 = polygon[i % n]
        if y > min(py1, py2):
            if y <= max(py1, py2):
                if x <= max(px1, px2):
                    if py1 != py2:
                        xinters = (y - py1) * (px2 - px1) / (py2 - py1) + px1
                    if px1 == px2 or x <= xinters:
                        inside = not inside
        px1, py1 = px2, py2
    return inside

def check_outrange(pos, map_size, radius=0):
    return not (
        pos[0] > radius and pos[0] < map_size[0] - radius \
        and pos[1] > radius and pos[1] < map_size[1] - radius
    )

def check_overlap(p1, r1, p2, r2):
    d = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
    return d < r1 + r2

def check_blocked(point, radius, 
                  obs_polygons=[], obs_circles=[]):
    for poly in obs_polygons:
        if radius != 0:
            robot_shape = Point(point).buffer(radius)
            flag = robot_shape.intersects(Polygon(poly))
        else:
            flag = point_in_polygon(point, poly)
        if flag:
            return True
    px, py = point
    for circle in obs_circles:
        cx, cy = circle["pos"]
        r = circle["radius"]
        if math.hypot(px - cx, py - cy) <= r + radius:
            return True
    return False


class Simu:
    def __init__(self, config):
        self.config = config
        # robot init
        self.robot_radius = config['robot']['radius']
        self.robot_pos = [config['robot']['pos'][0], config['robot']['pos'][1]]
        self.robot_angle = 0
        self.move_speed = 3
        self.rotate_speed = 5
        self.reset_mode = config['robot'].get('reset_mode', 'norm')
        # laser init
        laser_cfg = config['robot']['laser']
        self.laser_fov = laser_cfg['fov']
        self.laser_interval = laser_cfg['interval']
        self.laser_range = laser_cfg['max_range']
        self.laser_size = int(self.laser_range / self.laser_interval) + 1
        self.laser_dists = [self.laser_range for _ in range(self.laser_size)]
        # map info init 
        map_conf = config['map']
        self.map_size = map_conf['size']
        self.obs_polygons = map_conf['obstacles']['polygons']
        self.obs_circles = map_conf['obstacles']['circles']
        self.obstacles = map_conf['obstacles']

        self.goal_pos = config['robot']['goal']['pos']
        self.goal_radius = config['robot']['goal']['radius']
        self.reach_goal_cnt = 0
        self.step_cnt = 0
        dx, dy = self.calc_goal_info()
        self.min_dist_to_goal = np.hypot(dx, dy)
        self.last_pos = self.robot_pos
        self.last_angle = self.robot_angle

    def reset(self):
        if self.reset_mode == "rand":
            self.robot_pos = self.get_valid_pos(self.robot_radius)
            self.robot_angle = random.uniform(-180, 180)
            self.goal_pos = self.get_valid_pos(self.goal_radius)
        else:
            self.robot_pos = list(self.config['robot']['pos'])
            self.robot_angle = 0
            self.goal_pos = self.config['robot']['goal']['pos']

        dx, dy = self.calc_goal_info()
        self.min_dist_to_goal = np.hypot(dx, dy)
        self.reach_goal_cnt = 0
        self.step_cnt = 0

    def update(self, move_param):
        assert len(move_param) == 2, f"move param only support [move, rotate]"
        self.last_pos = self.robot_pos.copy()
        self.last_angle = self.robot_angle
        act_map = lambda m, r: (m -1, r - 1)
        move, rotate = act_map(*move_param)
        move = max(move, 0)

        self.robot_angle = (self.robot_angle + rotate * self.rotate_speed) % 360
        rad = math.radians(self.robot_angle)
        dx = move * self.move_speed * math.cos(rad)
        dy = -1 * move * self.move_speed * math.sin(rad)
        self.robot_pos[0] += dx
        self.robot_pos[1] += dy
        
        done, done_flag = self.check_env()
        if done and done_flag < 0:
            # collision or out range, action failed
            self.robot_pos = self.last_pos.copy()
            self.robot_angle = self.last_angle
        self.laser_hits, self.laser_dists = self.laser_scan()

        self.step_cnt += 1
        return done, done_flag

    def laser_scan(self):
        obs = self.config["map"]["obstacles"]
        robot_pos = self.robot_pos
        robot_angle = self.robot_angle
        laser_hits = []
        laser_dists = []
        start_angle = robot_angle - self.laser_fov / 2
        end_angle = robot_angle + self.laser_fov / 2

        for deg in range(int(start_angle), int(end_angle) + 1, self.laser_interval):
            rad = math.radians(deg)
            for d in range(0, self.laser_range, 2):
                x = robot_pos[0] + math.cos(rad) * d
                y = robot_pos[1] - math.sin(rad) * d
                if check_blocked([x, y], 0, self.obs_polygons, self.obs_circles):
                    laser_hits.append((x, y))
                    laser_dists.append(d)
                    break
                if check_outrange([x, y], self.map_size):
                    laser_hits.append((x, y))
                    laser_dists.append(d)
                    break
            else:
                x = robot_pos[0] + math.cos(rad) * self.laser_range
                y = robot_pos[1] - math.sin(rad) * self.laser_range
                laser_hits.append((x, y))
                laser_dists.append(self.laser_range)

        laser_dists += [self.laser_range] * max(self.laser_size - len(laser_dists), 0)
        laser_dists = laser_dists[:self.laser_size]
        return laser_hits, laser_dists

    def check_env(self):
        done = False
        done_flag = 0
        if self.step_cnt > 3000:
            done = True
            done_flag = 0
        if check_blocked(self.robot_pos, self.robot_radius,
                         self.obs_polygons, self.obs_circles):
            done = True
            done_flag = -1
        elif check_outrange(self.robot_pos, self.map_size, self.robot_radius):
            done = True
            done_flag = -2
        elif check_overlap(self.robot_pos, self.robot_radius, 
                            self.goal_pos, self.goal_radius):
            done = True
            done_flag = 1
        return done, done_flag

    def get_valid_pos(self, radius):
        cnt = 0
        while cnt < 10:
            x = np.random.uniform(radius, self.map_size[0]-radius)
            y = np.random.uniform(radius, self.map_size[1]-radius)
            if check_blocked([x, y], radius, 
                             self.obs_polygons, self.obs_circles):
                cnt += 1
            else:
                return [x, y]
    
    def calc_goal_info(self):
        dx = self.goal_pos[0] - self.robot_pos[0]
        dy = self.goal_pos[1] - self.robot_pos[1]
        return dx, dy

    def draw(self, screen):
        screen.fill((255, 255, 255))
        for obs in self.obs_polygons:
            points = [(int(x), int(y)) for x, y in obs]
            pygame.draw.polygon(screen, (0, 0, 0), points)
        for obs in self.obs_circles:
            cx, cy = obs["pos"]
            r = obs["radius"]
            pygame.draw.circle(screen, (0, 0, 0), (int(cx), int(cy)), int(r))

        # 绘制激光束
        for hit in self.laser_hits:
            pygame.draw.line(screen, (255, 0, 0), self.robot_pos, hit, 1)
        # 绘制机器人
        pygame.draw.circle(screen, (0, 0, 255), (int(self.robot_pos[0]), int(self.robot_pos[1])), self.robot_radius)
        # 绘制目标点
        pygame.draw.circle(screen, (0, 255, 0), (int(self.goal_pos[0]), int(self.goal_pos[1])), int(self.goal_radius))


class RobotEnv:
    def __init__(self, config, render=False):
        pygame.init()
        config = load_config(config)
        self.simu = Simu(config)
        self.need_render = render
        if self.need_render:
            self.screen = pygame.display.set_mode(config['map']['size'])
            self.clock = pygame.time.Clock()

    def reset(self):
        """ 重置环境，重新初始化机器人位置 """
        self.simu.reset()
        return self.get_obs()

    def step(self, move_param):
        """ 执行一步机器人动作，并返回观察结果和是否完成 """
        done, done_flag = self.simu.update(move_param)
        reward = self.get_reward(done, done_flag)
        if done:
            self.reset()
        next_obs = self.get_obs()
        return next_obs, reward, done, done_flag

    def get_obs(self):
        """ 获取当前的环境观测 """
        # 机器人数据
        robot_angle = self.simu.robot_angle        
        dx, dy = self.simu.calc_goal_info()
        dist = np.hypot(dx, dy)
        robot2goal_rad = np.arctan2(dy, dx)
        # 角度数据
        robot_rad = np.deg2rad(robot_angle)
        rad_diff = robot2goal_rad - robot_rad
        rad_diff = np.arctan2(np.sin(rad_diff), np.cos(rad_diff))
        rad_sin = np.sin(rad_diff)
        rad_cos = np.cos(rad_diff)
        # 激光数据
        laser = np.array(self.simu.laser_dists) / self.simu.laser_range
        
        # 归一化相对位置和距离
        map_size = self.simu.map_size
        dx /= map_size[0]  # 使用地图宽度归一化
        dy /= map_size[1]  # 使用地图高度归一化
        dist /= np.hypot(map_size[0], map_size[1])  # 使用地图对角线长度归一化
        rad_diff /= np.pi  # 归一化到 [-1, 1]
        
        # 将所有观测组合成numpy数组
        obs = np.concatenate([
            laser,                       # laser readings (n) [0,1]
            [dx, dy],                    # dx, dy (2) [-1,1]
            [dist],                      # distance to goal (1) [0,1]
            [rad_diff, rad_sin, rad_cos],                # angle difference (1) [-1,1]
        ])
        return obs

    def get_reward(self, done, done_flag):
        rewards = 0
        # Terminal rewards       
        if done:
            if done_flag == -1:  # Collision
                return -10
            elif done_flag == -2: # Out of range
                return -10
            elif done_flag == 1:
                return 100
        
        # Distance-based reward
        dx, dy = self.simu.calc_goal_info()
        curr_dist = np.hypot(dx, dy)
        # Update minimum distance if we're closer than before
        if curr_dist < self.simu.min_dist_to_goal:
            reward = (self.simu.min_dist_to_goal - curr_dist) / 20  # Normalize the reward
            self.simu.min_dist_to_goal = curr_dist
            rewards += reward

        # stack penalty
        last_pos = self.simu.last_pos
        cur_pos = self.simu.robot_pos
        move_dist = np.hypot(last_pos[0] - cur_pos[0], last_pos[1] - cur_pos[1])
        rewards += int(move_dist < 2.0) * -0.01      
        
        return rewards

    def render(self):
        """ 渲染环境状态 """
        self.simu.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        """ 关闭环境，清理pygame """
        pygame.quit()

    @property
    def state_dim(self):
        obs = self.get_obs()
        return obs.shape[0]
    
    @property
    def action_dim(self):
        return [3, 3]
