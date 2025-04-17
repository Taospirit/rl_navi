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

def is_point_blocked(point, obstacles):
    if any(point_in_polygon(point, poly) for poly in obstacles["polygons"]):
        return True
    for circ in obstacles["circles"]:
        cx, cy = circ["pos"]
        r = circ["radius"]
        if math.hypot(point[0] - cx, point[1] - cy) <= r:
            return True
    return False

def load_config(config_path):
    file_ext = config_path.split('.')[-1]
    if file_ext == "json":
        with open(config_path) as f:
            config = json.load(f)
    else:
        raise ValueError(f"Invalid config file extension: {file_ext}")
    return config

# class Robot:
#     def __init__(self, robot_conf):
#         self.conf = robot_conf
#         self.pos = robot_conf["pos"]
#         self.radius = robot_conf["radius"]
#         self.angle = robot_conf.get("angle", random.uniform(-180, 180))
#         self.move_speed = 3
#         self.rotate_speed = 5
#         laser_conf = robot_conf["laser"]
#         self.laser_fov = laser_conf["fov"]
#         self.laser_interval = laser_conf["interval"]
#         self.laser_range = laser_conf["max_range"]
#         self.laser_size = int(self.laser_range / self.laser_interval) + 1
#         self.laser_dists = []
#         goal_conf = robot_conf["goal"]
#         self.goal_pos = goal_conf["pos"]
#         self.goal_radius = goal_conf["radius"]

#     def move_step(self, move_param, obstacles, map_size):
#         assert len(move_param) == 2, f"move param only support [move, rotate]"
#         self.old_pos = self.pos.copy()
#         move, rotate = move_param
#         print(f'===> move_param {move_param}, angle {self.angle}')

#         self.angle = (self.angle + rotate * self.rotate_speed) % 360
#         rad = math.radians(self.angle)
#         dx = move * self.move_speed * math.cos(rad)
#         dy = -1 * move * self.move_speed * math.sin(rad)
#         self.pos[0] += dx
#         self.pos[1] += dy

#         self.laser_hits, self.laser_dists = self.laser_scan(obstacles, map_size)

#     def laser_scan(self, obstacles, map_size):
#         robot_pos = self.pos
#         robot_angle = self.angle
#         laser_hits = []
#         laser_dists = []
#         start_angle = robot_angle - self.laser_fov / 2
#         end_angle = robot_angle + self.laser_fov / 2

#         for deg in range(int(start_angle), int(end_angle) + 1, self.laser_interval):
#             rad = math.radians(deg)
#             for d in range(0, self.laser_range, 2):
#                 x = robot_pos[0] + math.cos(rad) * d
#                 y = robot_pos[1] - math.sin(rad) * d
#                 if is_point_blocked([x, y], obstacles):
#                     laser_hits.append((x, y))
#                     laser_dists.append(d)
#                     break
#                 if self.check_outrange([x, y], map_size):
#                     laser_hits.append((x, y))
#                     laser_dists.append(d)
#                     break
#             else:
#                 x = robot_pos[0] + math.cos(rad) * self.laser_range
#                 y = robot_pos[1] - math.sin(rad) * self.laser_range
#                 laser_hits.append((x, y))
#                 laser_dists.append(self.laser_range)

#         laser_dists += [self.laser_range] * max(self.laser_size - len(laser_dists), 0)
#         laser_dists = laser_dists[:self.laser_size]
#         return laser_hits, laser_dists

#     def reset(self, rand=False):
#         if rand:
#             pass
#         else:
#             self.pos = self.conf["pos"]
#             self.radius = self.conf["radius"]
#             self.angle = 90
#             # self.angle = self.conf.get("angle", random.uniform(-180, 180))

#     def get_laser(self):
#         return self.laser_dists
    
#     def check_outrange(self, pos, map_size):
#         return not (
#             pos[0] > 0 and pos[0] < map_size[0] and pos[1] > 0 and pos[1] < map_size[1]
#         )
    
class Simu:
    def __init__(self, config):
        self.config = config
        # robot init
        self.robot_radius = config['robot']['radius']
        self.robot_pos = [config['robot']['pos'][0], config['robot']['pos'][1]]
        self.robot_angle = 0
        self.move_speed = 3
        self.rotate_speed = 5
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
        # self.goal = self._load_goal(map_conf['goal'])
        self.goal_pos = config['robot']['goal']['pos']
        self.goal_radius = config['robot']['goal']['radius']
        self.done = False
        self.done_flag = 0
        self.reach_goal_cnt = 0
        dx, dy, _ = self.calc_goal_info()
        self.min_dist_to_goal = np.hypot(dx, dy)

    def reset(self):
        self.robot_pos = list(self.config['robot']['pos'])
        self.robot_angle = self.config['robot'].get('angle', random.uniform(-180, 180))
        dx, dy, _ = self.calc_goal_info()
        self.min_dist_to_goal = np.hypot(dx, dy)
        self.reach_goal_cnt = 0

    def update(self, move_param):
        assert len(move_param) == 2, f"move param only support [move, rotate]"
        def act_map(move, rotate):
            # map act from [0, 1, 2] -> [-1, 0, 1]
            return move - 1, rotate - 1
        self.old_pos = self.robot_pos.copy()
        move, rotate = act_map(*move_param)
        move = max(move, 0)

        self.robot_angle = (self.robot_angle + rotate * self.rotate_speed) % 360
        rad = math.radians(self.robot_angle)
        dx = move * self.move_speed * math.cos(rad)
        dy = -1 * move * self.move_speed * math.sin(rad)
        self.robot_pos[0] += dx
        self.robot_pos[1] += dy

        obs = self.config["map"]["obstacles"]
        self.laser_hits, self.laser_dists = self.laser_scan(obs)

    def laser_scan(self, obstacles):
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
                if is_point_blocked([x, y], obstacles):
                    laser_hits.append((x, y))
                    laser_dists.append(d)
                    break
                if self.check_outrange([x, y]):
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
    
    def get_laser_distances(self):
        return self.laser_dists
    
    def calc_goal_info(self):
        """计算机器人到目标点的距离"""
        robot_pos = self.robot_pos
        goal_pos = self.goal_pos
        dx = goal_pos[0] - robot_pos[0]
        dy = goal_pos[1] - robot_pos[1]
        angle = np.arctan2(dy, dx)
        return dx, dy, angle

    def get_done(self):
        done = False
        done_flag = 0
        if self.check_collision(self.robot_pos, self.robot_radius):
            done = True
            done_flag = -1
        elif self.check_outrange(self.robot_pos):
            done = True
            done_flag = -2
        elif self.check_goal(self.robot_pos, self.robot_radius, 
                            self.goal_pos, self.goal_radius):
            self.reach_goal_cnt += 1
            if self.reach_goal_cnt > 1:
                done = True
            else:
                done_flag = 1
        return done, done_flag

    def check_collision(self, pos, radius):
        robot_shape = Point(pos).buffer(radius)
        for poly in self.obs_polygons:
            if robot_shape.intersects(Polygon(poly)):
                return True
        for circle in self.obs_circles:
            dx = pos[0] - circle["pos"][0]
            dy = pos[1] - circle["pos"][1]
            distance = math.hypot(dx, dy)
            if distance < radius + circle["radius"]:
                return True
        return False
    
    def check_outrange(self, pos):
        return not (
            pos[0] > 0 and pos[0] < self.map_size[0] and pos[1] > 0 and pos[1] < self.map_size[1]
        )

    def check_goal(self, pos, pos_radius, goal_pos, goal_radius):
        cx, cy = goal_pos
        dist = math.sqrt((pos[0] - cx) ** 2 + (pos[1] - cy) ** 2)
        return dist <= goal_radius + pos_radius
    
    def get_info(self):
        return 0

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
        self.simu.update(move_param)
        done, done_flag = self.simu.get_done()
        reward = self.get_reward(done, done_flag)
        info = self.simu.get_info()
        if done:
            self.reset()
        next_obs = self.get_obs()
        return next_obs, reward, done, info

    def get_obs(self):
        """ 获取当前的环境观测 """
        # 获取机器人位置和角度
        robot_angle = self.simu.robot_angle        
        # 计算机器人和终点之间的差值
        dx, dy, angle = self.simu.calc_goal_info()
        dist = np.hypot(dx, dy)
    
        # 将机器人角度从度转换为弧度
        robot_angle_rad = np.deg2rad(robot_angle)
        diff_angle = angle - robot_angle_rad
        # 将角度归一化到 [-pi, pi]
        diff_angle = np.arctan2(np.sin(diff_angle), np.cos(diff_angle))
        
        # 获取激光雷达数据并归一化
        laser = np.array(self.simu.laser_dists) / self.simu.laser_range
        
        # 归一化相对位置和距离
        map_size = self.simu.map_size
        dx = dx / map_size[0]  # 使用地图宽度归一化
        dy = dy / map_size[1]  # 使用地图高度归一化
        dist = dist / np.sqrt(map_size[0]**2 + map_size[1]**2)  # 使用地图对角线长度归一化
        diff_angle = diff_angle / np.pi  # 归一化到 [-1, 1]
        
        # 将所有观测组合成numpy数组
        obs = np.concatenate([
            laser,                       # laser readings (n) [0,1]
            [dx, dy],                    # dx, dy (2) [-1,1]
            [dist],                      # distance to goal (1) [0,1]
            [diff_angle],                # angle difference (1) [-1,1]
        ])
        return obs

    def get_reward(self, done, done_flag):
        if done_flag == 1:  # Reached goal
            return 10
        # Terminal rewards
        if done:
            if done_flag == -1:  # Collision
                return -10
            elif done_flag == -2: # Out of range
                return -10
        
        # Distance-based reward
        dx, dy, _ = self.simu.calc_goal_info()
        curr_dist = np.hypot(dx, dy)
        # Update minimum distance if we're closer than before
        if curr_dist < self.simu.min_dist_to_goal:
            reward = (self.simu.min_dist_to_goal - curr_dist) / 20  # Normalize the reward
            self.simu.min_dist_to_goal = curr_dist
            return reward
        
        return 0

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
