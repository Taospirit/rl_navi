import pygame
import math
import numpy as np
import random
import cv2
from shapely.geometry import Point, Polygon
from core.utils import load_config

# -------------------------------
# 工具函数
# -------------------------------
def center_to_polygon(center, width, height, rotation=0):
    """将中心点和长宽转换为多边形顶点"""
    half_w, half_h = width/2, height/2
    vertices = [[-half_w, -half_h], [half_w, -half_h], 
                [half_w, half_h], [-half_w, half_h]]
    
    rad = math.radians(rotation)
    cos_val, sin_val = math.cos(rad), math.sin(rad)
    
    return [[x*cos_val - y*sin_val + center[0], 
             x*sin_val + y*cos_val + center[1]] 
            for x, y in vertices]

def point_in_polygon(point, polygon):
    """判断点是否在多边形内"""
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
    """检查位置是否超出地图边界"""
    return not (radius < pos[0] < map_size[0]-radius and 
                radius < pos[1] < map_size[1]-radius)

def check_overlap(p1, r1, p2, r2):
    """检查两个圆形是否重叠"""
    return np.hypot(p1[0]-p2[0], p1[1]-p2[1]) < r1 + r2

def check_blocked(point, radius, obs_polygons=[], obs_circles=[]):
    """检查点是否与障碍物重叠"""
    for poly in obs_polygons:
        if radius != 0:
            if Point(point).buffer(radius).intersects(Polygon(poly)):
                return True
        elif point_in_polygon(point, poly):
            return True
            
    px, py = point
    for circle in obs_circles:
        cx, cy = circle["pos"]
        if math.hypot(px-cx, py-cy) <= circle["radius"] + radius:
            return True
    return False


class Simu:
    def __init__(self, config):
        # 初始化配置
        self.config = config
        self.map_size = config.map.size 
        self.original_obstacles = config.map.obstacles
        self.obstacles_reset_mode = config.map.obstacles.reset_mode \
            if hasattr(config.map.obstacles, 'reset_mode') else 'norm'
        
        # 初始化机器人
        robot_cfg = config.robot
        self.robot_radius = robot_cfg.radius
        self.robot_pos = list(robot_cfg.pos)
        self.robot_deg = 0
        self.move_speed = 3
        self.rotate_speed = 5
        self.reset_mode = robot_cfg.reset_mode \
            if hasattr(robot_cfg, 'reset_mode') else 'norm'
        
        # 初始化激光
        laser_cfg = robot_cfg.laser
        self.laser_fov = laser_cfg.fov
        self.laser_interval = laser_cfg.interval
        self.laser_range = laser_cfg.max_range
        self.laser_size = int(self.laser_fov / self.laser_interval) + 1
        self.laser_dists = [self.laser_range] * self.laser_size
        
        # 初始化目标
        self.goal_pos = robot_cfg.goal.pos
        self.goal_radius = robot_cfg.goal.radius
        
        # 初始化状态
        self.reach_goal_cnt = 0
        self.step_cnt = 0
        self.last_pos = self.robot_pos
        self.last_angle = self.robot_deg
        
        # 初始化障碍物
        self.obs_polygons = []
        self.obs_circles = []
        self.update_obstacles()
        self.min_dist_to_goal = np.hypot(*self.calc_goal_info())

    def update_obstacles(self):
        """更新障碍物位置和旋转"""
        # 更新多边形障碍物
        self.obs_polygons = []
        for poly in self.original_obstacles.polygons:
            if self.obstacles_reset_mode == "rand":
                width, height = poly.width, poly.height
                min_x, max_x = width/2, self.map_size[0]-width/2
                min_y, max_y = height/2, self.map_size[1]-height/2
                center = [random.uniform(min_x, max_x), 
                         random.uniform(min_y, max_y)]
                rotation = random.uniform(0, 360)
            else:
                center = poly.center
                rotation = poly.rotation if hasattr(poly, 'rotation') else 0
                width = poly.width
                height = poly.height
            
            self.obs_polygons.append(
                center_to_polygon(center, width, height, rotation))
        
        # 更新圆形障碍物
        self.obs_circles = []
        for circle in self.original_obstacles.circles:
            if self.obstacles_reset_mode == "rand":
                radius = circle.radius
                min_x, max_x = radius, self.map_size[0]-radius
                min_y, max_y = radius, self.map_size[1]-radius
                pos = [random.uniform(min_x, max_x), 
                      random.uniform(min_y, max_y)]
            else:
                pos = circle.pos
                radius = circle.radius
            
            self.obs_circles.append({"pos": pos, "radius": radius})

    def get_valid_pos(self, radius):
        """获取一个有效的位置，确保不会与任何障碍物重叠
        Args:
            radius: 物体的半径，用于确保位置不会与边界和障碍物重叠
        Returns:
            list: 有效的位置 [x, y]，如果找不到有效位置则返回地图中心点
        """
        for _ in range(100):
            pos = [random.uniform(radius, self.map_size[0]-radius),
                  random.uniform(radius, self.map_size[1]-radius)]
            
            if not any(check_blocked(pos, radius, [poly], []) 
                      for poly in self.obs_polygons) and \
               not any(check_overlap(pos, radius, circle["pos"], circle["radius"]) 
                      for circle in self.obs_circles):
                return pos
                
        print("Warning: Failed to find valid position after 100 attempts, returning center position")
        # 返回地图中心点，确保不会与边界重叠
        return [self.map_size[0]/2, self.map_size[1]/2]

    def reset(self):
        """重置环境"""
        if self.reset_mode == "rand":
            self.robot_pos = self.get_valid_pos(self.robot_radius)
            self.robot_deg = random.uniform(-180, 180)
            self.goal_pos = self.get_valid_pos(self.goal_radius)
        else:
            self.robot_pos = list(self.config.robot.pos)
            self.robot_deg = 0
            self.goal_pos = self.config.robot.goal.pos
        
        self.update_obstacles()
        self.laser_hits, self.laser_dists = self.laser_scan()
        self.min_dist_to_goal = np.hypot(*self.calc_goal_info())
        self.reach_goal_cnt = 0
        self.step_cnt = 0

    def calc_goal_info(self):
        """计算与目标的相对位置"""
        return (self.goal_pos[0] - self.robot_pos[0],
                self.goal_pos[1] - self.robot_pos[1])

    def update(self, move_param):
        """更新机器人状态"""
        assert len(move_param) == 2, "move param only support [move, rotate]"
        self.last_pos = self.robot_pos.copy()
        self.last_angle = self.robot_deg
        
        # 计算移动
        move, rotate = (move_param[0]-1, move_param[1]-1)
        self.robot_deg = (self.robot_deg + rotate * self.rotate_speed) % 360
        
        # 更新位置
        rad = np.deg2rad(self.robot_deg)
        dx = move * self.move_speed * math.cos(rad)
        dy = -1 * move * self.move_speed * math.sin(rad)
        self.robot_pos[0] += dx
        self.robot_pos[1] += dy
        
        # 检查碰撞
        done, done_flag = self.check_env()
        if done and done_flag < 0:
            self.robot_pos = self.last_pos.copy()
            self.robot_deg = self.last_angle
            
        self.laser_hits, self.laser_dists = self.laser_scan()
        self.step_cnt += 1
        return done, done_flag

    def check_env(self):
        """检查环境状态"""
        if check_blocked(self.robot_pos, self.robot_radius, 
                        self.obs_polygons, self.obs_circles):
            return True, -1
        if check_outrange(self.robot_pos, self.map_size, self.robot_radius):
            return True, -2
        if check_overlap(self.robot_pos, self.robot_radius, 
                        self.goal_pos, self.goal_radius):
            return True, 1
        return False, 0

    def laser_scan(self):
        """激光扫描"""
        hits, dists = [], []
        start_angle = self.robot_deg - self.laser_fov / 2
        end_angle = self.robot_deg + self.laser_fov / 2
        
        for deg in range(int(start_angle), int(end_angle) + 1, self.laser_interval):
            rad = math.radians(deg)
            for d in range(0, self.laser_range, 2):
                x = self.robot_pos[0] + math.cos(rad) * d
                y = self.robot_pos[1] - math.sin(rad) * d
                if check_blocked([x, y], 0, self.obs_polygons, self.obs_circles) or \
                   check_outrange([x, y], self.map_size):
                    hits.append((x, y))
                    dists.append(d)
                    break
            else:
                x = self.robot_pos[0] + math.cos(rad) * self.laser_range
                y = self.robot_pos[1] - math.sin(rad) * self.laser_range
                hits.append((x, y))
                dists.append(self.laser_range)
                
        dists += [self.laser_range] * max(self.laser_size - len(dists), 0)
        return hits, dists[:self.laser_size]

    def draw(self, screen):
        """绘制环境"""
        screen.fill((255, 255, 255))
        
        # 绘制障碍物
        for poly in self.obs_polygons:
            pygame.draw.polygon(screen, (0, 0, 0), 
                              [(int(x), int(y)) for x, y in poly])
        for circle in self.obs_circles:
            cx, cy = circle["pos"]
            pygame.draw.circle(screen, (0, 0, 0), 
                             (int(cx), int(cy)), int(circle["radius"]))
        
        # 绘制激光
        for hit in self.laser_hits:
            pygame.draw.line(screen, (255, 0, 0), 
                           self.robot_pos, hit, 1)
        
        # 绘制机器人和目标
        pygame.draw.circle(screen, (0, 0, 255), 
                         (int(self.robot_pos[0]), int(self.robot_pos[1])), 
                         self.robot_radius)
        pygame.draw.circle(screen, (0, 255, 0), 
                         (int(self.goal_pos[0]), int(self.goal_pos[1])), 
                         int(self.goal_radius))


class RobotEnv:
    def __init__(self, config_path, render=False, save_video=None):
        pygame.init()
        config = load_config(config_path)
        self.simu = Simu(config)
        self.need_render = render
        self.save_video = save_video
        if self.need_render:
            self.screen = pygame.display.set_mode(self.simu.map_size)
            self.clock = pygame.time.Clock()
            self.frames = []  # 用于存储视频帧
            self.frame_skip = 2  # 每2帧保存一次，降低帧率
            self.frame_counter = 0

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
        robot_pos = self.simu.robot_pos
        robot_deg = self.simu.robot_deg
        laser_dist = self.simu.laser_dists
        ##### 转换坐标，由 y 轴向上变成 y 轴向下, 归一化到 -pi -> pi ####
        robot_deg_norm = ((-robot_deg + 180) % 360) - 180
        map_size = self.simu.map_size
        # 机器人数据
        px, py = robot_pos  
        dx, dy = self.simu.calc_goal_info()
        dist = np.hypot(dx, dy)
        robot2goal_deg = np.rad2deg(np.arctan2(dy, dx))
        # 角度数据
        deg_diff = robot2goal_deg - robot_deg_norm
        rad_diff = np.deg2rad(deg_diff)
        rad_diff = np.arctan2(np.sin(rad_diff), np.cos(rad_diff))
        robot_rad = np.deg2rad(robot_deg_norm)
        rad_sin = np.sin(rad_diff)
        rad_cos = np.cos(rad_diff)
        # 激光数据，转换成从左数
        laser_dist = laser_dist[::-1]

        debug = False
        if debug:
            info_dict = {
                "robot": robot_pos,
                "goal": self.simu.goal_pos,
                "deg": robot_deg_norm,
                "deg2goal": robot2goal_deg.item(),
                "deg_diff": np.rad2deg(rad_diff).item(),
                "pos_diff": (dx, dy),
                "pos_dist": dist.item(),
                "laser_dist": laser_dist,
            }
            print(info_dict)

        # 激光数据
        laser_feat = np.array(laser_dist) / self.simu.laser_range
        # 归一化
        px /= map_size[0]
        py /= map_size[1]
        dx /= map_size[0]  # 使用地图宽度归一化
        dy /= map_size[1]  # 使用地图高度归一化
        dist /= np.hypot(map_size[0], map_size[1])  # 使用地图对角线长度归一化
        rad_diff /= np.pi  # 归一化到 [-1, 1]
        robot_rad /= np.pi
        
        obs = np.concatenate([
            laser_feat,
            [px, py, dx, dy],
            [dist],
            [robot_rad, rad_diff, rad_sin, rad_cos],
        ])
        return obs
    
    def get_action_mask(self):
        """获取动作掩码，默认为全1"""
        masks = np.ones(self.action_dim, dtype=bool)
        # move mask, ban backward
        masks[0][0] = 0

        max_len = max(len(x) for x in masks)
        padded_masks = np.zeros((len(masks), max_len), dtype=bool)
        for i, m in enumerate(masks):
            padded_masks[i, :len(m)] = m
        return padded_masks

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
        stack_calc_dist = 5.0
        move_thr = 2.0
        last_pos = self.simu.last_pos
        cur_pos = self.simu.robot_pos
        move_dist = np.hypot(last_pos[0] - cur_pos[0], last_pos[1] - cur_pos[1])
        is_stack = curr_dist > stack_calc_dist and move_dist < move_thr
        rewards += int(is_stack) * -0.01      
        
        return rewards
    
    def get_rand_obs(self):
        return np.random.rand(self.state_dim)
    
    def get_rand_act(self):
        return np.random.randint(0, self.action_dim)
    
    def get_step_cnt(self):
        return self.simu.step_cnt

    def close(self):
        """ 关闭环境，清理pygame """
        if self.save_video is not None and self.frames:
            # 获取第一帧的尺寸
            height, width = self.frames[0].shape[:2]
            # 使用更高效的编码
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264编码
            out = cv2.VideoWriter(self.save_video, fourcc, 45, (width, height))  # 降低帧率到15fps
            # 写入所有帧
            for frame in self.frames:
                out.write(frame)                
            # 释放视频写入器
            out.release()
            print(f"视频已保存到: {self.save_video}")

        pygame.quit()
        self.frames = []  # 清空帧列表

    def render(self):
        """ 渲染环境状态 """
        self.simu.draw(self.screen)
        pygame.display.flip()
        # 捕获当前帧并转换为OpenCV格式
        if self.need_render and self.save_video:
            self.frame_counter += 1
            if self.frame_counter % self.frame_skip == 0:
                # 使用更高效的图像转换方式
                frame = pygame.surfarray.array3d(self.screen)
                frame = np.transpose(frame, (1, 0, 2))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.frames.append(frame)
            
        self.clock.tick(60)  # 保持游戏运行在60fps

    @property
    def state_dim(self):
        obs = self.get_obs()
        return obs.shape[0]
    
    @property
    def action_dim(self):
        return np.array([3, 3])
