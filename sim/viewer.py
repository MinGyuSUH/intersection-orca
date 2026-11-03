"""pygame viewer.

Controls
  SPACE        play / pause
  R            reset (new simulation, same seed)
  N / P        add / remove a vehicle
  + / -        simulation speed
  T            show paths
  S            show braking-distance outline ("stadium")
  I            show vehicle ids
  B / F        1 s back / forward (Dev mode only)
  mouse drag   pan,  wheel: zoom
  Q / ESC      quit

The run pauses by itself when a body overlaps a wall or another body ("STOPPED" in the side panel).
"""
import copy
import math
from collections import deque

import numpy as np
import pygame

from .simulation import ALGORITHMS

WIN_W, WIN_H = 1280, 860
SIDE_W = 200
TOPBAR_H = 44
MAP_PAD = 30

BG = (18, 20, 28)
COL_OBSTACLE = (90, 95, 105)
COL_OBSTACLE_EDGE = (50, 55, 65)
COL_TEXT = (220, 225, 230)
COL_BAD = (230, 90, 90)
COL_DIM = (110, 115, 125)
COL_PANEL = (30, 32, 40)
COL_BTN = (55, 65, 80)
COL_BTN_HOVER = (75, 85, 100)
AMR_COLORS = [
    (240, 90, 90), (90, 200, 120), (90, 150, 240), (240, 190, 60),
    (200, 100, 220), (90, 220, 220), (240, 140, 60), (160, 220, 90),
    (255, 120, 180), (140, 120, 255), (235, 235, 235), (175, 120, 70),
    (0, 160, 140), (255, 235, 130), (130, 60, 190), (110, 150, 60),
]

SIM_SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
DEFAULT_SPEED_IDX = 2
SIM_SECONDS_PER_SEC_AT_1X = 5.0     # at "x1", 5 simulated seconds pass per real second
MAX_STEPS_PER_FRAME = 500
MAX_REAL_DT = 0.25                  # cap the frame time so a slow frame does not snowball
HISTORY_SECONDS = 40.0              # how far Dev mode can rewind


class Camera:
    def __init__(self, bbox, w, h, top=0, pad=MAP_PAD):
        x0, y0, x1, y1 = bbox
        mw, mh = x1 - x0, y1 - y0
        self.scale = min((w - 2 * pad) / mw, (h - 2 * pad) / mh)
        self.offset = [pad + (w - 2 * pad - mw * self.scale) / 2 - x0 * self.scale,
                       top + pad + (h - 2 * pad - mh * self.scale) / 2 - y0 * self.scale]
        self._drag = None

    def to_screen(self, x, y):
        return int(x * self.scale + self.offset[0]), int(y * self.scale + self.offset[1])

    def px(self, length):
        return max(2, length * self.scale)

    def on_down(self, pos):
        self._drag = (pos, list(self.offset))

    def on_move(self, pos):
        if self._drag is not None:
            (sx, sy), (ox, oy) = self._drag
            self.offset = [ox + pos[0] - sx, oy + pos[1] - sy]

    def on_up(self):
        self._drag = None

    def on_scroll(self, pos, up):
        f = 1.15 if up else 1 / 1.15
        self.offset = [pos[0] + (self.offset[0] - pos[0]) * f, pos[1] + (self.offset[1] - pos[1]) * f]
        self.scale *= f


class Button:
    def __init__(self, rect, label, action):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.action = action
        self.hover = False

    def draw(self, surf, font):
        pygame.draw.rect(surf, COL_BTN_HOVER if self.hover else COL_BTN, self.rect, border_radius=4)
        pygame.draw.rect(surf, COL_DIM, self.rect, 1, border_radius=4)
        lbl = font.render(self.label, True, COL_TEXT)
        surf.blit(lbl, lbl.get_rect(center=self.rect.center))


class Viewer:
    def __init__(self, config):
        self.config = config
        self.algo_names = list(ALGORITHMS)
        self.algo = self.algo_names[0]
        self.running = False
        self.speed_idx = DEFAULT_SPEED_IDX
        self.step_credit = 0.0
        self.show_path = False
        self.show_stadium = True
        self.show_names = True
        self.dev_mode = False
        self.history = deque()
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
        self.font = pygame.font.SysFont("segoeui,arial,helvetica", 16)
        self.font_small = pygame.font.SysFont("segoeui,arial,helvetica", 13)
        self.clock = pygame.time.Clock()
        self.new_sim()

    # ------------------------------------------------------------------ state

    def new_sim(self):
        self.sim = ALGORITHMS[self.algo](self.config)
        self.running = False
        self.step_credit = 0.0
        self.history.clear()
        pygame.display.set_caption(f"intersection-orca - {self.sim.map.name} - {self.algo}")
        self.make_camera(*self.screen.get_size())
        self.layout(*self.screen.get_size())

    def make_camera(self, win_w, win_h):
        self.cam = Camera(self.sim.map.bounds, win_w - SIDE_W, win_h - TOPBAR_H, top=TOPBAR_H)

    def _snapshot(self):
        s = self.sim
        mutable = copy.deepcopy((s.agents, getattr(s, "_off_since", None)))
        return (mutable, s.rng.bit_generator.state, s.t, s.arrivals, s.collision_stop, s.collision_stop_reason)

    def _restore(self, snap):
        s = self.sim
        (s.agents, off), s.rng.bit_generator.state, s.t, s.arrivals, s.collision_stop, s.collision_stop_reason = snap
        if off is not None:
            s._off_since = off

    def step(self):
        if self.dev_mode:
            self.history.append(self._snapshot())
            while len(self.history) > HISTORY_SECONDS / self.sim.dt:
                self.history.popleft()
        self.sim.step()

    def step_back(self):
        if not self.dev_mode or not self.history:
            return
        for _ in range(min(round(1.0 / self.sim.dt), len(self.history))):
            snap = self.history.pop()
        self._restore(snap)
        self.pause()

    def step_forward(self):
        if self.dev_mode:
            for _ in range(round(1.0 / self.sim.dt)):
                self.step()
            self.pause()

    # ------------------------------------------------------------------ actions

    def toggle_running(self):
        self.running = not self.running
        self.buttons["play"].label = "Pause" if self.running else "Play"

    def pause(self):
        self.running = False
        self.step_credit = 0.0
        self.buttons["play"].label = "Play"

    def switch_algo(self):
        self.algo = self.algo_names[(self.algo_names.index(self.algo) + 1) % len(self.algo_names)]
        self.new_sim()

    def change_speed(self, d):
        self.speed_idx = max(0, min(len(SIM_SPEEDS) - 1, self.speed_idx + d))

    def toggle(self, attr):
        setattr(self, attr, not getattr(self, attr))
        self.layout(*self.screen.get_size())

    def toggle_dev(self):
        self.dev_mode = not self.dev_mode
        self.history.clear()
        self.layout(*self.screen.get_size())

    def layout(self, win_w, win_h):
        self.map_w = win_w - SIDE_W
        x = self.map_w + 20
        on = lambda flag: "ON" if flag else "OFF"
        rows = [
            [("play", "Pause" if self.running else "Play", self.toggle_running)],
            [("reset", "Reset", self.new_sim)],
            [("del", "-AMR", lambda: self.sim.remove_agent()), ("add", "+AMR", lambda: self.sim.add_agent())],
            [("slow", "speed -", lambda: self.change_speed(-1)), ("fast", "speed +", lambda: self.change_speed(1))],
            [("path", f"Path: {on(self.show_path)}", lambda: self.toggle("show_path"))],
            [("stadium", f"Stadium: {on(self.show_stadium)}", lambda: self.toggle("show_stadium"))],
            [("names", f"Names: {on(self.show_names)}", lambda: self.toggle("show_names"))],
            [("dev", f"Dev: {on(self.dev_mode)}", self.toggle_dev)],
        ]
        if self.dev_mode:
            rows.append([("back", "<< Back", self.step_back), ("fwd", "Fwd >>", self.step_forward)])
        self.buttons = {"algo": Button((14, 8, 260, 28), f"Algo: {self.algo}", self.switch_algo)}
        y = TOPBAR_H + 12
        for row in rows:
            w = 160 if len(row) == 1 else 75
            for k, (key, label, action) in enumerate(row):
                self.buttons[key] = Button((x + k * 85, y, w, 34), label, action)
            y += 42
        self.info_y = y + 10

    # ------------------------------------------------------------------ main loop

    def run(self):
        keys = {
            pygame.K_SPACE: self.toggle_running, pygame.K_r: self.new_sim,
            pygame.K_n: lambda: self.sim.add_agent(), pygame.K_p: lambda: self.sim.remove_agent(),
            pygame.K_EQUALS: lambda: self.change_speed(1), pygame.K_PLUS: lambda: self.change_speed(1),
            pygame.K_KP_PLUS: lambda: self.change_speed(1), pygame.K_MINUS: lambda: self.change_speed(-1),
            pygame.K_KP_MINUS: lambda: self.change_speed(-1), pygame.K_t: lambda: self.toggle("show_path"),
            pygame.K_s: lambda: self.toggle("show_stadium"), pygame.K_i: lambda: self.toggle("show_names"),
            pygame.K_b: self.step_back, pygame.K_f: self.step_forward,
        }
        while True:
            real_dt = min(self.clock.tick(60) / 1000.0, MAX_REAL_DT)
            mouse = pygame.mouse.get_pos()
            for b in self.buttons.values():
                b.hover = b.rect.collidepoint(mouse)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_q, pygame.K_ESCAPE)):
                    pygame.quit()
                    return
                if ev.type == pygame.KEYDOWN and ev.key in keys:
                    keys[ev.key]()
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    hit = next((b for b in self.buttons.values() if b.rect.collidepoint(ev.pos)), None)
                    if hit is not None:
                        hit.action()
                    elif ev.pos[0] < self.map_w and ev.pos[1] >= TOPBAR_H:
                        self.cam.on_down(ev.pos)
                elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                    self.cam.on_up()
                elif ev.type == pygame.MOUSEMOTION:
                    self.cam.on_move(ev.pos)
                elif ev.type == pygame.MOUSEWHEEL:
                    self.cam.on_scroll(mouse, up=ev.y > 0)
                elif ev.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode(ev.size, pygame.RESIZABLE)
                    self.make_camera(*ev.size)
                    self.layout(*ev.size)

            if self.running:
                self.step_credit += real_dt * SIM_SECONDS_PER_SEC_AT_1X * SIM_SPEEDS[self.speed_idx] / self.sim.dt
                n = 0
                while self.step_credit >= 1.0 and n < MAX_STEPS_PER_FRAME:
                    self.step()
                    self.step_credit -= 1.0
                    n += 1
                    if self.sim.collision_stop:
                        self.pause()
                        break
            self.draw()

    # ------------------------------------------------------------------ drawing

    def draw(self):
        scr, cam, sim = self.screen, self.cam, self.sim
        scr.fill(BG)
        for poly in sim.map.obstacles:
            pts = [cam.to_screen(x, y) for x, y in poly.exterior.coords]
            pygame.draw.polygon(scr, COL_OBSTACLE, pts)

        r_goal = max(2, int(cam.px(4.0)))
        for amr in sim.agents:
            color = AMR_COLORS[amr.id % len(AMR_COLORS)]
            if amr.goal is not None:                                     # goal: small x in the vehicle's color
                gx, gy = cam.to_screen(amr.goal.x, amr.goal.y)
                pygame.draw.line(scr, color, (gx - r_goal, gy - r_goal), (gx + r_goal, gy + r_goal), 2)
                pygame.draw.line(scr, color, (gx - r_goal, gy + r_goal), (gx + r_goal, gy - r_goal), 2)
            if self.show_path and amr.path is not None and len(amr.path.waypoints) > 1:
                pygame.draw.lines(scr, color, False, [cam.to_screen(x, y) for x, y in amr.path.waypoints], 1)
            p = cam.to_screen(amr.pos[0], amr.pos[1])
            r = int(cam.px(amr.radius))
            if self.show_stadium:
                # Body circle stretched forward by the braking distance.
                pygame.draw.circle(scr, color, p, r, 1)
                if amr.braking_distance > 0.5:
                    front = amr.pos + amr.heading * amr.braking_distance
                    perp = np.array([-amr.heading[1], amr.heading[0]]) * amr.radius
                    for side in (perp, -perp):
                        pygame.draw.line(scr, color, cam.to_screen(*(amr.pos + side)), cam.to_screen(*(front + side)), 1)
                    pygame.draw.circle(scr, color, cam.to_screen(*front), r, 1)
            c, s = math.cos(amr.theta), math.sin(amr.theta)
            hl, hw = amr.length / 2.0, amr.width / 2.0
            body = [cam.to_screen(amr.pos[0] + lx * c - ly * s, amr.pos[1] + lx * s + ly * c)
                    for lx, ly in ((-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw))]
            pygame.draw.polygon(scr, color, body)
            pygame.draw.polygon(scr, COL_OBSTACLE_EDGE, body, 1)
            if self.show_names:
                lbl = self.font_small.render(str(amr.id), True, (255, 255, 255))
                scr.blit(lbl, (p[0] - lbl.get_width() // 2, p[1] - lbl.get_height() // 2))

        w, h = scr.get_size()
        pygame.draw.rect(scr, COL_PANEL, (self.map_w, 0, SIDE_W, h))
        pygame.draw.rect(scr, (30, 34, 46), (0, 0, w, TOPBAR_H))
        pygame.draw.line(scr, (66, 72, 92), (0, TOPBAR_H - 1), (w, TOPBAR_H - 1), 1)
        scr.blit(self.font.render(f"Map: {sim.map.name}", True, COL_DIM), (290, 12))
        for b in self.buttons.values():
            b.draw(scr, self.font)

        lines = [(f"t = {sim.t:.1f} s", COL_TEXT), (f"AMRs = {len(sim.agents)}", COL_TEXT),
                 (f"speed x{SIM_SPEEDS[self.speed_idx]}", COL_TEXT), (f"arrivals = {sim.arrivals}", COL_TEXT)]
        if sim.collision_stop:
            lines += [("STOPPED:", COL_BAD), (f"  {sim.collision_stop_reason}", COL_BAD)]
        for i, (text, col) in enumerate(lines):
            scr.blit(self.font_small.render(text, True, col), (self.map_w + 20, self.info_y + i * 22))
        pygame.display.flip()


def run_viewer(config):
    Viewer(config).run()
