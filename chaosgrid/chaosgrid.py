import pygame
from pygame import gfxdraw
import mido
import random
import math
from abc import ABC, abstractmethod
from uuid import uuid4, UUID
from typing import List, Dict, Tuple, Callable, Any, Mapping, Optional
from collections import defaultdict


class PhysicsNode:
    def __init__(
        self, x: int, y: int, radius: int, mass: float, layer_name: str, id: UUID
    ) -> None:
        self.px = float(x)
        self.py = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.mass = mass
        self.radius = radius
        self.layer_name = layer_name
        self.id = id

    def apply_force(self, fx: float, fy: float, max_velocity: float = 15) -> None:
        self.vx += fx / self.mass
        self.vy += fy / self.mass
        # ダンピング
        self.vx *= 0.99
        self.vy *= 0.99
        self.vx = min(max(self.vx, -max_velocity), max_velocity)
        self.vy = min(max(self.vy, -max_velocity), max_velocity)

    def apply_velocity(self, dt: float) -> None:
        self.px += self.vx * dt
        self.py += self.vy * dt

    def get_id(self) -> UUID:
        return self.id


class PhysicsField:
    ATTRACTIVE_FORCE_CONST = 3000
    MAX_ATTRACTIVE_FORCE = 80
    MIN_DIST_SQ = 0

    def __init__(
        self, px: int, py: int, width: int, height: int, nodes: Dict[UUID, PhysicsNode]
    ) -> None:
        self.px = px
        self.py = py
        self.width = width
        self.height = height
        self.nodes = nodes
        self.last_time_elapsed: float = 0.0

    def add_node(self, node: PhysicsNode) -> None:
        self.nodes[node.get_id()] = node

    def remove_node(self, node: PhysicsNode) -> None:
        try:
            self.nodes.pop(node.get_id())
        except KeyError:
            print(f"Node {node} does not exist in the field.")

    def update(self, dt: float) -> None:
        for node in self.nodes.values():
            fx, fy = self.get_force(node)
            node.apply_force(fx, fy)
        for node in self.nodes.values():
            self.check_boundary(node)
            node.apply_velocity(dt)

    def get_force(self, node: PhysicsNode) -> Tuple[float, float]:
        fx = 0.0
        fy = 0.0

        for other_node in self.nodes.values():
            if node == other_node:
                continue

            dx = other_node.px - node.px
            dy = other_node.py - node.py
            dist_sq = dx**2 + dy**2
            dist = math.sqrt(dist_sq)

            if dist_sq <= self.MIN_DIST_SQ:
                continue

            attractive_force = min(
                self.ATTRACTIVE_FORCE_CONST / dist_sq, self.MAX_ATTRACTIVE_FORCE
            )

            fx += attractive_force * dx / dist
            fy += attractive_force * dy / dist

        return fx, fy

    def check_boundary(self, node: PhysicsNode) -> None:
        if node.px < self.px:
            node.px = self.px
            node.vx = -node.vx
        if node.px > self.px + self.width:
            node.px = self.px + self.width
            node.vx = -node.vx
        if node.py < self.py:
            node.py = self.py
            node.vy = -node.vy
        if node.py > self.py + self.height:
            node.py = self.py + self.height
            node.vy = -node.vy


class Cell:
    def __init__(
        self, x: int, y: int, width: int, height: int, attributes: Dict[str, Any]
    ) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.attributes = attributes
        self.glow = 0
        self.color = (60, 60, 60)


class CellView:
    def __init__(self, cell: Cell) -> None:
        self.cell = cell
        self.cell_inner_offset = 2

    def draw(self, screen: pygame.Surface, offset_x: int, offset_y: int) -> None:
        color_intensity = max(0, min(255, self.cell.glow * 2))
        color = (
            self.cell.color[0],
            self.cell.color[1],
            self.cell.color[2],
            color_intensity,
        )
        gfxdraw.box(
            screen,
            pygame.Rect(
                self.cell.x + offset_x + self.cell_inner_offset,
                self.cell.y + offset_y + self.cell_inner_offset,
                self.cell.width - 2 * self.cell_inner_offset,
                self.cell.height - 2 * self.cell_inner_offset,
            ),
            color,
        )
        gfxdraw.rectangle(
            screen,
            (
                self.cell.x + offset_x + self.cell_inner_offset,
                self.cell.y + offset_y + self.cell_inner_offset,
                self.cell.width - 2 * self.cell_inner_offset,
                self.cell.height - 2 * self.cell_inner_offset,
            ),
            (0, 0, 0),
        )

        self.cell.glow = max(0, self.cell.glow - 5)


class Grid:
    def __init__(
        self,
        width: int,
        height: int,
        cell_width: int,
        cell_height: int,
        rows: int,
        cols: int,
    ) -> None:
        self.width = width
        self.height = height
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.rows = rows
        self.cols = cols
        self.cells = [
            [
                Cell(x * cell_width, y * cell_height, cell_width, cell_height, {})
                for x in range(cols)
            ]
            for y in range(rows)
        ]

    def get_cell_at(self, x_pos: int, y_pos: int) -> Optional[Cell]:
        x = x_pos // self.cell_width
        y = y_pos // self.cell_height

        if x < 0 or x >= self.cols or y < 0 or y >= self.rows:
            return None
        return self.cells[y][x]


class GridView:
    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self.cell_views = [[CellView(cell) for cell in row] for row in self.grid.cells]

    def draw(self, screen: pygame.Surface, offset_x: int, offset_y: int) -> None:
        for row in self.cell_views:
            for cell_view in row:
                cell_view.draw(screen, offset_x, offset_y)


class NoteInterface(ABC):
    @abstractmethod
    def get_tick(self) -> int:
        pass


class Note(NoteInterface):
    def __init__(self, tick: int) -> None:
        self.tick = tick

    def get_tick(self) -> int:
        return self.tick


class Sequencer:
    def __init__(self, length: int, callback: Callable[[NoteInterface], None]) -> None:
        self.length = length
        self.callback = callback
        self.notes: List[NoteInterface] = []
        self.last_tick = 0

    def update(self, tick: int) -> None:
        if tick < self.last_tick:
            self.last_tick = tick  # Handle tick overflow if necessary

        current_tick = tick % self.length
        last_tick_mod = self.last_tick % self.length

        for note in self.notes:
            note_tick = note.get_tick()
            if last_tick_mod <= current_tick:
                if last_tick_mod < note_tick <= current_tick:
                    self.callback(note)
            else:
                if last_tick_mod < note_tick or note_tick <= current_tick:
                    self.callback(note)

        self.last_tick = tick


class SoundNodeInterface(ABC):
    @abstractmethod
    def play_note(self, note: NoteInterface) -> None:
        pass

    @abstractmethod
    def update(self, tick: int) -> None:
        pass

    @abstractmethod
    def get_id(self) -> UUID:
        pass


class MidiInstrument:
    def __init__(self, output_port: mido.ports.BaseOutput) -> None:
        self.output_port = output_port
        self.playing: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.last_tick = 0

    def note_on(self, channel: int, note: int, gate_time: int, velocity: float) -> None:
        self.output_port.send(
            mido.Message(
                "note_on",
                channel=channel,
                note=note,
                velocity=min(max(int(velocity * 127), 0), 127),
            )
        )
        if gate_time > 0:
            self.playing[(channel, note)] = {"tick_elapsed": 0, "gate_time": gate_time}

    def channel_note_off(self, channel: int) -> None:
        for (ch, note), info in list(self.playing.items()):
            if ch == channel:
                self.output_port.send(
                    mido.Message("note_off", channel=channel, note=note)
                )
                del self.playing[(ch, note)]

    def update(self, tick: int) -> None:
        delta_tick = max(tick - self.last_tick, 0)
        self.last_tick = tick

        for (channel, note), info in list(self.playing.items()):
            info["tick_elapsed"] += delta_tick
            if info["tick_elapsed"] >= info["gate_time"]:
                self.output_port.send(
                    mido.Message("note_off", channel=channel, note=note)
                )
                del self.playing[(channel, note)]

    def cc(self, channel: int, control: int, value: float) -> None:
        self.output_port.send(
            mido.Message(
                "control_change",
                channel=channel,
                control=control,
                value=int(min(max(value * 127, 0), 127)),
            )
        )


class SoundNodeList:
    def __init__(
        self,
        sound_nodes: Mapping[UUID, SoundNodeInterface],
    ) -> None:
        self.sound_nodes = sound_nodes

    def update(self, tick: int) -> None:
        for sound_node in self.sound_nodes.values():
            sound_node.update(tick)


class SoundNode(SoundNodeInterface):
    def __init__(
        self,
        loop_length: int,
        physics_field: PhysicsField,
        grid: Grid,
        id: UUID,
        layer: str,
        instrument: MidiInstrument,
        sequence: List[NoteInterface] = [],
        global_note: Dict[str, Any] = {},
        color: Tuple[int, int, int] = (60, 60, 60),
        enable: bool = True,
    ) -> None:
        super().__init__()
        self.loop_length = loop_length
        self.physics_field = physics_field
        self.grid = grid
        self.id = id
        self.layer = layer
        self.instrument = instrument
        self.glow = 0
        self.velocity = 100
        self.sequence: List[NoteInterface] = sequence
        self.sequencer = Sequencer(loop_length, self.play_note)
        self.on_play = lambda node, cell: None
        self.sequencer.notes = self.sequence
        self.global_note = global_note
        self.color = color
        self.enable = enable

    def set_enable(self, enable: bool) -> None:
        self.enable = enable

    def play_note(self, note: NoteInterface) -> None:
        if not self.enable:
            return

        # Set glow effect
        self.glow = 100

        physics_node = self.physics_field.nodes.get(self.id)
        if physics_node is None:
            print("physics node not found")
            return
        cell = self.grid.get_cell_at(int(physics_node.px), int(physics_node.py))
        if cell is None:
            print("cell not found")
            return

        self.on_play(self, cell)

        attributes = cell.attributes.get(self.layer)
        if attributes is None:
            print("attributes not found")
            return

        cell.glow = 100
        cell.color = self.color

        self.instrument.channel_note_off(0)

        notes = attributes.get("notes", [])
        if attributes.get("use_global_notes"):
            print(self.global_note.get(self.layer))
            notes = [
                {
                    "note": n,
                    "channel": attributes.get("global_channel", 0),
                    "gate_time": attributes.get("global_gate_time", 10),
                }
                for n in self.global_note.get(self.layer, notes)
            ]
            print(notes)

        if attributes.get("arpeggio") and len(notes) > 1:
            notes = [random.choice(notes)]

        for note_attributes in notes:
            self.instrument.note_on(
                channel=note_attributes["channel"],
                note=note_attributes["note"],
                gate_time=note_attributes["gate_time"],
                velocity=0.8,
            )

        self.sequencer.notes = attributes.get("sequence", self.sequence)

    def update(self, tick: int) -> None:
        self.sequencer.update(tick)

    def get_id(self) -> UUID:
        return self.id


class NodeViewInterface(ABC):
    @abstractmethod
    def draw(self, screen: pygame.Surface, offset_x: int, offset_y: int) -> None:
        pass


class NodeView(NodeViewInterface):
    def __init__(
        self,
        physics_node: PhysicsNode,
        sound_node: SoundNode | None = None,
    ) -> None:
        self.sound_node = sound_node
        self.physics_node = physics_node

    def draw(self, screen: pygame.Surface, offset_x: int, offset_y: int) -> None:
        color_intensity = 0
        ring_color = (0, 0, 0)
        if self.sound_node is not None:
            color_intensity = max(0, min(255, self.sound_node.glow * 2))
            self.sound_node.glow = max(0, self.sound_node.glow - 8)
            ring_color = self.sound_node.color
            if not self.sound_node.enable:
                ring_color = (240, 240, 240)
        color = (255, 255, 255, color_intensity)
        gfxdraw.filled_circle(
            screen,
            int(self.physics_node.px) + offset_x,
            int(self.physics_node.py) + offset_y,
            self.physics_node.radius,
            color,
        )
        gfxdraw.aacircle(
            screen,
            int(self.physics_node.px) + offset_x,
            int(self.physics_node.py) + offset_y,
            self.physics_node.radius,
            ring_color,
        )
        gfxdraw.aacircle(
            screen,
            int(self.physics_node.px) + offset_x,
            int(self.physics_node.py) + offset_y,
            self.physics_node.radius - 2,
            ring_color,
        )


class UIElement(ABC):
    @abstractmethod
    def update(self, event: pygame.event.Event) -> None:
        pass

    @abstractmethod
    def draw(self, screen: pygame.Surface) -> None:
        pass


class ButtonWidget(UIElement):
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        text: str,
        font: pygame.font.Font,
        on_click: Callable[[], None] = lambda: None,
    ) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.margin = 2
        self.text = text
        self.font = font
        self.on_click = on_click
        self.hover_alpha = 0
        self.click_alpha = 0
        self.alpha_increment = 20
        self.is_clicked = False

    def draw(self, screen: pygame.Surface) -> None:
        rect = pygame.Rect(
            self.x + self.margin,
            self.y + self.margin,
            self.width - 2 * self.margin,
            self.height - 2 * self.margin,
        )

        # 透明度の反映
        hover_surface = pygame.Surface(
            (self.width - 2 * self.margin, self.height - 2 * self.margin),
            pygame.SRCALPHA,
        )
        click_surface = pygame.Surface(
            (self.width - 2 * self.margin, self.height - 2 * self.margin),
            pygame.SRCALPHA,
        )

        # マウスオーバー時の色
        hover_surface.fill((255, 255, 255, self.hover_alpha))
        # クリック時の色
        click_surface.fill((60, 60, 60, self.click_alpha))

        screen.blit(hover_surface, (self.x + self.margin, self.y + self.margin))
        screen.blit(click_surface, (self.x + self.margin, self.y + self.margin))

        # 通常時のボタンの枠
        gfxdraw.rectangle(screen, rect, (30, 30, 30))

        text = self.font.render(self.text, True, (0, 0, 0))
        text_rect = text.get_rect(
            center=(self.x + self.width / 2, self.y + self.height / 2)
        )
        screen.blit(text, text_rect)

        mouse_pos = pygame.mouse.get_pos()
        mouse_over = (
            self.x <= mouse_pos[0] <= self.x + self.width
            and self.y <= mouse_pos[1] <= self.y + self.height
        )

        if mouse_over:
            if self.hover_alpha < 250:
                self.hover_alpha = min(250, self.hover_alpha + self.alpha_increment)
        else:
            if self.hover_alpha > 0:
                self.hover_alpha = max(0, self.hover_alpha - self.alpha_increment)

        if self.is_clicked:
            self.click_alpha = 150
            self.is_clicked = False
        else:
            if self.click_alpha > 0:
                self.click_alpha = max(0, self.click_alpha - self.alpha_increment)

    def update(self, event: pygame.event.Event) -> None:
        mouse_pos = pygame.mouse.get_pos()
        mouse_over = (
            self.x <= mouse_pos[0] <= self.x + self.width
            and self.y <= mouse_pos[1] <= self.y + self.height
        )

        if mouse_over and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.is_clicked = True
            self.on_click()


class KnobWidget(UIElement):
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        initial_angle: float,
        on_change: Callable[[float], None] = lambda angle: None,
    ) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.radius = min(width, height) // 2 - 10
        self.angle = initial_angle
        self.on_change = on_change
        self.is_dragging = False
        self.min_angle = -math.pi * 5 / 6  # -150 degrees
        self.max_angle = math.pi * 5 / 6  # 150 degrees

    def draw(self, screen: pygame.Surface) -> None:
        center_x = self.x + self.width // 2
        center_y = self.y + self.height // 2

        # ノブの背景（円の縁）
        gfxdraw.aacircle(screen, center_x, center_y, self.radius, (30, 30, 30))

        # ノブのハンドル
        handle_x = center_x + self.radius * math.cos(self.angle)
        handle_y = center_y + self.radius * math.sin(self.angle)
        handle_inner_x = center_x + (self.radius // 2) * math.cos(self.angle)
        handle_inner_y = center_y + (self.radius // 2) * math.sin(self.angle)
        pygame.draw.aaline(
            screen,
            (0, 0, 0),
            (int(handle_inner_x), int(handle_inner_y)),
            (int(handle_x), int(handle_y)),
        )

        gfxdraw.aacircle(screen, center_x, center_y, self.radius // 2, (0, 0, 0))

    def update(self, event: pygame.event.Event) -> None:
        mouse_pos = pygame.mouse.get_pos()
        center_x = self.x + self.width // 2
        center_y = self.y + self.height // 2
        dx = mouse_pos[0] - center_x
        dy = mouse_pos[1] - center_y
        distance = math.sqrt(dx**2 + dy**2)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if distance <= self.radius:
                self.is_dragging = True

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.is_dragging = False

        if self.is_dragging and event.type == pygame.MOUSEMOTION:
            angle = math.atan2(dy, dx)
            if self.min_angle <= angle <= self.max_angle:
                self.angle = angle
                normalized_value = (self.angle - self.min_angle) / (
                    self.max_angle - self.min_angle
                )
                self.on_change(normalized_value)


class RotateListWidget(UIElement):
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        options: List[str],
        font: pygame.font.Font,
        on_select: Callable[[str], None] = lambda option: None,
    ) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.margin = 2
        self.options = options
        self.font = font
        self.on_select = on_select
        self.current_index = 0
        self.hover_alpha = 0
        self.alpha_increment = 20
        self.is_hovered = False

    def draw(self, screen: pygame.Surface) -> None:
        mouse_pos = pygame.mouse.get_pos()
        self.is_hovered = (
            self.x <= mouse_pos[0] <= self.x + self.width
            and self.y <= mouse_pos[1] <= self.y + self.height
        )

        if self.is_hovered:
            if self.hover_alpha < 250:
                self.hover_alpha = min(250, self.hover_alpha + self.alpha_increment)
        else:
            if self.hover_alpha > 0:
                self.hover_alpha = max(0, self.hover_alpha - self.alpha_increment)

        rect = pygame.Rect(
            self.x + self.margin,
            self.y + self.margin,
            self.width - 2 * self.margin,
            self.height - 2 * self.margin,
        )

        # 背景の塗りつぶし
        hover_surface = pygame.Surface(
            (self.width - 2 * self.margin, self.height - 2 * self.margin),
            pygame.SRCALPHA,
        )
        hover_surface.fill((255, 255, 255, self.hover_alpha))
        screen.blit(hover_surface, (self.x + self.margin, self.y + self.margin))

        # 現在の選択肢を表示
        current_text = self.font.render(
            self.options[self.current_index], True, (30, 30, 30)
        )
        current_text_rect = current_text.get_rect(
            center=(self.x + self.width / 2, self.y + self.height / 2)
        )
        screen.blit(current_text, current_text_rect)

        if self.is_hovered:
            # 前後の選択肢を表示
            prev_index = (self.current_index - 1) % len(self.options)
            next_index = (self.current_index + 1) % len(self.options)

            prev_text = self.font.render(
                self.options[prev_index],
                True,
                (
                    200 - self.hover_alpha // 5,
                    200 - self.hover_alpha // 5,
                    200 - self.hover_alpha // 5,
                ),
            )
            prev_text_rect = prev_text.get_rect(
                center=(self.x + self.width / 2, self.y + self.height / 2 - 30)
            )
            screen.blit(prev_text, prev_text_rect)

            next_text = self.font.render(
                self.options[next_index],
                True,
                (
                    200 - self.hover_alpha // 5,
                    200 - self.hover_alpha // 5,
                    200 - self.hover_alpha // 5,
                ),
            )
            next_text_rect = next_text.get_rect(
                center=(self.x + self.width / 2, self.y + self.height / 2 + 30)
            )
            screen.blit(next_text, next_text_rect)

        # 枠を描画
        gfxdraw.rectangle(screen, rect, (60, 60, 60))

    def update(self, event: pygame.event.Event) -> None:
        mouse_pos = pygame.mouse.get_pos()
        self.is_hovered = (
            self.x <= mouse_pos[0] <= self.x + self.width
            and self.y <= mouse_pos[1] <= self.y + self.height
        )

        if self.is_hovered and event.type == pygame.MOUSEWHEEL:
            if event.y > 0:  # マウスホイールアップ
                self.current_index = (self.current_index - 1) % len(self.options)
            elif event.y < 0:  # マウスホイールダウン
                self.current_index = (self.current_index + 1) % len(self.options)
            self.on_select(self.options[self.current_index])


class TextWidget(UIElement):
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        text: str,
        font: pygame.font.Font,
        color: tuple = (0, 0, 0),
    ) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.font = font
        self.color = color
        self.margin = 2

    def draw(self, screen: pygame.Surface) -> None:
        rect = pygame.Rect(
            self.x + self.margin,
            self.y + self.margin,
            self.width - 2 * self.margin,
            self.height - 2 * self.margin,
        )
        gfxdraw.rectangle(screen, rect, (30, 30, 30))
        text_surface = self.font.render(self.text, True, self.color)
        text_rect = text_surface.get_rect(
            center=(self.x + self.width / 2, self.y + self.height / 2)
        )
        screen.blit(text_surface, text_rect)

    def set_text(self, new_text: str) -> None:
        self.text = new_text

    def update(self, event: pygame.event.Event) -> None:
        pass


class ContainerWidget(UIElement):
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        widgets: List[UIElement] = [],
    ) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.widgets = widgets
        self.hover_alpha = 0
        self.alpha_increment = 20
        self.is_hovered = False

    def draw(self, screen: pygame.Surface) -> None:
        rect = pygame.Rect(self.x, self.y, self.width, self.height)

        # 背景の塗りつぶし
        hover_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        hover_surface.fill((255, 255, 255, self.hover_alpha))
        screen.blit(hover_surface, (self.x, self.y))

        # 枠の描画
        gfxdraw.rectangle(screen, rect, (30, 30, 30))

        # 各ウィジェットの描画
        for widget in self.widgets:
            widget.draw(screen)

        mouse_pos = pygame.mouse.get_pos()
        self.is_hovered = (
            self.x <= mouse_pos[0] <= self.x + self.width
            and self.y <= mouse_pos[1] <= self.y + self.height
        )

        if self.is_hovered:
            if self.hover_alpha < 250:
                self.hover_alpha = min(250, self.hover_alpha + self.alpha_increment)
        else:
            if self.hover_alpha > 0:
                self.hover_alpha = max(0, self.hover_alpha - self.alpha_increment)

    def update(self, event: pygame.event.Event) -> None:
        for widget in self.widgets:
            widget.update(event)


class SequencerWidget:
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        sequencer: Sequencer,
        color: Tuple[int, int, int] = (60, 60, 60),
        enable: bool = True,
    ) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.center_x = self.width // 2 + self.x
        self.center_y = self.height // 2 + self.y
        self.radius = min(self.width, self.height) // 2 - 10
        self.sequencer = sequencer
        self.color = color
        self.enabled = enable
        self.background_alpha = 0
        self.alpha_increment = 5
        self.margin = 2

    def draw(self, screen: pygame.Surface) -> None:
        self.draw_sequence(screen)
        self.draw_border(screen)

    def draw_sequence(self, screen: pygame.Surface) -> None:
        sequencer = self.sequencer
        notes = sorted(sequencer.notes, key=lambda note: note.get_tick())
        if len(notes) < 1:
            pygame.draw.circle(
                screen,
                self.color,
                (self.center_x, self.center_y),
                5,
            )
            return

        angles = [2 * math.pi * note.get_tick() / sequencer.length for note in notes]
        points = [
            (
                self.center_x + self.radius * math.cos(angle - math.pi / 2),
                self.center_y + self.radius * math.sin(angle - math.pi / 2),
            )
            for angle in angles
        ]

        if len(notes) == 1:
            gfxdraw.aacircle(
                screen, self.center_x, self.center_y, self.radius, (60, 60, 60)
            )
            angle = 2 * math.pi * sequencer.last_tick / sequencer.length - math.pi / 2
            gfxdraw.filled_circle(
                screen,
                int(self.center_x + math.cos(angle) * self.radius),
                int(self.center_y + math.sin(angle) * self.radius),
                5,
                self.color,
            )
            gfxdraw.aacircle(
                screen,
                int(self.center_x + math.cos(angle) * self.radius),
                int(self.center_y + math.sin(angle) * self.radius),
                5,
                self.color,
            )
            return

        for i in range(len(points) - 1):
            pygame.draw.aaline(
                screen,
                (60, 60, 60),
                (int(points[i][0]), int(points[i][1])),
                (int(points[i + 1][0]), int(points[i + 1][1])),
            )

        pygame.draw.aaline(
            screen,
            (60, 60, 60),
            (int(points[-1][0]), int(points[-1][1])),
            (int(points[0][0]), int(points[0][1])),
        )

        # 現在の位置を計算して線の上に小さな円を描画
        current_tick = sequencer.last_tick % sequencer.length
        current_index = 0
        for i, note in enumerate(notes):
            if note.get_tick() >= current_tick:
                current_index = i
                break

        next_point = points[current_index]
        prev_point = points[current_index - 1]
        lerp_ratio = (
            (current_tick - notes[current_index - 1].get_tick() + sequencer.length)
            % sequencer.length
            / (
                (
                    notes[current_index].get_tick()
                    - notes[current_index - 1].get_tick()
                    + sequencer.length
                )
                % sequencer.length
            )
        )

        lerp_x = prev_point[0] + (next_point[0] - prev_point[0]) * lerp_ratio
        lerp_y = prev_point[1] + (next_point[1] - prev_point[1]) * lerp_ratio

        gfxdraw.filled_circle(screen, int(lerp_x), int(lerp_y), 5, self.color)
        gfxdraw.aacircle(screen, int(lerp_x), int(lerp_y), 5, self.color)

    def draw_border(self, screen: pygame.Surface) -> None:
        rect = pygame.Rect(
            self.x + self.margin,
            self.y + self.margin,
            self.width - 2 * self.margin,
            self.height - 2 * self.margin,
        )
        gfxdraw.rectangle(screen, rect, (30, 30, 30))

        hover_surface = pygame.Surface(
            (self.width - 2 * self.margin, self.height - 2 * self.margin),
            pygame.SRCALPHA,
        )
        hover_surface.fill((220, 220, 220, self.background_alpha))
        screen.blit(hover_surface, (self.x + self.margin, self.y + self.margin))

        if self.enabled:
            self.background_alpha -= self.alpha_increment
        else:
            self.background_alpha += self.alpha_increment

        self.background_alpha = max(
            min(200, self.background_alpha),
            0,
        )

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def update(self, event: pygame.event.Event) -> None:
        pass


def extract_chords_from_midi(midi_file_path: str) -> Dict[int, List[List[int]]]:
    mid = mido.MidiFile(midi_file_path)
    channel_notes: Dict[int, List[tuple]] = defaultdict(list)
    channel_chords: Dict[int, List[List[int]]] = defaultdict(list)

    for track in mid.tracks:
        current_time = 0
        for msg in track:
            current_time += msg.time
            if not msg.is_meta:
                if msg.type == "note_on" and msg.velocity > 0:
                    channel_notes[msg.channel].append((current_time, msg.note))

    for channel, notes in channel_notes.items():
        # Group notes by their timing
        time_to_notes: Dict[int, List[int]] = defaultdict(list)
        for time, note in notes:
            time_to_notes[time].append(note)

        # Extract chords
        for time, extracted_notes in sorted(time_to_notes.items()):
            if (
                len(extracted_notes) > 0
            ):  # More than one note at the same time is a chord
                channel_chords[channel].append(extracted_notes)
                if len(channel_chords[channel]) >= 16:
                    break

    return channel_chords


if __name__ == "__main__":
    # Pygameの初期化
    pygame.init()

    # ウィンドウサイズ
    screen_width = 240 * 4 // 9 * 16
    screen_height = 240 * 4
    field_size = screen_height

    # Pygameのウィンドウを作成
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("CHAOSGRID")

    # フォントの設定
    pygame.font.init()
    font = pygame.font.SysFont("swisse.ttf", 24)

    # ゲームの設定
    clock = pygame.time.Clock()
    running = True

    # Grid, PhysicsField, Playerの初期化
    grid = Grid(screen_width, screen_height, field_size // 4, field_size // 4, 4, 4)

    physics_field = PhysicsField(0, 0, field_size, field_size, {})

    output_ports = mido.get_output_names()
    out_port_num = 0
    for i, port_name in enumerate(output_ports):
        print(f"{i}: {port_name}")
    while True:
        try:
            out_port_num = int(input("Select output port number: "))
            break
        except ValueError:
            print("Please enter a valid number.")
    output_port = mido.open_output(output_ports[out_port_num])

    instrument = MidiInstrument(output_port)

    keys = {
        "theme": uuid4(),
        "kick": uuid4(),
        "snare": uuid4(),
        "hihat": uuid4(),
        "shaker": uuid4(),
        "perc": uuid4(),
        "chord": uuid4(),
        "bass": uuid4(),
        "arp": uuid4(),
        "pad": uuid4(),
    }

    sound_nodes: Dict[UUID, SoundNode] = {}

    # note listの初期化
    score = extract_chords_from_midi("./chaosgrid/score.mid")
    global_notes_list: List[Dict[str, Any]] = []
    print(score[0])

    for i in range(16):
        global_notes_list.append(
            {
                "chord": score.get(0, [[] for _ in range(16)])[i],
                "bass": score.get(1, [[] for _ in range(16)])[i],
                "arp": score.get(2, [[] for _ in range(16)])[i],
                "pad": score.get(0, [[] for _ in range(16)])[i],
            }
        )
    print(global_notes_list)

    theme_node = SoundNode(
        loop_length=1920,
        physics_field=physics_field,
        grid=grid,
        id=keys["theme"],
        layer="theme",
        instrument=instrument,
        sequence=[Note(0)],
        color=(0xF2, 0x01, 0x2F),
        enable=True,
    )

    def set_key(node: SoundNode, cell: Cell) -> None:
        idx = cell.attributes.get(node.layer, {}).get("index", 0)

        for n in sound_nodes.values():
            n.global_note = global_notes_list[idx]
            print(n.global_note)

    theme_node.on_play = set_key

    kick_node = SoundNode(
        loop_length=1920,
        physics_field=physics_field,
        grid=grid,
        id=keys["kick"],
        layer="kick",
        instrument=instrument,
        sequence=[
            Note(0),
            Note(480),
            Note(960),
            Note(1440),
        ],
        enable=True,
    )

    snare_node = SoundNode(
        loop_length=1920,
        physics_field=physics_field,
        grid=grid,
        id=keys["snare"],
        layer="snare",
        instrument=instrument,
        sequence=[Note(480), Note(1440)],
        enable=False,
    )

    hihat_node = SoundNode(
        loop_length=1920,
        physics_field=physics_field,
        grid=grid,
        id=keys["hihat"],
        layer="hihat",
        instrument=instrument,
        sequence=[Note(120 * i) for i in range(16) if i % 4 != 0],
        enable=False,
    )

    shaker_node = SoundNode(
        loop_length=1920,
        physics_field=physics_field,
        grid=grid,
        id=keys["shaker"],
        layer="shaker",
        instrument=instrument,
        sequence=[
            Note(120 * 1),
            Note(120 * 4),
            Note(120 * 7),
            Note(120 * 10),
            Note(120 * 13),
        ],
        enable=False,
    )

    perc_node = SoundNode(
        loop_length=1920,
        physics_field=physics_field,
        grid=grid,
        id=keys["perc"],
        layer="perc",
        instrument=instrument,
        sequence=[
            Note(120 * 2),
            Note(120 * 5),
            Note(120 * 8),
            Note(120 * 11),
            Note(120 * 14),
        ],
        enable=False,
    )

    chord_node = SoundNode(
        loop_length=1920,
        physics_field=physics_field,
        grid=grid,
        id=keys["chord"],
        layer="chord",
        instrument=instrument,
        sequence=[Note(0)],
        color=(0xFB, 0x51, 0x05),
        enable=False,
    )

    bass_node = SoundNode(
        loop_length=1920,
        physics_field=physics_field,
        grid=grid,
        id=keys["bass"],
        layer="bass",
        instrument=instrument,
        sequence=[Note(240 * i) for i in range(8)],
        color=(0x18, 0x61, 0x63),
        enable=False,
    )

    arp_node = SoundNode(
        loop_length=1920,
        physics_field=physics_field,
        grid=grid,
        id=keys["arp"],
        layer="arp",
        instrument=instrument,
        sequence=[Note(120 * i) for i in range(16)],
        color=(0xEB, 0xC6, 0x02),
        enable=False,
    )

    pad_node = SoundNode(
        loop_length=1920,
        physics_field=physics_field,
        grid=grid,
        id=keys["pad"],
        layer="pad",
        instrument=instrument,
        sequence=[Note(0)],
        color=(0x8D, 0x4B, 0x0E),
        enable=False,
    )

    for i in range(4):
        for j in range(4):
            cell = grid.cells[i][j]
            cell.attributes["theme"] = {"index": i * 4 + j}
            cell.attributes["kick"] = {
                "notes": [
                    {"channel": 0, "note": 36, "gate_time": 10},
                ]
            }
            cell.attributes["snare"] = {
                "notes": [
                    {"channel": 0, "note": random.choice([38, 40]), "gate_time": 10},
                ]
            }
            cell.attributes["shaker"] = {
                "notes": [
                    {
                        "channel": 0,
                        "note": random.choice(
                            [48, 45, 47]
                        ),  # 50: Hi Tom, 45: Low Tom, 48: Mid Tom, 43: Floor Tom
                        "gate_time": 10,
                    },
                ]
            }
            cell.attributes["hihat"] = {
                "notes": [
                    {"channel": 0, "note": 42, "gate_time": 10},
                ],
                "sequence": [
                    Note(120 * k + (30 if k % 2 == 1 else 0))
                    for k in range(16)
                    if k % 4 != i
                ],
            }

            timings = []
            if j == 0:
                timings = [False, False, True, False, False, False, False, False] * 2
            if j == 1:
                timings = [False, False, True, False] * 4
            if j == 2:
                timings = [] + [False, False, True, False, False, False, True, True] * 2
            if j == 3:
                timings = [] + [True, False, True, False, True, False, True, True] * 2

            cell.attributes["perc"] = {
                "notes": [{"channel": 5, "note": 60 + i * 4 + j, "gate_time": 10}],
                "sequence": [Note(t * 120) for t, b in enumerate(timings) if b],
            }

            timings = []
            if j == 0:
                timings = [True, False, False, False] * 2 + [False] * 8
            if j == 1:
                timings = [True, False, False, True, False, False, True, False] + [
                    False
                ] * 8
            if j == 2:
                timings = [True, False, False] * 4 + [False] * 4
            if j == 3:
                timings = [True, False, False] * 4 + [True, False] * 2

            cell.attributes["chord"] = {
                "global_channel": 1,
                "use_global_notes": True,
                "global_gate_time": 480,
                "sequence": [Note(t * 120) for t, b in enumerate(timings) if b],
            }

            timings = []
            if j == 0:
                timings = [False, False, True, False] * 4
            if j == 1:
                timings = [False, False, True, False, False, True, False, True] * 2
            if j == 2:
                timings = [True, False] * 8
            if j == 3:
                timings = [False, True, True, True] * 4

            cell.attributes["bass"] = {
                "global_channel": 2,
                "use_global_notes": True,
                "global_gate_time": 100,
                "sequence": [Note(t * 120) for t, b in enumerate(timings) if b],
            }

            random_sequence = [True] * 4 * (j + 1) + [False] * 4 * (3 - j)
            random.shuffle(random_sequence)

            cell.attributes["arp"] = {
                "global_channel": 3,
                "use_global_notes": True,
                "global_gate_time": 50 * j,
                "sequence": [
                    Note(120 * k)
                    for idx, k in enumerate(range(16))
                    if random_sequence[idx]
                ],
                "arpeggio": True,
            }
            cell.attributes["pad"] = {
                "global_channel": 4,
                "use_global_notes": True,
                "global_gate_time": 1900,
                "sequence": [Note(0)],
            }

    sound_nodes[keys["theme"]] = theme_node
    sound_nodes[keys["kick"]] = kick_node
    sound_nodes[keys["snare"]] = snare_node
    sound_nodes[keys["hihat"]] = hihat_node
    sound_nodes[keys["shaker"]] = shaker_node
    sound_nodes[keys["perc"]] = perc_node
    sound_nodes[keys["chord"]] = chord_node
    sound_nodes[keys["bass"]] = bass_node
    sound_nodes[keys["arp"]] = arp_node
    sound_nodes[keys["pad"]] = pad_node

    sound_node_list = SoundNodeList(sound_nodes)

    # PhysicsNodesの作成と追加
    for key, id in keys.items():
        node = PhysicsNode(
            x=random.randint(0, field_size),
            y=random.randint(0, field_size),
            radius=50,
            mass=10.0,
            layer_name=key,
            id=id,
        )
        physics_field.add_node(node)

    # NodeViewsの作成
    node_views = [
        NodeView(node, sound_nodes.get(node.id))
        for node in physics_field.nodes.values()
    ]

    bpm = 120
    bps = bpm / 60.0
    ticks_per_beat = 480

    print(sound_nodes)

    # UIの初期化
    ui_elements: List[UIElement] = []

    seq_widgets = [
        SequencerWidget(
            960 + 24,
            96 * i,
            96,
            96,
            sound_nodes[k].sequencer,
            color=sound_nodes[k].color,
            enable=sound_nodes[k].enable,
        )
        for i, k in enumerate(keys.values())
    ]

    for widget in seq_widgets:
        ui_elements.append(widget)

    text_widgets = [
        TextWidget(960 + 120, 96 * i, 160, 96, k, font, sound_nodes[id].color)
        for i, (k, id) in enumerate(keys.items())
    ]

    for widget in text_widgets:
        ui_elements.append(widget)

    # メインゲームループ
    while running:
        # イベント処理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and pygame.K_1 <= event.key <= pygame.K_9:
                key_number = event.key - pygame.K_0
                key_list = list(keys.values())
                sound_nodes[key_list[key_number]].enable = (
                    False if sound_nodes[key_list[key_number]].enable else True
                )
                seq_widgets[key_number].set_enabled(
                    sound_nodes[key_list[key_number]].enable
                )
            for ui_element in ui_elements:
                ui_element.update(event)

        current_tick = int(pygame.time.get_ticks() / 1000.0 * bps * ticks_per_beat)

        # 時間経過の計算
        dt = 1 + min(
            40 / (min(current_tick % 480 + 1, 480 - current_tick % 480 + 1)), 10
        )  # 秒単位での経過時間

        # 物理フィールドの更新
        physics_field.update(dt)

        for i, phy_node in enumerate(physics_field.nodes.values()):
            instrument.cc(0, 20 + i, phy_node.px / field_size)
            instrument.cc(0, 30 + i, phy_node.py / field_size)
            instrument.cc(0, 40 + i, math.sqrt(phy_node.vx**2 + phy_node.vy**2) / 20)

        for _ in range(4):
            clock.tick(240)
            # sound_node_listの更新（サウンドノードの更新）
            current_tick = int(pygame.time.get_ticks() / 1000.0 * bps * ticks_per_beat)
            sound_node_list.update(current_tick)

            # 楽器の更新
            instrument.update(current_tick)

        # 画面のクリア
        screen.fill((220, 220, 220))

        # グリッドの描画
        grid_view = GridView(grid)
        grid_view.draw(screen, 0, 0)

        # ノードの描画
        for node_view in node_views:
            node_view.draw(screen, 0, 0)

        # UIの描画
        for ui_element in ui_elements:
            ui_element.draw(screen)

        # Pygameの画面更新
        pygame.display.flip()

    for channel in range(16):
        instrument.channel_note_off(channel)

    # Pygame終了処理
    pygame.quit()
    output_port.close()
