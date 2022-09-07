import numpy as np
from math import cos, sin, sqrt, pi
from manim import *
from typing import Callable, Union as TUnion
from helpers import create_mut_obj, get_vector, get_mut_dot, get_line, get_vector_title, get_vec_info, cut, deep_arrange


class Task2(Scene):
    # Configs
    AXES_CONFIG = {
        'x_range': [-120, 40],
        'y_range': [-40, 40],
        'x_length': 9,
        'y_length': 4.5,
        'axis_config': {
            'stroke_width': 1,
            'tip_width': 0.15,
            'tip_height': 0.15,
            'tick_size': 0.01,
        },
        'x_axis_config': {
            'numbers_to_include': np.arange(
                -120, 40, 10),
            'numbers_with_elongated_ticks': np.arange(
                -120, 40, 10),
            'font_size': 16,
            'line_to_number_buff': MED_SMALL_BUFF / 3,
        },
        'y_axis_config': {
            'numbers_to_include': np.arange(
                -40, 40, 10),
            'numbers_with_elongated_ticks': np.arange(
                -40, 40, 10),
            'font_size': 16,
            'line_to_number_buff': MED_SMALL_BUFF / 3,
        },
    }
    VECTORS_KWARGS = {
        'stroke_width': 3,
        'buff': 0,
        'max_tip_length_to_length_ratio': 0.12,
        'max_stroke_width_to_length_ratio': 6,
    }
    MATRIX_CONFIG = {
        'stroke_width': 1,
        'element_alignment_corner': LEFT,
        'element_to_mobject_config': {
            'num_decimal_places': 3,
            'font_size': 28,
        }
    }

    # System properties
    AB_LENGTH = 80
    AO_LENGTH = 25
    BC_LENGTH = 60  # AB - AC
    ANG_VEL = 1

    PHI_2 = 5 * pi / 6  # Initial angle in auxiliary coord. system
    O_Y_2 = AO_LENGTH * sqrt(3) / 2  # Point O coordinates in auxiliary coord. system

    # Transformation matrix from the original coord. system to the auxiliary coord. system
    ROTATION_ANG = 7 * pi / 6
    TRANS_MATRIX = np.array(
        [[cos(ROTATION_ANG), -sin(ROTATION_ANG), -AO_LENGTH * sqrt(3) / 4],
         [sin(ROTATION_ANG), cos(ROTATION_ANG), AO_LENGTH * 3 / 4],
         [0, 0, 1]],
    )

    FONT_SIZE = 22

    VEL_S = 1.0  # Scale for velocities
    ACC_S = 1.0  # Scale for acceleration

    VEL_COLOR = BLUE
    ACC_COLOR = RED

    # Simulation time
    INITIAL_TIME = 0
    END_TIME = TAU

    # Point A coordinates in the auxiliary coord. system
    @staticmethod
    def a_coord_2(t: float) -> np.ndarray:
        source = Task2
        return np.array(
            (
                -source.AO_LENGTH * sin(source.ANG_VEL * t + source.PHI_2),
                source.AO_LENGTH * cos(source.ANG_VEL * t + source.PHI_2) + source.O_Y_2,
                0.0,
            )
        )

    # Point A velocity in the auxiliary coord. system
    @staticmethod
    def a_vel_2(t: float) -> np.ndarray:
        source = Task2
        return np.array(
            (
                - source.AO_LENGTH * source.ANG_VEL * cos(source.ANG_VEL * t + source.PHI_2),
                - source.AO_LENGTH * source.ANG_VEL * sin(source.ANG_VEL * t + source.PHI_2),
                0.0,
            )
        )

    # Point A acceleration in the auxiliary coord. system
    @staticmethod
    def a_acc_2(t: float) -> np.ndarray:
        source = Task2
        return np.array(
            (
                source.AO_LENGTH * (source.ANG_VEL ** 2) * sin(source.ANG_VEL * t + source.PHI_2),
                - source.AO_LENGTH * (source.ANG_VEL ** 2) * cos(source.ANG_VEL * t + source.PHI_2),
                0.0,
            )
        )

    # Point B coordinates in the auxiliary coord. system
    @staticmethod
    def b_coord_2(t: float) -> np.ndarray:
        source = Task2
        a_2 = source.a_coord_2(t)
        return np.array(
            (
                a_2[0] + sqrt(source.AB_LENGTH ** 2 - a_2[1] ** 2),
                0.0,
                0.0,
            )
        )

    # Point B velocity in the auxiliary coord. system
    @staticmethod
    def b_vel_2(t: float) -> np.ndarray:
        source = Task2
        a_c_2 = source.a_coord_2(t)
        a_v_2 = source.a_vel_2(t)
        return np.array(
            (
                a_v_2[0] - a_c_2[1] * a_v_2[1] / sqrt(source.AB_LENGTH ** 2 - a_c_2[1] ** 2),
                0.0,
                0.0,
            )
        )

    # Point B acceleration in the auxiliary coord. system
    @staticmethod
    def b_acc_2(t: float) -> np.ndarray:
        source = Task2
        a_c_2 = source.a_coord_2(t)
        a_v_2 = source.a_vel_2(t)
        a_a_2 = source.a_acc_2(t)
        return np.array(
            (
                a_a_2[0]
                - a_c_2[1] * a_a_2[1] / sqrt(source.AB_LENGTH ** 2 - a_c_2[1] ** 2)
                - (source.AB_LENGTH ** 2 / (source.AB_LENGTH ** 2 - a_c_2[1] ** 2))
                * (a_v_2[1] ** 2) / sqrt(source.AB_LENGTH ** 2 - a_c_2[1] ** 2),
                0.0,
                0.0,
            )
        )

    # Point C coordinates in the auxiliary coord. system
    @staticmethod
    def c_coord_2(t: float) -> np.ndarray:
        source = Task2
        alpha = source.BC_LENGTH / source.AB_LENGTH

        return alpha * source.a_coord_2(t) + (1 - alpha) * source.b_coord_2(t)

    # Point C velocity in the auxiliary coord. system
    @staticmethod
    def c_vel_2(t: float) -> np.ndarray:
        source = Task2
        alpha = source.BC_LENGTH / source.AB_LENGTH

        return alpha * source.a_vel_2(t) + (1 - alpha) * source.b_vel_2(t)

    # Point C acceleration in the auxiliary coord. system
    @staticmethod
    def c_acc_2(t: float) -> np.ndarray:
        source = Task2
        alpha = source.BC_LENGTH / source.AB_LENGTH

        return alpha * source.a_acc_2(t) + (1 - alpha) * source.b_acc_2(t)

    # Convert coordinates from the auxiliary coord. system to the original one
    @staticmethod
    def transform(r: np.ndarray, with_shift: bool = True) -> np.ndarray:
        source = Task2

        if with_shift:
            res = source.TRANS_MATRIX.dot(r + np.array((0.0, 0.0, 1.0)))
            res[2] = 0.0
            return res

        return source.TRANS_MATRIX.dot(r)

    # Point A coordinates in the original coord. system
    @staticmethod
    def a_coord_1(t: float) -> np.ndarray:
        source = Task2
        return source.transform(source.a_coord_2(t))

    # Point A velocity in the original coord. system
    @staticmethod
    def a_vel_1(t: float) -> np.ndarray:
        source = Task2
        return source.transform(source.a_vel_2(t), False)

    # Point A acceleration in the original coord. system
    @staticmethod
    def a_acc_1(t: float) -> np.ndarray:
        source = Task2
        return source.transform(source.a_acc_2(t), False)

    # Point A tangent acc. in the original coord. system
    @staticmethod
    def a_acc_t_1(t: float) -> np.ndarray:
        source = Task2
        a_vel = source.a_vel_1(t)
        n_a_vel = np.linalg.norm(a_vel)
        return (source.a_acc_1(t).dot(a_vel) / n_a_vel) * normalize(a_vel)

    # Point A normal acc. in the original coord. system
    @staticmethod
    def a_acc_n_1(t: float) -> np.ndarray:
        source = Task2
        return source.a_acc_1(t) - source.a_acc_t_1(t)

    # Point B coordinates in the original coord. system
    @staticmethod
    def b_coord_1(t: float) -> np.ndarray:
        source = Task2
        return source.transform(source.b_coord_2(t))

    # Point B velocity in the original coord. system
    @staticmethod
    def b_vel_1(t: float) -> np.ndarray:
        source = Task2
        return source.transform(source.b_vel_2(t), False)

    # Point B acceleration in the original coord. system
    @staticmethod
    def b_acc_1(t: float) -> np.ndarray:
        source = Task2
        return source.transform(source.b_acc_2(t), False)

    # Point B tangent acc. in the original coord. system
    @staticmethod
    def b_acc_t_1(t: float) -> np.ndarray:
        source = Task2
        b_vel = source.b_vel_1(t)
        n_b_vel = np.linalg.norm(b_vel)
        return (source.b_acc_1(t).dot(b_vel) / n_b_vel) * normalize(b_vel)

    # Point B normal acc. in the original coord. system
    @staticmethod
    def b_acc_n_1(t: float) -> np.ndarray:
        source = Task2
        return source.b_acc_1(t) - source.b_acc_t_1(t)

    # Point C coordinates in the original coord. system
    @staticmethod
    def c_coord_1(t: float) -> np.ndarray:
        source = Task2
        return source.transform(source.c_coord_2(t))

    # Point C velocity in the original coord. system
    @staticmethod
    def c_vel_1(t: float) -> np.ndarray:
        source = Task2
        return source.transform(source.c_vel_2(t), False)

    # Point C acceleration in the original coord. system
    @staticmethod
    def c_acc_1(t: float) -> np.ndarray:
        source = Task2
        return source.transform(source.c_acc_2(t), False)

    # Point C tangent acc. in the original coord. system
    @staticmethod
    def c_acc_t_1(t: float) -> np.ndarray:
        source = Task2
        c_vel = source.c_vel_1(t)
        n_c_vel = np.linalg.norm(c_vel)
        return (source.c_acc_1(t).dot(c_vel) / n_c_vel) * normalize(c_vel)

    # Point C normal acc. in the original coord. system
    @staticmethod
    def c_acc_n_1(t: float) -> np.ndarray:
        source = Task2
        return source.c_acc_1(t) - source.c_acc_t_1(t)

    # Point B trajectory line
    @staticmethod
    def b_line_eq(x: float) -> float:
        return (x * sqrt(3)) / 3 + Task2.AO_LENGTH

    # Get all points (A, B, C, O)
    def get_all_points(self, t: ValueTracker, axes: Axes):
        c = WHITE
        laws = [lambda v: np.array((0.0, 0.0, 0.0)), self.a_coord_1, self.b_coord_1, self.c_coord_1]
        radius = DEFAULT_DOT_RADIUS * 0.75
        return (
            get_mut_dot(t, axes, c, radius, law)
            for law in laws
        )

    # Get all mechanism lines (AO, AB)
    def get_all_mechanism_lines(self, t: ValueTracker, axes: Axes):
        c = GREEN
        laws = [
            (lambda v: np.array((0.0, 0.0, 0.0)), self.a_coord_1),
            (self.a_coord_1, self.b_coord_1)
        ]
        return (
            get_line(t, axes, c, s_law, e_law)
            for s_law, e_law in laws
        )

    def get_vector(self, t: ValueTracker, axes: Axes, c: str,
                   start_function: Callable[[float], np.ndarray],
                   end_function: Callable[[float], np.ndarray],
                   scale: float, kwargs: dict = None) -> Arrow:
        a_kwargs = self.VECTORS_KWARGS

        if kwargs:
            a_kwargs = a_kwargs.copy()
            a_kwargs.update(kwargs)

        return get_vector(t, axes, c, start_function, end_function, scale, a_kwargs)

    def get_vectors(self, t: ValueTracker, axes: Axes, c: str, scale: TUnion[float, list[float]],
                    e_functions: list[Callable[[float], np.ndarray]], l_kwargs: list[dict] = None):
        s_functions = [self.a_coord_1, self.b_coord_1, self.c_coord_1]

        if isinstance(scale, float):
            scale = [scale] * len(e_functions)

        if not l_kwargs:
            l_kwargs = [None] * len(e_functions)

        return (
            self.get_vector(t, axes, c, s_fun, e_fun, scale_, kwargs)
            for s_fun, e_fun, scale_, kwargs in zip(s_functions, e_functions, scale, l_kwargs)
        )

    # Get V_A, V_B, V_C
    def get_all_vel(self, t: ValueTracker, axes: Axes):
        e_functions = [self.a_vel_1, self.b_vel_1, self.c_vel_1]
        return self.get_vectors(t, axes, self.VEL_COLOR, self.VEL_S, e_functions)

    # Get a_A, a_B, a_C
    def get_all_acc(self, t: ValueTracker, axes: Axes):
        e_functions = [self.a_acc_1, self.b_acc_1, self.c_acc_1]
        return self.get_vectors(t, axes, self.ACC_COLOR, self.ACC_S, e_functions)

    # Get a_tA, a_tB, a_tC
    def get_all_acc_t(self, t: ValueTracker, axes: Axes):
        e_functions = [self.a_acc_t_1, self.b_acc_t_1, self.c_acc_t_1]
        return self.get_vectors(
            t, axes, self.ACC_COLOR, self.ACC_S, e_functions,
            l_kwargs=[
                None,
                None,
                {   # Special settings for a_tC
                    'max_stroke_width_to_length_ratio': 15,
                    'stroke_width': 4,
                    'max_tip_length_to_length_ratio': 0.2,
                }
            ]
        )

    # Get a_nA, a_nB, a_nC
    def get_all_acc_n(self, t: ValueTracker, axes: Axes):
        e_functions = [self.a_acc_n_1, self.b_acc_n_1, self.c_acc_n_1]
        return self.get_vectors(t, axes, self.ACC_COLOR, self.ACC_S, e_functions)

    def get_vectors_titles(self, t: ValueTracker, titles: list[str],
                           shift_laws: list[Callable[[np.ndarray], np.ndarray]], vectors: list[Arrow]):
        return (
            get_vector_title(t, vector, title, self.FONT_SIZE, shift_law)
            for title, shift_law, vector in zip(titles, shift_laws, vectors)
        )

    def get_all_vel_titles(self, t: ValueTracker, vectors: list[Arrow]):
        shift_laws = [lambda v: normalize(v) * 0.2] * 3
        shift_laws[1] = lambda v: np.array((0, 0.28, 0))

        titles = ['\\vec{V_A}', '\\vec{V_B}', '\\vec{V_C}']
        return self.get_vectors_titles(t, titles, shift_laws, vectors)

    def get_all_acc_titles(self, t: ValueTracker, vectors: list[Arrow]):
        shift_laws = [lambda v: normalize(v)] * 3

        shift_laws[0] = lambda v: normalize(v) * 0.23
        shift_laws[1] = lambda v: np.array((0, 0.28, 0))
        shift_laws[2] = lambda v: normalize(v) * 0.2

        titles = ['\\vec{a_A}', '\\vec{a_B}', '\\vec{a_C}']
        return self.get_vectors_titles(t, titles, shift_laws, vectors)

    def get_all_acc_t_titles(self, t: ValueTracker, vectors: list[Arrow]):
        shift_laws = [lambda v: normalize(v)] * 3

        shift_laws[0] = lambda v: -np.array((0, 0.28, 0))
        shift_laws[1] = lambda v: -np.array((-0.05, 0.25, 0))
        shift_laws[2] = lambda v: -np.array((0, 0.2, 0))

        titles = ['\\vec{a_{t,A}}', '\\vec{a_{t,B}}', '\\vec{a_{t,C}}']
        return self.get_vectors_titles(t, titles, shift_laws, vectors)

    def get_all_acc_n_titles(self, t: ValueTracker, vectors: list[Arrow]):
        shift_laws = [lambda v: normalize(v)] * 3

        shift_laws[0] = lambda v: \
            normalize(v) * 0.2 + np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).dot(normalize(v) * 0.24)
        shift_laws[1] = lambda v: np.array((1, -sqrt(3), 0)) * 0.18
        shift_laws[2] = lambda v: normalize(v) * 0.2

        titles = ['\\vec{a_{n,A}}', '\\vec{a_{n,B}}', '\\vec{a_{n,C}}']
        return self.get_vectors_titles(t, titles, shift_laws, vectors)

    # For telemetry point state
    def get_point_info(self, t: ValueTracker, point_name: str, font_size: int, scale: float,
                       row_buff: float, laws: list[Callable[[float], np.ndarray]]):
        labels = [
            f'\\vec{{r_{point_name}}}=',
            f'\\vec{{V_{point_name}}}=',
            f'\\vec{{a_{point_name}}}=',
            f'\\vec{{a_{{t,{point_name}}}}}=',
            f'\\vec{{a_{{n,{point_name}}}}}='
        ]

        return VGroup(
            *[
                get_vec_info(t, label, font_size, law, self.MATRIX_CONFIG, scale=scale).arrange(RIGHT, buff=row_buff)
                for label, law in zip(labels, laws)
            ]
        )

    def get_all_points_infos(self, t: ValueTracker):
        points = ['A', 'B', 'C']
        points_laws = [
            [self.a_coord_1, self.a_vel_1, self.a_acc_1, self.a_acc_t_1, self.a_acc_n_1],
            [self.b_coord_1, self.b_vel_1, self.b_acc_1, self.b_acc_t_1, self.b_acc_n_1],
            [self.c_coord_1, self.c_vel_1, self.c_acc_1, self.c_acc_t_1, self.c_acc_n_1]
        ]
        scale = 0.5
        row_buff = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * 0.95
        column_buff = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * 0.75

        return deep_arrange(
            VGroup(
                *[
                    self.get_point_info(
                        t, point, self.FONT_SIZE, scale, row_buff, [cut(law) for law in laws]
                    ).arrange(RIGHT, buff=row_buff)
                    for point, laws in zip(points, points_laws)
                ]
            ),
            DOWN, buff=column_buff, center=False, aligned_edge=LEFT
        )

    def get_point_trajectory(self, axes: Axes, law: Callable[[float], np.ndarray]):
        curve = axes.plot_parametric_curve(
            law, color=WHITE, t_range=np.array([self.INITIAL_TIME, self.END_TIME]),
            stroke_width=DEFAULT_STROKE_WIDTH / 5
        )
        return DashedVMobject(curve, num_dashes=30)

    def construct(self):
        t = ValueTracker(self.INITIAL_TIME)

        axes_pos = UP * 1.5
        axes = Axes(**self.AXES_CONFIG).move_to(axes_pos)
        x_label, y_label = axes_labels = axes.get_axis_labels()
        x_label.set(font_size=self.FONT_SIZE * 2)
        y_label.set(font_size=self.FONT_SIZE * 2)

        b_line = axes.plot(self.b_line_eq, color=BLUE, x_range=[-120, 20], stroke_width=1)

        o_point, a_point, b_point, c_point = self.get_all_points(t, axes)

        body_height = 0.3
        body_width = 0.6
        stroke_width = DEFAULT_STROKE_WIDTH / 2

        # Create rectangle that represents the body
        body_rectangle = create_mut_obj(
            lambda tracker: Rectangle(
                height=body_height, width=body_width, stroke_width=stroke_width
            ).rotate(pi / 6).move_to(b_point),  # Rotated by 30 degrees
            t,
            lambda getter, tracker: lambda z: z.move_to(b_point)
        )

        ao_line, ab_line = self.get_all_mechanism_lines(t, axes)

        a_vel, b_vel, c_vel = self.get_all_vel(t, axes)
        a_acc, b_acc, c_acc = self.get_all_acc(t, axes)
        a_acc_t, b_acc_t, c_acc_t = self.get_all_acc_t(t, axes)
        a_acc_n, b_acc_n, c_acc_n = self.get_all_acc_n(t, axes)

        a_vel_text, b_vel_text, c_vel_text = self.get_all_vel_titles(t, [a_vel, b_vel, c_vel])
        a_acc_text, b_acc_text, c_acc_text = self.get_all_acc_titles(t, [a_acc, b_acc, c_acc])
        a_acc_t_text, b_acc_t_text, c_acc_t_text = self.get_all_acc_t_titles(t, [a_acc_t, b_acc_t, c_acc_t])
        a_acc_n_text, b_acc_n_text, c_acc_n_text = self.get_all_acc_n_titles(t, [a_acc_n, b_acc_n, c_acc_n])

        timer_pos = (LEFT + UP) * 3.5
        timer = get_vec_info(
            t, 't=', self.FONT_SIZE, lambda c_t: c_t, {'num_decimal_places': 3, 'font_size': self.FONT_SIZE}
        ).arrange(RIGHT, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER / 5).move_to(timer_pos)

        all_info_pos = DOWN * 2.5
        all_info = self.get_all_points_infos(t).move_to(all_info_pos)

        curve_a = self.get_point_trajectory(axes, self.a_coord_1)
        curve_c = self.get_point_trajectory(axes, self.c_coord_1)

        self.add(
            axes, axes_labels,
            b_line,
            curve_a, curve_c,
            body_rectangle,
            ao_line, ab_line,
            a_vel, b_vel, c_vel,
            a_acc, b_acc, c_acc,
            a_acc_t, b_acc_t, c_acc_t,
            a_acc_n, b_acc_n, c_acc_n,
            o_point, a_point, b_point, c_point,
            a_vel_text, b_vel_text, c_vel_text,
            a_acc_text, b_acc_text, c_acc_text,
            a_acc_t_text, b_acc_t_text, c_acc_t_text,
            a_acc_n_text, b_acc_n_text, c_acc_n_text,
            timer,
            all_info
        )
        self.play(t.animate.set_value(self.END_TIME), run_time=self.END_TIME - self.INITIAL_TIME, rate_func=linear)
