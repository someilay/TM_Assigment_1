from math import cos, sin, sqrt, pi
from manim import *
from helpers import sign, get_vector, get_vector_title, get_mut_dot, get_vec_info, deep_arrange


class Task1(Scene):
    # Configs
    AXES_CONFIG = {
        'x_range': [-20, 20],
        'y_range': [-5, 115],
        'x_length': 2.5,
        'y_length': 7.5,
        'axis_config': {
            'stroke_width': 1,
            'tip_width': 0.15,
            'tip_height': 0.15,
            'tick_size': 0.01
        },
        'x_axis_config': {
            'numbers_to_include': np.arange(-20, 21, 10),
            'numbers_with_elongated_ticks': np.arange(-20, 21, 10),
            'font_size': 18,
            'line_to_number_buff': MED_SMALL_BUFF / 3,
        },
        'y_axis_config': {
            'numbers_to_include': np.arange(-0, 111, 10),
            'numbers_with_elongated_ticks': np.arange(-0, 111, 10),
            'font_size': 16,
            'line_to_number_buff': MED_SMALL_BUFF / 3,
        }
    }
    VECTORS_KWARGS = {
        'stroke_width': 2,
        'buff': 0,
        'max_tip_length_to_length_ratio': 0.075,
        'max_stroke_width_to_length_ratio': 5,
    }
    MATRIX_CONFIG = {
        'stroke_width': 1,
        'element_alignment_corner': LEFT,
        'element_to_mobject_config': {
            'num_decimal_places': 3
        }
    }

    VEL_S = 0.7  # Scale for velocities
    ACC_S = 2.2  # Scale for acceleration

    # Simulation time
    INITIAL_TIME = -5
    END_TIME = 5

    # Trajectory graph
    @staticmethod
    def graph_fun(x: float) -> float:
        return (4 * x ** 2) / 9 + 1

    # Body coordinates
    @staticmethod
    def coord(t: float) -> np.ndarray:
        return np.array((3 * t, 4 * t ** 2 + 1, 0.0))

    # Body velocity
    @staticmethod
    def vel(t: float) -> np.ndarray:
        return np.array((3.0, 8 * t, 0.0))

    # Body acceleration
    @staticmethod
    def acc(t: float) -> np.ndarray:
        return np.array((0.0, 8.0, 0.0))

    # Tangent acceleration
    @staticmethod
    def acc_t(t: float) -> np.ndarray:
        return np.array((3.0, 8 * t, 0.0)) * ((64 * t) / (9 + 64 * t ** 2))

    # Normal acceleration
    @staticmethod
    def acc_n(t: float) -> float:
        return np.array((-8 * t, 3.0, 0.0)) * (24 / (9 + 64 * t ** 2))

    # Curvature of trajectory
    @staticmethod
    def curv(t: float) -> float:
        return 8 / (9 * (sqrt(1 + (64 * t ** 2) / 9)) ** 3)

    # Get all vectors (velocity, acceleration, tangent acceleration, normal acceleration)
    def get_all_vectors(self, t: ValueTracker, axes: Axes):
        return (
            get_vector(t, axes, c, self.coord, fun, s, self.VECTORS_KWARGS)
            for fun, c, s in zip([self.vel, self.acc, self.acc_t, self.acc_n],
                                 [BLUE] + [RED] * 3,
                                 [self.VEL_S] + [self.ACC_S] * 3)
        )

    # Get titles for all vectors
    @staticmethod
    def get_all_vector_titles(t: ValueTracker, vectors: list[Arrow]):
        titles = ['\\vec{V}', '\\vec{a}', '\\vec{a_t}', '\\vec{a_n}']
        font_size = 28
        r_shift_text = np.array((0.25, 0.0, 0.0))
        u_shift_text = np.array((0.0, 0.15, 0.0))

        # Shifts respect to vectors' endings
        shifts = [
            lambda v: r_shift_text,
            lambda v: u_shift_text,
            lambda v: 0.25 * r_shift_text * sign(v[0]) + normalize(v) * 0.2,
            lambda v: normalize(v) * 0.25,
        ]

        return (
            get_vector_title(t, vector, title, font_size, shift)
            for vector, title, shift in zip(vectors, titles, shifts)
        )

    def get_all_info(self, t: ValueTracker, pos: np.ndarray) -> VGroup:
        scalar_labels = ['t=', 'k=']
        scalar_laws = [lambda t_c: t_c, self.curv]
        vec_labels = ['\\vec{r}=', '\\vec{V}=', '\\vec{a}=', '\\vec{a_t}=', '\\vec{a_n}=']
        vec_laws = [
            lambda c_t: self.coord(c_t)[:-1],
            lambda c_t: self.vel(c_t)[:-1],
            lambda c_t: self.acc(c_t)[:-1],
            lambda c_t: self.acc_t(c_t)[:-1],
            lambda c_t: self.acc_n(c_t)[:-1]
        ]

        font_size = 28
        scale = 0.5
        row_buff = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * 0.95
        column_buff = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * 0.75
        number_config = {'num_decimal_places': 3, 'font_size': 28}

        res = VGroup(
            *
            [
                get_vec_info(
                    t, label, font_size, law, number_config, scale=scale
                ).arrange(RIGHT, buff=row_buff)
                for label, law in zip(scalar_labels, scalar_laws)
            ] + [
                get_vec_info(
                    t, label, font_size, law, self.MATRIX_CONFIG, scale=scale
                ).arrange(RIGHT, buff=row_buff)
                for label, law in zip(vec_labels, vec_laws)
            ]
        )

        # Arranging
        deep_arrange(res, DOWN, buff=column_buff, center=False, aligned_edge=LEFT).move_to(pos)

        return res

    def construct(self):
        t = ValueTracker(self.INITIAL_TIME)

        axes = Axes(
            **self.AXES_CONFIG
        )
        axes_labels = axes.get_axis_labels()

        graph = axes.plot(self.graph_fun, color=BLUE, x_range=[-15, 15], stroke_width=1)

        body = get_mut_dot(t, axes, WHITE, DEFAULT_DOT_RADIUS / 2, self.coord)

        velocity, acc, acc_t, acc_n = self.get_all_vectors(t, axes)

        graph_name_pos = LEFT * 2.2  # Position in axes coordinates
        graph_name = MathTex('y=\\frac{4}{9}x^2+1', font_size=32)
        graph_name.move_to(graph_name_pos)

        velocity_text, acc_text, acc_t_text, acc_n_text = self.get_all_vector_titles(t, [velocity, acc, acc_t, acc_n])

        info_table_pos = RIGHT * 4
        total_info = self.get_all_info(t, info_table_pos)

        # Add and animate all objects
        self.add(
            axes, graph, graph_name, axes_labels,
            velocity, velocity_text,
            acc, acc_text,
            acc_t, acc_t_text,
            acc_n, acc_n_text,
            body,
            total_info,
        )
        self.play(t.animate.set_value(self.END_TIME), run_time=self.END_TIME - self.INITIAL_TIME, rate_func=linear)
        self.wait()
