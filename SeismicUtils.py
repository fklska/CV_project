from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.image import AxesImage


_colors = {
    'map_projection': '#0072B2',
    'vertical_view_x_projection': '#FF8C00',
    'vertical_view_y_projection': '#3CB371'
}


def _make_color_rectangle(ax: Axes, edgecolor: str) -> Rectangle:
    (ax_x_min, ax_x_max), (ax_y_max, ax_y_min) = ax.get_xlim(), ax.get_ylim()

    return Rectangle(
        (ax_x_min, ax_y_min),
        ax_x_max - ax_x_min,
        ax_y_max - ax_y_min,
        edgecolor=edgecolor,
        facecolor='none',
        linewidth=5
    )


def _enable_grid(ax: Axes, grid: bool) -> None:
    if grid:
        ax.grid()


def _enable_colorbar(obj: AxesImage, colorbar: bool, ax: Axes) -> None:
    if colorbar:
        plt.colorbar(obj, ax=ax)


def visualize_three_plots(
        cube: np.ndarray,
        z_index: int,
        x_index: int,
        y_index: int,
        cmap: str = 'gray',
        min_max_norm: bool = False,
        colorbar: bool = False,
        grid: bool = False,
        figsize: tuple[int, int] = (12, 5)
) -> None:
    if len(cube.shape) != 3:
        raise ValueError('Input cubes must have 3 dimensions')

    local_cube = np.transpose(cube, (2, 0, 1))

    if z_index < 0 or z_index >= local_cube.shape[0]:
        raise ValueError('z_index out of range')

    if x_index < 0 or x_index >= local_cube.shape[2]:
        raise ValueError('x_index out of range')

    if y_index < 0 or y_index >= local_cube.shape[1]:
        raise ValueError('y_index out of range')

    if min_max_norm:
        minimum = np.nanmin(local_cube)
        maximum = np.nanmax(local_cube)
        local_cube = (local_cube - minimum) / (maximum - minimum + np.finfo(np.float32).eps)

    local_cube_shape = local_cube.shape
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    plt.suptitle(
        f'Seismic cube, '
        f'z∈[0, {local_cube_shape[0]}), '
        f'x∈[0, {local_cube_shape[2]}), '
        f'y∈[0, {local_cube_shape[1]})'
    )

    img_map = axes[0].imshow(local_cube[z_index, :, :], cmap=cmap)
    axes[0].add_patch(_make_color_rectangle(axes[0], _colors['map_projection']))
    axes[0].set_title(f'z_index={z_index}')
    axes[0].plot(
        [x_index, x_index],
        [0, local_cube_shape[1] - 1],
        c=_colors['vertical_view_x_projection']
    )
    axes[0].plot(
        [0, local_cube_shape[2] - 1],
        [y_index, y_index],
        c=_colors['vertical_view_y_projection']
    )
    _enable_colorbar(img_map, colorbar, ax=axes[0])
    _enable_grid(axes[0], grid)

    vertical_view_x = axes[1].imshow(local_cube[:, :, x_index], cmap=cmap)
    axes[1].add_patch(_make_color_rectangle(axes[1], _colors['vertical_view_x_projection']))
    axes[1].set_title(f'x_index={x_index}')
    axes[1].plot(
        [y_index, y_index],
        [0, local_cube_shape[0] - 1],
        c=_colors['vertical_view_y_projection']
    )
    axes[1].plot(
        [0, local_cube_shape[1] - 1],
        [z_index, z_index],
        c=_colors['map_projection']
    )
    _enable_colorbar(vertical_view_x, colorbar, ax=axes[1])
    _enable_grid(axes[1], grid)

    vertical_view_y = axes[2].imshow(local_cube[:, y_index, :], cmap=cmap)
    axes[2].add_patch(_make_color_rectangle(axes[2], _colors['vertical_view_y_projection']))
    axes[2].set_title(f'y_index={y_index}')
    axes[2].plot(
        [x_index, x_index],
        [0, local_cube_shape[0] - 1],
        c=_colors['vertical_view_x_projection']
    )
    axes[2].plot(
        [0, local_cube_shape[2] - 1],
        [z_index, z_index],
        c=_colors['map_projection']
    )
    _enable_colorbar(vertical_view_y, colorbar, ax=axes[2])
    _enable_grid(axes[2], grid)

    plt.show()
