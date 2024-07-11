import math
import shutil
import sys

import numpy as np

def clip(x, low, high):
    if x < low:
        x = low
    if x > high:
        x = high
    return x

GRAYSCALE_MAP = np.array(list(" .:-=+*#%@"))
def to_grayscale(values, z_range):
    values = np.clip(values, z_range[0], z_range[1]) - z_range[0]
    delta = z_range[1] - z_range[0]
    grayscale_indices = np.round(9 * (values / delta)).astype(np.uint8)
    return GRAYSCALE_MAP[grayscale_indices]

CSI = '\x1b['
def clear_screen_move_top():
    sys.stderr.write("\x1b[0J\x1b[H\n")

def get_term_size():
    """
    Return: (cols, rows) terminal size in characters
    """
    return shutil.get_terminal_size(0)


PLOT_CFG = {
    'y_label_width': 5,
    'x_label_width': 5,
    'horiz_padding': 1,
    'vert_padding': 0,
    'x_padding': 1,
    'y_padding': 1,
    'x_label_padding': 3,
    'y_label_padding': 3,
    'x_range': 'auto',
    'y_range': 'auto',
    'z_range': (0, 1)
}
def get_or_default(dict1, key, fallback):
    return dict1.get(key, fallback[key])

axis_thickness = 1
label_height = 1

class TerminalPlot:
    def __init__(self, rows=None, cols=None, **kwargs):
        if rows is None and cols is None:
            cols, rows = get_term_size()
        self.char_matrix = np.array([[' ']*cols for row in range(rows)])
        self.params = kwargs
        self.x_plot_start = 0
        self.x_plot_range = cols - 1
        self.y_plot_start = 0
        self.y_plot_range = rows - 1
        self.rows = rows
        self.cols = cols

        self.image_data = None
        self.z_range = None

        self.dirty_row_ranges = [(0, self.rows)]

    def move_to_cmd(self, row, col):
        return f'{CSI}{self.rows - row};{col+1}H'

    def create_axes(self, x_labels=True, y_labels=True, **kwargs):
        x_range = self.params['x_range']
        y_range = self.params['y_range']
        x_min, x_max = x_range
        y_min, y_max = y_range

        # Calculate x axis label size
        if x_labels:
            x_label_integer_size = max(len(str(math.floor(x_min))), len(str(math.ceil(x_max))))
            x_label_width = get_or_default(self.params, 'x_label_width', PLOT_CFG)
            x_label_width = max(x_label_width, x_label_integer_size)
            decimal_points = max(0, x_label_width - x_label_integer_size - 1)
            x_label_format = '{' + f':^{x_label_width}.{decimal_points}f' + '}'
        else:
            x_label_width = 0

        if y_labels:
            # Calculate y axis label size
            y_label_integer_size = max(len(str(math.floor(y_min))), len(str(math.ceil(y_max))))
            y_label_width = get_or_default(self.params, 'y_label_width', PLOT_CFG)
            y_label_width = max(y_label_width, y_label_integer_size)
            y_label_width = max(y_label_width, x_label_width // 2 - axis_thickness)
            decimal_points = max(0, y_label_width - y_label_integer_size - 1)
            y_label_format = '{' + f':>{y_label_width}.{decimal_points}f' + '}'
        else:
            y_label_width = 0

        # Set up the box
        horiz_padding = get_or_default(self.params, 'horiz_padding', PLOT_CFG)
        vert_padding = get_or_default(self.params, 'vert_padding', PLOT_CFG)
        x_padding = get_or_default(self.params, 'x_padding', PLOT_CFG)
        y_padding = get_or_default(self.params, 'y_padding', PLOT_CFG)

        self.x_plot_start = axis_thickness + horiz_padding + y_label_width + x_padding
        self.x_plot_range = self.cols - self.x_plot_start - horiz_padding - (x_label_width // 2) - 1
        self.y_plot_start = axis_thickness + label_height + vert_padding + y_padding
        self.y_plot_range = self.rows - self.y_plot_start - y_padding - vert_padding - 1

        if x_labels:
            # Calculate x label spacing
            x_label_padding = get_or_default(self.params, 'x_label_padding', PLOT_CFG)
            x_label_min_spacing = x_label_width + x_label_padding
            x_label_appearances = self.x_plot_range // x_label_min_spacing
            x_label_fractional_spacing = self.x_plot_range / x_label_appearances

        if y_labels:
            # Calculate y label spacing
            y_label_padding = get_or_default(self.params, 'y_label_padding', PLOT_CFG)
            y_label_min_spacing = label_height + y_label_padding
            y_label_appearances = self.y_plot_range // y_label_min_spacing
            y_label_fractional_spacing = self.y_plot_range / y_label_appearances

        # Draw axes
        #   horizontal
        row_target = vert_padding + label_height
        if horiz_padding == 0:
            self.char_matrix[row_target] = '-'
        else:
            self.char_matrix[row_target, horiz_padding:-horiz_padding] = '-'
        #   vertical
        col_target = horiz_padding + y_label_width
        if vert_padding == 0:
            self.char_matrix[:, col_target] = '|'
        else:
            self.char_matrix[vert_padding:-vert_padding, col_target] = '|'
        self.char_matrix[row_target, col_target] = '+'

        if y_labels:
            # Y axis labels
            y_row = self.y_plot_start
            y_labels = np.linspace(y_min, y_max, y_label_appearances + 1)
            for i, y_val in enumerate(y_labels):
                y_coord = math.floor(y_row)
                s = list(y_label_format.format(y_val))
                self.char_matrix[y_coord, horiz_padding:horiz_padding + y_label_width] = s
                self.char_matrix[y_coord, horiz_padding + y_label_width] = '+'
                y_row += y_label_fractional_spacing

        if x_labels:
            # X axis labels
            x_col_mid = self.x_plot_start
            x_col_start = x_col_mid - x_label_width // 2
            x_labels = np.linspace(x_min, x_max, x_label_appearances + 1)
            y_coord = vert_padding
            for i, x_val in enumerate(x_labels):
                x_start = math.floor(x_col_start)
                x_mid = math.floor(x_col_mid)
                s = list(x_label_format.format(x_val))
                self.char_matrix[y_coord, x_start:x_start + x_label_width] = s
                self.char_matrix[y_coord + label_height, x_mid] = '+'
                x_col_start += x_label_fractional_spacing
                x_col_mid += x_label_fractional_spacing

        self.dirty_row_ranges.append((0, self.rows))

    def max_binned_plot(self, xs, ys, symbol='*', **kwargs):
        x_range = self.params['x_range']
        y_range = self.params['y_range']
        x_min, x_max = x_range
        y_min, y_max = y_range

        plot_vals = [None] * (self.x_plot_range + 1)
        x_delta = x_max - x_min
        for x, y in zip(xs, ys):
            x_bin = round((x - x_min) / x_delta * self.x_plot_range)
            y_old = plot_vals[x_bin]
            if y_old is None:
                plot_vals[x_bin] = y
            elif y_old < y:
                plot_vals[x_bin] = y

        # Plot the binned values
        y_delta = y_max - y_min
        for i, val in enumerate(plot_vals):
            if val is None:
                continue
            val = clip(val, y_min, y_max)
            x_coord = self.x_plot_start + i
            y_coord = self.y_plot_start + round(self.y_plot_range * (val - y_min) / y_delta)
            self.char_matrix[y_coord, x_coord] = '*'
        self.dirty_row_ranges.append((self.y_plot_start, self.y_plot_start + self.y_plot_range + 1))


    def setup_image(self, image_width, image_height, plot_square=False, **kwargs):
        self.image_data = np.zeros((image_height, image_width))
        self.z_range = get_or_default(kwargs, 'z_range', PLOT_CFG)


    def plot_image_section(self, data, start_row=0):
        end_row = start_row+len(data)
        self.image_data[start_row:end_row, :] = data

        image_h, image_w = self.image_data.shape
        render_start_row = self.y_plot_start + math.floor(start_row * self.y_plot_range / image_h)
        render_end_row = self.y_plot_start + math.ceil(end_row * self.y_plot_range / image_h)
        self.render_image_section(render_start_row, render_end_row)


    def render_image_section(self, render_start_row, render_end_row):
        agg_values = np.zeros((render_end_row+1 - render_start_row, self.x_plot_range+1), dtype=np.float64)

        image_h, image_w = self.image_data.shape
        read_row = math.ceil((render_start_row - self.y_plot_start) * image_h / self.y_plot_range)
        next_read_row = read_row + 1
        next_render_row = self.y_plot_start + math.floor(next_read_row * self.y_plot_range / image_h)

        image_xs = np.linspace(0, 1, image_w)
        render_xs = np.linspace(0, 1, self.x_plot_range+1)
        for i, row in enumerate(range(render_start_row, render_end_row+1)):
            # TODO: fix artifacting at vertical pixel boundaries
            vals = np.interp(render_xs, image_xs, self.image_data[read_row, :])
            agg_values[i] += vals
            n = 1
            while next_render_row == row and read_row < image_h-1:
                read_row = next_read_row
                next_read_row = read_row + 1
                next_render_row = self.y_plot_start + math.floor(next_read_row * self.y_plot_range / image_h)
                vals = np.interp(render_xs, image_xs, self.image_data[read_row, :])
                agg_values[i] += vals
                n += 1
            agg_values[i] /= n
        ascii_art = to_grayscale(agg_values, self.z_range)
        self.char_matrix[
                render_start_row:render_end_row+1,
                self.x_plot_start:self.x_plot_start+self.x_plot_range+1
            ] = ascii_art

        self.dirty_row_ranges.append((render_start_row, render_end_row+1))


    def clear_plot_area(self):
        self.char_matrix[
                self.y_plot_start:self.y_plot_start+self.y_plot_range+1,
                self.x_plot_start:self.x_plot_start+self.x_plot_range+1
            ] = ' '
        self.dirty_row_ranges.append((self.y_plot_start, self.y_plot_start + self.y_plot_range + 1))


    def draw(self, **kwargs):
        # Print the character matrix
        row_dirty = np.zeros(self.rows, dtype=np.uint8)
        for range_min, range_max in self.dirty_row_ranges:
            row_dirty[range_min:range_max] = 1

        print_rows_inds = list(enumerate(self.char_matrix.tolist()))[::-1]
        output = None
        prev_line_output = False
        for ind, row in print_rows_inds:
            if not row_dirty[ind]:
                if output:
                    output.append('\n')
                prev_line_output = False
                continue
            if output is None:
                output = [self.move_to_cmd(ind, 0)]
            elif not prev_line_output:
                output.append('\n')
            output.append(''.join(row))
            prev_line_output = True

        output = ''.join(output)
        sys.stderr.write(output)
        self.dirty_row_ranges = []
    
def terminal_max_binned_plot(data, data2=None, **kwargs):
    if data2 is None:
        ys = data
        xs = range(len(data))
    else:
        xs = data
        ys = data2

    x_range = get_or_default(kwargs, 'x_range', PLOT_CFG)
    if x_range == 'auto':
        x_range = (min(xs), max(xs))
    kwargs['x_range'] = x_range

    y_range = get_or_default(kwargs, 'y_range', PLOT_CFG)
    if y_range == 'auto':
        y_range = (min(ys), max(ys))
    kwargs['y_range'] = y_range

    cols, rows = get_term_size()
    plotter = TerminalPlot(rows, cols, **kwargs)
    plotter.create_axes(**kwargs)
    plotter.max_binned_plot(xs, ys, **kwargs)
    plotter.draw(**kwargs)
    return plotter


def test_1d():
    plotter = TerminalPlot(x_range=[0, 2*np.pi], y_range=[0, 4])
    plotter.create_axes()
    xs = np.linspace(0, 2*np.pi, 200)
    total_time = 0
    while True:
        for t in np.linspace(-15, 15, 200):
            ys = (2 + 2*np.cos(xs - t)) * np.exp(-0.05*(xs - t)**2)

            t1 = time.time()
            plotter.clear_plot_area()
            plotter.max_binned_plot(xs, ys)
            plotter.draw()
            t2 = time.time()
            total_time += t2-t1
            time.sleep(0.01)
    #input(total_time)

def test_2d():
    plotter = TerminalPlot(x_range=[-np.pi, 3*np.pi], y_range=[0, 99])
    plotter.create_axes()

    plotter.setup_image(800, 100, z_range=[0, 4])
    xs = np.linspace(-np.pi, 3*np.pi, 800)
    total_time = 0
    for i, t in enumerate(np.linspace(0, 8, 100)):
        ys = (2 + 2*np.cos(xs - t)) * np.exp(-0.05*(xs - t)**2)

        t1 = time.time()
        plotter.plot_image_section(ys.reshape((1, -1)), start_row=i)
        plotter.draw()
        t2 = time.time()
        total_time += t2-t1
        time.sleep(0.01)
    #input(total_time)

if __name__ == '__main__':
    import time
    test_1d()
    #test_2d()
