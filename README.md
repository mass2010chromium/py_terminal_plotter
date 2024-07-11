# Python Terminal Plotter

## Why
Why not

## Quickstart

Simple plot
```python
import numpy as np
from terminal_plotter import terminal_max_binned_plot

# Generate a plot with y values only (just will be a line)
data = np.linspace(0, 2*np.pi, 200)
terminal_max_binned_plot(data)

# Enter to continue
input()

# Generate a plot with x and y values (sin plot)
ys = np.sin(data)
terminal_max_binned_plot(data, ys)

# Enter to continue
input()
```

Performant dynamic plotting
```python
import time
import numpy as np
from terminal_plotter import TerminalPlot

# Plot range needs to be setup beforehand
plotter = TerminalPlot(x_range=[0, 2*np.pi], y_range=[0, 4])
# Axes are optional and off by default
plotter.create_axes()

xs = np.linspace(0, 2*np.pi, 200)
# Plot a moving wave
while True:
    for t in np.linspace(-15, 15, 200):
        ys = (2 + 2*np.cos(xs - t)) * np.exp(-0.05*(xs - t)**2)
        plotter.clear_plot_area()
        plotter.max_binned_plot(xs, ys)
        plotter.draw()
        time.sleep(0.01)
```


## TODO
- Make it an actual installable thing
