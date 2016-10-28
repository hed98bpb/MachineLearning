def plot_perf(names, fig=None, ax=None, xlim=None, ylim=None, figsize=(15, 10)):
    import matplotlib
    from IPython.display import display, clear_output

    formats = ['%s: %%s' % n for n in names]
    n = len(formats)

    if ax is None or isinstance(ax, int):
        if fig is None:
            fig = matplotlib.figure.Figure(figsize=figsize)
            fig.canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(fig)

        if isinstance(ax, int):
            ax = fig.add_subplot(ax)
        else:
            ax = fig.gca()
        if xlim is None:
            ax.set_autoscalex_on(True)
        else:
            ax.set_autoscalex_on(False)
            ax.set_xlim(*xlim)
        if ylim is None:
            ax.set_autoscaley_on(True)
        else:
            ax.set_autoscaley_on(False)
            ax.set_ylim(*ylim)
        ax.grid()
    lines = [([], [], ax.plot([], [])[0]) for _ in range(n)]

    def add(*points):
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        for y, (xs, ys, line) in zip(points, lines):
            xs.append(len(xs))
            ys.append(y)
            line.set_data(xs, ys)
        ax.legend([f % p for f, p in zip(formats, points)])
        ax.relim()
        ax.autoscale(enable=None)
        clear_output(wait=True)
        display(fig)

    return fig, add