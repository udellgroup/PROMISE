from matplotlib.ticker import ScalarFormatter

class CustomScalarFormatter(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.1f"