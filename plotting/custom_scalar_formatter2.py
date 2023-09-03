from matplotlib.ticker import ScalarFormatter

class CustomScalarFormatter2(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.2f"