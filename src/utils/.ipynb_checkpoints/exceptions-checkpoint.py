class ArchitectureError(Exception):
    def __init__(self, arch, *args):
        super().__init__(args)
        self.arch = arch

    def __str__(self):
        return f"The architecture '{self.arch}' is not a valid model."
