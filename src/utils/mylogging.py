from rich.console import Console
from rich.syntax import Syntax
from omegaconf import OmegaConf

console = Console()

def pretty_print_config(cfg):
    """Prints the Hydra configuration with syntax highlighting."""
    config_str = OmegaConf.to_yaml(cfg, resolve=True)
    syntax = Syntax(config_str, "yaml", theme="monokai", line_numbers=False)
    console.print(syntax)