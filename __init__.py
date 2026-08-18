"""ComfyUI entry point -- lets this repo be cloned straight into
ComfyUI/custom_nodes/ and load as a node pack. Everything is pure
stdlib (plus ComfyUI's own numpy/Pillow), so there is nothing to
pip-install; only a running Ollama is needed."""

from .comfyui_node.h3_music_prompter import (NODE_CLASS_MAPPINGS,
                                             NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
