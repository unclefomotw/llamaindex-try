from typing import Optional

from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent


class NaiveEventHandler(BaseEventHandler):
    log_file: str

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "NaiveEventHandler"

    @classmethod
    def _get_pretty_dict_str(
            cls,
            _dict: dict,
            skip_keys: Optional[list] = None,
            indent_str: str = ""
    ) -> str:
        _skip_keys = skip_keys or []

        ret = ""
        for _k, _v in _dict.items():
            if _k in _skip_keys:
                continue
            ret += f"{indent_str}{_k}: {_v}\n"
        return ret

    @classmethod
    def _get_pretty_even_str(cls, event: BaseEvent) -> str:
        _indent = "    "
        ret = ""

        for ek, ev in event.dict().items():
            if ek == "model_dict":
                # dict
                ret += f"{ek}:\n"
                ret += cls._get_pretty_dict_str(
                    ev, skip_keys=["api_key"], indent_str=_indent
                )
            elif ek == "embeddings":
                # List[List[float]]
                ret += f"{ek}: "
                ret += ",".join([f"<{len(_embedding)}-dim>" for _embedding in ev])
                ret += "\n"
            elif ek == "nodes":
                # List[NodeWithScore]
                # NodeWithScore is still too long; cannot think of a good repr in pure text
                ret += f"{ek}:\n"
                for _n in ev:
                    ret += f"{_indent}{_n}\n"
            elif ek == "messages":
                # List[ChatMessage]
                ret += f"{ek}:\n"
                for _n in ev:
                    ret += f"{_indent}{_n}\n"
            else:
                ret += f"{ek}: {ev}\n"

        return ret

    def handle(self, event: BaseEvent, **kwargs):
        """Logic for handling event."""
        with open(self.log_file, "a") as f:
            f.write(self._get_pretty_even_str(event))
            f.write("\n")
