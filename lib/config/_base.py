from typing import TypedDict, List, Dict


class DictInit(TypedDict):
    name: str
    args: List
    kwargs: Dict
    dynamic_args: List[str]
