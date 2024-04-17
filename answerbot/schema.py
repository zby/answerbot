import inspect
from typing import Annotated, Callable, OrderedDict, get_origin, Type
import pydantic as pd
from pydantic_core import PydanticUndefined


def parameters_basemodel_from_function(function: Callable) -> Type[pd.BaseModel]:
    fields = OrderedDict()
    for name, parameter in inspect.signature(function).parameters.items():
        description = None
        type_ = parameter.annotation
        if get_origin(parameter.annotation) is Annotated:
            if parameter.annotation.__metadata__:
                description = parameter.annotation.__metadata__[0]
            type_ = parameter.annotation.__args__[0]
        default = PydanticUndefined if parameter.default is inspect.Parameter.empty else parameter.default
        fields[name] = (type_, pd.Field(default, description=description))

    return pd.create_model(f'{function.__name__}_ParameterModel', **fields)


def get_function_openai_schema(function: Callable) -> dict:
    model = parameters_basemodel_from_function(function)
    return {
            'type': 'function',
            'function': {
                'name': function.__name__,
                'description': (function.__doc__ or '').strip(),
                'parameters': model.model_json_schema()
                }
            }
