import functools
from typing import Callable


class CallOrderChecker:
    def __init__(self):
        self.called_functions = set()

    def assert_called_before(self, target_func_name, func_name):
        if target_func_name not in self.called_functions:
            raise AssertionError(f"{target_func_name} should be called before calling {func_name}.")

    def assert_not_called_before(self, func_name, current_func_name):
        if func_name in self.called_functions:
            raise AssertionError(
                f"{func_name} should not be called before {current_func_name} to have the desired effect.")

    def record_call(self, func_name):
        self.called_functions.add(func_name)


def assert_called_before(target_func_name: str):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            self.call_order_checker.assert_called_before(target_func_name, func.__name__)
            result = func(self, *args, **kwargs)
            self.call_order_checker.record_call(func.__name__)
            return result

        return wrapper

    return decorator


def assert_not_called_before(func_name: str):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            self.call_order_checker.assert_not_called_before(func_name, func.__name__)
            result = func(self, *args, **kwargs)
            self.call_order_checker.record_call(func.__name__)
            return result

        return wrapper

    return decorator


def log_call(func: Callable):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self.call_order_checker.record_call(func.__name__)
        return result

    return wrapper


class LogFunctionCallsMeta(type):
    call_order_checker: CallOrderChecker

    def __new__(cls, name, bases, attrs):
        for key, value in attrs.items():
            if callable(value):
                attrs[key] = log_call(value)
        return super().__new__(cls, name, bases, attrs)

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls.call_order_checker = CallOrderChecker()
