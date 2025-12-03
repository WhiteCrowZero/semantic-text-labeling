import functools
import time


def build_new_messages():
    """
    构建新的对话消息列表
    """
    new_messages = [
        {
            "role": "system",
            "content": "You are an AI voice assistant. Respond concisely in short sentences.",
        }
    ]

    return new_messages


def split_reply(full_reply: str, flag="</think>"):
    """
    分割AI回复文本，根据指定分隔符将完整回复分割为纯文本部分

    Args:
        full_reply: 完整的AI回复文本
        flag: 分割标记（默认为"</think>"）

    Returns:
        分割后的纯文本内容
    """
    parts = full_reply.split(sep=flag)

    return parts[1] if len(parts) > 1 else parts[0]


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # 高精度计时器
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"{func.__qualname__} 执行耗时: {duration:.3f} 秒")
        return result

    return wrapper
