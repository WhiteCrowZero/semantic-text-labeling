import os
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError

from src.common.log_utils import init_logger
from src.config.settings import CONFIG, LOG_DIR
from src.common.llm_utils import build_new_messages, split_reply, timeit


class MainOpenAIModule:
    """
    基于 OpenAI GPT-4o 的主语言处理模块

    继承自主语言处理基类，实现与 OpenAI API 的交互，处理用户输入并生成自然语言回复。
    """

    def __init__(self):
        self.name = "LLM OpenAI Module"
        self.model = CONFIG.get("model_name", "gpt-4o")
        self.max_completion_tokens = CONFIG.get("max_tokens", 2000)
        self.temperature = CONFIG.get("temperature", 0.7)
        self.logger = init_logger(
            name="llm",
            module_name=str(self.__class__.__name__),
            log_dir=os.path.join(LOG_DIR, "model"),
        )
        # 加载 API KEY
        load_dotenv()
        self.__api_key = os.getenv("OPENAI_API_KEY")
        # 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=self.__api_key)

    @timeit
    def process(self) -> str:
        """
        处理对话消息并生成回复
        """
        messages = build_new_messages()
        try:
            # 调用 OpenAI GPT-4o 接口获取回复
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
            )
        except RateLimitError as e:
            self.logger.error(f"OpenAI API 调用失败: {e}")
            return "OpenAI API 达到额度限制，请稍后再试。"
        except APIError as e:
            self.logger.error(f"OpenAI API 错误: {e}")
            return "OpenAI API 发生错误，请稍后再试。"
        except Exception as e:
            self.logger.error(f"未知错误: {e}")
            return "发生未知错误，请稍后再试。"

        # 数据处理：去除分隔符或元信息，只保留纯文本
        ai_reply = response.choices[0].message.content
        pure_reply = split_reply(ai_reply)
        return pure_reply


if __name__ == "__main__":
    main_openai_module = MainOpenAIModule()
    print(main_openai_module.process())
