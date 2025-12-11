import os
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError

from src.common.log_utils import init_logger
from src.config.settings import CONFIG, LOG_DIR
from src.common.llm_utils import build_new_messages, split_reply, timeit


class OpenAILLM:
    """
    基于 OpenAI GPT-4o 的主语言处理模块

    继承自主语言处理基类，实现与 OpenAI API 的交互，处理用户输入并生成自然语言回复。
    """

    def __init__(self):
        self.name = "LLM OpenAI Module"
        self.config = CONFIG.get("NLP", {})
        self.model = self.config.get("model_name", "gpt-4o")
        self.max_completion_tokens = self.config.get("max_tokens", 2000)
        self.temperature = self.config.get("temperature", 0.7)

        self.logger = init_logger(
            name="nlp_llm",
            module_name=str(self.__class__.__name__),
            log_dir=os.path.join(LOG_DIR, "model"),
        )
        # 加载 API KEY
        load_dotenv()
        self.__api_key = os.getenv("OPENAI_API_KEY")
        # self.__api_key = os.getenv("OPENAI_API_KEY_REAL")
        # 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=self.__api_key)

    @timeit
    def process(self, text: str) -> str:
        """
        处理对话消息并生成回复
        """
        messages = build_new_messages()
        messages.append({"role": "user", "content": f"Text: {text}"})
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

    def process_batch(self, texts: list[str]) -> dict[int, str]:
        """
        批量处理对话消息并生成回复
        """
        messages = build_new_messages()
        messages.append(
            {
                "role": "system",
                "content": "For each input text, Return the result strictly in the format:\n[id]: [label]",
            }
        )

        # 构造编号输入
        items = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])
        messages.append(
            {"role": "user", "content": f"Here are the texts to classify:\n{items}"}
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_completion_tokens=self.max_completion_tokens,
            )
        except Exception as e:
            self.logger.error(f"API 调用错误: {e}")
            return {}

        ai_reply = response.choices[0].message.content
        raw = split_reply(ai_reply)

        # 解析输出：  "1: label\n2: label"
        result = {}
        for line in raw.split("\n"):
            if ":" in line:
                idx, label = line.split(":", 1)
                idx = int(idx.strip())
                label = label.strip()
                result[idx] = label

        return result


if __name__ == "__main__":
    openai_module = OpenAILLM()
    # res = openai_module.process("这个鱼肉真难吃，蔬菜少的可怜，我吃的一点不开心，下次不会再来了，餐厅服务也不好")
    # print(res)

    res2 = openai_module.process_batch(
        [
            "这个鱼肉真难吃，蔬菜少的可怜，我吃的一点不开心，下次不会再来了，餐厅服务也不好",
            "这个餐厅的菜味道很不错，服务也很好，下次还会再来",
        ]
    )
    print(res2)

    """
    result:
    {1: '食物质量', 2: '食物味道'}
    """