import os
import stanza
from typing import List, Dict, Any

from src.common.log_utils import init_logger
from src.config.settings import CONFIG, LOG_DIR


class NLPProcessor:
    def __init__(self):
        # === 初始化 NLP pipeline ===
        self.config = CONFIG.get("NLP", {})
        self.processors = self.config.get("processors", "tokenize,pos,lemma,ner,depparse")
        self.lang = self.config.get("lang", "zh")
        self.use_gpu = self.config.get("use_gpu", False)
        self.local_model_path = self.config.get("local_model_path", "")

        self.logger = init_logger(
            name="nlp_ner",
            module_name=str(self.__class__.__name__),
            log_dir=os.path.join(LOG_DIR, "model"),
        )

        self.logger.info(f"Initializing Stanza pipeline: {self.processors}")
        self.nlp = None
        self.load_model()
        self.logger.info("NLPProcessor initialized.")

    def load_model(self):
        """
        根据配置加载 Stanza NLP 模型：
          - 若 local_model_dir 存在且已下载模型 => 从本地加载
          - 否则自动下载模型并写入 local_model_dir
        """

        model_name = self.lang  # 如 'zh'

        if self.local_model_path is not None:
            os.makedirs(self.local_model_path, exist_ok=True)

        # ----------- 1. 判断本地是否已下载 -----------
        model_exists = False
        if self.local_model_path:
            model_folder = os.path.join(self.local_model_path, model_name)
            resources_json = os.path.join(model_folder, "resources.json")
            if os.path.exists(resources_json):
                model_exists = True

        # ----------- 2. 若存在本地模型，则直接加载 -----------
        if model_exists:
            self.logger.info(f"[NLP] 从本地加载 Stanza 模型: {self.local_model_path}")

        # ----------- 3. 本地不存在，开始下载模型 -----------
        else:
            self.logger.info(f"[NLP] 本地未找到 Stanza 模型，正在从远程下载: {model_name}，到: {self.local_model_path}")

            try:
                stanza.download(
                    model_name, model_dir=self.local_model_path, verbose=False  # 不要输出太多内容
                )
                self.logger.info(f"[NLP] 已成功下载 Stanza 模型")
            except Exception as e:
                self.logger.error(f"[NLP] 下载 Stanza 模型失败: {e}")
                raise e

        # 初始化 pipeline
        self.nlp = stanza.Pipeline(
            model_name,
            processors=self.processors,
            use_gpu=self.use_gpu,
            model_dir=self.local_model_path,
        )

    # ============================================================
    #                   单条文本处理主接口
    # ============================================================
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        对外接口：给一句话，输出结构化信息与向量。
        """
        doc = self.nlp(text)

        entities: List[tuple] = []
        subjects: List[str] = []
        adj_pairs: List[tuple] = []
        keywords: List[str] = []
        main_event = None

        # ---- 命名实体 ----
        for ent in doc.ents:
            entities.append((ent.text, ent.type))

        # ---- 依存句法、词性 ----
        for sent in doc.sentences:
            words = sent.words
            id2w = {w.id: w for w in words}

            # 主语：nsubj
            for w in words:
                if w.deprel.startswith("nsubj"):
                    subjects.append(w.text)

            # 形容词 -> 名词修饰
            for w in words:
                if w.upos == "ADJ" and w.head != 0:
                    head = id2w[w.head]
                    if head.upos in ("NOUN", "PROPN", "PRON"):
                        adj_pairs.append((w.text, head.text))

            # 关键词：名词、专有名词、动词
            for w in words:
                if w.upos in ("NOUN", "PROPN", "VERB") and len(w.text) > 1:
                    keywords.append(w.text)

            # 找 root 动词作为主要事件
            if main_event is None:
                roots = [w for w in words if w.head == 0]
                event = None
                for r in roots:
                    if r.upos == "VERB":
                        event = r.text
                        break
                if event is None and roots:
                    event = roots[0].text
                main_event = event

        result = {
            "text": text,
            "entities": entities,
            "subjects": sorted(set(subjects)),
            "adj_pairs": adj_pairs,
            "main_event": main_event,
            "keywords": sorted(set(keywords)),
        }

        return result

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.analyze(text) for text in texts]


if __name__ == "__main__":
    nlp = NLPProcessor()
    res = nlp.analyze("今天天气真好，我们一起去公园玩吧。")
    print(res)

    # res2 = nlp.analyze_batch(["明天天气不好，我们不去公园了。", "今天天气真好，我们一起去公园玩吧。"])
    # print(res2)

    """
    example result:
    {
        "text": "今天天气真好，我们一起去公园玩吧。",
        "entities": [("今天", "DATE")],
        "subjects": ["天气", "我们"],
        "adj_pairs": [],
        "main_event": "玩吧",
        "keywords": ["一起", "今天", "公园", "天气", "玩吧"],
    }
    """
