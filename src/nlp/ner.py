import os
import stanza
from typing import List, Dict, Any

from src.common.log_utils import init_logger
from src.config.settings import CONFIG, LOG_DIR


class NLPProcessor:
    def __init__(self):
        self.logger = init_logger(
            name="ner",
            module_name=str(self.__class__.__name__),
            log_dir=os.path.join(LOG_DIR, "model"),
        )

        # === 初始化 NLP pipeline ===
        nlp_cfg = CONFIG.get("NLP", {})
        processors = nlp_cfg.get("processors", "tokenize,pos,lemma,ner,depparse")
        lang = nlp_cfg.get("lang", "zh")
        use_gpu = nlp_cfg.get("use_gpu", False)

        self.logger.info(f"Initializing Stanza pipeline: {processors}")
        self.nlp = stanza.Pipeline(lang, processors=processors, use_gpu=use_gpu)

        self.logger.info("NLPProcessor initialized.")

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


if __name__ == "__main__":
    nlp = NLPProcessor()
    print(nlp.analyze("今天天气真好，我们一起去公园玩吧。"))

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
