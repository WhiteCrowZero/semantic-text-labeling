import os
from typing import List

from src.common.log_utils import init_logger
from src.config.settings import CONFIG, LOG_DIR

from src.clustering.label_clustering import LabelClusterer
from src.embeddings.bert import BertEmbedder
from src.nlp.ner import NLPProcessor
from src.nlp.llm import OpenAILLM


class TextModelService:
    def __init__(self):
        self.logger = init_logger(
            name="service",
            module_name=__name__,
            log_dir=os.path.join(LOG_DIR, "web"),
        )
        self._load_model()
        self.config = CONFIG.get("WEB", {})
        self._llm_batch_size = self.config.get("llm_batch_size", 30)
        self._clustering_batch_size = self.config.get("clustering_batch_size", 60)
        self.logger.info("[Service] 服务初始化完成")

    def _load_model(self):
        self.logger.info("[Service] 加载模型")
        self.bert_embedder = BertEmbedder()
        self.logger.info("[Service] Bert模型加载完成")
        self.label_clusterer = LabelClusterer()
        self.logger.info("[Service] 标签聚类器加载完成")
        self.llm = OpenAILLM()
        self.logger.info("[Service] LLM加载完成")
        self.nlp_processor = NLPProcessor()
        self.logger.info("[Service] NLP处理器加载完成")

    def _llm_tag(self, texts: List[str]):
        self.logger.info("[Service] LLM开始处理文本标签")
        tags = self.llm.process_batch(texts)
        return tags

    def _clustering_tag(self, tags: List[str], algo: str, k: int):
        self.logger.info("[Service] 聚类开始处理文本标签")
        embeddings = self.bert_embedder.embed_batch(tags)
        clustering_tags = self.label_clusterer.fit_predict(embeddings, algo, k)
        return clustering_tags

    def batch_tag_analysis(self, texts: List[str], algo: str, k: int):
        self.logger.info(f"[Service] 开始处理文本，待处理文本{len(texts)}条")
        # tags_list = []
        # for i in range(0, len(texts), self._llm_batch_size):
        #     self.logger.info(
        #         "[Service] 处理第{}-{}条文本".format(
        #             i + 1, i + self._llm_batch_size + 1
        #         )
        #     )
        #     llm_texts = texts[i : i + self._llm_batch_size]
        #     llm_tags = self._llm_tag(llm_texts)
        #     tags_list.extend(llm_tags.values())
        tags_list = texts
        self.logger.info("[Service] LLM处理完成")

        if len(tags_list) < self._clustering_batch_size:
            self.logger.warning("[Service] 标签数量不足，不适合进行聚类，直接返回")
            return {"tags": tags_list, "clustering_res": None}

        clustering_res = self._clustering_tag(tags_list, algo, k)
        self.logger.info("[Service] 聚类完成")
        res = {"tags": tags_list, "clustering_res": clustering_res}
        return res

    def ner_analysis(self, text: str):
        self.logger.info(f"[Service] 开始处理文本NER")
        ner_res = self.nlp_processor.analyze(text)
        return ner_res

    def sentiment_analysis(self, text: str):
        pass
