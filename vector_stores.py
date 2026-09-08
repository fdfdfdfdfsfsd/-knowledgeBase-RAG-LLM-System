from langchain_chroma import Chroma
import config_data as config


class VectorStorService(object):
    """
     :param embedding: 嵌入模型的传入

    """
    def __init__(self,embedding):
        
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.collection_name,      #数据库表名
            embedding_function=self.embedding, #嵌入模型
            persist_directory=config.persist_directory,   #数据库本地存储文件夹
        )

    def get_retriever(self):
        """返回向量存储器，方便放入连。
        带 score_threshold 相关性过滤：低于阈值的检索结果会被过滤，视为“无相关资料”，
        便于 AI 在资料不足时用自身通用知识回答。"""
        search_kwargs = {"k": config.similarity_threshold}
        search_type = "similarity"
        score_threshold = getattr(config, "min_relevance_score", None)
        if score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold
            # 必须指定 similarity_score_threshold 类型，score_threshold 过滤才会生效
            search_type = "similarity_score_threshold"
        return self.vector_store.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )


if __name__ == '__main__':
    from langchain_community.embeddings import DashScopeEmbeddings
    r = VectorStorService(DashScopeEmbeddings(model="text-embedding-v3")).get_retriever()

    res = r.invoke("春天，衣服推荐")
    print(res)