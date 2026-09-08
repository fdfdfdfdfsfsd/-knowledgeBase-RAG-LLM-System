#配置文件
md5_path = "./md5.text"

#chroma
collection_name = "rag"
persist_directory ="./chroma.db"

# spliter
chunk_size= 1000
chunk_overlap= 100
separators =["\n\n","\n",".","!","?","。","！","？"," ",""]
max_spliter_char_number= 1000  # 文本分割阈值

#
similarity_threshold = 1 #检索返回匹配的文档数量

# 相关性最低阈值（0~1，越接近1越相关）。检索结果低于该分数会被过滤，
# 视为“无相关资料”，此时 AI 会用自身通用知识回答。设为 None 则不过滤。
# 可根据实际命中效果调整：设太低(如0.2)可能塞入无关片段，设太高(如0.6)可能漏掉有效资料。
min_relevance_score = 0.3

embedding_model_name= "text-embedding-v3"
chat_model_name = "deepseek-v4-pro"

#session_config
session_config={
    "configurable":{
                "session_id":"user_001",
            }
}