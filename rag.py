from vector_stores import VectorStorService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from file_history_store import get_history
from langchain_core.runnables.history import RunnableWithMessageHistory

def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)

    return prompt

class RagService(object):
    def __init__(self):

        self.vector_service = VectorStorService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

        self.prompt_temople = ChatPromptTemplate.from_messages(
            [
                ("system","你是专业的智能客服助手。下面提供与用户问题可能相关的参考资料：{context}。"
                "回答规则：1) 若参考资料能回答用户问题，请优先以参考资料为依据，简洁、专业地作答；"
                "2) 若参考资料为空、不相关或无法回答用户问题，请基于你自己的通用知识正常回答，"
                "不要局限于参考资料，必要时可提示该回答并非来自当前资料。"
                "同时为你提供用户的对话历史记录，如下："),
                MessagesPlaceholder("history"),
                ("user","请回答用户提问：{input}")

            ]
        )

        self.chat_model = ChatOpenAI(model = config.chat_model_name,
                                    base_url="https://ws-fhe63d55pbuquh68.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")

        self.chain = self.__get_chain()

    def __get_chain(self):
        """获取最终的执行链"""
        retriever = self.vector_service.get_retriever()


        def format_document(docs: list[Document]):
            if not docs:
                return "无相关资料"

            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"
            return formatted_str

        def format_for_retriever(value: dict)->str:
            return value["input"]

        def format_for_prompt_template(value):
            # {input, context, history}
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["content"]
            new_value["history"] = value["input"]["history"]
            return new_value


        chain =(
            {
                "input": RunnablePassthrough(),
                "content": RunnableLambda(format_for_retriever) |retriever | format_document
            } | RunnableLambda(format_for_prompt_template) | self.prompt_temople | print_prompt | self.chat_model | StrOutputParser()
        )
        #增强链，添加历史记忆
        conversation_chain = RunnableWithMessageHistory(       # 增强的链
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain

     

if __name__ == "__main__":
    #session_id配置
    session_config ={
        "configurable":{
            "session_id":"user_001",
        }
    }
    res = RagService().chain.invoke({"input":"我之前问了什么"},session_config)
    print(res)