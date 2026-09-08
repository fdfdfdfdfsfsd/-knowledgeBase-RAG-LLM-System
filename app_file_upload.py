"""
基于Streamlit完成WEB网页上传服务

Streamlit ： 当WEB页面元素变化，则代码重新执行一遍，无法维护状态
"""
import streamlit as st
from knowlege_base import KnowledgeBaseService
import time

#添加标题
st.title("知识库更新服务")

uploader_flie = st.file_uploader(
    label = "请上传TxT文件",
    type = ['txt'],
    accept_multiple_files= False,#False表示仅接受一个文件的上传
)

# #如果不使用 session_state(字典），
# # 每次刷新都会新建 KnowledgeBaseService，导致数据丢失
service = KnowledgeBaseService()

if "service" not in st.session_state:
     st.session_state["service"] = KnowledgeBaseService()

if uploader_flie is not None:
     # 提取文件信息
    file_name = uploader_flie.name
    file_size = uploader_flie.size / 1024
    file_type = uploader_flie.type

    #在 Streamlit 应用中显示一个二级标题，其中动态包含文件名
    st.subheader(f"文件名：{file_name}")
    st.write(f"格式：{file_type} | 大小：{file_size:.2f}KB")

    #获取文件内容getvalue()-->bytes
    #通过decode-->(utf-8)
    text = uploader_flie.getvalue().decode("utf-8")

    with st.spinner("载入知识库中。。。"):
        time.sleep(1)

        result= st.session_state["service"].upload_by_str(text,file_name)

        st.write(result)

