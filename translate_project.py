import streamlit as st
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate

# ========== 页面配置 ==========
st.set_page_config(
    page_title="AI翻译助手",
    page_icon="🌐",
    layout="centered"
)

# ========== 侧边栏设置 ==========
with st.sidebar:
    st.header("🔑 API 设置")
    user_api_key = st.text_input(
        "请输入您的 DashScope API Key",
        type="password",
        placeholder="sk-xxxxxxxxxxxxxxxx",
        help="Key 仅保存在您的浏览器会话中，不会上传到任何服务器。获取地址：https://dashscope.console.aliyun.com/"
    )

    st.divider()
    st.header("⚙️ 翻译设置")

    model_name = st.selectbox(
        "选择模型",
        ["qwen-turbo", "qwen-plus", "qwen-max"],
        index=0,
        help="推荐日常使用 qwen-turbo，成本最低"
    )

    temperature = st.slider(
    "创意程度",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1
    )

    translate_mode = st.selectbox(
        "翻译模式",
        ["标准翻译", "官方翻译", "商务翻译", "文学翻译"],
        index=0
    )

    st.divider()
    st.markdown("**使用说明：**")
    st.markdown("1. 填入您自己的 DashScope API Key")
    st.markdown("2. 输入中文文本，点击开始翻译")
    st.markdown("3. 查看翻译结果与专业解析")
    st.caption("⚠️ 单次输入上限 1000 字，请勿频繁调用")

# ========== 标题区域 ==========
st.title("🌐 AI翻译助手")
st.markdown("输入中文，帮你翻译成英文，并且做出解析 | 🔒 本地会话安全模式")

# ========== 模型初始化（带 Key 校验）==========
@st.cache_resource
def init_model(api_key, model_name, temperature):
    return ChatTongyi(
        api_key=api_key,
        model=model_name,
        model_kwargs={"temperature": temperature}
    )

# ========== 主界面 ==========
input_text = st.text_area(
    "📝 输入中文",
    placeholder="请输入要翻译的中文文本...",
    height=150,
    max_chars=1000
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    translate_button = st.button("🚀 开始翻译", type="primary", use_container_width=True)

# ========== 翻译逻辑 ==========
if translate_button:
    # 前置校验
    if not user_api_key:
        st.warning("⚠️ 请先在左侧边栏输入您的 DashScope API Key")
        st.stop()

    if not input_text or not input_text.strip():
        st.warning("⚠️ 请先输入要翻译的文本")
        st.stop()

    with st.spinner("🤔 AI 正在思考中..."):
        try:
            tongyi = init_model(user_api_key, model_name, temperature)

            mode_prompts = {
                "标准翻译": "你是一位专业的汉译英翻译专家，擅长将中文翻译成地道、准确的英文。",
                "文学翻译": "你是一位文学翻译大师，擅长保留原文意境和美感，译文富有文采。",
                "商务翻译": "你是一位商务翻译专家，译文专业规范，适合合同、邮件等场景。",
                "官方翻译": "你是一位政务翻译专家，译文严谨正式，符合官方表述习惯。"
            }

            prompt = ChatPromptTemplate.from_messages([
                ('system', f"{mode_prompts[translate_mode]}\n输出纯文本，不要使用任何 Markdown 标记符号。"),
                ("human", "请将以下中文翻译成英文，并给出简明专业解析：\n\n{question}")
            ])

            messages = prompt.format_messages(question=input_text.strip())
            resp = tongyi.invoke(messages)

            st.divider()
            st.success("✅ 翻译完成！")

            with st.container(border=True):
                st.markdown("### 📄 翻译结果")
                st.markdown(resp.content)

            st.info("💡 选中上方文字即可复制，或右键点击复制链接")

        except Exception as e:
            error_msg = str(e)
            if "InvalidApiKey" in error_msg or "Unauthorized" in error_msg:
                st.error("❌ API Key 无效，请检查后重新输入")
            elif "Quota" in error_msg or "Insufficient" in error_msg:
                st.error("❌ API Key 额度不足，请前往 DashScope 控制台充值")
            else:
                st.error(f"❌ 翻译出错：{error_msg}")

# ========== 底部信息 ==========
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    基于通义千问大模型 · LangChain 驱动 · 开源安全版 · API Key 仅存于本地会话
</div>
""", unsafe_allow_html=True)