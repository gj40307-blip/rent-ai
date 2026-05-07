import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
def get_secret(name):
    return os.getenv(name) or st.secrets.get(name)
import chromadb
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

# ================= 1. 網頁基本設定 =================
st.set_page_config(
    page_title="房客租屋法律防線",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= 2. 全域 CSS 樣式 =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans TC', sans-serif; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e8e8e8;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── 統計卡片 ── */
.stat-card {
    background: #f8fdf9;
    border: 1px solid #c8ecd8;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.stat-number { font-size: 28px; font-weight: 700; margin-bottom: 2px; }
.stat-label  { font-size: 12px; color: #6b7280; }
.stat-green  { color: #0F6E56; }
.stat-red    { color: #A32D2D; }
.stat-amber  { color: #854F0B; }
.stat-blue   { color: #185FA5; }

/* ── 結果卡片 ── */
.result-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.result-card.danger  { border-left: 4px solid #E24B4A; }
.result-card.warning { border-left: 4px solid #EF9F27; }
.result-card.safe    { border-left: 4px solid #1D9E75; }

/* ── 判定標籤 ── */
.verdict-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 14px; border-radius: 20px;
    font-size: 12px; font-weight: 700;
    margin-bottom: 12px;
}
.verdict-illegal { background: #FCEBEB; color: #A32D2D; }
.verdict-warning { background: #FAEEDA; color: #854F0B; }
.verdict-legal   { background: #E1F5EE; color: #0F6E56; }

/* ── 法律依據 ── */
.law-ref {
    border-left: 3px solid #1D9E75;
    padding: 6px 10px;
    background: #f0faf5;
    border-radius: 0 6px 6px 0;
    font-size: 13px; color: #374151;
    margin: 8px 0;
}

/* ── 總結卡片 ── */
.summary-card {
    background: #f8fdf9;
    border: 1px solid #9FE1CB;
    border-radius: 14px;
    padding: 20px 24px;
    margin-top: 20px;
}
.summary-title {
    font-size: 16px; font-weight: 700;
    color: #0F6E56; margin-bottom: 12px;
}

/* ── 側邊欄 debug 區 ── */
.debug-box {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px; padding: 12px;
    font-size: 12px; color: #374151;
}
.debug-row   { display: flex; justify-content: space-between; margin-bottom: 4px; }
.debug-bar-wrap { height: 4px; background: #e5e7eb; border-radius: 4px; margin: 4px 0 8px; overflow: hidden; }
.debug-bar { height: 100%; background: #1D9E75; border-radius: 4px; }

/* ── 輸入提示 ── */
.input-hint { font-size: 12px; color: #9ca3af; text-align: center; margin-top: 6px; }

/* ── 分隔線 ── */
hr.divider { border: none; border-top: 1px solid #f0f0f0; margin: 8px 0; }

/* ── 歡迎頁 ── */
.welcome-box {
    text-align: center; padding: 56px 24px; color: #9ca3af;
}

/* ── Streamlit 按鈕覆寫 ── */
.stButton > button {
    border-radius: 20px !important;
    border: 1px solid #e5e7eb !important;
    font-size: 13px !important;
    padding: 4px 14px !important;
    background: #fff !important;
    color: #374151 !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #E1F5EE !important;
    color: #0F6E56 !important;
    border-color: #9FE1CB !important;
}
</style>
""", unsafe_allow_html=True)


# ================= 3. 金鑰與模型設定 =================



Settings.llm = OpenAI(
    model="gpt-4o",
    temperature=0.0,
    api_key=get_secret("OPENAI_API_KEY")
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ================= 4. 核心 Prompt 定義 =================
DETAIL_PROMPT = PromptTemplate(
"你是一位極端維護房客權益的法律顧問，專精於台灣《住宅租賃條例》與《民法》。請針對以下條款逐一進行深度分析。\n\n"
    "【⚠️ 法律推論與判斷死指令】：\n"
    "1. 押金上限：住宅租賃押金上限為『2個月』租金。若合約寫 3 個月或以上，必須判定為『顯失公平/違法無效』。\n"
    "2. 社會通念原則：判斷條款合理性時，必須基於『台灣現今的一般社會常理與市場行情』。絕對禁止憑空假設極端、罕見或不合常理的未來情境（例如假設物價無底線下跌）。\n"
    "3. 有利房客推定：若合約約定的收費項目（如水電費），依一般社會常理判斷，明顯低於正常收費標準，應直接判定為『合法且對房客有利』，並提醒防範隱藏費用即可。\n"
    "4. 標籤嚴格規定：你只能從以下四個標籤中擇一：【合法】、【合理但需留意】、【合法且對房客有利】、【顯失公平/違法無效】。\n\n"
    "格式要求：\n\n"
    "### [條款名稱]\n\n"
    "🏠 **判斷結果**：【標籤】\n\n"
    "📝 **小助手白話分析**：(從房客角度揭露風險)\n\n"
    "📖 **法律依據**：(引用具體法條)\n\n"
    "💡 **建議修改方案**：(提供具體談判建議)\n\n"
    "📍 **參考法規來源**：(簡述法規名稱)\n\n"
    "待審核條款：{query_str}\n"
    "法規背景資料：{context_str}\n"
)

SUMMARY_PROMPT = PromptTemplate(
    "請根據上方的『詳細審核結果』，為房客製作一份『一分鐘風險報告』。\n"
    "你必須以房客的利益為最高準則，直接告訴他這份合約能不能簽。\n\n"
    "輸出格式要求：\n"
    "--- \n"
    "## 📊 房客合約總結摘要\n"
    "* **總體安全性評估**：(🔴危險 / 🟡偏低 / 🟢極高)\n"
    "* **合約亮點**：(對房客非常有利的條款)\n"
    "* **高風險陷阱**：(違法或壓榨房客的最嚴重3點)\n"
    "* **簽署建議**：(具體行動方針)\n\n"
    "審核內容：{query_str}\n"
)


# ================= 5. 資料庫連線 =================
@st.cache_resource
def get_index():
    try:
        import chromadb

        client = chromadb.CloudClient(
            api_key=get_secret("CHROMA_API_KEY"),
            tenant=get_secret("CHROMA_TENANT"),
            database=get_secret("CHROMA_DATABASE")
        )
        client.get_user_identity()
    except Exception as e:
        st.error(f"Chroma Cloud 連線失敗：{e}")
        st.stop()

    chroma_collection = client.get_or_create_collection(name="LlamaIndex_Contracts_v1")
    st.write("=== Chroma count ===")
    st.write(chroma_collection.count())
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


# ================= 6. Session State 初始化 =================
if "history" not in st.session_state:
    st.session_state.history = []   # [{input, detail, summary, sources}]
if "stats" not in st.session_state:
    st.session_state.stats = {"total": 0, "legal": 0, "illegal": 0, "warning": 0}


# ================= 7. 側邊欄 =================
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="padding:16px 8px 12px; border-bottom:1px solid #e8e8e8; margin-bottom:8px;">
      <div style="display:flex;align-items:center;gap:10px;">
        <div style="width:36px;height:36px;background:#FCEBEB;border-radius:8px;
                    display:flex;align-items:center;justify-content:center;font-size:20px;">🛡️</div>
        <div>
          <div style="font-weight:700;font-size:15px;color:#111827;">房客租屋法律防線</div>
          <div style="font-size:11px;color:#6b7280;">專為台灣租屋族設計</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 連線狀態
    st.markdown("""
    <div style="display:flex;align-items:center;gap:6px;padding:6px 8px;background:#f0faf5;
                border-radius:8px;margin-bottom:12px;">
      <div style="width:8px;height:8px;border-radius:50%;background:#1D9E75;"></div>
      <span style="font-size:12px;color:#0F6E56;font-weight:500;">ChromaDB 已連線</span>
    </div>
    """, unsafe_allow_html=True)

    # Debug 診斷
    st.markdown("**🛠️ 後台診斷數據**")
    show_sources = st.checkbox("顯示法規原文來源", value=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # 清除 / 重啟
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🧹 清除對話", use_container_width=True):
            st.session_state.history = []
            st.session_state.stats = {"total": 0, "legal": 0, "illegal": 0, "warning": 0}
            st.rerun()
    

    st.markdown("""
    <div style="margin-top:16px;font-size:11px;color:#9ca3af;text-align:center;line-height:1.6;">
      基於《民法》《租賃住宅市場發展<br>及管理條例》審核<br>
      <span style="color:#1D9E75;">● LlamaIndex · ChromaDB · GPT-4o</span>
    </div>
    """, unsafe_allow_html=True)


# ================= 8. 主要內容區 =================
# 頁面標題列
st.markdown("""
<div style="background:#fff;border-bottom:1px solid #e8e8e8;padding:14px 24px;
            display:flex;align-items:center;justify-content:space-between;">
  <div>
    <h2 style="margin:0;font-size:18px;font-weight:700;color:#111827;">🛡️ 合約深度審核</h2>
    <p style="margin:0;font-size:12px;color:#6b7280;">
      自動過濾違法條款・保護房客權益・提供修改建議
    </p>
  </div>
  <div style="display:flex;gap:8px;align-items:center;">
    <span style="background:#E1F5EE;color:#0F6E56;padding:4px 10px;border-radius:20px;font-size:12px;font-weight:600;">
      ● 服務運行中
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# 統計卡片
stats = st.session_state.stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-number stat-blue">{stats['total']}</div>
      <div class="stat-label">已審核次數</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-number stat-green">{stats['legal']}</div>
      <div class="stat-label">✅ 無明顯問題</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-number stat-red">{stats['illegal']}</div>
      <div class="stat-label">🔴 發現違法條款</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-number stat-amber">{stats['warning']}</div>
      <div class="stat-label">🟡 需留意條款</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── 歷史審核結果顯示 ──
if st.session_state.history:
    for record in st.session_state.history:
        # 使用者輸入
        st.markdown(f"""
        <div style="background:#f3f4f6;border-radius:10px;padding:12px 16px;
                    margin-bottom:12px;font-size:13px;color:#374151;">
          <span style="font-size:11px;color:#9ca3af;font-weight:600;">📋 送出審核的條款</span><br>
          {record['input'][:200]}{'...' if len(record['input']) > 200 else ''}
        </div>
        """, unsafe_allow_html=True)

        # 詳細分析結果
        st.markdown(record["detail"])

        # 法規原文來源
        if show_sources and record.get("sources"):
            with st.expander("📚 點擊檢視系統引用之法規原文資料庫"):
                for i, src in enumerate(record["sources"]):
                    st.info(f"**來源項目 {i+1}**")
                    st.write(src)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="welcome-box">
      <div style="font-size:48px;margin-bottom:16px;">🛡️</div>
      <div style="font-size:17px;font-weight:700;color:#374151;margin-bottom:8px;">
        歡迎使用房客租屋法律防線
      </div>
      <div style="font-size:13px;line-height:1.8;max-width:400px;margin:0 auto;">
        將合約條款貼入下方輸入框，系統將自動比對法規資料庫<br>
        找出 <strong>違法條款</strong>、<strong>不公平約定</strong>，並提供修改建議
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── 輸入區 ──
st.markdown("**📝 貼入契約內容**")

user_input = st.text_area(
    label="",
    height=200,
    placeholder="例如：\n第一條：押金收取三個月租金...\n第二條：電費固定每度7.5元...\n第三條：房東可隨時進屋檢查...",
    label_visibility="collapsed",
    key="contract_input"
)

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    submit = st.button("⚖️ 開始審核", type="primary", use_container_width=True)

st.markdown('<div class="input-hint">⌨️ 支援整份合約貼入，系統將逐條分析並給出風險報告</div>',
            unsafe_allow_html=True)


# ================= 9. 執行審核 =================
if submit and user_input.strip():
    with st.spinner("⚖️ 正在檢索法規庫並產出專業分析..."):
        try:
            index = get_index()

            # 第一階段：詳細逐條分析
            detail_engine = index.as_query_engine(
                text_qa_template=DETAIL_PROMPT,
                similarity_top_k=3
            )
            detail_response = detail_engine.query(user_input)

            st.write("=== detail_response ===")
            st.write(str(detail_response))

            st.write("=== source_nodes 數量 ===")
            st.write(len(detail_response.source_nodes))

            st.write("=== source_nodes ===")
            st.write(detail_response.source_nodes)

            st.stop()
            
            detail_res = str(detail_response)

            # 第二階段：總結報告
            summary_engine = index.as_query_engine(text_qa_template=SUMMARY_PROMPT)
            summary_res = str(summary_engine.query(detail_res))

            # 顏色渲染
            detail_rendered = detail_res.replace(
                "【顯失公平/違法無效】", "🔴 **:red[【顯失公平/違法無效】]**"
            ).replace(
                "【合理但需留意】", "🟡 **:orange[【合理但需留意】]**"
            ).replace(
                "【合法且對房客有利】", "🟢 **:green[【合法且對房客有利】]**"
            ).replace(
                "【合法】", "🟢 **:green[【合法】]**"
            )

            # 更新統計
            stats["total"] += 1
            if "顯失公平" in detail_res or "違法無效" in detail_res:
                stats["illegal"] += 1
            elif "合理但需留意" in detail_res:
                stats["warning"] += 1
            else:
                stats["legal"] += 1

            # 收集來源
            sources = [node.node.get_content() for node in detail_response.source_nodes]

            # 存入歷史
            st.session_state.history.append({
                "input": user_input,
                "detail": detail_rendered,
                "summary": summary_res,
                "sources": sources
            })

        except Exception as e:
            st.error(f"⚠️ 分析失敗，請檢查金鑰或網路連線：{e}")

    #st.rerun()