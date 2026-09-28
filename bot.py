import streamlit as st
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Konfigurasi Halaman Utama
st.set_page_config(page_title="Enterprise Multi-Persona AI Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 Enterprise Multi-Persona AI Chatbot System")
st.markdown("Chatbot interaktif berbasis **LangChain** & **Groq Cloud (Llama 3.3)** dengan fitur Memory, Dynamic Persona, dan Export History.")

# --- SIDEBAR CONTROL: KONFIGURASI PARAMETER KREATIF ---
st.sidebar.header("🎭 Konfigurasi AI Agent")

# Pilihan Persona / Kepribadian Bot
persona_pilihan = st.sidebar.selectbox(
    "Pilih Kepribadian Bot:",
    ["Konsultan Bisnis (Formal)", "Supervisor Tegas (Galak)", "Asisten Santai (Kasual)"]
)

# Pemetaan System Prompt berdasarkan Persona yang dipilih
if persona_pilihan == "Konsultan Bisnis (Formal)":
    system_prompt = "Anda adalah seorang Konsultan Bisnis profesional, analitis, dan ahli strategi perusahaan. Jawablah setiap pertanyaan pengguna menggunakan Bahasa Indonesia yang formal, terstruktur, sopan, dan berfokus pada solusi bisnis makro."
elif persona_pilihan == "Supervisor Tegas (Galak)":
    system_prompt = "Anda adalah seorang Supervisor operasional yang sangat tegas, disiplin, dan berorientasi pada efisiensi kerja tinggi. Jawab dengan singkat, padat, langsung pada inti masalah, dan gunakan nada bicara yang tegas atau menuntut perbaikan performa jika diperlukan."
else:
    system_prompt = "Anda adalah asisten virtual yang sangat ramah, santai, dan asyik diajak mengobrol. Gunakan Bahasa Indonesia yang kasual, gunakan bahasa sehari-hari atau sedikit bahasa gaul (slang) yang sopan, serta buat suasana obrolan menjadi menyenangkan."

# Tombol untuk Reset Memory Chat
if st.sidebar.button("🧹 Hapus Riwayat Chat (Reset)"):
    st.session_state.langchain_messages = []
    st.rerun()

# --- 1. INISIALISASI SESSION STATE (MEMORY BERBASIS LANGCHAIN) ---
if "langchain_messages" not in st.session_state:
    st.session_state.langchain_messages = []

# Jika memory kosong, berikan pesan sambutan default sesuai persona
if len(st.session_state.langchain_messages) == 0:
    if persona_pilihan == "Konsultan Bisnis (Formal)":
        sambutan = "Halo, selamat datang. Saya adalah AI Business Consultant Anda. Apa yang bisa saya bantu untuk mengoptimalkan strategi perusahaan Anda hari ini?"
    elif persona_pilihan == "Supervisor Tegas (Galak)":
        sambutan = "Lapor! Saya siap mengaudit kinerja. Jangan buang waktu, sebutkan kendala operasional Anda sekarang!"
    else:
        sambutan = "Halo! Kenalin, aku asisten santai kamu hari ini. Mau ngobrolin atau nanya-nanya soal apa nih kita?"
        
    st.session_state.langchain_messages.append({"role": "assistant", "content": sambutan})

# Tampilkan riwayat obrolan dari Memory ke UI Streamlit
for message in st.session_state.langchain_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 2. LOGIKA UTAMA CHATBOT (INTEGRASI LANGCHAIN & GROQ) ---
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
    
    # Menerima input dari pengguna
    if user_input := st.chat_input("Ketik pesan Anda di sini..."):
        # Simpan input user ke memory lokal UI
        st.session_state.langchain_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        # Panggil LangChain untuk memproses respon
        with st.chat_message("assistant"):
            with st.spinner("AI sedang merumuskan jawaban..."):
                try:
                    # Inisialisasi Model ChatGroq dari LangChain
                    llm = ChatGroq(
                        temperature=0.7,
                        model_name="llama-3.3-70b-specdec",
                        groq_api_key=api_key
                    )
                    
                    # Menyusun payload pesan menggunakan skema ChatMessage LangChain
                    payload_langchain = [SystemMessage(content=system_prompt)]
                    
                    for msg in st.session_state.langchain_messages:
                        if msg["role"] == "user":
                            payload_langchain.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            payload_langchain.append(AIMessage(content=msg["content"]))
                    
                    # Eksekusi pemanggilan LLM lewat rantai pesan LangChain
                    response = llm.invoke(payload_langchain)
                    ai_response = response.content
                    
                    # Tampilkan hasil di UI
                    st.markdown(ai_response)
                    
                    # Simpan respon AI ke memory
                    st.session_state.langchain_messages.append({"role": "assistant", "content": ai_response})
                    
                except Exception as e:
                    st.error(f"Gagal mendapatkan respon dari LangChain-Groq. Error: {e}")
else:
    st.warning("⚠️ Kunci API Groq (`GROQ_API_KEY`) belum dikonfigurasi di Streamlit Secrets. Fitur Chatbot dinonaktifkan.")

# --- 3. FITUR TAMBAHAN: EXPORT & UNDUH RIWAYAT CHAT ---
st.markdown("---")
if len(st.session_state.langchain_messages) > 1:
    st.subheader("📥 Manajemen Data Riwayat Chat")
    
    chat_transcript = f"TRANSKRIP OBROLAN AI CHATBOT (LANGCHAIN INTEGRATION)\nDibuat pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    chat_transcript += f"Persona Aktif: {persona_pilihan}\n"
    chat_transcript += "="*50 + "\n\n"
    
    for msg in st.session_state.langchain_messages:
        role_label = "USER" if msg["role"] == "user" else "AI ASSISTANT"
        chat_transcript += f"[{role_label}]:\n{msg['content']}\n\n"
        
    st.download_button(
        label="📥 Ekspor & Unduh Riwayat Chat (.TXT)",
        data=chat_transcript,
        file_name=f"Riwayat_Chatbot_Langchain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )
