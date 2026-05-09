import streamlit as st
import requests
import json

st.set_page_config(page_title="Управление недвижимостью", layout="wide")

API_URL = "http://localhost:8000"

# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("⚙️ Настройки сессии")
    thread_id = st.text_input("ID сессии", value="main")

    st.divider()

    st.header("🤖 Последний агент")
    agent_box = st.empty()

    st.divider()

    st.header("🏢 История арендаторов")
    if st.button("Обновить данные"):
        try:
            resp = requests.get(f"{API_URL}/state/{thread_id}")
            data = resp.json()
            tenant_history = data.get("tenant_history", {})
            if tenant_history:
                for tenant, history in tenant_history.items():
                    with st.expander(f"📋 {tenant}"):
                        for entry in history:
                            st.text(entry)
            else:
                st.info("История пуста")
        except Exception as e:
            st.error(f"Ошибка: {e}")

# =========================================================
# Chat
# =========================================================
st.title("💬 Управление недвижимостью")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Отображаем историю
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "agent" in msg:
            st.caption(f"🤖 {msg['agent']}")

# Ввод пользователя
if prompt := st.chat_input("Напишите сообщение..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            with requests.post(
                f"{API_URL}/chat",
                json={"message": prompt, "thread_id": thread_id},
                stream=True,
                timeout=60
            ) as r:
                for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

            # Получаем метаданные после завершения стрима
            state_resp = requests.get(f"{API_URL}/state/{thread_id}")
            if state_resp.ok:
                state_data = state_resp.json()
                last_route = state_data.get("last_route", "unknown")
                agent_box.info(f"🤖 {last_route}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "agent": last_route
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })

        except Exception as e:
            st.error(f"Ошибка соединения с API: {e}")