import streamlit as st
from langchain import chat_models
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage


def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'model' not in st.session_state:
        st.session_state.model = chat_models.init_chat_model(
            model="ollama:gpt-oss:20b",
            temperature=0.7,
            base_url="http://localhost:11434"
        )
    if 'prompt' not in st.session_state:
        st.session_state.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("placeholder", "{messages}")
        ])


def reset_conversation():
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.model = chat_models.init_chat_model(
        model="ollama:gpt-oss:20b",
        temperature=0.7,
        base_url="http://localhost:11434"
    )
    st.session_state.prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("placeholder", "{messages}")
    ])


def main():
    st.set_page_config(page_title="Chat Client", page_icon="💬", layout="wide")

    # Custom CSS for larger text input
    st.markdown("""
        <style>
        /* Make chat input auto-expand */
        textarea[data-testid="stChatInputTextArea"] {
            min-height: 50px !important;
            max-height: 300px !important;
            font-size: 15px !important;
            resize: vertical !important;
            overflow-y: auto !important;
        }
        /* Adjust input container position */
        div[data-testid="stChatInput"] {
            position: fixed;
            bottom: 2rem;
            left: 0;
            right: 0;
            background-color: var(--background-color);
            padding: 0.5rem 1rem;
            z-index: 999;
        }
        /* Adjust for sidebar - don't overlap */
        section[data-testid="stSidebar"] ~ div div[data-testid="stChatInput"] {
            margin-left: 21rem;
        }
        /* When sidebar is collapsed */
        section[data-testid="stSidebar"][aria-expanded="false"] ~ div div[data-testid="stChatInput"] {
            margin-left: 0;
        }
        /* Add padding to chat messages container to avoid overlap */
        section[data-testid="stAppViewContainer"] {
            padding-bottom: 150px !important;
        }
        /* Add padding to chat messages */
        .stChatMessage {
            margin-bottom: 1rem;
        }
        /* Preserve line breaks in chat messages */
        .stChatMessage p {
            white-space: pre-wrap !important;
        }
        </style>
        <script>
        // Auto-resize textarea on input
        const observer = new MutationObserver(() => {
            const textarea = document.querySelector('textarea[data-testid="stChatInputTextArea"]');
            if (textarea && !textarea.dataset.autoResizeAttached) {
                textarea.dataset.autoResizeAttached = 'true';

                const autoResize = () => {
                    textarea.style.height = '50px';
                    const newHeight = Math.min(Math.max(textarea.scrollHeight, 50), 300);
                    textarea.style.height = newHeight + 'px';
                };

                textarea.addEventListener('input', autoResize);
                textarea.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' && e.shiftKey) {
                        setTimeout(autoResize, 0);
                    }
                });

                autoResize();
            }
        });

        observer.observe(document.body, { childList: true, subtree: true });
        </script>
    """, unsafe_allow_html=True)

    initialize_session_state()

    st.title("💬 Chat Client")

    if st.sidebar.button("새 대화", use_container_width=True):
        reset_conversation()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 설정")
    st.sidebar.markdown(f"**모델:** gpt-oss:20b")
    st.sidebar.markdown(f"**메시지 수:** {len(st.session_state.messages)}")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("메시지를 입력하세요...", key="chat_input"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            chain = st.session_state.prompt | st.session_state.model

            try:
                for chunk in chain.stream({"messages": st.session_state.chat_history}):
                    chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    full_response += chunk_text
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)
            except Exception as e:
                error_msg = f"오류가 발생했습니다: {str(e)}"
                message_placeholder.markdown(error_msg)
                full_response = error_msg

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.chat_history.append(AIMessage(content=full_response))

        st.rerun()


if __name__ == "__main__":
    main()
