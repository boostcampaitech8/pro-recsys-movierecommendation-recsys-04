# C4 Diagram - Chat Client

## Level 1: System Context Diagram

```mermaid
C4Context
    title System Context diagram for Chat Client

    Person(user, "User", "A person who wants to chat with AI")

    System(chatClient, "Chat Client", "Streamlit-based web application for AI chatting")

    System_Ext(ollama, "Ollama Server", "Local LLM server running gpt-oss:20b model")

    Rel(user, chatClient, "Interacts with", "HTTPS")
    Rel(chatClient, ollama, "Sends prompts and receives responses", "HTTP/REST")
```

## Level 2: Container Diagram

```mermaid
C4Container
    title Container diagram for Chat Client

    Person(user, "User", "A person who wants to chat with AI")

    Container_Boundary(c1, "Chat Client Application") {
        Container(webUI, "Streamlit Web UI", "Python/Streamlit", "Provides chat interface and handles user interactions")
        Container(langchain, "Langchain Layer", "Python/Langchain", "Manages conversation chain and message history")
        Container(sessionMemory, "Session Memory", "In-Memory", "Stores conversation history for the current session")
    }

    System_Ext(ollama, "Ollama Server", "Local LLM server (gpt-oss:20b)")

    Rel(user, webUI, "Types messages, clicks buttons", "Browser")
    Rel(webUI, langchain, "Invokes chain with user input", "Python API")
    Rel(langchain, sessionMemory, "Reads/Writes conversation history", "In-Memory")
    Rel(langchain, ollama, "Sends messages, streams responses", "HTTP/REST (port 11434)")
```

## Level 3: Component Diagram

```mermaid
C4Component
    title Component diagram for Chat Client Application

    Person(user, "User")

    Container_Boundary(c1, "Streamlit Web Application") {
        Component(mainUI, "Main UI Controller", "Python/Streamlit", "Renders chat interface and handles user events")
        Component(sessionState, "Session State Manager", "Streamlit Session State", "Manages application state across reruns")
        Component(chatDisplay, "Chat Display Component", "Streamlit Chat UI", "Renders messages in chat format")
        Component(inputComponent, "Chat Input Component", "Streamlit Chat Input", "Multi-line text input with auto-resize")

        Component(chatModel, "Chat Model", "langchain.chat_models", "Initializes and manages Ollama chat model")
        Component(promptTemplate, "Prompt Template", "ChatPromptTemplate", "Formats messages for LLM")
        Component(messageHistory, "Message History", "List[HumanMessage, AIMessage]", "Stores conversation as message objects")
        Component(streamHandler, "Stream Handler", "Python Generator", "Handles streaming responses from LLM")
    }

    System_Ext(ollama, "Ollama Server")

    Rel(user, mainUI, "Interacts with")
    Rel(mainUI, sessionState, "Reads/Updates state")
    Rel(mainUI, chatDisplay, "Renders messages")
    Rel(mainUI, inputComponent, "Handles user input")

    Rel(inputComponent, chatModel, "Sends user message")
    Rel(chatModel, promptTemplate, "Formats with template")
    Rel(promptTemplate, messageHistory, "Includes history")
    Rel(chatModel, ollama, "Invokes via chain.stream()")
    Rel(ollama, streamHandler, "Returns response chunks")
    Rel(streamHandler, chatDisplay, "Updates UI incrementally")
    Rel(messageHistory, sessionState, "Persists in session")
```

## Level 4: Code Diagram (Key Functions)

```mermaid
classDiagram
    class ChatClientApp {
        +main()
    }

    class SessionStateManager {
        -messages: List
        -chat_history: List
        -model: ChatModel
        -prompt: ChatPromptTemplate
        +initialize_session_state()
        +reset_conversation()
    }

    class UIController {
        +render_sidebar()
        +render_chat_messages()
        +handle_chat_input()
    }

    class LangchainIntegration {
        -model: ChatOllama
        -prompt: ChatPromptTemplate
        +init_chat_model()
        +create_chain()
        +stream_response()
    }

    class MessageManager {
        -messages: List~dict~
        -chat_history: List~Message~
        +add_user_message()
        +add_assistant_message()
        +get_history()
    }

    ChatClientApp --> SessionStateManager
    ChatClientApp --> UIController
    UIController --> LangchainIntegration
    UIController --> MessageManager
    LangchainIntegration --> MessageManager
```

## Data Flow

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI
    participant State as Session State
    participant Chain as Langchain Chain
    participant Model as ChatOllama
    participant Ollama as Ollama Server

    User->>UI: Enter message
    UI->>State: Save user message
    UI->>State: Add HumanMessage to history
    UI->>Chain: Invoke with chat_history
    Chain->>Model: Format prompt with template
    Model->>Ollama: POST /api/generate (stream)

    loop Streaming Response
        Ollama-->>Model: Response chunk
        Model-->>Chain: Process chunk
        Chain-->>UI: Update placeholder
        UI-->>User: Display partial response
    end

    Model-->>Chain: Complete response
    Chain-->>State: Save assistant message
    State-->>UI: Trigger rerun
    UI-->>User: Display complete chat
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit | Web UI framework |
| **LLM Framework** | Langchain 1.x | Chain management, message handling |
| **LLM Provider** | Ollama | Local LLM server |
| **Model** | gpt-oss:20b | Language model |
| **Memory** | In-Memory (Python List) | Session-based conversation history |
| **Language** | Python 3.10+ | Application code |

## Key Design Decisions

1. **Streaming Responses**: Uses `chain.stream()` for real-time token display
2. **Session State**: Leverages Streamlit's session_state for persistence across reruns
3. **Dual Message Storage**:
   - `messages` (dict): For UI rendering
   - `chat_history` (Message objects): For Langchain processing
4. **Auto-resize Input**: JavaScript-based textarea auto-expansion
5. **CSS Customization**: Custom styles for better UX (multi-line input, sidebar handling)
