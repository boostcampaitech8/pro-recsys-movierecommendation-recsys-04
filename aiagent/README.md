# Chat Client

Ollama와 Langchain을 사용한 Streamlit 기반 채팅 클라이언트입니다.

## 요구사항

- Python 3.8+
- Ollama가 로컬에 설치되어 있어야 함
- gpt-oss:20b 모델이 Ollama에 설치되어 있어야 함

## 설치

```bash
pip install -r requirements.txt
```

## Ollama 설정

1. Ollama 설치 (이미 설치되어 있지 않은 경우):
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

2. gpt-oss:20b 모델 다운로드:
```bash
ollama pull gpt-oss:20b
```

3. Ollama 서버 실행 확인:
```bash
ollama serve
```

## 실행

```bash
streamlit run chat_client.py
```

브라우저에서 자동으로 열리지 않으면 http://localhost:8501 로 접속하세요.

## 기능

- **채팅 인터페이스**: 일반적인 채팅 UI로 대화 가능
- **새 대화**: 사이드바의 "새 대화" 버튼으로 대화 기록 초기화
- **세션 메모리**: 대화 컨텍스트를 유지하여 이전 대화 내용 기억
- **실시간 응답**: LLM의 응답을 실시간으로 표시

## 구성

- **LLM**: Ollama gpt-oss:20b (http://localhost:11434)
- **Memory**: ConversationBufferMemory (세션 기반)
- **Framework**: Langchain ConversationChain
