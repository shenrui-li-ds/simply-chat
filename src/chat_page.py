import streamlit as st

import openai
import anthropic
import google.generativeai as genai
from google.api_core.exceptions import InvalidArgument
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

def chat_openai(temperature, top_p, max_tokens, system_prompt, openai_api_key):
    # Initialize client
    client = openai.OpenAI(api_key=openai_api_key)
    messages_for_llm = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    # Stream response
    stream = client.chat.completions.create(
        model=st.session_state["selected_model"],
        messages=messages_for_llm,
        stream=True,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    # Stream response to streamlit
    response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})

def chat_anthropic(temperature, top_p, max_tokens, system_prompt, anthropic_api_key):
    # Anthropic requires first message must use the "user" role
    while st.session_state.messages and st.session_state.messages[0]["role"] != "user":
        st.session_state.messages.pop(0)
    if st.session_state.messages is None or len(st.session_state.messages) == 0:
        st.error(
            (
                "Anthropic encounters an issue with multi-turn conversations, "
                "please consider switching to a different provider."
            ), 
            icon="🚨"
        )
    # Initialize client
    client = anthropic.Anthropic(api_key=anthropic_api_key)
    messages_for_llm = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    # Stream response
    with client.messages.stream(
            model=st.session_state["selected_model"],
            system=system_prompt,
            messages=messages_for_llm,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        ) as stream:
        # Stream response to streamlit
        response = st.write_stream(stream.text_stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

def chat_ollama(temperature, top_p, max_tokens, system_prompt):
    # Initialize client
    client = openai.OpenAI(
        base_url = 'http://localhost:11434/v1',
        api_key='ollama', # required, but unused
    )
    messages_for_llm = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    # Stream response
    stream = client.chat.completions.create(
        model=st.session_state["selected_model"],
        messages=messages_for_llm,
        stream=True,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    # Stream response to streamlit
    response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})

def chat_google(google_api_key):
    # Weird Google API - your context turns can only be odd numbers smh...
    if len(st.session_state.messages)%2 == 0:
        st.session_state.messages.pop(0)
    # Initialize client
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel('gemini-pro')
    messages_for_genai = [
        {
            "role": m["role"] if m["role"] == "user" else "model",
            "parts": [m["content"]]
        } for m in st.session_state.messages
    ]
    # Stream response
    def generate_responses():
        for chunk in model.generate_content(messages_for_genai, stream=True):
            yield chunk.text
    # Stream response to streamlit
    try:
        response = st.write_stream(generate_responses)
        st.session_state.messages.append({"role": "assistant", "content": response})
    except InvalidArgument:
        st.error(
            (
                "Google encounters an issue with multi-turn conversations, "
                "please consider switching to a different provider."
            ), 
            icon="🚨"
        )

def chat_mistral(temperature, top_p, max_tokens, system_prompt, mistral_api_key):
    # Initialize client
    client = MistralClient(api_key=mistral_api_key)
    # messages = [
    #     ChatMessage(role="user", content="What is the best French cheese?")
    # ]
    messages_for_llm = [
        ChatMessage(role="system", content=system_prompt)
    ] + [
        ChatMessage(role=m["role"], content=m["content"])
        for m in st.session_state.messages
    ]
    # With streaming
    stream_response = client.chat_stream(
        model=st.session_state["selected_model"], 
        messages=messages_for_llm,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        )
    # Stream response
    def generate_responses():
        for chunk in stream_response:
            yield chunk.choices[0].delta.content
    # Stream response to streamlit
    response = st.write_stream(generate_responses)
    st.session_state.messages.append({"role": "assistant", "content": response})

def chat_page():
    st.title("Chat Interface")
    if "api_provider" in st.session_state and "selected_model" in st.session_state:
        api_provider = st.session_state["api_provider"]
        selected_model = st.session_state["selected_model"]
        st.caption((
            f"Current model: [{api_provider}] - [{selected_model}]. "
            "LLM can make mistakes. Consider checking important information."
        ))
    else:
        st.caption((
            "No API provider or model selected. "
            "LLM can make mistakes. Consider checking important information."
        ))
    # st.sidebar.markdown("# Chat Interface")
    if "system_prompt" not in st.session_state:
        st.session_state['system_prompt'] = (
            "You are a helpful assistant and your job is to answer any questions the user may have. "
            "Please follow the guidelines and be respectful. Don't provide any false information."
        )

    # Chat interface code goes here
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        # st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if user_input := st.chat_input("Say something"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            if "api_provider" in st.session_state and (st.session_state["api_provider"] is not None) and "system_prompt" in st.session_state:
                api_provider = st.session_state["api_provider"]
                system_prompt = st.session_state["system_prompt"]

                if api_provider == "OpenAI":
                    if "selected_model" not in st.session_state:
                        # Set default model
                        st.session_state["selected_model"] = "gpt-3.5-turbo"
                    
                    # Set model parameters
                    temperature = st.session_state["temperature"]
                    top_p = st.session_state["top_p"]
                    max_tokens = st.session_state["max_tokens"]
                    openai_api_key = st.session_state.secrets["your_api_key"]
                    # OpenAI chat completion
                    chat_openai(temperature, top_p, max_tokens, system_prompt, openai_api_key)

                elif api_provider == "Anthropic":
                    if "selected_model" not in st.session_state:
                        # Set default model
                        st.session_state["selected_model"] = "claude-3-haiku-20240307"

                    # Set model parameters
                    temperature = st.session_state["temperature"]
                    top_p = st.session_state["top_p"]
                    max_tokens = st.session_state["max_tokens"]
                    anthropic_api_key = st.session_state.secrets["your_api_key"]
                    # Anthropic chat completion
                    chat_anthropic(temperature, top_p, max_tokens, system_prompt, anthropic_api_key)

                elif api_provider == "Google":
                    if "selected_model" not in st.session_state:
                        # Set default model
                        st.session_state["selected_model"] = "gemini-pro"

                    google_api_key = st.session_state.secrets["your_api_key"]
                    # Google chat completion
                    chat_google(google_api_key)

                elif api_provider == "Ollama":
                    if "selected_model" not in st.session_state:
                        # Set default model
                        st.session_state["selected_model"] = "mistral:7b-instruct"
                    # Set model parameters
                    temperature = st.session_state["temperature"]
                    top_p = st.session_state["top_p"]
                    max_tokens = st.session_state["max_tokens"]
                    # Ollama chat completion
                    chat_ollama(temperature, top_p, max_tokens, system_prompt)
                
                elif api_provider == "Mistral":
                    if "selected_model" not in st.session_state:
                        # Set default model
                        st.session_state["selected_model"] = "open-mistral-7b"
                    
                    # Set model parameters
                    temperature = st.session_state["temperature"]
                    top_p = st.session_state["top_p"]
                    max_tokens = st.session_state["max_tokens"]
                    mistral_api_key = st.session_state.secrets["your_api_key"]
                    # Mistral chat completion
                    chat_mistral(temperature, top_p, max_tokens, system_prompt, mistral_api_key)

                else:
                    st.write("Invalid API provider selected.")
            else:
                st.write("Please select the API provider first.")
