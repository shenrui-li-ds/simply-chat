import streamlit as st
import openai
import anthropic
# import google.generativeai as genai


def chat_page():
    st.title("Chat Interface")

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
            if "api_provider" in st.session_state and "system_prompt" in st.session_state:
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
                    response = st.write_stream(stream)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                elif api_provider == "Anthropic":
                    if "selected_model" not in st.session_state:
                        # Set default model
                        st.session_state["selected_model"] = "claude-3-haiku-20240307"

                    # Set model parameters
                    temperature = st.session_state["temperature"]
                    top_p = st.session_state["top_p"]
                    max_tokens = st.session_state["max_tokens"]
                    anthropic_api_key = st.session_state.secrets["your_api_key"]
                    
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

                        response = st.write_stream(stream.text_stream)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                # elif api_provider == "Google":
                #     if "selected_model" not in st.session_state:
                #         # Set default model
                #         st.session_state["selected_model"] = "gemini-pro"

                #     google_api_key = st.session_state.secrets["your_api_key"]
                #     # Initialize client
                #     genai.configure(api_key=google_api_key)
                #     model = genai.GenerativeModel('gemini-pro')
                #     # chat = model.start_chat(history=[])
                #     messages_for_genai = [
                #         {
                #             "role": m["role"] if m["role"]=="user" else "model",
                #             "parts": [m["content"]]
                #         } for m in st.session_state.messages
                #     ]
                #     # Stream response
                #     # response = chat.send_message(messages_for_genai, stream=True)
                #     response = model.generate_content(messages_for_genai, stream=True)
                #     st.session_state.messages.append({"role": "assistant", "content": response.text})

                elif api_provider == "Ollama":
                    if "selected_model" not in st.session_state:
                        # Set default model
                        st.session_state["selected_model"] = "mistral:7b-instruct"
                    # Set model parameters
                    temperature = st.session_state["temperature"]
                    top_p = st.session_state["top_p"]
                    max_tokens = st.session_state["max_tokens"]
                    
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
                    response = st.write_stream(stream)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                else:
                    st.write("Invalid API provider selected.")
            else:
                st.write("Please select the API provider first.")