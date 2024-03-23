# def api_settings_page():
#     st.title("API Settings")

#     api_provider = st.selectbox("Select API Provider", ("OpenAI", "Anthropic", "Google", "Ollama"))
#     openai_api_key = st.text_input("OpenAI API Key", type="password")
#     anthropic_api_key = st.text_input("Anthropic API Key", type="password")
#     google_api_key = st.text_input("Google API Key", type="password")

#     # Stateful "Save" button    
#     if 'api_saved' not in st.session_state:
#         st.session_state.api_saved = False
    
#     def save_api_settings():
#         st.session_state.api_saved = True
    
#     st.button('Save', on_click=save_api_settings)

#     if st.session_state.api_saved:
#         st.session_state.api_saved = True
#         st.session_state["api_provider"] = api_provider
#         st.session_state.secrets = {
#             "openai_api_key": openai_api_key,
#             "anthropic_api_key": anthropic_api_key,
#             "google_api_key": google_api_key,
#         }
#         st.success(" API settings saved!", icon='✅')
#     else:
#         st.warning(' Please enter your credentials!', icon='⚠️')