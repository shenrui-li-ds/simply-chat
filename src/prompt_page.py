import streamlit as st

def system_prompt_page():
    st.title("System Prompt Configuration")
    # Prompt page code goes here
    system_prompt = st.text_area("Enter your prompt")
    if st.button("Save Prompt"):
        st.session_state['system_prompt'] = system_prompt
        st.success("System prompt saved!")

def few_shot_prompting():
    (
        "A 'whatpu' is a small, furry animal native to Tanzania. An example of a sentence that uses the word whatpu is: "
        "We were traveling in Africa and we saw these very cute whatpus. "
        "To do a 'farduddle' means to jump up and down really fast. "
        "An example of a sentence that uses the word farduddle is: "
    )
    return

def chain_of_thought_prompting():
    pass