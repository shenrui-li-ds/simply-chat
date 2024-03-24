# SimplyChat: Streamlit Chatbot UI
SimplyChat is a minimalistic chatbot user interface built with Streamlit, designed to provide an easy-to-use platform for chatting, prompt engineering, and one-click multi-agent research tasks. It supports a range of API providers including OpenAI, Anthropic, and Ollama, catering to both open and closed source environments.

## 🌟 Features
- Chat Interface: Engage in conversations with AI models from various providers.
- Prompt Engineering: Customize prompts to refine interactions with AI models.
- Multi-Agent Research: Utilize multiple AI agents to generate comprehensive research reports.
- Model Selection: Choose from a variety of models provided by OpenAI, Anthropic, Google, and Ollama.
- API Flexibility: Compatible with both open and closed source AI APIs.
- Docker Support: Deploy and run the application using Docker for ease of use and scalability.

## 🌐 Demo App
Check out the [Demo](https://simply-chat.streamlit.app/) here.

## 💾 Installation
1. Clone the repository to your local machine.
```bash
git clone https://github.com/shenrui-li-ds/simply-chat
```
2. Navigate to the cloned repository directory.
```bash
cd <repository-directory>
```
3. Install the required Python packages.
```bash
pip install -r ./requirements/requirements.txt
```
4. For Ollama environment setup, refer to [Ollama's official blog](https://ollama.com/blog) for detailed instructions.

## 🧑‍💻 Usage
To start the SimplyChat application, run the following command in the terminal:
```bash
streamlit run src/app.py
```
This will launch the Streamlit application in your default web browser. From there, you can navigate through the application's features using the sidebar, which includes Chat, Prompt, and Multi-Agents pages.

### 💬 Chat Interface
- Engage with AI models in a chat-like interface.
- Select the API provider and model of your choice.
- Customize parameters such as temperature, top P, and max tokens for the conversation.

### 📝 Prompt Engineering
- Customize system prompts to guide the AI's responses.
- Save prompts for future use.

### 🕵️‍♂️ Multi-Agent Research
- Generate comprehensive reports by leveraging multiple AI agents.
- Supported APIs: OpenAI and Tavily (currently).
- Download reports in Markdown format.

### 🐳 Docker Deployment
To deploy SimplyChat using Docker, use the provided Dockerfile to build and run the application in a containerized environment. This ensures compatibility and ease of deployment across various systems.

## 📋 Notes
Ensure you have the necessary API keys for the AI providers you intend to use.
The application is initially inspired by [dataprofessor/llama2](https://github.com/dataprofessor/llama2).

## 👏 Acknowledgments
Thanks to the developers of Streamlit for providing an amazing platform to build web applications with ease.
Gratitude to OpenAI, Anthropic, and Ollama for their state-of-the-art AI models and APIs.
