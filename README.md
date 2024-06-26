# SimplyChat: Streamlit Chatbot UI
SimplyChat is a minimalistic chatbot user interface built with Streamlit, designed to provide a comprehensive platform for real-time chatting with AI models, engineering prompts, conducting multi-agent research, and leveraging retrieval-augmented generation for in-depth query responses. It offers a seamless integration with various AI providers such as OpenAI, Anthropic, Google, and Mistral, making it a versatile tool for both open and closed source environments.

## 🌟 Features
- **Chat Interface**: Engage in conversations with AI models from various providers.
- **Prompt Engineering**: Customize prompts to refine interactions with AI models.
- **Retrieval-Augmented Generation (RAG)**: Enhance query responses with contextually relevant information extracted from uploaded documents. The RAG feature supports various file types and leverages advanced models like Mistral for content retrieval and generation.
- **Multi-Agent Research**: Generate detailed research reports by aggregating insights from multiple AI agents. This feature supports OpenAI and Tavily API currently.
- **API Flexibility**: Compatible with both open and closed source LLM APIs and a variety of models provided by OpenAI, Anthropic, Google, Mistral, and Ollama.

## 🌐 Demo App

Experience SimplyChat in action [here](https://simply-chat.streamlit.app/).

## 💾 Installation

To get started with SimplyChat, follow these steps:
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
    pip install -r requirements.txt
    ```
4. For Ollama environment setup, refer to [Ollama's official blog](https://ollama.com/blog) for detailed instructions.

## 🧑‍💻 Usage
To start the SimplyChat application, run the following command in the terminal:
```bash
streamlit run src/app.py
```
This will launch the Streamlit application in your default web browser. From there, you can navigate through the application's features using the sidebar, which includes Chat, Prompt, RAG and Multi-Agents pages.

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
Quickly set up SimplyChat with Docker:

1. **Preparation**:
   - Ensure Docker is installed from [Docker's site](https://www.docker.com/products/docker-desktop/).
   - Navigate to your project's root directory where the Dockerfile is.
     ```bash
     cd <repository-directory>
     ```

2. **Build the Image**:
   - Create a Docker image named `simplychat-image`.
     ```bash
     docker build -t simplychat-image .
     ```

3. **Run the Container**:
   - Launch a container named `simplychat-container` using the built image, mapping port `8501` to access the app.
     ```bash
     docker run --name simplychat-container -p 8501:8501 -d simplychat-image
     ```

4. **Access the App**:
   - Visit `http://localhost:8501` in your browser.

5. **Cleanup** (Optional):
   - Stop and remove the container when done.
     ```bash
     docker stop simplychat-container
     docker rm simplychat-container
     ```

## 📋 Notes
- Ensure you have the necessary API keys for the AI providers you intend to use.
- For detailed instructions on setting up specific environments or APIs, refer to the official documentation of the respective providers.
- The application is initially inspired by [dataprofessor/llama2](https://github.com/dataprofessor/llama2).

## 👏 Acknowledgments
Thanks to the developers of Streamlit for providing an amazing platform to build web applications with ease.
Gratitude to OpenAI, Anthropic, Google, Mistral and Ollama for their state-of-the-art AI models and APIs.
Additionally, I want to acknowledge the invaluable contributions of open-source resources like [GPT-Researcher](https://github.com/assafelovic/gpt-researcher), among others, that make this project possible.
