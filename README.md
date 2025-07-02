# Maestro Testing Agent

 <!-- It's highly recommended to create a simple banner for visual appeal -->

**Transform your mobile testing workflow by converting plain English into executable Maestro test flows with the power of local and server-side AI.**

This repository contains the complete toolkit for the **Maestro Testing Agent**, a system designed to accelerate mobile UI test automation. It includes scripts for local execution with Ollama, a full-featured server for GPU-powered inference, and the training data used to create the specialized AI model.

---

## 📖 Full Documentation

For detailed guides on setup, prompt engineering, server deployment, and model training, please visit our official documentation website:

**[https://qoneqt-testing-agent.gitbook.io/qoneqt-testing-agent-docs](https://qoneqt-testing-agent.gitbook.io/qoneqt-testing-agent-docs)**

---

## ✨ Features

*   **Natural Language to YAML:** Write test commands like `"Launch the app and tap the login button"` and get production-ready Maestro YAML in seconds.
*   **Dual Execution Modes:**
    *   **Local Mode:** Run the entire system on your local machine using **Ollama** for complete privacy and offline capability.
    *   **Server Mode:** Deploy the model on a powerful GPU server (e.g., RunPod) for faster inference and team-wide access.
*   **Comprehensive Command Support:** The agent understands a wide range of Maestro commands, from simple taps and inputs to complex flow control like retries, conditionals, and running other flows.
*   **AI-Powered Analysis:** The server component includes an endpoint to analyze test results and generate a human-readable QA report.
*   **Open and Extensible:** All scripts and training data are provided, allowing developers to understand, modify, and retrain the agent for their specific needs.

---

## 🚀 Quick Start: Local Execution with Ollama

Get up and running in minutes on your local machine.

### Prerequisites

1.  **Python 3.8+**
2.  **[Maestro](https://maestro.mobile.dev/getting-started/installing-maestro)**
3.  **[Ollama](https://ollama.com/)** (Make sure the application is running)

### Setup Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/pyandcpp-coder/Maestro_Testing_Agent.git
    cd Maestro_Testing_Agent
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download and Set Up the AI Model:**
    The local agent uses a custom GGUF model. You need to download it and register it with Ollama.
    *   **Download the model file** from Hugging Face: **[yrevash/maestro-agent-gguf](https://huggingface.co/yrevash/maestro-agent-gguf)**
    *   Place the downloaded `.gguf` file in a convenient location (e.g., a new `models` folder inside this project).
    *   **Create a `Modelfile`** in the root of this repository with the following content, making sure to update the `FROM` path to your downloaded file:
        ```modelfile
        # Modelfile for maestro-agent
        FROM ./models/maestro-agent-Q4_K_M.gguf

        PARAMETER temperature 0.1
        SYSTEM """You are an expert Maestro automation assistant...""" # (Use full system prompt)
        ```
    *   **Create the model in Ollama:**
        ```bash
        ollama create maestro-agent -f ./Modelfile
        ```

4.  **Run the Local Agent:**
    ```bash
    python run_local_maestro_agent.py
    ```
    The script will prompt you to enter a command. It will generate the YAML, save it in the `maestro_tests` folder, and ask if you want to execute it.

> 📚 For a more detailed walkthrough, see the **[Local Execution Guide](https://qoneqt-testing-agent.gitbook.io/qoneqt-testing-agent-docs/setup-and-local-execution-guide)** in our documentation.

---

## ☁️ Server-Side Deployment

For team environments or to leverage more powerful hardware, you can deploy the agent on a GPU server. The `complete_api_server.py` script automates this entire process.

### Key Components

*   **`complete_api_server.py`**: An all-in-one setup script for a GPU-enabled Linux environment (like RunPod). It installs dependencies, downloads the model adapter, and generates the `api_server.py` file.
*   **`run_remote_agent.py`**: The client script you run on your local machine to interact with the deployed server.

### High-Level Deployment Steps

1.  **Launch a GPU Pod:** Use a platform like RunPod with a PyTorch template and a GPU (e.g., RTX A4000). **Remember to expose TCP port 8000.**
2.  **Run the Setup Script:** Copy `complete_api_server.py` to the pod and execute it (`python3 complete_api_server.py`).
3.  **Start the Server:** The setup script will prompt you to start the server. The first launch will download the base model and may take several minutes.
4.  **Configure and Run the Client:** Update the `POD_URL` in `run_remote_agent.py` on your local machine to point to your new server and run it.

> 📚 For a complete, step-by-step tutorial, please refer to the **[Server-Side Deployment Guide](https://qoneqt-testing-agent.gitbook.io/qoneqt-testing-agent-docs/server-side-setup-and-deployment-guide)**.

---

## 🤖 Prompt Engineering

The quality of the generated YAML is directly related to the quality of your prompt. The AI has been trained on the provided `dataset.json`.

**Golden Rule:** Describe the user's actions clearly and sequentially.

*   **Good:** `Launch the app, scroll until 'Submit' is visible, and then tap on it.`
*   **Bad:** `Test the submit button.`

> 🗣️ To become an expert, study the **[Complete Prompt Engineering Guide](https://qoneqt-testing-agent.gitbook.io/qoneqt-testing-agent-docs/maestro-agent-the-complete-prompt-engineering-guide)** for dozens of examples and best practices.

---

## 🛠️ For Developers: Model Training

This repository is fully open for extension. If you wish to retrain or further fine-tune the model with your own data, you can follow our developer guide.

The process involves using a cloud GPU environment and leveraging libraries like Hugging Face `transformers`, `peft`, and `trl` to perform Parameter-Efficient Fine-Tuning (PEFT) with a LoRA adapter on the `mistralai/Mistral-7B-Instruct-v0.2` base model.

> 🧠 Dive into the technical details in our **[Developer Guide: Training the Adapter](https://qoneqt-testing-agent.gitbook.io/qoneqt-testing-agent-docs/developer-guide-training-the-maestro-agent-adapter)**.


