# Agentic Math Question Generator

This repository contains a Python script that leverages a multi-agent system of Large Language Models (LLMs) to autonomously generate novel, difficult, and verified math problems.

> [!WARNING]
> **This is a Prototype!**
>
> This project is an experimental demonstration and should be considered a prototype. The generation process is complex, and success is not guaranteed. Runs may be unstable, get stuck in loops, or fail to produce a satisfactory question. Please see the [Examples & Logs](#-examples--logs) section to understand the range of possible outcomes.

The system uses an agentic architecture with three distinct LLM roles—**Talker**, **Creator**, and **Solver**—to iteratively brainstorm, refine, harden, and verify a math question until it meets a high standard of difficulty and correctness.

## Core Concept: The Three Agents

The entire process is driven by the interaction between three specialized LLM agents:

1.  🧠 **The Talker LLM (The Architect)**: This is the "brain" of the operation. It acts as the project manager, possessing full autonomy over the workflow. Its responsibilities include:
    *   **Guiding** the Creator LLM with specific, strategic instructions.
    *   **Analyzing** the conversation history and feedback to decide on the next logical step.
    *   **Performing quality control** by identifying flaws, trivialities, or errors in the generated content.
    *   **Managing the verification process**, including requesting Python code and interpreting its execution results.
    *   **Deciding when a question is complete** by triggering a final "Finalization" sequence.

2.  ✍️ **The Creator LLM (The Specialist)**: This is the "hands" of the operation. It's a creative mathematician that executes the Talker's instructions. Its tasks include:
    *   **Generating** initial math problems, solutions, and hardening "blueprints."
    *   **Refining** mathematical formulations into abstract, textbook-style problems.
    *   **Implementing** changes and corrections based on the Talker's feedback.
    *   **Writing** Python verification code using `numpy` and `sympy` to prove a solution's correctness.

3.  🎯 **The Solver LLM (The Adversary)**: This agent acts as the ultimate difficulty benchmark. Its role is simple but crucial:
    *   **Attempting to solve** the finalized question.
    *   If the Solver succeeds, the question is deemed **not hard enough**, and the Talker is instructed to continue the hardening process.
    *   If the Solver fails, the question is considered **successfully difficult**, and the generation process concludes.

## The Agentic Workflow

The script operates in a loop of "turns," managed by the Talker. The goal is to produce a question that is both mathematically sound and too difficult for the Solver LLM.

```mermaid
graph TD
    A[Start: Provide Topic] --> B{Talker: Instruct Creator to generate initial problem};
    B --> C[Creator: Responds with Question/Solution/Ideas];
    C --> D{System: External Checks};
    D -- Code Found --> E[Execute Python Code & Capture Output];
    E --> F[Inject '[SYSTEM FEEDBACK]' into History];
    D -- No Code --> F;
    F --> G{Talker: Analyze History & Feedback};
    G --> H{Talker decides: 'FINALIZE' command?};
    H -- No --> B;
    H -- Yes --> I{Final Integrity Check};
    I -- Fails --> J[Inject '[SYSTEM FEEDBACK]' with failure reason];
    J --> B;
    I -- Passes --> K{Final Difficulty Check with Solver LLM};
    K -- Solver Succeeds --> L[Inject '[SYSTEM FEEDBACK]' with 'Not Hard Enough'];
    L --> B;
    K -- Solver Fails --> M([🏆 Success! Question is Verified & Difficult]);
```

## Key Features

-   **True Agentic Control**: The Talker LLM is not following a rigid script. It uses a detailed persona and analyzes the full context to make intelligent, autonomous decisions at each step.
-   **Multi-Provider Support**: Seamlessly use models from **OpenAI** or **Google (Gemini)** for any of the three agent roles.
-   **Automated Code Execution**: The system automatically extracts and runs Python verification code in a sandboxed environment with a timeout, feeding the results back to the Talker for analysis.
-   **Adversarial Difficulty Testing**: Uses a Solver LLM as a practical, objective measure of a problem's difficulty.
-   **Conversation History Compression**: To handle very long generation processes, the system automatically summarizes the oldest parts of the conversation, retaining key information while staying within context limits.
-   **Detailed Logging**: Saves a complete JSON log of all LLM interactions for each run, which is essential for debugging and analysis.

## 📂 Examples & Logs

To get a feel for what this prototype can do, please explore the included examples:

-   `example_questions/`: This folder contains sample outputs from previous runs. You will find both **good** examples (novel, complex questions) and **bad** examples (flawed, trivial, or incorrect questions).

-   `example_logs/`: This folder contains the full JSON logs from the runs that produced the questions in the `example_questions` folder. To understand *how* a specific question was created (or why a run failed), find the corresponding log file. These transcripts are the best way to see the agentic process in action.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

    **`requirements.txt` contents:**
    ```
    python-dotenv
    numpy
    sympy
    langchain-openai
    langchain-google-genai
    ```

3.  **Set up your environment variables:**
    Create a file named `.env` in the root directory by copying the example file:
    ```bash
    cp .env.example .env
    ```
    Now, edit the `.env` file with your API keys and desired configuration.

## ⚙️ Configuration

Your `.env` file controls the entire behavior of the script.

```dotenv
# .env.example

# --- Provider & Model Selection for each Agent ---
# Options: "openai" or "google"
CREATOR_PROVIDER="openai"
CREATOR_MODEL="gpt-4-turbo" # Or "o3" or any other valid model name

TALKER_PROVIDER="openai"
TALKER_MODEL="gpt-4-turbo"

SOLVER_PROVIDER="openai"
SOLVER_MODEL="gpt-4-turbo"

# --- API Keys ---
# Required for OpenAI models
OPENAI_API_KEY="sk-..."

# Required for Google models
GOOGLE_API_KEY="AIza..."

# --- Role-Specific Overrides (Optional) ---
# You can use different keys or API endpoints for each role.
# Useful for routing to different local models or API providers.
# TALKER_OPENAI_API_KEY="sk-talker-specific-key"
# CREATOR_OPENAI_BASE_URL="http://localhost:11434/v1" # Example for a local model via Ollama

# --- Generation Parameters ---
# The initial topic for the math question
MATH_TOPIC="Polynomials"

# The maximum number of agentic turns before stopping
MAX_TURNS=40

# The number of turns after which the oldest turn's history is compressed into a summary.
# Set to 0 to disable compression.
HISTORY_COMPRESSION_TURNS=10

# Global timeout for the entire script process in seconds.
# Note: This functionality requires a Unix-like OS (Linux, macOS).
GLOBAL_TIMEOUT_SECONDS=3600

# Timeout for the execution of a single block of generated Python code in seconds.
CODE_EXECUTION_TIMEOUT_SECONDS=60
```

## ▶️ Usage

Once your `requirements.txt` is installed and your `.env` file is configured, simply run the script:

```bash
python prototype.py
```

The script will print the agentic conversation to the console in real-time. You will see the Talker's instructions, the Creator's responses, and system feedback from code execution.

## 📄 Output

-   **Console Output**: A live-stream of the agentic process.
-   **Generated Question File**: If the process succeeds, a markdown file will be saved in the `generated_questions/` directory. The file will be named like `agentic_q_topic_YYYYMMDD_HHMMSS.md` and will contain the final question and solution.
-   **LLM Interaction Log**: A detailed `agentic_llm_log_YYYYMMDD_HHMMSS.json` file will be created in the root directory. This log is the most important artifact for debugging a failed run.
