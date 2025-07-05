import os
import re
import time
import traceback
import io
import contextlib
from dotenv import load_dotenv
from typing import Tuple, Optional, Dict, Any, List
import json
import datetime
import logging
import signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Required Math Libraries ---
try:
    import numpy
    import sympy
    import math
    import random
    import decimal
    import fractions
    MATH_LIBS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import required math library: {e}")
    logging.warning("Please install numpy and sympy: pip install numpy sympy")
    MATH_LIBS_AVAILABLE = False

# --- Langchain Setup ---
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    HAS_OPENAI = True
except ImportError:
    logging.warning("langchain_openai not installed. OpenAI models unavailable.")
    HAS_OPENAI = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_GOOGLE = True
except ImportError:
    logging.warning("langchain_google_genai not installed. Google models unavailable.")
    HAS_GOOGLE = False

# --- Configuration Loading ---
load_dotenv()

# --- Global Log List ---
llm_outputs_log: List[Dict[str, Any]] = []

# --- Helper Functions ---

def initialize_llm(
    provider: str,
    model_name: str,
    role: str,
    temperature: Optional[float] = None,
    **kwargs):
    """
    Initializes an LLM instance based on provider and model name for a specific role.
    """
    provider = provider.lower()
    role_upper = role.upper()

    logging.info(f"Initializing {role_upper} LLM:")
    logging.info(f"  Provider: {provider}")
    logging.info(f"  Model: {model_name}")

    if provider == "openai":
        if not HAS_OPENAI:
            raise ImportError(f"OpenAI provider selected for {role_upper}, but langchain_openai is not installed.")

        base_url_env_var = f"{role_upper}_OPENAI_BASE_URL"
        api_key_env_var = f"{role_upper}_OPENAI_API_KEY"

        base_url = kwargs.pop('base_url', os.getenv(base_url_env_var, None))
        api_key = kwargs.pop('api_key', os.getenv(api_key_env_var, os.getenv("OPENAI_API_KEY", None)))

        if not api_key and not base_url:
            logging.warning(f"No specific API key found for {role_upper} ({api_key_env_var} or OPENAI_API_KEY) and no base URL override. OpenAI call might fail.")
        elif not api_key and base_url:
            logging.warning(f"No specific API key found for {role_upper} ({api_key_env_var} or OPENAI_API_KEY), but using base URL '{base_url}'. Assuming key is optional/handled by endpoint.")

        openai_kwargs = {
            "model": model_name,
            "openai_api_key": api_key,
            "max_tokens": 64000,
            "timeout": 1200,
            **kwargs
        }

        if base_url:
            logging.info(f"  Using OpenAI Base URL: {base_url}")
            openai_kwargs["base_url"] = base_url

        if temperature is not None:
            logging.info(f"  Setting temperature: {temperature}")
            #openai_kwargs["temperature"] = temperature
        else:
            logging.info("  Using default temperature.")

        return ChatOpenAI(**openai_kwargs)

    elif provider == "google":
        if not HAS_GOOGLE:
            raise ImportError(f"Google provider selected for {role_upper}, but langchain_google_genai is not installed.")

        api_key_env_var = "GOOGLE_API_KEY"
        api_key = kwargs.pop('google_api_key', os.getenv(api_key_env_var))
        if not api_key:
            raise ValueError(f"{api_key_env_var} not found in environment variables or kwargs for {role_upper}.")

        safety_settings = kwargs.pop('safety_settings', None)

        google_kwargs = {
            "model": model_name,
            "google_api_key": api_key,
            "safety_settings": safety_settings,
            "max_output_tokens": 8192,
            **kwargs
        }

        if temperature is not None:
            logging.info(f"  Setting temperature: {temperature}")
            google_kwargs["temperature"] = temperature
        else:
            logging.info("  Using default temperature.")

        if safety_settings:
            logging.info("  Using custom safety settings.")

        return ChatGoogleGenerativeAI(**google_kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider for {role_upper}: {provider}. Choose 'openai' or 'google'.")


def extract_code_blocks(text: str, block_type: str) -> Optional[str]:
    """Extracts content from the first code block of a specific type."""
    pattern = rf"```{block_type}\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def run_verification_code(code: str, timeout_seconds: int = 30) -> Tuple[bool, str]:
    """
    Executes the provided Python code string, capturing its output with a dedicated timeout.
    This timeout is specific to the code execution and will not terminate the entire script.
    """
    if not code:
        return False, "No code provided to execute."
    if not MATH_LIBS_AVAILABLE:
        return False, "Execution aborted: Required math libraries (numpy, sympy) not available."

    # This local timeout mechanism only works on Unix-like systems
    is_code_timeout_supported = hasattr(signal, 'SIGALRM')

    class CodeTimeoutException(Exception):
        pass

    def code_timeout_handler(signum, frame):
        raise CodeTimeoutException

    if is_code_timeout_supported:
        # Store the old handler to restore it later, so we don't interfere
        # with the main script's global timeout handler.
        old_handler = signal.signal(signal.SIGALRM, code_timeout_handler)
        signal.alarm(timeout_seconds)

    logging.info(f"\n--- Attempting to Run Verification Code (Timeout: {timeout_seconds}s) ---")
    output_buffer = io.StringIO()
    try:
        # A restricted but powerful execution environment for the math code
        exec_globals = {
            'math': math, 'numpy': numpy, 'np': numpy, 'sympy': sympy, 'sp': sympy,
            'random': random, 'Decimal': decimal.Decimal, 'Fraction': fractions.Fraction,
            'print': print,
            "ValueError": ValueError, "TypeError": TypeError, "ZeroDivisionError": ZeroDivisionError,
            "Exception": Exception, "AssertionError": AssertionError,
        }
        with contextlib.redirect_stdout(output_buffer):
            exec(code, exec_globals)
        output = output_buffer.getvalue()
        logging.info("--- Code Execution Successful ---")
        logging.info(f"Output:\n{output}")
        return True, output
    except CodeTimeoutException:
        error_msg = f"Code execution timed out after {timeout_seconds} seconds."
        logging.error(f"--- Code Execution Failed: {error_msg} ---")
        return False, error_msg
    except Exception:
        error_msg = f"Error during code execution: {traceback.format_exc()}"
        logging.error("--- Code Execution Failed ---")
        logging.error(error_msg)
        return False, error_msg
    finally:
        # Crucially, cancel the local alarm and restore the original handler
        if is_code_timeout_supported:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

# --- Core Logic ---

class AgenticMathQuestionGenerator:
    """
    Manages the agentic process of generating difficult math questions.
    The Talker LLM directs the Creator LLM, and the Solver LLM verifies difficulty.
    """
    def __init__(self, creator_llm, talker_llm, solver_llm, history_compression_turns: int = 10, code_execution_timeout: int = 30):
        self.creator_llm = creator_llm
        self.talker_llm = talker_llm
        self.solver_llm = solver_llm
        self.history_compression_turns = history_compression_turns
        self.code_execution_timeout = code_execution_timeout

        # This persona is the "brain" of the agent. It defines its mission and capabilities.
        self.agentic_talker_persona = """
You are an expert mathematician and AI assistant (Agentic Talker LLM) with full autonomy over the question generation process.
Your mission is to guide a Creator LLM to generate a novel, very difficult math problem that is NOT a proof and has a verifiable numerical answer. You must intelligently decide what actions to take at each step to ensure the final question is correct, verified, and appropriately difficult.
The generated question MUST: not be obvious (Not give hints), only ask for one thing and have a numerical or symbolic answer (Can be as complex as pleased.).
When trying blueprints, keep building on top of blueprints iteratively, don't turn back to try other ones seperately! Generate new blueprints and add them. Only revert if the question is clearly not working.
Send each task one by one, instead of wanting everything at once. Ask the Creator LLM bit by bit.
Avoid using overly large numbers (More than 10 digits for example) to make a question hard.
The Creator LLM cannot run the code it generates, it will only run after the Creator LLM gives its response to you.

**CORE RESPONSIBILITIES:**
1.  **Guide and Instruct:** Direct the Creator LLM through the process of question generation, refinement, and hardening.
2.  **Analyze and Decide:** Analyze the conversation history, including the Creator's responses and external verification results (from the solver or code execution), to decide the next logical step.
3.  **Quality Control:** Detect and correct errors, logical flaws, or trivialities in the Creator's output. Ensure questions are non-obvious and not solvable by simple brute-force.
4.  **Verification:** Assess question difficulty by reviewing solver attempts. Request Python verification code and **analyze the results provided in [SYSTEM FEEDBACK]** to confirm the solution's correctness. This includes handling code that fails or times out.
5.  **Workflow Management:** Manage the entire workflow from initial idea to final, polished question.

**POSSIBLE ACTIONS YOU CAN INSTRUCT THE CREATOR TO DO:**
*   `GENERATE_INITIAL_PROBLEM`: Ask for an initial mathematical problem formulation on a given topic.
*   `REFINE_AS_TEXTBOOK_PROBLEM`: Ask the Creator to turn a mathematical formulation into an engaging, abstract textbook-style problem, hiding the core insight.
*   `REQUEST_HARDENING_IDEAS`: If a question seems too easy (e.g., the Solver solved it), ask the Creator for blueprints/ideas to make it harder.
*   `CREATE_HARDER_VERSION`: Instruct the Creator to generate a new, harder question by combining the previous version with a selected hardening idea.
*   `GENERATE_VERIFICATION_CODE`: Request a Python script to numerically verify the solution. The system will automatically run this code and provide you with the output. The code must print the output, not return it.
*   `CORRECT_FLAW`: If you spot a mathematical error or a flaw in the question/solution (or in the verification code's output, including timeout errors), provide a detailed correction and ask the Creator to regenerate it.
*   `FINALIZE`: If you are satisfied with the question's difficulty, correctness, and novelty, you can state that the process is complete. The system will then perform one final check with the Solver. If the Solver fails, the process ends successfully. If it succeeds, you will be prompted to make the question even harder.

**HOW TO OPERATE:**
1.  Review the entire conversation history. This may include a `[PREVIOUS HISTORY SUMMARY]`.
2.  **Pay close attention to `[SYSTEM FEEDBACK]` messages which provide results from the solver or code execution.** This is new, critical information for you.
3.  Based on the current state, decide on the best next action.
4.  Formulate a clear, specific, and actionable prompt for the Creator LLM. Tell it exactly what you want it to do next.
"""

        self.creator_persona = """
You are a highly creative mathematician and AI assistant (Creator LLM).
You specialize in generating complex mathematical problems based on instructions from the Talker LLM.
You will work iteratively with the Talker LLM to refine problems, generate solutions, and create verification code.
Avoid using overly large numbers (More than 10 digits for example) to make a question hard.

If wanted, the generated question MUST: not be obvious (Not give hints), only ask for one thing and have a numerical or symbolic answer (Can be as complex as pleased.).
- Focus on creating non-proof problems with clear numerical answers.
- Follow the Talker's instructions carefully.
- Format your output clearly, using ```question```, ```solution```, and ```python``` tags as requested.
- Ensure your mathematics are rigorous and correct.
- When creating verification code, use libraries like `numpy` (as np) and `sympy` (as sp) to test the solution robustly. Make the code self-contained and print the final numerical result.
"""

    def _get_model_info(self, llm_instance) -> Tuple[str, str]:
        # Helper to extract provider and model name for logging
        if isinstance(llm_instance, ChatOpenAI):
            return "openai", getattr(llm_instance, 'model', 'unknown_openai')
        if isinstance(llm_instance, ChatGoogleGenerativeAI):
            return "google", getattr(llm_instance, 'model', 'unknown_google')
        return "unknown", "unknown"

    def _llm_chat(self, llm: Any, role: str, system_prompt: str, history: List[Dict[str, str]], user_message: str, max_retries: int = 2) -> str:
        """
        Conducts a chat turn, correctly using history, and includes logging and retries.
        """
        messages_for_llm = [SystemMessage(content=system_prompt)]
        for msg in history:
            if msg["role"] == "user":
                messages_for_llm.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages_for_llm.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system": # System feedback is also part of the context
                 messages_for_llm.append(HumanMessage(content=msg["content"]))
        messages_for_llm.append(HumanMessage(content=user_message))

        content = ""
        provider, model_name = self._get_model_info(llm)
        final_error = None
        attempt = 0 # Initialize attempt counter

        for attempt in range(max_retries + 1):
            try:
                response = llm.invoke(messages_for_llm)
                content = response.content
                if content and content.strip():
                    break
                else:
                    raise ValueError("LLM returned empty content.")
            except Exception as e:
                logging.warning(f"LLM call attempt {attempt+1} for role '{role}' failed: {e}")
                last_exception = e
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                else:
                    final_error = f"LLM call failed after {max_retries + 1} attempts. Last error: {last_exception}"
                    content = f"Error: {final_error}"

        # Log the full interaction
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(), "role": role, "provider": provider,
            "model_name": model_name, "system_prompt": system_prompt,
            "history_len": len(history), "user_message": user_message,
            "response_content": content, "error": final_error, "attempts_made": attempt + 1
        }
        llm_outputs_log.append(log_entry)
        return content

    def _update_summary(self, existing_summary: List[Dict[str, str]], messages_to_add: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Updates the summary with new turns."""
        logging.info("Iteratively updating conversation history summary...")

        existing_summary_text = ""
        if existing_summary:
            # Extract content from "[PREVIOUS HISTORY SUMMARY]\n..."
            content = existing_summary[0].get('content', '')
            match = re.search(r"\[PREVIOUS HISTORY SUMMARY\]\n(.*)", content, re.DOTALL)
            if match:
                existing_summary_text = match.group(1).strip()

        new_turns_text = "\n---\n".join([f"<{msg['role'].upper()}>:\n{msg['content']}" for msg in messages_to_add])

        summarization_prompt = f"""
You are a History Condenser AI. Your task is to integrate a new turn of a conversation into an existing summary.

The summary must remain concise but dense with all critical information. Preserve key events like new questions, solutions, verification results, and major decisions.

EXISTING SUMMARY:
---
{existing_summary_text if existing_summary_text else "No summary yet. This is the first turn to be summarized."}
---

NEW CONVERSATION TURN TO INTEGRATE:
---
{new_turns_text}
---

Provide the new, updated summary below. It should seamlessly combine the old summary with the new information.
"""
        # Use the Talker LLM for summarization
        summary_content = self._llm_chat(
            self.talker_llm,
            'history_compressor',
            "You are an expert at updating conversation summaries to retain key tactical information.",
            [], # Summarizer gets a clean slate
            summarization_prompt,
            max_retries=1
        )

        if summary_content.startswith("Error:"):
            logging.error("Failed to update summary. The turn to be compressed will be prepended to recent history to avoid data loss.")
            # Prepend a warning and the messages that failed to compress.
            warning_msg = {
                "role": "system",
                "content": "[SYSTEM WARNING] History summary update failed. The following turn was not compressed:"
            }
            return existing_summary + [warning_msg] + messages_to_add

        # Create the new history list with the summary as the first message
        updated_summary = [{
            "role": "system",
            "content": f"[PREVIOUS HISTORY SUMMARY]\n{summary_content}"
        }]

        logging.info("History summary updated successfully.")
        return updated_summary

    def _solver_attempt(self, question: str) -> str:
        """Solver LLM attempts to solve the question."""
        logging.info("\n--- Solver LLM Attempting Question ---")
        solver_prompt = f"""Solve the following mathematical problem. Provide a step-by-step derivation. Conclude with the final numerical answer clearly marked as 'ANSWER: [your answer]'.

Problem:
```question
{question}
```
"""
        return self._llm_chat(
            self.solver_llm, 'solver',
            "You are a highly capable math problem solving AI. Your goal is to find the correct numerical answer.",
            [], solver_prompt
        )

    def _verify_question_and_solution_integrity(self, question: str, solution: str) -> Tuple[bool, str]:
        """
        Uses the Creator LLM to perform a final integrity check on the question and solution.
        It checks for contradictions, correctness, and alignment between the two.
        Returns a boolean for success/failure and a string with the full verifier response for context.
        """
        logging.info("\n--- Performing Final Integrity Check with Verifier LLM ---")

        verification_prompt = f"""
You are a mathematical problem verifier. Your task is to check if a given question and solution pair is coherent and correct.

Analyze the following question and solution.
1. Is the question self-contradictory?
2. Is the solution mathematically correct?
3. Does the solution correctly answer the question?

Question:
```question
{question}
```

Solution:
```solution
{solution}
```

Based on your analysis, provide a final decision inside a code block. The decision must be ONLY "YES" or "NO".
If the decision is "NO", provide a brief explanation of the problem BEFORE the code block.

Example for a "NO" response:
The question asks for the perimeter, but the solution calculates the area.
```
NO
```

Example for a "YES" response:
```
YES
```
"""
        evaluator_persona = "You are a meticulous and impartial mathematical problem evaluator. Your task is to check questions and solutions for correctness, clarity, and consistency. You must be strict and precise in your judgment."

        # Use the creator_llm for this check, with no conversation history.
        response = self._llm_chat(
            self.creator_llm,
            'final_verifier',
            evaluator_persona,
            [], # No history
            verification_prompt,
            max_retries=1
        )

        logging.info(f"Verifier LLM raw response:\n{response}")

        decision_block = extract_code_blocks(response, "") # Extracts from a plain ``` block

        # We need a clear "YES" in a block to pass. Anything else is a failure.
        if decision_block and "YES" in decision_block.strip().upper():
            return True, response # Success, return the response for logging/printing
        else:
            return False, response # Failure, return the response for feedback

    def _talker_verify_solver(self, question: str, expected_solution: str, solver_answer_full: str) -> bool:
        """Uses the Talker LLM to check if the solver's answer is correct."""
        logging.info("\n--- Talker Verifying Solver's Answer ---")

        # If the solver errored out or couldn't find an answer, it's not solved.
        if solver_answer_full.startswith("Error:") or "ANSWER:" not in solver_answer_full.upper():
            logging.info("Solver did not produce a valid answer. Considered 'not solved'.")
            return False

        checker_prompt = f"""
You are a meticulous Answer Verifier. Your task is to determine if the 'Solver's Answer' is numerically equivalent to the 'Ground Truth Solution'.

- The 'Ground Truth Solution' is the correct, definitive answer.
- The 'Solver's Answer' is what another AI produced. It might have a different derivation, but you must focus only on whether the final numerical result is the same.

**Question:**
```question
{question}
```

**Ground Truth Solution:**
```solution
{expected_solution}
```

**Solver's Answer to Verify:**
```solver_answer
{solver_answer_full}
```

Based on your comparison of the final numerical answers, provide a final decision inside a code block. The decision must be ONLY "YES" (if the answers are equivalent) or "NO" (if they are not).

Example for a correct answer:
```
YES
```

Example for an incorrect answer:
```
NO
```
"""
        # Using the Talker LLM as it's good at analytical tasks.
        checker_response = self._llm_chat(
            self.talker_llm,
            'solver_checker',
            "You are a strict and impartial numerical answer comparator.",
            [], # No history needed for this atomic task
            checker_prompt,
            max_retries=1
        )

        logging.info(f"Solver Answer Checker Raw Response:\n{checker_response}")

        decision_block = extract_code_blocks(checker_response, "") # Extracts from a plain ``` block

        # Return True only if the block contains "YES".
        # Any other response (NO, malformed, error) means it's not solved correctly.
        if decision_block and "YES" in decision_block.strip().upper():
            logging.info("--- Solver Answer Verification: YES (Answers are equivalent) ---")
            return True
        else:
            logging.info("--- Solver Answer Verification: NO (Answers are not equivalent or check failed) ---")
            return False

    def _extract_latest_question_and_solution(self, history: List[Dict[str, str]]) -> Tuple[Optional[str], Optional[str]]:
        """
        Uses an LLM to extract the most recent question and solution from the conversation history.
        This is used during the finalization step to ensure the latest versions are being evaluated.
        """
        logging.info("\n--- Attempting to extract latest Q&A from history for finalization ---")

        # Format the history for the LLM
        formatted_history = "\n---\n".join([f"<{msg['role'].upper()}>:\n{msg['content']}" for msg in history])

        extractor_prompt = f"""
Review the following conversation history. Your task is to identify the single most recent and definitive math question and its corresponding definitive solution. Do not get distracted by older versions or discussions about hardening. Find the latest complete problem statement and its answer.

CONVERSATION HISTORY:
---
{formatted_history}
---

If you find a valid question and solution, format your response ONLY with the following structure, and nothing else:
```question
[The full text of the question]
```
```solution
[The full text of the solution]
```

If no definitive question or solution can be found in the history, return an empty response.
"""
        # Using the Talker LLM as it's good at understanding conversation context.
        extraction_response = self._llm_chat(
            self.talker_llm,
            'extractor',
            "You are a specialized text extraction AI. Your task is to find the most recent, complete, and final version of a math question and its corresponding solution from a conversation history.",
            [], # Extractor gets a clean slate with the history in the prompt
            extractor_prompt,
            max_retries=1
        )

        if extraction_response.startswith("Error:"):
            logging.error(f"Failed to extract Q&A from history: {extraction_response}")
            return None, None

        question = extract_code_blocks(extraction_response, "question")
        solution = extract_code_blocks(extraction_response, "solution")

        if question and solution:
            logging.info("--- Successfully extracted latest Q&A from history. ---")
        else:
            logging.warning("--- Could not extract a valid Q&A pair from history for finalization. ---")

        return question, solution

    def generate_hard_question_agentic(self, topic: str, max_turns: int = 40) -> Optional[Tuple[str, str]]:
        """
        Manages the agentic conversation to generate a hard math question.
        """
        logging.info(f"\n=== Starting Agentic Question Generation (Topic: {topic}) ===")

        summary_history: List[Dict[str, str]] = []
        recent_history: List[Dict[str, str]] = []
        turn_message_counts: List[int] = []

        current_question = None
        current_solution = None

        user_prompt = f"Let's begin. Your mission is to generate a very difficult math question about '{topic}'. Start by instructing the Creator LLM to generate an initial problem formulation."

        for turn in range(max_turns):
            logging.info(f"\n{'='*20} Agentic Turn {turn + 1}/{max_turns} {'='*20}")

            # --- Iterative History Compression ---
            if self.history_compression_turns > 0 and turn >= self.history_compression_turns:
                num_messages_to_compress = turn_message_counts.pop(0)

                messages_to_compress = recent_history[:num_messages_to_compress]
                recent_history = recent_history[num_messages_to_compress:]

                logging.info(f"\n--- Compressing {num_messages_to_compress} messages from turn {turn - self.history_compression_turns + 1} into summary ---")

                summary_history = self._update_summary(summary_history, messages_to_compress)

                user_prompt = "The conversation history has been updated with a summary of the oldest turn. Based on the summary and recent events, decide on the next logical action and instruct the Creator. Don't forget that the topic is {topic}."

            messages_in_this_turn = []
            full_history = summary_history + recent_history

            # 1. Talker's Turn: Decide what to do next
            logging.info("\n--- Agentic Talker is thinking... ---")
            talker_instruction = self._llm_chat(
                self.talker_llm, 'agentic_talker',
                self.agentic_talker_persona,
                full_history,
                user_prompt
            )

            if talker_instruction.startswith("Error:"):
                logging.error("Agentic talker failed. Aborting.")
                return None

            logging.info(f"\n--- Agentic Talker's Instruction to Creator ---")
            print(talker_instruction)
            talker_msg = {"role": "assistant", "content": talker_instruction}
            messages_in_this_turn.append(talker_msg)

            # Check for finalization command
            if "FINALIZE" in talker_instruction.upper():
                logging.info("\n--- Agentic Talker is attempting to finalize. Performing final checks. ---")

                # Run any code in the talker's finalize message before other checks
                code_from_talker = extract_code_blocks(talker_instruction, "python") or extract_code_blocks(talker_instruction, "")
                if code_from_talker:
                    logging.info("\n--- Running code block from Talker's FINALIZE instruction ---")
                    run_verification_code(
                        code_from_talker,
                        timeout_seconds=self.code_execution_timeout
                    )

                # Extract the latest Q&A from history to ensure it's up-to-date.
                logging.info("Extracting latest question and solution from history to ensure they are current.")
                latest_q, latest_s = self._extract_latest_question_and_solution(full_history)
                if latest_q:
                    current_question = latest_q
                if latest_s:
                    current_solution = latest_s

                if not current_question or not current_solution:
                    logging.warning("Talker wants to finalize, but no valid question/solution is available. Instructing to continue.")
                    system_feedback = "[SYSTEM FEEDBACK]: You tried to finalize, but no valid question and solution are stored or could be extracted from the history. Please continue the process until a verifiable question is ready."
                else:
                    # NEW VERIFICATION STEP: Check for integrity before checking for difficulty.
                    integrity_ok, verifier_response = self._verify_question_and_solution_integrity(current_question, current_solution)

                    if integrity_ok:
                        logging.info("\n--- FINAL INTEGRITY CHECK PASSED. Proceeding to difficulty check with Solver. ---")
                        # The verifier LLM has output its decision. We print the decision block.
                        print("\n--- Verifier LLM Decision ---")
                        print(f"```{extract_code_blocks(verifier_response, '')}```")
                        print("---------------------------\n")

                        solver_response = self._solver_attempt(current_question)
                        is_solved = self._talker_verify_solver(current_question, current_solution, solver_response)

                        if not is_solved:
                            logging.info("\n--- FINAL DIFFICULTY CHECK PASSED: Solver was unable to solve the question. ---")
                            return current_question, current_solution
                        else:
                            logging.warning("\n--- FINAL DIFFICULTY CHECK FAILED: Solver successfully solved the question. ---")
                            system_feedback = "[SYSTEM FEEDBACK]: You tried to finalize, and the question passed the integrity check. However, the Solver LLM solved it. It is not hard enough. Instruct the Creator to make it significantly harder."
                    else:
                        # Integrity check failed. Provide the full response from the verifier to the talker.
                        logging.warning("\n--- FINAL INTEGRITY CHECK FAILED. ---")
                        system_feedback = f"[SYSTEM FEEDBACK]: You tried to finalize, but the problem failed an integrity check. Review the evaluator's feedback and instruct the Creator to fix the issues.\n\n--- Evaluator Feedback ---\n{verifier_response}"

                system_msg = {"role": "system", "content": system_feedback}
                messages_in_this_turn.append(system_msg)
                recent_history.extend(messages_in_this_turn)
                turn_message_counts.append(len(messages_in_this_turn))
                user_prompt = "Based on the latest [SYSTEM FEEDBACK], decide on the next action."
                continue

            # 2. Creator's Turn: Execute the Talker's instruction
            logging.info("\n--- Creator is working based on instruction... ---")
            creator_response = self._llm_chat(
                self.creator_llm, 'creator',
                self.creator_persona,
                full_history,
                talker_instruction
            )

            if creator_response.startswith("Error:"):
                logging.error("Creator LLM failed. The agent will be notified.")
                system_feedback = f"[SYSTEM FEEDBACK]: The Creator LLM failed to respond. Please try a different instruction or rephrase your request."
                creator_msg = {"role": "user", "content": creator_response} # Log the failure
                system_msg = {"role": "system", "content": system_feedback}
                messages_in_this_turn.extend([creator_msg, system_msg])
            else:
                logging.info(f"\n--- Creator's Response ---")
                print(creator_response[:1000] + ('...' if len(creator_response) > 1000 else ''))
                creator_msg = {"role": "user", "content": creator_response}
                messages_in_this_turn.append(creator_msg)

                # 3. System Turn: Extract artifacts and perform external checks
                new_question = extract_code_blocks(creator_response, "question")
                new_solution = extract_code_blocks(creator_response, "solution")
                # Run any code block, prioritizing 'python' then a generic block.
                new_code = extract_code_blocks(creator_response, "python") or extract_code_blocks(creator_response, "")


                feedback_additions = []
                if new_question:
                    logging.info("\n--- Extracted new Question ---")
                    current_question = new_question
                if new_solution:
                    logging.info("\n--- Extracted new Solution ---")
                    current_solution = new_solution
                if new_code:
                    logging.info("\n--- Extracted new Verification Code ---")
                    code_success, code_output = run_verification_code(
                        new_code,
                        timeout_seconds=self.code_execution_timeout
                    )
                    code_feedback = "SUCCESS" if code_success else "FAILED"
                    feedback_additions.append(f"Verification Code Status: The generated code was executed with the result: {code_feedback}.")
                    feedback_additions.append(f"Code Output/Error: {code_output}")

                if feedback_additions:
                    full_feedback = "[SYSTEM FEEDBACK]\n" + "\n".join(feedback_additions)
                    system_msg = {"role": "system", "content": full_feedback}
                    messages_in_this_turn.append(system_msg)

            # Commit all messages from this turn to history
            recent_history.extend(messages_in_this_turn)
            turn_message_counts.append(len(messages_in_this_turn))

            user_prompt = "Based on the updated conversation history, especially the latest Creator response and any new [SYSTEM FEEDBACK], decide on the next action and instruct the Creator."

        logging.warning("Maximum turns reached. Returning the last valid question.")
        return (current_question, current_solution) if current_question and current_solution else None


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Agentic Math Question Generator ---")

    # --- Timeout Setup (for Unix-like systems) ---
    # This is the GLOBAL timeout for the entire process.
    class TimeoutException(BaseException):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutException

    is_timeout_supported = hasattr(signal, 'SIGALRM')
    if is_timeout_supported:
        signal.signal(signal.SIGALRM, timeout_handler)
    else:
        logging.warning("Timeout functionality is not supported on this platform (e.g., Windows). The process-wide timeout limit will not be enforced.")

    # --- Configuration ---
    CREATOR_PROVIDER = os.getenv("CREATOR_PROVIDER", "openai").lower()
    CREATOR_MODEL = "o3"
    CREATOR_TEMP = 0.7

    TALKER_PROVIDER = os.getenv("TALKER_PROVIDER", "openai").lower()
    TALKER_MODEL = "o3"
    TALKER_TEMP = 0.2

    SOLVER_PROVIDER = os.getenv("SOLVER_PROVIDER", "openai").lower()
    SOLVER_MODEL = "o3"
    SOLVER_TEMP = 0.0

    MATH_TOPIC = os.getenv("MATH_TOPIC", "Polynomials")
    MAX_TURNS = int(os.getenv("MAX_TURNS", "400"))
    HISTORY_COMPRESSION_TURNS = int(os.getenv("HISTORY_COMPRESSION_TURNS", "10"))
    GLOBAL_TIMEOUT_SECONDS = 1000 * 60  # 10 minutes global limit for the whole process
    CODE_EXECUTION_TIMEOUT_SECONDS = 300 # 300 seconds limit for running generated code

    print("\n--- Configuration ---")
    print(f"Creator: {CREATOR_PROVIDER}/{CREATOR_MODEL} (Temp: {CREATOR_TEMP})")
    print(f"Talker:  {TALKER_PROVIDER}/{TALKER_MODEL} (Temp: {TALKER_TEMP})")
    print(f"Solver:  {SOLVER_PROVIDER}/{SOLVER_MODEL} (Temp: {SOLVER_TEMP})")
    print(f"Topic:   {MATH_TOPIC}")
    print(f"Max Turns: {MAX_TURNS}")
    print(f"History Compression Turns: {HISTORY_COMPRESSION_TURNS}")
    print(f"Global Process Timeout: {GLOBAL_TIMEOUT_SECONDS}s")
    print(f"Code Execution Timeout: {CODE_EXECUTION_TIMEOUT_SECONDS}s")
    print("---------------------\n")

    try:
        creator = initialize_llm(CREATOR_PROVIDER, CREATOR_MODEL, role='creator', temperature=CREATOR_TEMP)
        talker = initialize_llm(TALKER_PROVIDER, TALKER_MODEL, role='talker', temperature=TALKER_TEMP)
        solver = initialize_llm(SOLVER_PROVIDER, SOLVER_MODEL, role='solver', temperature=SOLVER_TEMP)

        print("\n--- LLMs Initialized Successfully ---\n")

        generator = AgenticMathQuestionGenerator(
            creator_llm=creator,
            talker_llm=talker,
            solver_llm=solver,
            history_compression_turns=HISTORY_COMPRESSION_TURNS,
            code_execution_timeout=CODE_EXECUTION_TIMEOUT_SECONDS
        )

        start_time = time.time()
        result = None
        try:
            if is_timeout_supported:
                signal.alarm(GLOBAL_TIMEOUT_SECONDS)
            result = generator.generate_hard_question_agentic(MATH_TOPIC, MAX_TURNS)
        except TimeoutException:
            print("\n---------------------------------------------------------------------------------")
            print(f"!!! GLOBAL TIMEOUT: The entire process took too long to finish (>{GLOBAL_TIMEOUT_SECONDS / 60:.1f} minutes) and was terminated.")
            print("---------------------------------------------------------------------------------")
        finally:
            if is_timeout_supported:
                signal.alarm(0) # Ensure the alarm is cancelled
        end_time = time.time()

        print(f"\n\n--- Agentic Generation Process Finished (Duration: {end_time - start_time:.2f} seconds) ---")
        print("===================================")
        if result:
            final_question, final_solution = result
            print("\n=== Final Generated Question ===\n")
            print(final_question)
            print("\n\n=== Final Solution ===\n")
            print(final_solution)

            output_dir = "generated_questions"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            topic_slug = re.sub(r'\W+', '_', MATH_TOPIC).lower()
            filename = os.path.join(output_dir, f"agentic_q_{topic_slug}_{timestamp}.md")
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(f"# Generated Math Question\n\n**Topic:** {MATH_TOPIC}\n\n")
                    f.write("## Question\n\n")
                    f.write(f"```\n{final_question}\n```\n\n")
                    f.write("## Solution\n\n")
                    f.write(f"```\n{final_solution}\n```\n")
                print(f"\n\nQuestion saved to: {filename}")
            except IOError as e:
                print(f"\nError saving question to file: {e}")
        else:
            print("\n=== Failed to generate a suitable question (or the process timed out). ===")
        print("===================================")

    except (ImportError, ValueError) as e:
        logging.critical(f"\nError during setup: {e}", exc_info=True)
        print("Please check library installations, model names, and API keys.")
    except Exception:
        logging.critical(f"\nAn unexpected error occurred:", exc_info=True)

    finally:
        log_filename = f"agentic_llm_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(log_filename, "w", encoding="utf-8") as f:
                json.dump(llm_outputs_log, f, indent=2)
            print(f"\n--- LLM interaction log saved to: {log_filename} ---")
        except Exception as e:
            print(f"\n--- Error saving LLM log: {e} ---")