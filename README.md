# Local LLM Cookbook

The Local LLM Cookbook offers a collection of examples and best practices for improving the usability and robustness of local Large Language Models (LLMs). It provides practical examples, optimization tips, and tutorials to ensure reliability.

This cookbook includes example code for building applications with various frameworks, such as LangChain, LangGraph, CrewAI, AutoGen, and LlamaIndex, with a focus on practical and local LLM use cases.

Notebook | Description
:- | :-
[llm_chessmaster.ipynb](https://github.com/r95222023/Local-LLM-Cookbook/tree/main/Langgraph/llm_chessmaster.ipynb) | Build a chess application that is designed to provide an interactive chess-playing experience with a large language model(LLM).
[robust_react_agent.ipynb](https://github.com/r95222023/Local-LLM-Cookbook/tree/main/Langgraph/robust_react_agent.ipynb) | Perform retrieval-augmented generation (rag) on documents with semi-structured data, including text and tables, using unstructured for parsing, multi-vector retriever for storing, and lcel for implementing chains.
[code_assistant.ipynb](https://github.com/r95222023/Local-LLM-Cookbook/tree/main/Langgraph/code_assistant.ipynb) | Build a code assistant that generates boilerplate code, templates, or even entire functions/classes based on high-level descriptions.
[llm_function_call.ipynb](https://github.com/r95222023/Local-LLM-Cookbook/tree/main/Langgraph/llm_function_call.ipynb) | Build an agent that is designed for robust tool-calling functionality. If an error occurs during the execution of a tool, the LLM model will be asked to carefully examine each generated argument and attempt to fix the problem.
[llm_structured_output.ipynb](https://github.com/langchain-ai/langchain/tree/master/cookbook/llm_structured_output.ipynb) | Build an robust agent with structured outputs. If an error occurs during the execution of a tool, the LLM model will be asked to carefully examine each generated argument and attempt to fix the problem.
[crewai_chessmaster.ipynb](https://github.com/r95222023/Local-LLM-Cookbook/tree/main/Crewai/crewai_chessmaster.ipynb) | Build a chess application with crewAi designed to offer an interactive chess-playing experience using a large language model (LLM). The crewAi agent can search the internet to find the best move.
[stock_analysis.ipynb](https://github.com/r95222023/Local-LLM-Cookbook/tree/main/Crewai/stock_analysis.ipynb) | This notebook demonstrates how to use the CrewAI framework to automate stock analysis. CrewAI coordinates autonomous AI agents, allowing them to collaborate and perform complex tasks efficiently.
[booking_agent.ipynb](https://github.com/r95222023/Local-LLM-Cookbook/tree/main/LlamaIndex/booking_agent.ipynb) | For powerful online LLMs such as GPT-4 or Claude, `llama_index.core.agent.FunctionCallingAgentWorker` combined with tools with direct returns can perform the tasks of a simple booking agent. However, FunctionCallingAgentWorker does not yet support local LLMs. To address this, a workaround using ReActAgent is demonstrated in this notebook.
[nested_chat.ipynb](https://github.com/r95222023/Local-LLM-Cookbook/tree/main/Autogen/nested_chat.ipynb) | A chess game built to demonstrate the use of nested chats to facilitate tool usage, integrating an LLM-based agent with a tool executor agent into a single cohesive unit.
[structured_output.ipynb](https://github.com/r95222023/Local-LLM-Cookbook/tree/main/Autogen/structured_output.ipynb) | AutoGen provides a unified multi-agent conversation framework as a high-level abstraction for using foundation models. However, it does not offer an easy way to produce structured output. This notebook presents a workaround that utilizes agent tool calling to structure the output.

