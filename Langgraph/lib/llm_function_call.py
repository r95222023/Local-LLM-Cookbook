from dotenv import load_dotenv
import re
from langgraph.graph import END, StateGraph, START
from langchain.globals import set_verbose, set_debug
from typing import List, TypedDict, Literal
import json
from langchain_core.tools import tool

from langchain_core.prompts import ChatPromptTemplate

# from langchain_experimental.llms.ollama_functions import OllamaFunctions


fix_tool_args_prompt = ChatPromptTemplate.from_messages( # {"tool_name", "reflections", "arg", "tool_description", args}
    [
        (
            "system",
            """You are an AI assistant equipped with various tools to help answer questions and solve problems. 
            \nYou have tried to use {tool_name}, but errors occurred. The reflections and tool informations are shown below:
            \nTool Description:
            \n\n```
            \n{tool_description}
            \n```
            \n
            \nTool Args Schema:
            \n\n```
            \n{args}
            \n```
            \n
            \nReflections:
            \n\n```
            \n{reflections}
            \n```
            \n
            \nWhat is correct {arg} parameter needed to resolve the issue? Only give the best one.
            \nThe answer should follow the following format.
            \n\nCorrect function call:
            \n\n```
            \nmy_function(arg_1=fixed_arg1, arg_2=fixed_arg2)
            \n```
            \n
            \n[INSERT YOUR ANSWER HERE]
            """,
        ),
    ]
)

# retry_prompt = ChatPromptTemplate.from_messages( # {"task", "reflections", "tool_calls"}
#     [
#         (
#             "system",
#             """You are an AI assistant equipped with various tools to help answer questions and solve problems.
#             \nFind the proper tool for the task given by the user.
#             \nWhenever the user provide a task or ask a question, please determine and use the most appropriate tool to achieve the best results.
#             \nFor example, if I ask for current events, use the browser tool to fetch the latest information. If I need calculations or data analysis, use the python tool.
#             \nFor long-term memory or context, utilize the bio tool to remember important details.
#             \nPlease respond with the solution or answer, and mention the tool you used. The tool you use must be one of the tools given below.
#             \nThe available tools are {tools}
#             \nHere is the user's request:
#             \n {task}
#             \n[INSERT YOUR ANSWER HERE]""",
#         ),
#     ]
# )

finish_prompt = ChatPromptTemplate.from_messages( # {"task", "tool_outputs"}
    [
        (
            "system",
            """You are an AI assistant equipped with various tools to help answer questions and solve problems.
            \nYou have chosen to use the tools with outputs in JSON format shown below, 
            where the keys are the tool names and the value are their outputs. 
            \n\n```
            \n{tool_outputs}
            \n\n```
            \n
            Use tool outputs to answer the user's question. Do not make up any data.\n Here is the user question:""",
        ),
        ("placeholder", "{task}"),
    ]
)

finish_wo_calls_prompt = ChatPromptTemplate.from_messages( # {"task", "tool_outputs"}
    [
        (
            "system",
            """You are an helpful AI assistant who is good at answering questions and solve problems.
            \n Here is the user question:""",
        ),
        ("placeholder", "{task}"),
    ]
)

reflect_prompt = ChatPromptTemplate.from_messages( # {"tool_calls", "messages"}
    [
        (
            "system",
            """ You are a debugging expert with expertise in Python, Langchain, and openai library. 
            \nThe following tool_calls failed during runtime.
            \nTool Calls
            \n\n```
            \n{tool_calls}
            \n\n```
            The error messages are listed as follows:""",
        ),
        ("placeholder", "{messages}"),
        ("system", "Provide the solution to the errors to aid in debugging.")
    ]
)


class GraphState(TypedDict):
    task: str
    tool_calls: list
    iterations: int
    reflections: str
    error: str
    messages: List
    answer: str


def parse_arg(ai_msg, arg):
    pattern_string = f'{arg}=([^\s,)]+)'
    regex = re.compile(pattern_string)
    # pattern = r'location=([^\s,)]+)'
    # match = re.search(pattern, ai_msg.content)
    match = regex.search(ai_msg.content)
    if match:
        try:
            return eval(match.group(1))
        except Exception as e:
            return match.group(1) or ""

# Usage
# @tool
# def get_weather(city: Literal["nyc", "sf"]):
#     """Use this to get weather information."""
#     if city == "nyc":
#         return "cloudy, 30 degrees celcius"
#     elif city == "sf":
#         return "sunny, 30 degrees celcius"
#     else:
#         raise AssertionError("Unknown city")
#
#
# from langchain_experimental.llms.ollama_functions import OllamaFunctions
# tools = [get_weather]
# available_tools = {'get_weather': get_weather}
# llm_function = OllamaFunctions(model="llama3:instruct", format="json")
# app = ToolLlm(llm_function, available_tools=available_tools)
# app.invoke("give the weather in nyc")


class ToolLlm:
    def __init__(self, llm, available_tools={}, max_iterations=5, reflect=False, debug=False):
        set_debug(debug)

        self.compiled = False
        self.workflow = None
        self.available_tools = available_tools
        self.tools = []
        for k in available_tools:
            self.tools.append(available_tools[k])
        self.llm = llm
        self.max_iterations = max_iterations
        self.llm_with_tools = llm.bind_tools(self.tools)
        self.reflect = reflect
        # self.retry_cahin = retry_prompt | self.llm # {"task", "reflections", "tool_calls"}
        self.finish_chain = finish_prompt | self.llm  # {"task", "tool_outputs"}
        self.finish_wo_calls_chain = finish_wo_calls_prompt | self.llm  # {"task"}
        self.reflect_chain = reflect_prompt | self.llm  # {"tool_calls", "messages"}
        self.fix_tool_args_chain = fix_tool_args_prompt | self.llm  # {"task", "tool_name", "reflections", "arg", "tool_description", args}
        self.workflow = StateGraph(GraphState)

    def invoke(self, task):
        if not self.compiled:
            app = self.compile()
        else:
            app = self.workflow
        return app.invoke({"task": [("user", task)], "iterations": 0, "messages": []})

    def compile(self):
        workflow = self.workflow
        workflow.add_node("find_tool_node", self.find_tool_node)
        workflow.add_node("fix_tool_args_node", self.fix_tool_args_node)
        workflow.add_node("check_code_node", self.check_code_node)
        workflow.add_node("reflect_node", self.reflect_node)
        workflow.add_node("finish_node", self.finish_node)

        # Build graph
        workflow.add_edge(START, "find_tool_node")
        workflow.add_conditional_edges(
            "find_tool_node",
            self.decide_to_finish,
            {
                "finish": "finish_node",
                "check": "check_code_node",
            },
        )
        workflow.add_conditional_edges(
            "check_code_node",
            self.decide_to_reflect,
            {
                "finish": "finish_node",
                "reflect": "reflect_node",
                "retry": "find_tool_node"
            },
        )
        workflow.add_edge("finish_node", END)
        workflow.add_edge("reflect_node", "fix_tool_args_node")
        workflow.add_edge("fix_tool_args_node", "check_code_node")
        self.workflow = workflow.compile()
        self.compiled = True
        return self.workflow

    def finish_node(self, state: GraphState):
        """
        Find suitable tool to solve the problem

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation
        """
        # State
        task = state["task"]
        tool_calls = state["tool_calls"]
        messages = state["messages"]
        iterations = state["iterations"]
        if tool_calls:
            tool_outputs = {}
            for tool in tool_calls:
                tool_outputs[tool['name']] = tool['output']

            ai_msg = self.finish_chain.invoke({"task": task, "tool_outputs": tool_outputs})
            # answer = ai_msg.content
            # messages += [("system", f"{answer}")]
            # return {**state, "tool_calls": tool_calls, "iterations": iterations, "messages": messages, "answer": answer}
        else:
            ai_msg = self.finish_wo_calls_chain.invoke({"task": task})

        answer = ai_msg.content
        messages += [("system", f"{answer}")]
        return {**state, "tool_calls": tool_calls, "iterations": iterations, "messages": messages, "answer": answer}

    def find_tool_node(self, state: GraphState):
        """
        Find suitable tool to solve the problem

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation
        """

        print("---FINDING TOOL---")

        # State
        task = state["task"]
        iterations = state["iterations"]
        reflections = state["reflections"]
        error = state["error"]
        messages = state["messages"]

        # Increment
        iterations = iterations + 1
        if error == 'yes':
            # got error messages at check_code_node
            task = [messages[-1]] + [('user', f'Try again to complete the task: {task[0][1]}')]
        ai_msg = self.llm_with_tools.invoke(task)
        tool_calls = getattr(ai_msg, 'tool_calls', False)
        if tool_calls:
            return {**state, "tool_calls": tool_calls, "iterations": iterations, "messages": messages}
        else:
            return {**state, "tool_calls": [], "iterations": iterations, "messages": messages}

    def check_code_node(self, state: GraphState):
        """
        Check code

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, error
        """

        print("---CHECKING CODE---")

        # State
        task = state["task"]
        tool_calls = state["tool_calls"]
        messages = state["messages"]

        # Check execution
        try:
            for i in range(len(tool_calls)):
                tool = tool_calls[i]
                tool['output'] = self.available_tools[tool['name']].invoke(tool['args'])
                tool_calls[i] = tool
        except Exception as e:
            print("---TOOL EXECUTION: FAILED---")
            error_message = [("user", f"Your solution failed the test: {e}")]
            messages += error_message
            return {
                **state,
                "messages": messages,
                "error": "yes",
            }

        # No errors
        print("---NO CODE TEST FAILURES---")
        return {
            **state,
            "error": "no",
            "tool_calls": tool_calls
        }

    def reflect_node(self, state: GraphState):
        """
        Reflect on errors

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation
        """

        print("---GENERATING CODE SOLUTION---")

        # State
        # task = state["task"]
        tool_calls = state["tool_calls"]
        messages = state["messages"]

        # Add reflection
        reflections = self.reflect_chain.invoke(
            {"tool_calls": tool_calls, "messages": messages}
        ).content
        messages += [("assistant", f"Here are reflections on the error: {reflections}")]
        return {**state, "reflections": reflections, "messages": messages}

    def fix_tool_args_node(self, state: GraphState):
        """
        Fix the parameters for the selected tool

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation
        """

        print("---FINDING TOOL ARGS---")

        # State
        task = state["task"]
        tool_calls = state["tool_calls"]
        iterations = state["iterations"]
        iterations += 1
        reflections = state["reflections"]
        error = state["error"]
        messages = state["messages"]
        for i in range(len(tool_calls)):
            tool = tool_calls[i]
            fixed_args = {}
            for arg in tool.args:
                # {"tool_name", "reflections", "arg", "tool_description", args}
                ai_msg = self.fix_tool_args_chain.invoke({"tool_name": tool['name'], "reflections": reflections,
                                                          "arg": arg, "tool_description": tool.description,
                                                          "arg_info": json.dumps(tool.args[arg])})
                fixed_args[arg] = parse_arg(ai_msg, arg)

            tool['args'] = fixed_args
            tool_calls[i] = tool

        return {**state, "tool_calls": tool_calls, "iterations": iterations, "messages": messages}

    ### Edges

    def decide_to_finish(self, state: GraphState):
        """
        Determines whether to finish.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """
        tool_calls = state["tool_calls"]

        if tool_calls:
            print("---DECISION: CHECK CODE---")
            return "check"
        else:
            print("---DECISION: FINISH---")
            return "finish"

    def decide_to_reflect(self, state: GraphState):
        """
        Determines whether to reflect.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        error = state["error"]
        iterations = state["iterations"]

        if error == "no" or iterations == self.max_iterations:
            print("---DECISION: FINISH---")
            return "finish"
        else:
            if self.reflect == "reflect":
                print("---DECISION: RE-TRY SOLUTION---")
                return "reflect"
            else:
                return "retry"