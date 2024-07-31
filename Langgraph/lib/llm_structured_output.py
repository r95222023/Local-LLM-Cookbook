from dotenv import load_dotenv
import re
from pydantic import ValidationError
from langgraph.graph import END, StateGraph, START
from langchain.globals import set_verbose, set_debug
from typing import List, TypedDict, Literal, Union, Annotated
import json
from langchain_core.tools import tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, schema_json_of

from langchain_experimental.llms.ollama_functions import OllamaFunctions

llm_function = OllamaFunctions(model="llama3:instruct", format="json")


fix_structure_output_prompt = ChatPromptTemplate.from_messages(  # {"tool_name": tool.name, "reflections": reflections,
    # "arg": arg, "tool_description": tool.description, "arg_info": json.dumps(tool.args[arg])}
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

reflect_prompt = ChatPromptTemplate.from_messages( # {"task": task, "schema": schema, "messages": messages}
    [
        (
            "system",
            """ You are a debugging expert with expertise in Python and Pydantic.
            \nAn agent tried to convert a description into an output with structured format:
            \nDescription:
            \n\n```
            \n{task}
            \n\n```
            \nStructured Format:
            \n\n```
            \n{schema}
            \n\n```
            \nThe error messages received by the agent are listed as follows:
            \n {messages}
            \n\n Provide reflections on these errors and the corrected structured output.
            \n[YOUR ANSWER]""",
        ),
    ]
)


class GraphState(TypedDict):
    task: str
    structured_output: dict
    iterations: int
    reflections: str
    error: str
    messages: List


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

    
class StructuredOutputLlm:
    def __init__(self, llm, structured_model, max_iterations=5, reflect=False):
        self.structured_model = structured_model
        self.compiled = False
        self.workflow = None
        self.llm = llm
        self.max_iterations = max_iterations
        self.llm_with_structured_output = llm.with_structured_output(structured_model)
        self.reflect = reflect
        self.reflect_prompt = reflect_prompt
        self.reflect_chain = reflect_prompt | self.llm # {"task": task, "structured_output": structured_output, "messages": messages}
        self.fix_structure_output_chain =  fix_structure_output_prompt | self.llm # {"task", "tool_name", "reflections", "arg", "tool_description", args}

        # self.workflow = StateGraph(GraphState)
        self.structured_output_tool = StructuredTool.from_function(
            func=self.structured_model,
            name="StructuredOutput",
            description="Return args in pydantic way",
            # coroutine= ... <- you can specify an async method if desired as well
        )

    def invoke(self, task):
        if not self.compiled:
            app = self.compile()
        else:
            app = self.workflow
        return app.invoke({"task": [("user", task)], "iterations": 0, "messages":[]})
            
    def compile(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("create_structure_output_node", self.create_structure_output_node)
        workflow.add_node("fix_structure_output_node", self.fix_structure_output_node)
        workflow.add_node("reflect_node", self.reflect_node)
        
        # Build graph
        workflow.add_edge(START, "create_structure_output_node")        
        workflow.add_conditional_edges(
            "create_structure_output_node",
            self.decide_to_reflect,
            {
                "finish": END,
                "reflect": "reflect_node",
                "retry": "create_structure_output_node"
            },
        )
        workflow.add_edge("reflect_node", "fix_structure_output_node")
        workflow.add_conditional_edges(
            "fix_structure_output_node",
            self.decide_to_reflect,
            {
                "finish": END,
                "reflect": "reflect_node",
            },
        )
        self.workflow = workflow.compile()
        self.compiled = True
        return self.workflow

    def create_structure_output_node(self, state: GraphState):
        """
        Find suitable tool to solve the problem
    
        Args:
            state (dict): The current graph state
    
        Returns:
            state (dict): New key added to state, generation
        """
    
        # State
        task = state["task"]
        structured_output = state["structured_output"]
        messages = state["messages"]
        iterations = state["iterations"]
        iterations += 1        
        error = state["error"]
        structured_output = None
        try:
            if error == 'yes':
                task = [messages[-1]] + [('user', f'Try again to complete the task: {task[0][1]}')]
            structured_output = self.structured_model.validate(self.llm_with_structured_output.invoke(task))
            error = 'no'
        except Exception as e:
            print(f'Error occured: {e}')
            messages += [("system", f"Errors occured in your last attempt: {e}")]
            error = 'yes'
        
        return {**state, "error": error, "iterations": iterations, "messages": messages, "structured_output": structured_output}

    

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
        task = state["task"]
        structured_output = state["structured_output"]
        messages = state["messages"]
        schema = schema_json_of(self.structured_model)

    
        # Add reflection
        err_msg = self.reflect_prompt.invoke({"task": task, "messages": messages, "schema": schema})
        reflections = self.llm.invoke(
            err_msg[0].content
        ).content
        messages += [("assistant", f"Here are reflections on the error: {reflections}")]
        return {**state, "reflections": reflections, "messages": messages}

    def fix_structure_output_node(self, state: GraphState):
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
        iterations = state["iterations"]
        iterations += 1
        reflections = state["reflections"]
        error = state["error"]
        messages = state["messages"]
        
        tool = self.structured_output_tool
        fixed_args = {}
        for arg in tool.args:
        # {"tool_name", "reflections", "arg", "tool_description", args}
            ai_msg = self.fix_structure_output_chain.invoke({"tool_name": tool.name, "reflections": reflections,
                                          "arg": arg, "tool_description": tool.description, "arg_info": json.dumps(tool.args[arg])})
            fixed_args[arg] = parse_arg(ai_msg, arg)

        try:
            fixed_structured_model = self.structured_model(**fixed_args)
            fixed_structured_model= self.structured_model.validate(fixed_structured_model)
            error = 'no'
        except ValidationError as e:
            messages += [("system", f"{e}")]
            error = 'yes'

        return {**state, "error": error, "iterations": iterations, "messages": messages, "structured_output": fixed_structured_model}
        
    ### Edges
    
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
            if self.reflect:
                print("---DECISION: REFLECTING SOLUTION---")
                return "reflect"
            else:
                print(f"---DECISION: RETRY (times:{iterations})---")
                return "retry"
