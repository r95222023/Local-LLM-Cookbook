import re
from langgraph.graph import END, StateGraph, START
from langchain.globals import set_verbose, set_debug
from typing import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate

solution_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",""" You are a coding expert with expertise in Python, LangChain and LangGraph. 
        \nHere is the user question: \n {task}
        \nHere is the documentation for a library:  \n ------- \n  {context} \n ------- \n Answer the user question based on the \n 
        above provided documentation. Ensure any code you provide can be executed with all required imports and variables defined.
        \n\n[INSERT YOUR CODE HERE]
        """),
    ]
)

solution_prompt_reflect = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ You are a coding expert with expertise in Python, LangChain and LangGraph. \n 
    Here is the documentation for a library:  \n ------- \n  {context} \n ------- \n You tried to answer the user question based on the \n 
    above provided documentation, but your last attempt was not successful! Here is the reflections for your last attempt 
    \n ------- \n  {reflections} \n ------- \n
    \n Answer the user question again. Fix the code and ensure any code you provide can be executed with all required imports and variables \n
    defined.\n Here is the user question:""",
        ),
        ("placeholder", "{task}"),
    ]
)

description_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ You are a coding expert with expertise in Python, LangChain and LangGraph. \n 
    The solution you provided in the last step is
    \n ------- \n  {solution} \n ------- \n 
    \n\n Here is the user question:
    \n\n{task}
    \n
    \nGive description for this code, and ensure that the description is clear and concise.
    \n[INSERT DESCRIPTION HERE]""",
        ),
    ]
)

imports_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ You are a coding expert with expertise in Python, LangChain and LangGraph. \n 
    The solution you provided in the last step is
    \n ------- \n  {solution} \n ------- \n 
    \n Extract the imports part for this code.
    \nEXAMPLE:
    \n\n
    ```
    \nfrom dotenv import load_dotenv
    \nfrom langchain_community.tools.tavily_search import TavilySearchResults
    \nfrom langchain import hub
    ```
    \n\n[INSERT IMPORTS HERE]
    """,
        ),
    ]
)

code_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ You are a coding expert with expertise in Python, LangChain and LangGraph. \n 
    The solution you provided in the last step is
    \n ------- \n  {solution} \n ------- \n 
    \n Extract the code block without imports part for this code, and ensure any code you provide can be executed.
    \nThe output should be formatted as follows:
    \nEXAMPLE: 
    \n\n
    ```
    \nexpt_llm = "claude-3-opus-20240229"
    \nllm = ChatAnthropic(
    \nmodel=expt_llm,
    \ndefault_headers=None,)

    \nstructured_llm_claude = llm.with_structured_output(code, include_raw=True)
    \n
    ```
    \n\n[INSERT CODE HERE]""",
        ),
    ]
)

reflect_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ You are a coding expert with expertise in Python, LangChain and LangGraph. \n 
    The solution you provided in the last step is
    \n ------- \n  {solution} \n ------- \n 
    \n Extract the code block for this code, and ensure any code you provide can be executed.""",
        ),
        ("placeholder", "{messages}"),
    ]
)


class GraphState(TypedDict):
    solution: str
    description: str
    # prefix: str
    imports: str
    code: str
    iterations: int
    error: str
    task: str
    messages: List
    reflections: str


def extract_code(text):
    pattern = r"```(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0]


class FormattedOutput:
    def __init__(self, output):
        self.description = output['description']
        self.imports = extract_code(output['imports'])
        self.code = extract_code(output['code'])
        self.solution = extract_code(output['solution'])


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

    
class CodeAssistant:
    def __init__(self, llm, context="", max_iterations=5, reflect=False, check_code=False):
        self.compiled = False
        self.workflow = None
        self.context = context
        
        self.llm = llm
        self.max_iterations = max_iterations
        self.reflect = reflect
        self.check_code = check_code
        # self.retry_cahin = retry_prompt | self.llm # {"task", "reflections", "tool_calls"}

        self.solution_gen_chain = solution_prompt| llm
        self.solution_gen_chain_reflect = solution_prompt_reflect | llm
        self.description_gen_chain = description_prompt | llm
        self.imports_gen_chain = imports_prompt | llm
        self.code_gen_chain = code_prompt | llm
        self.workflow = StateGraph(GraphState)

    def invoke(self, task):
        if not self.compiled:
            app = self.compile()
        else:
            app = self.workflow
        return app.invoke({"context":concatenated_content, "task": task, "iterations": 0, "messages":[]})

    def compile(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("generate_node", self.generation_node)
        workflow.add_node("description_node", self.description_node)
        workflow.add_node("imports_node", self.imports_node)
        workflow.add_node("code_node", self.code_node)
        workflow.add_node("code_check_node", self.code_check_node)
        workflow.add_node("reflect", self.reflect_node)
        
        # Build graph
        workflow.add_edge(START, "generate_node")
        workflow.add_edge("generate_node", "description_node")
        workflow.add_edge("description_node", "imports_node")
        workflow.add_edge("imports_node", "code_node")
        # workflow.add_edge("code_node", END)
        workflow.add_edge("code_node", "code_check_node")
        workflow.add_conditional_edges(
            "code_check_node",
            self.decide_to_finish,
            {
                "end": END,
                "reflect": "reflect",
                "retry": "description_node",
            },
        )
        workflow.add_edge("reflect", "generate_node")

        
        self.workflow = workflow.compile()
        self.compiled = True
        return self.workflow
        
    def generation_node(self, state: GraphState):
        """
        Generate a code solution
    
        Args:
            state (dict): The current graph state
    
        Returns:
            state (dict): New key added to state, generation
        """
    
        print("---GENERATING SOLUTION---")
    
        # State
        task = state["task"]
        iterations = state["iterations"]
        reflections = state["reflections"]
        error = state["error"]
        messages = state["messages"]
    
        # Increment
        iterations = iterations + 1
        
        # We have been routed back to generation with an error
        if error == "yes":
            messages += [
                (
                    "user",
                    "Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:",
                )
            ]
            # Solution
            code_solution = self.solution_gen_chain_reflect.invoke({"context":self.context, "task": task, "reflections": reflections}).content
        else:
            code_solution = self.solution_gen_chain.invoke({"context":self.context, "task": task}).content
        
        return {**state, "solution": code_solution, "iterations": iterations, "messages": messages}


    def description_node(self, state: GraphState):
        """
        Generate a code description
    
        Args:
            state (dict): The current graph state
    
        Returns:
            state (dict): New key added to state, generation
        """
    
        print("---GENERATING CODE DESCRIPTION---")
    
        # State
        task = state["task"]
        solution = state["solution"]
    
        # Description
        code_description = self.description_gen_chain.invoke(
            {"solution": solution, "task": task}
        ).content
    
        return {**state, "description": code_description}
    
    
    def imports_node(self, state: GraphState):
        """
        Generate a code imports
    
        Args:
            state (dict): The current graph state
    
        Returns:
            state (dict): New key added to state, generation
        """
    
        print("---GENERATING IMPORTS BLOCK---")
    
        # State
        task = state["task"]
        solution = state["solution"]
    
        # Description
        code_imports = self.imports_gen_chain.invoke(
            {"solution": solution}
        ).content
    
        return {**state, "imports": code_imports}
    
    
    def code_node(self, state: GraphState):
        """
        Generate a code parts without imports
    
        Args:
            state (dict): The current graph state
    
        Returns:
            state (dict): New key added to state, generation
        """
    
        print("---GENERATING CODE BLOCK---")
    
        # State
        task = state["task"]
        solution = state["solution"]
    
        # Description
        code_block = self.code_gen_chain.invoke(
            {"solution": solution}
        ).content
    
        return {**state, "code": code_block}
    
    
    def code_check_node(self, state: GraphState):
        """
        Check code
    
        Args:
            state (dict): The current graph state
    
        Returns:
            state (dict): New key added to state, error
        """
        if not self.check_code:
            return {
                **state,
                "error": "no",
            }
            
        print("---CHECKING CODE---")
    
        # State
        imports = state["imports"]
        code = state["code"]
    
        # Check imports
        try:
            exec(imports)
        except Exception as e:
            print("---CODE IMPORT CHECK: FAILED---")
            error_message = [("user", f"Your solution failed the import test: {e}")]
            messages += error_message
            return {
                **state,
                "messages": messages,
                "error": "yes",
            }
    
        # Check execution
        try:
            exec(imports + "\n" + code)
        except Exception as e:
            print("---CODE BLOCK CHECK: FAILED---")
            error_message = [("user", f"Your solution failed the code execution test: {e}")]
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
        messages = state["messages"]
        solution = state["solution"]
        task = state["task"]
    
        # Prompt reflection
    
        # Add reflection
        reflections = reflect_gen_chain.invoke(
            {"task": task, "solution": solution, "messages": messages}
        ).content
        messages += [("assistant", f"Here are reflections on the error: {reflections}")]
        return {**state, "reflections": reflections, "messages": messages}
    
    
    ### Edges
    
    
    def decide_to_finish(self, state: GraphState):
        """
        Determines whether to finish.
    
        Args:
            state (dict): The current graph state
    
        Returns:
            str: Next node to call
        """
        error = state["error"]
        iterations = state["iterations"]
    
        if error == "no" or iterations == max_iterations:
            print("---DECISION: FINISH---")
            return "end"
        else:
            print("---DECISION: RE-TRY SOLUTION---")
            if self.reflect:
                return "reflect"
            else:
                return "retry"