# Check Langgraph/robust_react_agent.ipynb for detail
import logging
import re
from typing import List, TypedDict
from langgraph.graph import END, StateGraph, START
from langchain_core.prompts import ChatPromptTemplate

class GraphState(TypedDict):
    actions: list
    thoughts: list
    observations: list
    iterations: int
    error: str
    task: str
    messages: List
    tool_calls: list
    reflections: list
    answer: str


thinking_prompt = ChatPromptTemplate.from_messages( # {"task": task, "tools": tool_list, "reflection": reflections}
    [
        (
            "system",
            """You are an helpful AI assistant equipped with various tools to help answer questions and solve problems.
            You have access to the following tools:
            \n{tools}
            \nDo not make up any data.\n Try to use provided tools only. Keep your answer clear and concise. Here is the user question:
            \n{task}
            \n
            \nWhat you should do to solve this question?
            \n[YOUR ANSWER HERE]""",
        ),
    ]
)

finish_prompt = ChatPromptTemplate.from_messages( # {"task":task, "react_str": react_str}
    [
        (
            "system",
            """You are a helpful AI assistant with expertiese in summarizing. 
            The task provided by the user is here:
            \n
            \n{task}
            \n
            The reasonings and actions done by other agents.
            \n
            \n{react_str}
            \n
            Based on these results. Provide the final answer to the user's task. 
            Ensure that everything in your final answer is based on the previous results and that no data is fabricated.\n
            [YOUR ANSWER HERE]""",
        ),
    ]
)

reflect_prompt = ChatPromptTemplate.from_messages( # {"task", "error_message": messages[-1], "react_str": react_str}
    [
        (
            "system",
            """ You are a helpful reflection assistant with expertiese in Synergizing Reasoning and Acting in Language Models (ReAct), 
            You give insightful critique and recommendations to help other agents complete the task. 
            \nThe task:
            \n{task}
            \n
            \nThe ReAct processes:
            \n\n
            \n{react_str}
            \n\n
            Any error occured during the last Thought/Action/Observation cycle is listed below:
            \n\n{error_message}\n
            \nProvide a detailed critique and recommendations on the results above, and include comprehensive instructions to help the agent complete the task.
            \nIf you think the task is already completed, give your final answer to the task and add an __END__ flag to the end.
            \n\n[YOUR ANSWER HERE]""",
        ),
    ]
)


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

# if llm parameter is a dict. The agent will uset llm['llm'] for reasoning and llm['llm_with_tools'] for function calling
class ReAct:
    def __init__(self, llm, available_tools={}, max_iterations=3, 
                 prompts=None, verbose=False):
        logger = logging.getLogger()
        if verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.ERROR)
        
        self.compiled = False
        self.max_iterations = max_iterations
        self.workflow = None
        self.available_tools = available_tools
        self.tools = []
        for k in available_tools:
            self.tools.append(available_tools[k])
        if type(llm) == dict:
            self.llm = llm['llm']
            self.llm_with_tools = llm['llm_with_tools'].bind_tools(self.tools)
        else:
            self.llm = llm
            self.llm_with_tools = llm.bind_tools(self.tools)
        if prompts == None:
            self.prompts = {"thinking": thinking_prompt, "reflect": reflect_prompt, "finish":finish_prompt}
        self.thinking_chain = self.prompts['thinking'] | self.llm
        self.finish_chain = self.prompts['finish'] | self.llm # {"task", "tool_outputs"}
        self.reflect_chain = self.prompts['reflect'] | self.llm # {"messages": messages, "react_str": react_str}
        self.workflow = StateGraph(GraphState)

    def invoke(self, task):
        if not self.compiled:
            app = self.compile()
        else:
            app = self.workflow
        return app.invoke({"task": task, "iterations": 0, "messages":[], 
                           "thoughts": [], "actions": [], "observations": [], "reflections": [],
                           "tool_calls": [] })

    def compile(self):
        workflow = self.workflow
        workflow.add_node("thinking_node", self.thinking_node)
        workflow.add_node("action_node", self.action_node)
        workflow.add_node("execute_code_node", self.execute_code_node)
        workflow.add_node("observation_node", self.observation_node)
        workflow.add_node("reflect_node", self.reflect_node)
        workflow.add_node("finish_node", self.finish_node)
        
        # Build graph
        workflow.add_edge(START, "thinking_node")
        workflow.add_edge("thinking_node", "action_node")
        workflow.add_edge("action_node", "execute_code_node")
        # workflow.add_conditional_edges(
        #     "execute_code_node",
        #     self.execute_code_conditional_edges,
        #     {
        #         "observe": "observation_node",
        #         "retry": "action_node",
        #     },
        # )
        workflow.add_edge("execute_code_node", "observation_node")
        workflow.add_edge("observation_node", "reflect_node")
        workflow.add_conditional_edges(
            "reflect_node",
            self.decide_finish,
            {
                "finish": "finish_node",
                "retry": "thinking_node",
            },
        )
        workflow.add_edge("finish_node", END)
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
        thoughts = state["thoughts"]
        actions = state["actions"]
        observations = state["observations"]
        reflections = state["reflections"]

        react_str = ""
        react_str += f"[THOUGHT]: {thoughts[-1]}\n\n"
        react_str += f"[ACTION]: {actions[-1]}\n\n"
        react_str += f"[OBSERVATION]: {observations[-1]}\n\n"
        react_str += f"[REFLECTION]: {reflections[-1]}"
            
        # messages = state["messages"]
        finish_msg = self.finish_chain.invoke({"task":task, "react_str": react_str})
        logging.info(f"[ANSWER]: {finish_msg.content}")
            
        return {**state, "answer": finish_msg.content}

    def thinking_node(self, state: GraphState):
        """
        Find suitable tool to solve the problem
    
        Args:
            state (dict): The current graph state
    
        Returns:
            state (dict): New key added to state, generation
        """
        
        # State
        task = state["task"]
        iterations = state["iterations"]
        reflections = state["reflections"]
        thoughts = state["thoughts"]
        tool_list = ",".join([f"{tool.name} : {tool.description}\n" for tool in self.tools])
            
        # New Cycle
        iterations = iterations + 1
        error = "no" 
        
        thought_msg = self.thinking_chain.invoke({"task": task, "tools": tool_list, "reflection": reflections})
        thought = thought_msg.content
        thoughts += [thought]
        logging.info(f"[THOUGHT]: {thought}")
        
        return {**state, "thoughts": thoughts, "iterations": iterations, "error": error}

    def action_node(self, state: GraphState):
        """
        Find suitable tool to solve the problem
    
        Args:
            state (dict): The current graph state
    
        Returns:
            state (dict): New key added to state, generation
        """
            
        # State
        task = state["task"]
        iterations = state["iterations"]
        reflections = state["reflections"]
        thoughts = state["thoughts"]
        thought = thoughts[-1]
        actions = state["actions"]
        error = state["error"]
        messages = state["messages"]
    
        # if error == 'yes':
        #     # got error messages at execute_code_node
        #     thought = messages[-1] + f'Try again to complete the task: {thought}'
        tool_msg = self.llm_with_tools.invoke(thought)
        tool_calls = getattr(tool_msg,'tool_calls', False)
        logging.info(f"[ACTION]: {tool_calls}")
        if tool_calls:
            action = ""
            for tool in tool_calls:
                args = ""
                for arg in tool['args']:
                    args += f"{arg}={tool['args'][arg]}, "
                action += f"{tool['name']}({args}); "
            actions += [action]
            return {**state, "actions": actions, "tool_calls": tool_calls, "iterations": iterations}
        else:
            actions += ["no tool is used"]
            return {**state, "actions": actions, "tool_calls": [], "iterations": iterations}

    def execute_code_node(self, state: GraphState):
        """
        Check code
    
        Args:
            state (dict): The current graph state
    
        Returns:
            state (dict): New key added to state, error
        """        
        
        # State
        # task = state["task"]
        tool_calls = state["tool_calls"]
        messages = state["messages"]
        error = "no"
    
        error_message = "Your solution failed to execute:\n"
        for i in range(len(tool_calls)):
            tool = tool_calls[i]
            try:
                tool['output'] = self.available_tools[tool['name']].invoke(tool['args'])
            except Exception as e:
                logging.info(f"{tool['name']} ERROR: \n{e}\n")
                error_message += f"{tool['name']} ERROR: \n{e}\n"
                error = "yes"
                tool['output'] = "error occured"
            tool_calls[i] = tool
        if error == "yes":
            messages += [("user", error_message)]
            return {
                **state,
                "messages": messages,
                "tool_calls": tool_calls,
                "error": "yes",
            }
    
        # No errors
        messages += [("user", "No error occured.")]
        return {
            **state,
            "messages": messages,
            "error": "no",
            "tool_calls": tool_calls
        }
        
    def observation_node(self, state: GraphState):
        """
        Find suitable tool to solve the problem
    
        Args:
            state (dict): The current graph state
    
        Returns:
            state (dict): New key added to state, generation
        """
        # State
        task = state["task"]
        thoughts = state["thoughts"]
        observations = state["observations"]
        thought = thoughts[-1]
        tool_calls = state["tool_calls"]
        # messages = state["messages"]
        iterations = state["iterations"]

        # tool_outputs = {}
        # for tool in tool_calls:
        #     tool_outputs[tool['name']] = tool['output']
        observation = ""
        for tool in tool_calls:
            args = ""
            for arg in tool['args']:
                args += f"{arg}={tool['args'][arg]}, "
            observation += f"{tool['name']}({args})={tool['output']}; "
    
        # obs_msg = self.observation_chain.invoke({"task":task, "thought":thought, "tool_outputs": tool_outputs}).content

        observations += [observation]
        logging.info(f"[OBSERVATION]: {observation}")
        return {**state, "observation": observations}

        
    # Summarize thought1 observations1, thought2, observation2 .... to get a better result or instruction for the next round.
    def reflect_node(self, state: GraphState):
        """
        Reflect on errors
    
        Args:
            state (dict): The current graph state
    
        Returns:
            state (dict): New key added to state, generation
        """
    
        # State
        task = state["task"]
        thoughts = state["thoughts"]
        actions = state["actions"]
        observations = state["observations"]
        reflections = state["reflections"]
        messages = state["messages"]
        error = state["error"]
        # if self.history_cut < self.iterations:
        #     thoughts = thoughts[-self.history_cut,:]
        #     actions = actions[-self.history_cut,:]
        #     observations = observations[-self.history_cut,:]

        react_str = ""
        for i in range(len(thoughts)):
            react_str += f"[THOUGHT{i}]: {thoughts[i]}\n\n"
            react_str += f"[ACTION{i}]: {actions[i]}\n\n"
            react_str += f"[OBSERVATION{i}]: {observations[i]}\n\n"
            react_str += f"[REFLECTION{i}]: {observations[i]}" if i < (len(thoughts)-1) else ""
        # Add reflection
        reflection = self.reflect_chain.invoke(
            {"error_message": messages[-1], "react_str": react_str, "task": task}
        ).content
        
        reflections += [reflection]
        logging.info(f"[REFLECTION]: {reflection}")
        return {**state, "reflections": reflections}

    ### Edges
    def decide_finish(self, state: GraphState):
        """
        Determines whether to reflect.
    
        Args:
            state (dict): The current graph state
    
        Returns:
            str: Next node to call
        """

        # error = state["error"]
        iterations = state["iterations"]
        reflection = state["reflections"][-1]
    
        if "__END__" in reflection or iterations == self.max_iterations:
            logging.info("---FINISH---")
            return "finish"
        else:
            return "retry"
