# %%
# Import required libraries
import yaml
import os
import inspect
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import numpy as np

# Import custom modules
from apicall import get_embedding, get_reply, Reply
from tools import (
    DataframeLoader, ContentExtractor, EmojiTranslator, EmojiMixer,
    KeypointExtractor, ContentExpander, TextExtractor, ContentReformatter,
    MultilingualTranslator, CosineSimilarityCalculator,StylePreserver
)

# %%
# Function to load YAML configuration file
def load_yaml(file_path):
    """Load and parse a YAML file."""
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return None
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            return data
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None

# %%
# Load the agent configuration from YAML
agents_config = load_yaml('agents_behaviour.yaml')

# %%
# Extract agent information
class AgentInfo(BaseModel):
    """Model for storing agent information."""
    name: str
    description: str
    id: str
    tools: List[Dict[str, Any]]
    parameters: Optional[Dict[str, Any]] = None
    embedding: Optional[Any] = None

def extract_agent_info(agents_data):
    """Extract detailed information for each agent."""
    agents = []
    for agent_data in agents_data:
        agent = AgentInfo(
            name=agent_data.get('name', 'Unknown'),
            description=agent_data.get('description', 'No description provided'),
            id=agent_data.get('id', 'unknown-id'),
            tools=agent_data.get('tools', []),
            parameters=agent_data.get('parameters', {})
        )
        agents.append(agent)
    return agents

# Create agent information objects
agents = extract_agent_info(agents_config)

# %%
# Generate embeddings for each agent based on name and description
def generate_agent_embeddings(agents_list):
    """Generate embeddings for each agent based on their name and description."""
    for agent in agents_list:
        embedding = get_embedding(f"Name: {agent.name}.\nDescription: {agent.description}")
        agent.embedding = embedding
    return agents_list

# Generate embeddings for all agents
agents_with_embeddings = generate_agent_embeddings(agents)

# %%
# Function to find the most relevant agent based on a query
def find_relevant_agents(query, agents_list, top_n=3):
    """
    Find the most relevant agents based on the query embedding.
    Returns the top N agents sorted by similarity score.
    """
    query_embedding = get_embedding(query)
    sim_scores = []
    
    for agent in agents_list:
        similarity = CosineSimilarityCalculator.calculate_similarity(
            embedding1=query_embedding, 
            embedding2=agent.embedding
        )
        sim_scores.append({
            'agent': agent,
            'similarity': similarity
        })
    
    # Sort agents by similarity score in descending order
    sim_scores.sort(key=lambda x: x['similarity'], reverse=True)
    
    return sim_scores[:top_n]

# %%
# Function to get input schema for a tool function
def get_input_schema(func):
    """Get the input schema for a function based on its signature."""
    signature = inspect.signature(func)
    schema = {}
    for param in signature.parameters.values():
        schema[param.name] = str(param.annotation)
    return schema

# Function to map tool names to actual functions
# Updated function to map tool names to actual functions
def get_tool_function(tool_name):
    """Map a tool name to its actual function implementation."""
    # First define all tool methods with appropriate wrappers for error handling
    # Map tools to their wrapped implementations
    tool_map = {
        "DataframeLoader": DataframeLoader.load_csv,
        "ContentExtractor": ContentExtractor.extract_narrative,
        "EmojiTranslator": EmojiTranslator.translate_to_emoji,
        "EmojiMixer": EmojiMixer.create_emoji_mix,
        "KeypointExtractor": KeypointExtractor.extract_keypoints,
        "ContentExpander": ContentExpander().expand_content,
        "TextExtractor": TextExtractor.extract_text_from_pdf,
        "ContentReformatter": ContentReformatter.reformat_content,
        "MultilingualTranslator": MultilingualTranslator().translate,
        "StylePreserver": StylePreserver().preserve_style,
    }
    
    
    return tool_map.get(tool_name)

# %%
# Class for handling tool sequence generation
class ToolsSeqFinder(BaseModel):
    """Model for finding the best sequence of tools to complete a task."""
    tools_name_in_seq: List[str]

# Function to determine the best tool sequence for a task
# Function to determine the best tool sequence for a task
def determine_tool_sequence(agent, query):
    """Determine the best sequence of tools to use for completing a task."""
    # tool_names = [tool['name'] for tool in agent.tools]
    
    message = [
        {
            "role": "system",
            "content": f"You have to find the best sequence for list of tools to complete the task. Available tools: {agent}"
        },
        {
            "role": "user",
            "content": query
        }
    ]
    
    tools_order = get_reply(message, ToolsSeqFinder)
    # Access the attribute directly from the Pydantic model
    return tools_order.tools_name_in_seq

# %%
# Class for gathering information required by tools
class ToolsInput(BaseModel):
    """Model for gathering tool input information."""
    information_tillnow: str
    all_information_gathered: bool
    flow_of_question: str

# Improved function to gather input information for a tool
def gather_tool_inputs(tool_name, tool_function, context="", previous_outputs=None):
    """
    Gather inputs required for a specific tool by asking user questions.
    Returns a dictionary of inputs for the tool.
    """
    if previous_outputs is None:
        previous_outputs = {}
    
    # Get input schema for the tool function
    input_schema = get_input_schema(tool_function)
    
    history = ''
    
    # Create initial message
    message = [
        {
            "role": "system",
            "content": f"You have to ask for the details of the tools required to complete the task. The tool required is {tool_name} with inputs: {input_schema}."
        },
        {
            "role": "user",
            "content": f"I want to use the {tool_name} tool. {context}"
        }
    ]
    
    # Set initial state
    all_information_gathered = False
    
    # Interactive loop to gather all required inputs
    while not all_information_gathered:
        # Get the reply with questions
        reply = get_reply(message, ToolsInput)
        
        # Use all_information_gathered attribute directly from the Pydantic model
        all_information_gathered = reply.all_information_gathered
        
        if all_information_gathered:
            break
        else:
            # Update the message with the flow of question
            history = history + f"Context:{context}" + reply.flow_of_question + '\n'
            message[0]['content'] = f"You have to ask for the details required to complete the task. The tool required is {tool_name} with inputs: {input_schema}. History of questions: {history}"
            
            # Get input from user
            query = input(f"[Tool: {tool_name}] {reply.flow_of_question} ")
            message[1]['content'] = query + f"History of questions: {history}"

            # append the user query to the history
            history += f"User input: {query}\n"
    
    # Debug print
    print(f"Information gathered: {reply.information_tillnow}")
    
    
    # Convert user input to function input format
    message = [
        {
            "role": "system",
            "content": f"Convert the user's input into a valid JSON object that matches this function schema: {input_schema}. Return ONLY the JSON object and nothing else."
        },
        {
            "role": "user",
            "content": f"User's input: {reply.information_tillnow}. Create a JSON object that matches the function input schema."
        }
    ]
    
    class FunctionInput(BaseModel):
        function_input: str
    
    function_input = get_reply(message, FunctionInput)
    
    return function_input.function_input

# %%
# Improved function to execute a tool with given inputs
def execute_tool(tool_name, inputs):
    """Execute a tool with the given inputs and return the output."""
    # tool_function = get_tool_function(tool_name)
    tool_function = get_tool_function(tool_name)
    print(f"Tool function for {tool_name}: {tool_function}")
    if not tool_function:
        return f"Error: Tool '{tool_name}' not found."
    
    # Debug information
    print(f"Executing {tool_name} with inputs: {inputs}")
    
    try:
        # Check if inputs is a string and convert to dict if necessary
        if isinstance(inputs, str):
            inputs = yaml.safe_load(inputs)
        elif not isinstance(inputs, dict):
            raise ValueError("Inputs must be a dictionary or a YAML string.")
        # Call the tool function with the provided inputs
        output = tool_function(**inputs)
        return output
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"

# %%
# Function to handle process flow between tools
def process_user_query(query):
    """
    Process a user query by finding relevant agents and executing their tools.
    
    Args:
        query: User's query or request
        
    Returns:
        Dict containing the processing results
    """
    # Find the most relevant agents for the query
    relevant_agents = find_relevant_agents(query, agents_with_embeddings)
    
    if not relevant_agents:
        return {"error": "No relevant agents found for the query."}
    
    # Get the most relevant agent
    top_agent = relevant_agents[0]['agent']
    print(f"Selected agent: {top_agent.name} (Similarity: {relevant_agents[0]['similarity']:.4f})")
    print(f"Description: {top_agent.description}")
    
    # Determine the best sequence of tools
    tool_sequence = determine_tool_sequence(top_agent, query)
    print(f"Tool sequence: {tool_sequence}")
    
    # Execute tools in sequence
    results = {}
    context = query
    shared_state = {}  # Shared state to pass data between tools
    
    for i, tool_name in enumerate(tool_sequence):
        print(f"\nExecuting tool {i+1}/{len(tool_sequence)}: {tool_name}")
        
        # Get the tool function
        tool_function = get_tool_function(tool_name)
        if not tool_function:
            results[tool_name] = f"Error: Tool '{tool_name}' not found."
            continue
        
        # Update context with previous results
        tool_context = context
        if i > 0:
            prev_tool = tool_sequence[i-1]
            if prev_tool in results:
                tool_context += f"\nOutput from previous tool ({prev_tool}): {results[prev_tool]}"
        
        # Gather inputs for the tool
        tool_inputs = gather_tool_inputs(tool_name, tool_function, tool_context, results)
        
        # Store the inputs in shared state for later use
        shared_state[tool_name] = tool_inputs
        
        # Execute the tool
        output = execute_tool(tool_name, tool_inputs)
        results[tool_name] = output
        
        print(f"Tool {tool_name} completed.")
    
    return {
        "agent": top_agent.name,
        "tool_sequence": tool_sequence,
        "results": results
    }
# %%
# Example usage of the complete workflow
if __name__ == "__main__":
    # Example query
    user_query = input("What would you like to do? ")
    
    # Process the query
    result = process_user_query(user_query)
    
    # Print the final result
    print("\nFinal Results:")
    print(f"Agent: {result['agent']}")
    print("Tool sequence:", ", ".join(result['tool_sequence']))
    print("\nOutputs:")
    for tool_name, output in result['results'].items():
        print(f"\n--- {tool_name} output ---")
        if isinstance(output, str) and len(output) > 500:
            print(output[:500] + "... (truncated)")
        else:
            print(output)
