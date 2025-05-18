import yaml
import os
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
# Import custom modules
from apicall import get_embedding, get_reply, Reply
from tools import (
    DataframeLoader, ContentExtractor, EmojiTranslator, EmojiMixer,
    KeypointExtractor, ContentExpander, TextExtractor, ContentReformatter,
    MultilingualTranslator, CosineSimilarityCalculator, StylePreserver, AdvanceCSVQuery
)

from prompts import NORMAL_CHAT, DETERMINE_TOOLS,TOOLS_REQUIRED, VALID_JSON, TASK_COMPLETION

# history
history = []

import yaml
import os
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
# Import custom modules
from apicall import get_embedding, get_reply, Reply
from tools import (
    DataframeLoader, ContentExtractor, EmojiTranslator, EmojiMixer,
    KeypointExtractor, ContentExpander, TextExtractor, ContentReformatter,
    MultilingualTranslator, CosineSimilarityCalculator, StylePreserver, AdvanceCSVQuery
)

from prompts import NORMAL_CHAT, DETERMINE_TOOLS,TOOLS_REQUIRED, VALID_JSON, TASK_COMPLETION

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

    # if greates similarity is less than 0.5, return empty list
    if sim_scores[0]['similarity'] < 0.5:
        class QA(BaseModel):
            """Model for storing answer."""
            answer: str
        message = [
            {
                "role": "system",
                "content": NORMAL_CHAT.format(agents)
            },           
            {
                "role": "user",
                "content": f"Query: {query}."
            }
        ]
        reply = get_reply(message, QA)
        print(reply.answer)
        history.append({"role": "system", "content": reply.answer})
        return []
    
    return sim_scores[:top_n]

class ToolsSeqFinder(BaseModel):
    """Model for finding the best sequence of tools to complete a task."""
    tools_name_in_seq: List[str]

# Function to determine the best tool sequence for a task
def determine_tool_sequence(agent, query):
    """Determine the best sequence of tools to use for completing a task."""
    # Include conversation history context for better tool sequence determination
    context_summary = "\n\nHistory" + str(history)
    
    message = [
        {
            "role": "system",
            "content": DETERMINE_TOOLS.format(agent,context_summary)
            #f"You have to find the best sequence for list of tools to complete the task. Available tools: {agent}"
            #f"\n\nConversation context: {context_summary}"
        },
        {
            "role": "user",
            "content": query
        }
    ]
    
    tools_order = get_reply(message, ToolsSeqFinder)
    # Add tool sequence to history
    history.append({"role": "system", "content": f"Tools sequence determined: {tools_order.tools_name_in_seq}"})
    
    return tools_order.tools_name_in_seq

# Function to map tool names to actual functions
def get_tool_function(tool_name):
    """Map a tool name to its actual function implementation."""
    # Map tools to their implementations
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
        "StylePreserver": StylePreserver.preserve_style,
        "AdvanceCSVQuery": AdvanceCSVQuery.query_dataframe,
    }
    
    return tool_map.get(tool_name)


import inspect
# Function to get input schema for a tool function
def get_input_schema(func):
    """Get the input schema for a function based on its signature."""
    signature = inspect.signature(func)
    schema = {}
    for param in signature.parameters.values():
        schema[param.name] = str(param.annotation)
    return schema


import json
# Class for gathering information required by tools
class ToolsInput(BaseModel):
    """Model for gathering tool input information."""
    information_tillnow: str
    all_information_gathered: bool
    flow_of_question: str

# Improved function to gather input information for a tool
# previous_outputs=tool_output, history=history
def gather_tool_inputs(tool_name, tool_function,history,previous_outputs, context=""):
    """
    Gather inputs required for a specific tool by asking user questions.
    Returns a dictionary of inputs for the tool.
    """
    # Get input schema for the tool function
    input_schema = get_input_schema(tool_function)
    
    # Get relevant context from conversation history
    full_context = str(history)

    
    # Create initial message
    message = [
        {
            "role": "system",
            "content": f"You are the {tool_name} tool. \n\n You have to ask for the details required to complete the task. The tool required is {tool_name} with inputs: {input_schema}. If required is path, send actual path.\n\n History of questions: {str(history)}\n\n Context from previous tools: {previous_outputs}\n\n Context from conversation: {context}\n\n"
            # TOOLS_REQUIRED.format(tool_name, input_schema, full_context)
            
        },
        {
            "role": "user",
            "content": f"User's query: {context}. \n\n\n Here is the past history:- {str(history)}"
        }
    ]
    history.append({"role": "system", "content": message[0]['content']})
    history.append({"role": "user", "content": message[1]['content']})
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
            history.append({"role": "system", "content": reply.flow_of_question})
            print(f"Flow of question: {reply.flow_of_question}")
            # Get input from user
            query = input(f"[Tool: {tool_name}] {reply.flow_of_question} ")
            history.append({"role": "user", "content": query})
            message[1]['content'] = query + f"\n\n\nHistory: {str(history)}"

    
    # Debug print
    print(f"Information gathered: {reply.information_tillnow}")
    history.append({"role": "system", "content": f"Information gathered: {reply.information_tillnow}"})
    
    # Convert user input to function input format
    message = [
        {
            "role": "system",
            "content": VALID_JSON.format(input_schema)
            # f"Convert the user's input into a valid JSON object that matches this function schema: {input_schema}. Return ONLY the JSON object and nothing else."
        },
        {
            "role": "user",
            "content": f"User's input: {reply.information_tillnow}. Create a JSON object that matches the function input schema. \n\n\nHistory: {str(history)}"
        }
    ]
    
    class FunctionInput(BaseModel):
        function_input: str
    
    function_input = get_reply(message, FunctionInput)
    
    # Parse the function input to ensure it's a proper dict
    try:
        if isinstance(function_input.function_input, str):
            parsed_input = json.loads(function_input.function_input)
        else:
            parsed_input = function_input.function_input
    except json.JSONDecodeError:
        # If JSON parsing fails, try YAML parsing (more lenient)
        try:
            parsed_input = yaml.safe_load(function_input.function_input)
        except yaml.YAMLError:
            # If all parsing fails, use as is
            parsed_input = function_input.function_input
    
    # Add tool input to history
    history.append({"role": "system", "content": f"Tool input for {tool_name}: {parsed_input}"})
    
    return parsed_input


# Improved function to execute a tool with given inputs
def execute_tool(tool_name, inputs):
    """Execute a tool with the given inputs and return the output."""
    tool_function = get_tool_function(tool_name)
    print(f"Tool function for {tool_name}: {tool_function}")
    if not tool_function:
        error_msg = f"Error: Tool '{tool_name}' not found."
        history.append({"role": "system", "content": error_msg})
        print(error_msg)
        return error_msg
    
    # Debug information
    print(f"Executing {tool_name} with inputs: {inputs}")
    
    try:
        # Check if inputs is a string and convert to dict if necessary
        if isinstance(inputs, str):
            try:
                inputs = json.loads(inputs)
            except json.JSONDecodeError:
                try:
                    inputs = yaml.safe_load(inputs)
                except yaml.YAMLError:
                    raise ValueError("Could not parse inputs as JSON or YAML.")
        elif not isinstance(inputs, dict):
            raise ValueError("Inputs must be a dictionary or a parseable JSON/YAML string.")
        
        # Call the tool function with the provided inputs
        output = tool_function(**inputs)
        
        # Add tool output to history
        history.append({"role": "system", "content": f"Tool output for {tool_name}: {output}"})
        print(f"Tool output for {tool_name}: {output}")
        
        return output
    except Exception as e:
        error_msg = f"Error executing tool '{tool_name}': {str(e)}"
        history.append({"role": "system", "content": error_msg})
        return error_msg
    

# Main function to handle user queries and tool execution
def handle_query(query):
    """Handle user query, find relevant agents, determine tool sequence, and execute tools."""
    history.append({"role": "user", "content": query})

    # Find the most relevant agents for the query
    relevant_agents = []
    # equal to 0 or data type is not list
    while len(relevant_agents) == 0 or not isinstance(relevant_agents, list):
        relevant_agents = find_relevant_agents(query, agents_with_embeddings)
        # conversation_history.add_system_message(relevant_agents)
        if len(relevant_agents) == 0:
            query = input("Enter your query ")
            # append conversation history with the query
            history.append({"role": "user", "content": query})
            query = query + f"\n\n\nHere is the past history:- {str(history)}"

    top_agent = relevant_agents[0]['agent']
    similarity = relevant_agents[0]['similarity']
    print(f"Most relevant agent: {top_agent.name} with similarity score: {similarity}")
    print('\n\n')
    history.append({"role": "system", "content": f"Most relevant agent: {top_agent.name} with similarity score: {similarity}"})

    # Determine the best sequence of tools
    tool_sequence = determine_tool_sequence(top_agent, query)
    print(f"Tool sequence: {tool_sequence}")
    print('\n\n')

    tool_output = []
    for i, tool_name in enumerate(tool_sequence):
        print(f"Executing tool {i+1}: {tool_name}")
        print('\n\n')
        tool_function = get_tool_function(tool_name)
        
        # Gather inputs for the tool
        tool_inputs = gather_tool_inputs(tool_name, tool_function, context=query, previous_outputs=tool_output, history=history)
        
        # Execute the tool with the gathered inputs
        output = execute_tool(tool_name, tool_inputs)
        
        # Add output to history and tool_output
        history.append({"role": "system", "content": f"Tool {i+1} ({tool_name}) output: {output}"})
        tool_output.append(output)

    # Final output after executing all tools
    print("Final output after executing all tools:")
    print('\n\n')
    for i, output in enumerate(tool_output):
        print(f"Output from tool {tool_sequence[i]}: {output}")
        print('\n\n')
    # Add final output to history
    history.append({"role": "system", "content": f"Final output after executing all tools: {tool_output}"})

    # save history to json file
    import json
    with open('conversation_history.json', 'w') as f:
        json.dump(history, f, indent=4)

# Example usage
if __name__ == "__main__":
    user_query = input("Enter your query: ")
    handle_query(user_query)
    print("Conversation history saved to conversation_history.json")