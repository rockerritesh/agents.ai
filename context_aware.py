# %%
# Import required libraries
import yaml
import os
import inspect
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import numpy as np
import json
from datetime import datetime

# Import custom modules
from apicall import get_embedding, get_reply, Reply
from tools import (
    DataframeLoader, ContentExtractor, EmojiTranslator, EmojiMixer,
    KeypointExtractor, ContentExpander, TextExtractor, ContentReformatter,
    MultilingualTranslator, CosineSimilarityCalculator, StylePreserver
)

from prompts import NORMAL_CHAT, DETERMINE_TOOLS,TOOLS_REQUIRED, VALID_JSON, TASK_COMPLETION

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
# Define a ConversationHistory class to track interaction history
class ConversationHistory:
    """
    Class to maintain conversation history and context throughout the process.
    """
    def __init__(self):
        self.history = []
        self.tool_outputs = {}
        self.agent_used = None
        self.current_query = None
        self.session_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def add_user_query(self, query):
        """Add user query to history."""
        self.current_query = query
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "user_query",
            "content": query
        }
        self.history.append(entry)
        return entry
    
    def add_agent_selection(self, agent_name, similarity_score):
        """Add agent selection to history."""
        self.agent_used = agent_name
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "agent_selection",
            "agent": agent_name,
            "similarity_score": similarity_score
        }
        self.history.append(entry)
        return entry
    
    def add_tool_sequence(self, tool_sequence):
        """Add selected tool sequence to history."""
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "tool_sequence",
            "tools": tool_sequence
        }
        self.history.append(entry)
        return entry
    
    def add_tool_input(self, tool_name, inputs):
        """Add tool input to history."""
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "tool_input",
            "tool": tool_name,
            "inputs": inputs
        }
        self.history.append(entry)
        return entry
    
    def add_tool_output(self, tool_name, output):
        """Add tool output to history and tool_outputs dictionary."""
        # Store in the tool_outputs dictionary for easy access
        self.tool_outputs[tool_name] = output
        
        # Add to chronological history
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "tool_output",
            "tool": tool_name,
            "output": output if isinstance(output, str) else str(output)[:500] + "..." if len(str(output)) > 500 else str(output)
        }
        self.history.append(entry)
        return entry
    
    def add_system_message(self, message):
        """Add system message to history."""
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "system_message",
            "message": message
        }
        self.history.append(entry)
        return entry
    
    def get_full_history(self):
        """Get the full conversation history."""
        return self.history
    
    def get_context_for_tool(self, tool_name, max_entries=10):
        """
        Get relevant context for a specific tool.
        Returns a formatted string with relevant history.
        """
        # Start with the original query
        context = f"Original query: {self.current_query}\n\n"
        
        # Add relevant tool outputs that might be useful for this tool
        context += "Previous tool outputs:\n"
        for name, output in self.tool_outputs.items():
            if name != tool_name:  # Don't include the tool's own previous output
                output_str = str(output)
                # Truncate long outputs
                if len(output_str) > 300:
                    output_str = output_str[:300] + "... (truncated)"
                context += f"- {name}: {output_str}\n"
        
        # Add recent relevant history entries
        recent_entries = []
        for entry in reversed(self.history[-max_entries:]):
            if entry["type"] in ["user_query", "tool_output"]:
                if entry["type"] == "user_query":
                    recent_entries.append(f"User asked: {entry['content']}")
                elif entry["type"] == "tool_output" and entry.get("tool") != tool_name:
                    tool = entry.get("tool")
                    output = entry.get("output", "")
                    if len(output) > 200:
                        output = output[:200] + "... (truncated)"
                    recent_entries.append(f"Tool {tool} produced: {output}")
        
        if recent_entries:
            context += "\nRecent relevant interactions:\n"
            context += "\n".join(reversed(recent_entries[-5:]))  # Show at most 5 recent entries
            
        return context
    
    def save_to_file(self, filename="conversation_history.json"):
        """Save the conversation history to a JSON file."""
        with open(filename, 'w') as f:
            json.dump({
                "session_start": self.session_start_time,
                "current_query": self.current_query,
                "agent_used": self.agent_used,
                "history": self.history
            }, f, indent=2)
        print(f"Conversation history saved to {filename}")

# Create a global conversation history object
conversation_history = ConversationHistory()

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

    # if greates similarity is less than 0.3, return empty list
    if sim_scores[0]['similarity'] < 0.4:
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
        return []
    
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
    }
    
    return tool_map.get(tool_name)

# %%
# Class for handling tool sequence generation
class ToolsSeqFinder(BaseModel):
    """Model for finding the best sequence of tools to complete a task."""
    tools_name_in_seq: List[str]

# Function to determine the best tool sequence for a task
def determine_tool_sequence(agent, query):
    """Determine the best sequence of tools to use for completing a task."""
    # Include conversation history context for better tool sequence determination
    context_summary = ""
    if conversation_history.history:
        # Summarize recent tool outputs if available
        if conversation_history.tool_outputs:
            context_summary += "Previous tool outputs:\n"
            for tool_name, output in conversation_history.tool_outputs.items():
                if isinstance(output, str):
                    # Truncate long outputs
                    output_summary = output[:100] + "..." if len(output) > 100 else output
                    context_summary += f"- {tool_name}: {output_summary}\n"
    
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
    conversation_history.add_tool_sequence(tools_order.tools_name_in_seq)
    
    return tools_order.tools_name_in_seq

# %%
# Class for gathering information required by tools
class ToolsInput(BaseModel):
    """Model for gathering tool input information."""
    information_tillnow: str
    all_information_gathered: bool
    flow_of_question: str

# Improved function to gather input information for a tool
def gather_tool_inputs(tool_name, tool_function, context=""):
    """
    Gather inputs required for a specific tool by asking user questions.
    Returns a dictionary of inputs for the tool.
    """
    # Get input schema for the tool function
    input_schema = get_input_schema(tool_function)
    
    # Get relevant context from conversation history
    history_context = conversation_history.get_context_for_tool(tool_name)
    full_context = f"{context}\n\n{history_context}" if context else history_context
    
    history = ''
    
    # Create initial message
    message = [
        {
            "role": "system",
            "content": TOOLS_REQUIRED.format(tool_name, input_schema, full_context)
            # f"You have to ask for the details of the tools required to complete the task. The tool required is {tool_name} with inputs: {input_schema}."
            # f"\n\nContext from previous interactions: {full_context}"
        },
        {
            "role": "user",
            "content": f"I want to use the {tool_name} tool."
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
            history = history + f"Context:{full_context}" + reply.flow_of_question + '\n'
            message[0]['content'] = TASK_COMPLETION.format(tool_name, input_schema, history)
            # f"You have to ask for the details required to complete the task. The tool required is {tool_name} with inputs: {input_schema}. History of questions: {history}"
            
            # Get input from user
            query = input(f"[Tool: {tool_name}] {reply.flow_of_question} ")
            message[1]['content'] = query + f"History of questions: {history}"

            # Add the user query to history with a special tag in conversation history
            conversation_history.add_system_message(f"User input for {tool_name}: {query}")
            
            # append the user query to the history
            history += f"User input: {query}\n"
    
    # Debug print
    print(f"Information gathered: {reply.information_tillnow}")
    
    # Convert user input to function input format
    message = [
        {
            "role": "system",
            "content": VALID_JSON.format(input_schema)
            # f"Convert the user's input into a valid JSON object that matches this function schema: {input_schema}. Return ONLY the JSON object and nothing else."
        },
        {
            "role": "user",
            "content": f"User's input: {reply.information_tillnow}. Create a JSON object that matches the function input schema."
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
    conversation_history.add_tool_input(tool_name, parsed_input)
    
    return parsed_input

# %%
# Improved function to execute a tool with given inputs
def execute_tool(tool_name, inputs):
    """Execute a tool with the given inputs and return the output."""
    tool_function = get_tool_function(tool_name)
    print(f"Tool function for {tool_name}: {tool_function}")
    if not tool_function:
        error_msg = f"Error: Tool '{tool_name}' not found."
        conversation_history.add_system_message(error_msg)
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
        conversation_history.add_tool_output(tool_name, output)
        
        return output
    except Exception as e:
        error_msg = f"Error executing tool '{tool_name}': {str(e)}"
        conversation_history.add_system_message(error_msg)
        return error_msg

# %%
# Function to create a summary of processing results
def create_results_summary(results, agent_name, tool_sequence):
    """Create a human-readable summary of processing results."""
    summary = f"Task Processing Summary:\n"
    summary += f"Agent: {agent_name}\n"
    summary += f"Tools used: {', '.join(tool_sequence)}\n\n"
    
    summary += "Results by tool:\n"
    for tool_name, output in results.items():
        summary += f"\n## {tool_name}\n"
        if isinstance(output, str):
            # Limit the length of the output in the summary
            if len(output) > 500:
                summary += output[:500] + "... (truncated)"
            else:
                summary += output
        else:
            summary += str(output)[:500] + "..." if len(str(output)) > 500 else str(output)
    
    # Add the summary to conversation history
    conversation_history.add_system_message("Process completed - Summary generated")
    
    return summary

# %%
# Improved function to handle process flow between tools
def process_user_query(query):
    """
    Process a user query by finding relevant agents and executing their tools.
    
    Args:
        query: User's query or request
        
    Returns:
        Dict containing the processing results
    """
    # Add user query to history
    conversation_history.add_user_query(query)
    
    
    # Find the most relevant agents for the query
    relevant_agents = []
    # equal to 0 or data type is not list
    while len(relevant_agents) == 0 or not isinstance(relevant_agents, list):
        relevant_agents = find_relevant_agents(query, agents_with_embeddings)
        conversation_history.add_system_message(relevant_agents)
        if len(relevant_agents) == 0:
            query = input("Enter your query ")
            # append conversation history with the query
            conversation_history.add_user_query(query)
            query = query + str(conversation_history.get_full_history())
    
    if not relevant_agents:
        error_msg = "No relevant agents found for the query."
        conversation_history.add_system_message(error_msg)
        return {"error": error_msg}
    
    # Get the most relevant agent
    top_agent = relevant_agents[0]['agent']
    similarity = relevant_agents[0]['similarity']
    print(f"Selected agent: {top_agent.name} (Similarity: {similarity:.4f})")
    print(f"Description: {top_agent.description}")
    
    # Add agent selection to history
    conversation_history.add_agent_selection(top_agent.name, similarity)
    
    # Determine the best sequence of tools
    tool_sequence = determine_tool_sequence(top_agent, query)
    print(f"Tool sequence: {tool_sequence}")
    
    # Execute tools in sequence
    results = {}
    context = query
    
    for i, tool_name in enumerate(tool_sequence):
        print(f"\nExecuting tool {i+1}/{len(tool_sequence)}: {tool_name}")
        
        # Get the tool function
        tool_function = get_tool_function(tool_name)
        if not tool_function:
            results[tool_name] = f"Error: Tool '{tool_name}' not found."
            conversation_history.add_system_message(f"Error: Tool '{tool_name}' not found.")
            continue
        if i >= 1:
            tool_context += f"\n\nPrevious tool outputs: {tool_context}"
        if i == 0:
            # Update context with previous results and conversation history
            tool_context = f"Original query: {context}\n\n"
        if i > 0:
            prev_tools = tool_sequence[:i]
            for prev_tool in prev_tools:
                if prev_tool in results:
                    prev_result = results[prev_tool]
                    prev_result_str = str(prev_result)
                    # Truncate long results for context clarity
                    if len(prev_result_str) > 300:
                        prev_result_str = prev_result_str#[:300] + "... (truncated)"
                    tool_context += f"Output from previous tool ({prev_tool}): {prev_result_str}\n\n"
        
        # Gather inputs for the tool
        tool_inputs = gather_tool_inputs(tool_name, tool_function, tool_context)
        
        # Execute the tool
        output = execute_tool(tool_name, tool_inputs)
        results[tool_name] = output
        
        print(f"Tool {tool_name} completed.")
    
    # Create a summary of results and add to history
    summary = create_results_summary(results, top_agent.name, tool_sequence)
    
    # Save conversation history to file
    # conversation_history.save_to_file()
    
    return {
        "agent": top_agent.name,
        "tool_sequence": tool_sequence,
        "results": results,
        "summary": summary
    }

# %%
# Example usage of the complete workflow
if __name__ == "__main__":
    while True:
        # Example query
        user_query = input("\nWhat would you like to do? (type 'exit' to quit): ")
        
        if user_query.lower() == 'exit':
            print("Exiting the system. Conversation history has been saved.")
            break
        
        # Process the query
        result = process_user_query(user_query)
        
        # Print the final result
        print("\nFinal Results:")
        print(f"Agent: {result['agent']}")
        print("Tool sequence:", ", ".join(result['tool_sequence']))
        print("\nSummary:")
        print(result.get('summary', 'No summary available'))
        
        # Ask if user wants to continue with another query
        continue_query = input("\nDo you want to continue with another query? (y/n): ")
        if continue_query.lower() != 'y':
            print("Exiting the system. Conversation history has been saved.")
            break
