import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import re
import os
import sys
from IPython.display import display, HTML
import io
import base64

class CSVAnalyzer:
    def __init__(self, api_key=None):
        """Initialize the CSV Analyzer with optional API key for the language model."""
        self.df = None
        self.csv_info = {}
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("Warning: No API key provided. You'll need to set it before calling the model.")
        # Initialize conversation history
        self.conversation_history = [
            {"role": "system", "content": "You are a data analysis assistant. Respond with executable Python code."}
        ]
    
    def load_csv(self, file_path):
        """Load a CSV file and return basic information about it."""
        try:
            self.df = pd.read_csv(file_path)
            print(f"Successfully loaded CSV file: {file_path}")
            print(f"Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return self.df
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None
    
    def analyze_csv(self):
        """Analyze the CSV and gather basic information about it."""
        if self.df is None:
            print("No CSV file loaded. Please load a CSV file first.")
            return None
        
        # Capture information in a buffer to include in the prompt
        buffer = io.StringIO()
        sys.stdout = buffer
        
        print("CSV ANALYSIS SUMMARY")
        print("=" * 50)
        
        print("\nFIRST 5 ROWS:")
        print(self.df.head())
        
        print("\nCOLUMNS:")
        for col in self.df.columns:
            print(f"- {col}")
        
        print("\nDATA TYPES:")
        print(self.df.dtypes)
        
        print("\nSUMMARY STATISTICS:")
        print(self.df.describe())
        
        print("\nMISSING VALUES:")
        print(self.df.isnull().sum())
        
        print("\nINFORMATION:")
        self.df.info(buf=buffer)
        
        # Restore standard output
        csv_info = buffer.getvalue()
        ## append df.info, df.describe(), df.head(), df.columns, df.dtypes, df.isnull().sum() in csv_info
        
        csv_info  = f"self.df.info()\n{self.df.describe()}\n{self.df.head()}\n{self.df.columns}\n{self.df.dtypes}\n{self.df.isnull().sum()}"
        buffer.close()

        sys.stdout = sys.__stdout__
        
        self.csv_info = csv_info
        # Add the CSV info to the conversation history
        self.conversation_history.append({
            "role": "user", 
            "content": f"I have a pandas DataFrame with the following information:\n\n{csv_info}"
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": "I've analyzed your CSV data. What would you like to know about it?"
        })
        
        return csv_info
    
    def query_model(self, user_query, error_message=None):
        """Query the language model with the CSV info and user query."""
        if not self.api_key:
            print("No API key provided. Please set the API key before querying the model.")
            return None
        
        if not self.csv_info and not self.conversation_history:
            print("No CSV analysis available. Please run analyze_csv() first.")
            return None
        
        # Build the prompt for the model
        if error_message:
            prompt = f"""
USER QUERY: {user_query}

The previous code generated resulted in the following error:
{error_message}

Please provide corrected Python code that addresses this query using the DataFrame (assume it's named 'df').
Put the code between ```python and ``` tags.
Include only the necessary code to accomplish the task. Assume pandas, matplotlib, seaborn, and numpy are already imported.
"""
        else:
            prompt = f"""
USER QUERY: {user_query}

Please provide Python code that addresses this query using the DataFrame (assume it's named 'df').And also check lower case and upper case of the column names and vlause inside the columns and query, If query is to find/search somethings.
Put the code between ```python and ``` tags.
Include only the necessary code to accomplish the task. Assume pandas, matplotlib, seaborn, and numpy are already imported.
"""
        
        # Add the user query to conversation history
        self.conversation_history.append({"role": "user", "content": prompt})
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": "gpt-4o-mini", #"deepcoder:1.5b", #   
            "messages": self.conversation_history,
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions", #"http://127.0.0.1:11434/v1/chat/completions", #
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            
            # Add the model's response to conversation history
            self.conversation_history.append({"role": "assistant", "content": content})
            
            return content
        except Exception as e:
            print(f"Error querying model: {e}")
            return None
    
    def extract_code(self, model_response):
        """Extract Python code from the model's response."""
        if not model_response:
            return None
        
        # Look for code between ```python and ``` tags
        pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(pattern, model_response, re.DOTALL)
        
        if matches:
            return matches[0]
        else:
            # Try without language specification
            pattern = r"```\s*(.*?)\s*```"
            matches = re.findall(pattern, model_response, re.DOTALL)
            if matches:
                return matches[0]
            else:
                print("No code found in model response. Using full response.")
                return model_response
    
    def execute_code(self, code):
        """Execute the extracted Python code and display results."""
        if not code:
            print("No code to execute.")
            return None
        
        if self.df is None:
            print("No DataFrame available. Please load a CSV file first.")
            return None
        
        # Create a copy of the dataframe to work with
        df_copy = self.df.copy()
        
        # Check if code contains plot commands
        has_plot = any(cmd in code for cmd in ['plt.show()', 'plt.savefig', '.plot(', '.hist(', '.bar(', 'sns.'])
        
        # Create a string buffer to capture print outputs
        output_buffer = io.StringIO()
        sys.stdout = output_buffer
        
        # If plotting, redirect matplotlib output to a file
        if has_plot:
            plt.figure(figsize=(10, 6))
        
        try:
            # Add locals to include df_copy
            exec_locals = {"df": df_copy, "pd": pd, "plt": plt, "sns": sns}
            
            # Execute the code
            exec(code, exec_locals)
            
            # Capture print outputs
            output = output_buffer.getvalue()
            
            # Restore standard output
            sys.stdout = sys.__stdout__
            
            result = {"output": output, "plot": None}
            
            # If there's a plot, save it to a bytes buffer
            if has_plot:
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', bbox_inches='tight')
                img_buffer.seek(0)
                result["plot"] = base64.b64encode(img_buffer.read()).decode('utf-8')
                plt.close()
            
            return result
            
        except Exception as e:
            sys.stdout = sys.__stdout__
            error_message = str(e)
            print(f"Error executing code: {error_message}")
            print("Code that failed to execute:")
            print(code)
            return {"error": error_message, "code": code}
    
    def display_results(self, result):
        """Display the execution results, including any plots."""
        if not result:
            return
        
        if "error" in result:
            return False
        
        # Display text output
        if result.get("output"):
            print("Output:")
            print(result["output"])
        
        # Display plot if there is one
        if result.get("plot"):
            display(HTML(f'<img src="data:image/png;base64,{result["plot"]}" />'))
        
        return True, result.get("output", ""), result.get("plot", None)
    
    def process_query(self, user_query):
        """Process a user query from start to finish with multiple retries."""
        print(f"Processing query: {user_query}")
        
        # Step 1: Analyze CSV (if not already done)
        if not self.csv_info:
            self.analyze_csv()
        
        # Initialize retry counter and error message
        retries = 0
        max_retries = 3
        error_message = None
        success = False
        
        while retries < max_retries and not success:
            # Step 2: Query the model (with error message if retry)
            model_response = self.query_model(user_query, error_message)
            if not model_response:
                return
            
            print(f"\n--- MODEL RESPONSE (Attempt {retries+1}/{max_retries}) ---")
            print(model_response)
            print("---------------------\n")
            
            # Step 3: Extract code
            code = self.extract_code(model_response)
            if not code:
                return
            
            print(f"\n--- EXTRACTED CODE (Attempt {retries+1}/{max_retries}) ---")
            print(code)
            print("----------------------\n")
            
            # Step 4: Execute code and display results
            result = self.execute_code(code)
            
            if result and "error" not in result:
                success, text, plot = self.display_results(result)
                break
            elif result:
                error_message = f"Error: {result['error']}\nCode that failed: {result['code']}"
                print(f"\nRetrying due to error... (Attempt {retries+1}/{max_retries})")
                retries += 1
            else:
                break
        
        if not success:
            print("\nFailed to execute the code successfully after all retry attempts.")

        return result

# Example usage
if __name__ == "__main__":
    analyzer = CSVAnalyzer()
    
    # Check if file path is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file_path> [query]")
        sys.exit(1)
    
    # Load the CSV file
    csv_path = sys.argv[1]
    analyzer.load_csv(csv_path)
    analyzer.analyze_csv()
    
    # If query is provided, process it
    if len(sys.argv) > 2:
        query = sys.argv[2]
        analyzer.process_query(query)
    else:
        # Interactive mode
        print("\nEnter your queries (type 'exit' to quit):")
        while True:
            query = input("\nQuery: ")
            if query.lower() == 'exit':
                break
            analyzer.process_query(query)


