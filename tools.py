import pandas as pd
import fitz  # pymupdf
import os
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import numpy as np

from apicall import get_reply, Reply

from prompts import DATAFRAME_PATH, NARRATIVE, EMOJI_TRANSLATE

# --- Tool Implementations ---

class DataframeLoader:
    """
    Loads CSV files into pandas dataframes for content extraction.
    Input: file_path (str) - Path to the CSV file
    Output: pandas DataFrame or None if there's an error
    """
    @staticmethod
    def load_csv(file_path: str) -> bool:
        """
        Loads a CSV file into a pandas DataFrame.
        
        Args:
            file_path (str): Path to the CSV file to be loaded
            
        Returns:
            pd.DataFrame or None: The loaded DataFrame or None if there's an error
        """
        class Path(BaseModel):
            full_path: str

        message = [
            {
                "role": "system",
                "content": DATAFRAME_PATH
                # "Here is the user information about the file path to load a CSV. Please extract the information of full_path of dataframe(like csv, excel). If more than one file, return the first one."
            },
            {
                "role": "user",
                "content": f"Load CSV from: {file_path}"
            }
        ]
        response = get_reply(message, Path)
        file_path = response.full_path
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded {file_path}")
            return True
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return False
        except Exception as e:
            print(f"An error occurred while loading the CSV: {e}")
            return False

class ContentExtractor:
    """
    Extracts narrative content from structured data using an LLM.
    Input: dataframe (pd.DataFrame), columns (List[str]), sample_rows (int)
    Output: str - Extracted narrative
    """
    @staticmethod
    def extract_narrative(dataframe_path: str, columns: str, sample_rows: int = 5) -> str:
        """
        Extracts narrative content from specified columns in a DataFrame
        by sampling rows and using an LLM to summarize or describe.
        
        Args:
            file_path: CSV file path to load the DataFrame
            columns (List[str]): List of column names to extract from
            sample_rows (int, optional): Number of rows to sample. Defaults to 5.
            
        Returns:
            str: The narrative extracted from the data
        """
        dataframe = pd.read_csv(dataframe_path)
        class bestColumns(BaseModel):
            columns: List[str]

        message = [
            {
                "role": "system",
                "content": f"From the user provided columns retunr the best match columns list.Head of dataframe is {dataframe.head(1).to_dict(orient='records')}"
            },
            {
                "role": "user",
                "content": f"Here is the user info: {columns}. Please return the best match columns."
            }
        ]
        response = get_reply(message, bestColumns)
        columns = response.columns
        if isinstance(columns, str):
            columns = [col.strip() for col in columns.split(',')]
        elif not isinstance(columns, list):
            raise ValueError("Expected columns to be a list or comma-separated string.")

        # Sample some rows to provide context to the LLM
        sampled_data = dataframe[columns].sample(min(sample_rows, len(dataframe)))
        data_string = sampled_data.to_string()

        message = [
            {
                "role": "system",
                "content": NARRATIVE
                # "Analyze the provided data snippet from a table and extract the main narrative or key information presented in the selected columns. Focus on describing the trends, patterns, or key points evident in this sample."
            },
            {
                "role": "user",
                "content": f"Extract the narrative from the following data snippet:\n\n{data_string}"
            }
        ]

        response = get_reply(message, Reply)
        return response.reply

class EmojiTranslator:
    """
    Translates words and concepts to relevant emojis using an LLM.
    Input: text (str) - The text to translate
    Output: str - String of emojis
    """
    @staticmethod
    def translate_to_emoji(text: str) -> str:
        """
        Translates text or concepts into a sequence of relevant emojis.
        
        Args:
            text (str): The text or concept to translate to emojis
            
        Returns:
            str: A string of emojis representing the input text
        """
        message = [
            {
                "role": "system",
                "content": EMOJI_TRANSLATE
                # "Translate the following text or concept into a sequence of relevant emojis. Provide only the emojis."
            },
            {
                "role": "user",
                "content": f"Translate to emojis: {text}"
            }
        ]

        response = get_reply(message, Reply)
        return response.reply.strip()  # Strip whitespace as LLM might add it

class EmojiMixer:
    """
    Creates custom emoji combinations for content using an LLM.
    Input: str - List of concepts to mix
    Output: str - Combined emoji string
    """
    @staticmethod
    def create_emoji_mix(text:str, concepts: str) -> str:
        """
        Creates a creative emoji combination based on a list of concepts.
        
        Args:
            concepts (str): A text string of concepts to combine into emojis
            
        Returns:
            str: A creative combination of emojis representing the concepts
        """
        concepts_string = ", ".join(concepts)
        message = [
            {
                "role": "system",
                "content": f"Create a unique and creative combination of emojis that represents the following concepts Here is the Concepts {concepts_string}. Combine them in an interesting way. Provide only the emoji combination."
            },
            {
                "role": "user",
                "content": f"Input text is {text}."
            }
        ]

        response = get_reply(message, Reply)
        return response.reply.strip()  # Strip whitespace

class KeypointExtractor:
    """
    Identifies key points from text for article writing using an LLM.
    Input: text (str) - The text to analyze
    Output: List[str] - List of extracted key points
    """
    @staticmethod
    def extract_keypoints(text: str) -> List[str]:
        """
        Extracts key bullet points from a given text.
        
        Args:
            text (str): The text to extract key points from
            
        Returns:
            List[str]: A list of extracted key points as strings
        """
        message = [
            {
                "role": "system",
                "content": "Read the following text and extract the most important key points as a bulleted list. Each key point should be concise."
            },
            {
                "role": "user",
                "content": f"Extract key points from:\n\n{text}"
            }
        ]

        response = get_reply(message, Reply)
        # Assuming the LLM returns a bulleted list, split by lines starting with a bullet
        keypoints = [line.strip() for line in response.reply.split('\n') if line.strip().startswith(('-', '*', '+'))]
        # Simple fallback if not bulleted: split by lines and filter non-empty
        if not keypoints:
             keypoints = [line.strip() for line in response.reply.split('\n') if line.strip()]

        return keypoints

class ContentExpander:
    """
    Expands bullet points or short text into full paragraphs using an LLM.
    Input: points (List[str] or str) - Points to expand
    Output: str - Expanded content
    """
    def expand_content(self, text: str, points: Union[List[str], str]) -> str:
        """
        Expands a list of points or a short text into coherent paragraphs.
        
        Args:
            text (str): The context or title for the expansion
            points (Union[List[str], str]): Either a list of bullet points or a short text that need to add.
            
        Returns:
            str: Expanded content as coherent paragraphs
        """
        if isinstance(points, list):
            input_text = "- " + "\n- ".join(points)
        else:
            input_text = points

        message = [
            {
                "role": "system",
                "content": f"Expand the following points or short text into well-written, coherent paragraphs. Ensure smooth transitions between ideas.{input_text}"
            },
            {
                "role": "user",
                "content": f"Expand the following text need to expand:\n\n{text}"
            }
        ]

        response = get_reply(message, Reply)
        return response.reply

class TextExtractor:
    """
    Extracts text content from PDFs for repurposing using pymupdf.
    Input: pdf_path (str) - Path to the PDF file
    Output: str or None - Extracted text
    """
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> Union[str, None]:
        """
        Extracts text from all pages of a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Union[str, None]: Extracted text as a string or None if an error occurs
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
            print(f"Successfully extracted text from {pdf_path}")
            return text
        except FileNotFoundError:
            print(f"Error: PDF file not found at {pdf_path}")
            return None
        except Exception as e:
            print(f"An error occurred while extracting text from PDF: {e}")
            return None

class ContentReformatter:
    """
    Reformats extracted content into new document types using an LLM.
    Input: text (str) - Text to reformat, format_description (str) - Desired format
    Output: str - Reformatted content
    """
    @staticmethod
    def reformat_content(text: str, format_description: str) -> str:
        """
        Reformats text content according to a specified format description.
        
        Args:
            text (str): The text content to reformat
            format_description (str): Description of the desired format
            
        Returns:
            str: The reformatted content
        """
        message = [
            {
                "role": "system",
                "content": f"Reformat the following text according to the user's specified format. The desired format is: {format_description}"
            },
            {
                "role": "user",
                "content": f"Reformat this text:\n\n{text}"
            }
        ]

        response = get_reply(message, Reply)
        return response.reply

class MultilingualTranslator:
    """
    Translates content between multiple languages using an LLM.
    Input: text (str) - Text to translate, target_language (str) - Target language, preserve_style (bool) - Whether to preserve style
    Output: str - Translated text
    """
    def translate(self, text: str, target_language: str) -> str:
        """
        Translates text to the target language, optionally preserving original style.
        
        Args:
            text (str): The text to translate
            target_language (str): The target language to translate to
            
        Returns:
            str: The translated text
        """
        system_prompt = f"Translate the following text into {target_language}."
        message = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Translate:\n\n{text}"
            }
        ]

        response = get_reply(message, Reply)
        return response.reply

class StylePreserver:
    """
    Maintains writing style during translation or content transformation.
    Input: original_text (str) - Original text, transformed_text (str) - Transformed text
    Output: str - Style-preserved transformed text
    """
    @staticmethod
    def preserve_style(original_text: str, transformed_text: str) -> str:
        """
        Adjusts transformed text to match the style of the original text.
        
        Args:
            original_text (str): The original text with the desired style
            transformed_text (str): The transformed text to adjust
            
        Returns:
            str: The transformed text with preserved style
        """
        message = [
            {
                "role": "system",
                "content": "Analyze the style of the original text and adjust the transformed text to match that style. Preserve tone, formality level, voice, and other stylistic elements."
            },
            {
                "role": "user",
                "content": f"Original text with desired style:\n\n{original_text}\n\nTransformed text to adjust:\n\n{transformed_text}"
            }
        ]

        response = get_reply(message, Reply)
        return response.reply

class CosineSimilarityCalculator:
    """
    Calculates cosine similarity between two embeddings.
    Input: embedding1 (np.ndarray), embedding2 (np.ndarray)
    Output: float - Similarity score
    """
    @staticmethod
    def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculates cosine similarity between two embeddings.
        
        Args:
            embedding1 (np.ndarray): First embedding vector
            embedding2 (np.ndarray): Second embedding vector
            
        Returns:
            float: Cosine similarity score between 0 and 1
        """
        if embedding1.shape != embedding2.shape:
            raise ValueError("Embeddings must have the same shape for cosine similarity calculation.")
        
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0  # Avoid division by zero
        
        return dot_product / (norm1 * norm2)

class AdvanceCSVQuery:
    """
    Performs advanced queries on CSV data using an LLM.
    Input: dataframe (pd.DataFrame), query (str)
    Output: pd.DataFrame - Resulting DataFrame after query
    """
    @staticmethod
    def query_dataframe(dataframe_path: str, text_query: str) -> str:
        """
        Performs an advanced query on a DataFrame using an LLM.
        
        Args:
            dataframe_path (pd.DataFrame): The DataFrame to query
            query (str): The query string
            
        Returns:
            pd.DataFrame: The resulting DataFrame after applying the query
        """
        # run following commands uv run script.py dataframe_path query
        import subprocess
        command = f"uv run tool/csv_script.py {dataframe_path} '{text_query}'"
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            print(f"Query executed successfully: {result.stdout}")
            # Assuming the script returns a CSV string or path
            return result.stdout.strip()  # Return first 10 rows for brevity
        except subprocess.CalledProcessError as e:
            print(f"Error executing query: {e.stderr}")
            return f"Error executing query: {e.stderr}"