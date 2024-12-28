from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import google.generativeai as genai
import json
from typing import Optional, Dict, List


class CharacterInfoExtractor:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.llm = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.1  # Lower temperature for more consistent outputs
        )

    def extract_character_info(self, character_name: str, story_chunks: List[str], story_title: str) -> Optional[Dict]:
        # Combine relevant chunks and limit context length
        context = "\n".join(story_chunks)
        if len(context) > 30000:  # Limit context to avoid token limits
            context = context[:30000] + "..."

        prompt_template = """
        Task: Extract information about the character {character_name} from the following story context.

        Story Title: {story_title}

        Context:
        {context}

        Instructions:
        1. If the character is not found in the context, return exactly: {{"error": "Character not found"}}
        2. If the character is found, provide information in the following JSON format:
        {{
            "name": "{character_name}",
            "storyTitle": "{story_title}",
            "summary": "Brief description of the character's role and journey in the story",
            "relations": [
                {{"name": "Name of related character", "relation": "Type of relationship"}}
            ],
            "characterType": "One of: protagonist, antagonist, supporting character"
        }}

        Important: Ensure the output is valid JSON format. Include only information that is explicitly mentioned in or can be directly inferred from the context.
        """

        # Create prompt
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["character_name", "story_title", "context"]
        )

        # Create chain
        chain = LLMChain(llm=self.llm, prompt=prompt)

        try:
            # Use invoke instead of predict
            response = chain.invoke({
                "character_name": character_name,
                "story_title": story_title,
                "context": context
            })

            # Extract the response text
            response_text = response.get('text', '')

            # Try to parse JSON from the response
            try:
                result = json.loads(response_text)

                # Check if we got an error response
                if "error" in result:
                    print(f"Character search result: {result['error']}")
                    return None

                return result
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response: {str(e)}")
                print(f"Raw response: {response_text}")
                return None

        except Exception as e:
            print(f"Error during character extraction: {str(e)}")
            return None