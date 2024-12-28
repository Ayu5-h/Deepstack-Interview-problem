import typer
import os
import chromadb
from dotenv import load_dotenv
from src.embeddings import EmbeddingProcessor
from src.character_info import CharacterInfoExtractor
from src.database import VectorStore
import json

app = typer.Typer()


@app.command()
def compute_embeddings(stories_dir: str):
    """Compute embeddings for all stories in the specified directory"""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        typer.echo("Error: GOOGLE_API_KEY not found in environment variables")
        raise typer.Exit(1)

    processor = EmbeddingProcessor(api_key)

    for file_name in os.listdir(stories_dir):
        if file_name.endswith('.txt'):
            file_path = os.path.join(stories_dir, file_name)
            result = processor.process_story(file_path)
            typer.echo(f"Processed {result['story_title']}: {result['chunks']} chunks")


@app.command()
def get_character_info(character_name: str):
    """Get information about a specific character"""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        typer.echo("Error: GOOGLE_API_KEY not found in environment variables")
        raise typer.Exit(1)

    # Search for character in vector store
    db = VectorStore()
    search_results = db.search_character(character_name)

    if not search_results["chunks"]:
        typer.echo(f"Character '{character_name}' not found in any story")
        raise typer.Exit(1)

    # Extract character information
    extractor = CharacterInfoExtractor(api_key)
    story_title = search_results["metadata"][0]["story_title"]

    typer.echo(f"Searching for information about '{character_name}' in {story_title}...")

    character_info = extractor.extract_character_info(
        character_name,
        search_results["chunks"],
        story_title
    )

    if character_info:
        typer.echo("\nCharacter Information:")
        typer.echo(json.dumps(character_info, indent=2))
    else:
        typer.echo(f"\nCould not extract information for character '{character_name}'")
        typer.echo("Please check if the character name is correct and exists in the stories.")
        raise typer.Exit(1)

@app.command()
def reset_database():
    """Reset the vector database (use this if you encounter embedding dimension issues)"""
    try:
        client = chromadb.PersistentClient(path="./data/chroma")
        client.reset()
        typer.echo("Database reset successfully")
    except Exception as e:
        typer.echo(f"Error resetting database: {str(e)}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()