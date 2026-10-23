# Gemini Knowledge Base Chatbot

A terminal chatbot that answers questions using only the documents in the
`knowledge/` folder. Documents are split into chunks, embedded with Gemini, and
the most relevant chunks are used as context for each answer.

## Setup

1. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

2. Put your Gemini API key in `.env`:

   ```
   GEMINI_API_KEY=your-api-key-here
   ```

   Get a key from <https://aistudio.google.com/apikey>.

3. Add your own `.md` or `.txt` files to `knowledge/`. The included files are
   sample content; replace them with your own.

4. Run the chatbot:

   ```powershell
   python chatbot.py
   ```

The first run creates `.kb_cache.json` with document embeddings. Later runs
reuse the cache and only re-embed when files change.

## Commands

- `/sources` - list the files in the knowledge base
- `/reload` - re-index the `knowledge/` folder after editing files
- `/reset` - clear the conversation history
- `/exit` or `/quit` - leave the chatbot

## How it works

1. Every `.md` and `.txt` file in `knowledge/` (including subfolders) is read
   and split into chunks of about 1000 characters.
2. Chunks are embedded with `gemini-embedding-001`; embeddings are cached.
3. Each question is embedded and compared to the chunks using cosine
   similarity. The top matches above `KB_MIN_SIMILARITY` are sent to Gemini as
   context.
4. Gemini is instructed to answer only from that context and to cite sources.
   If nothing relevant is found, the chatbot says so instead of guessing.

## Configuration

Optional variables in `.env`:

| Variable                 | Default                    | Description                                           |
| ------------------------ | -------------------------- | ----------------------------------------------------- |
| `GEMINI_MODEL`           | `gemini-3.8-flash`         | Chat model                                            |
| `GEMINI_FALLBACK_MODELS` | `gemini-3.5-flash-lite,...`| Models tried if the primary is busy or out of quota   |
| `EMBEDDING_MODEL`        | `gemini-embedding-001`     | Embedding model                                       |
| `EMBEDDING_DIM`          | `768`                      | Embedding size                                        |
| `KB_CHUNK_SIZE`          | `1000`                     | Target characters per chunk                           |
| `KB_TOP_K`               | `4`                        | Maximum chunks used as context                        |
| `KB_MIN_SIMILARITY`      | `0.55`                     | Minimum similarity to use a chunk                     |
| `SYSTEM_PROMPT`          | Grounded assistant         | Override the system instruction                       |
