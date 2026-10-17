"""One bounded batch per query/enrichment; injectable and no model downloads."""
class OpenAIEmbedder:
    def __init__(self, config, trace=None):
        self.config = config
        self.trace = trace if trace is not None else {}

    def __call__(self, texts):
        from openai import OpenAI
        from config.settings import OPENAI_API_KEY
        self.trace["embedding_calls"] = self.trace.get("embedding_calls", 0) + 1
        # Fixed dimension matches pgvector column. Change requires a schema/data migration.
        client = OpenAI(api_key=OPENAI_API_KEY or "not-configured", max_retries=0,
                        timeout=min(5, self.config.role_timeout))
        try:
            response = client.embeddings.create(model=self.config.embedding_model, dimensions=1536,
                                                input=[t[:8000] or "(empty)" for t in texts])
            self.trace["embedding_tokens"] = self.trace.get("embedding_tokens", 0) + response.usage.total_tokens
            return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        finally:
            client.close()
