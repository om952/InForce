import argparse
import json
import uuid
from collections.abc import Iterator
from functools import cache
from pathlib import Path

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

from inforce.config import Settings
from inforce.ingest import FOOTNOTE_REF_RE, OUT_DIR

LEXICAL_VECTOR = "bm25"
RRF_K = 60
CANDIDATES_PER_RETRIEVER = 20
RERANK_DEPTH = 10

DOCUMENT_FIELDS = (
    "doc_id",
    "title",
    "doc_type",
    "circular_number",
    "date",
    "effective_date",
    "source_url",
    "file",
)


@cache
def embedder(model_name: str, cache_dir: str) -> TextEmbedding:
    return TextEmbedding(model_name, cache_dir=cache_dir)


@cache
def lexical_embedder(model_name: str, cache_dir: str) -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name, cache_dir=cache_dir)


@cache
def reranker(model_name: str, cache_dir: str) -> TextCrossEncoder:
    return TextCrossEncoder(model_name, cache_dir=cache_dir)


def connect(settings: Settings) -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    return QdrantClient(path=settings.qdrant_path)


def load_chunks(processed_dir: Path) -> Iterator[dict]:
    for path in sorted(processed_dir.glob("*.json")):
        doc = json.loads(path.read_text())
        document = {field: doc[field] for field in DOCUMENT_FIELDS}
        for chunk in doc["chunks"]:
            yield {**document, **chunk}


def embedding_text(chunk: dict) -> str:
    text = FOOTNOTE_REF_RE.sub("", chunk["text"])
    section = chunk["section"]
    if section and not text.startswith(section):
        return f"{section}\n{text}"
    return text


def index(client: QdrantClient, settings: Settings, chunks: list[dict]) -> int:
    model = embedder(settings.embedding_model, settings.embedding_cache_dir)
    name = settings.qdrant_collection
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config={
            settings.embedding_model: models.VectorParams(
                size=TextEmbedding.get_embedding_size(settings.embedding_model),
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            LEXICAL_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
    texts = [embedding_text(chunk) for chunk in chunks]
    vectors = model.passage_embed(texts, batch_size=16)
    lexical = lexical_embedder(settings.lexical_model, settings.embedding_cache_dir)
    sparse_vectors = lexical.passage_embed(texts)
    client.upload_points(
        name,
        points=(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])),
                vector={
                    settings.embedding_model: vector.tolist(),
                    LEXICAL_VECTOR: models.SparseVector(
                        indices=sparse.indices.tolist(), values=sparse.values.tolist()
                    ),
                },
                payload=chunk,
            )
            for chunk, vector, sparse in zip(
                chunks, vectors, sparse_vectors, strict=True
            )
        ),
    )
    return len(chunks)


def search(
    client: QdrantClient, settings: Settings, query: str, top_k: int = 5
) -> list[dict]:
    model = embedder(settings.embedding_model, settings.embedding_cache_dir)
    vector = next(iter(model.query_embed(settings.embedding_query_prefix + query)))
    response = client.query_points(
        settings.qdrant_collection,
        query=vector.tolist(),
        using=settings.embedding_model,
        limit=top_k,
    )
    return [{"score": point.score, **point.payload} for point in response.points]


def lexical_search(
    client: QdrantClient, settings: Settings, query: str, top_k: int = 5
) -> list[dict]:
    model = lexical_embedder(settings.lexical_model, settings.embedding_cache_dir)
    sparse = next(iter(model.query_embed(query)))
    response = client.query_points(
        settings.qdrant_collection,
        query=models.SparseVector(
            indices=sparse.indices.tolist(), values=sparse.values.tolist()
        ),
        using=LEXICAL_VECTOR,
        limit=top_k,
    )
    return [{"score": point.score, **point.payload} for point in response.points]


def hybrid_search(
    client: QdrantClient,
    settings: Settings,
    query: str,
    top_k: int = 5,
    candidates: int = CANDIDATES_PER_RETRIEVER,
    rerank_depth: int = RERANK_DEPTH,
) -> list[dict]:
    pool = {}
    for retriever, hits in (
        ("dense", search(client, settings, query, candidates)),
        ("lexical", lexical_search(client, settings, query, candidates)),
    ):
        for rank, hit in enumerate(hits, start=1):
            entry = pool.setdefault(
                hit["chunk_id"],
                {**hit, "dense_rank": None, "lexical_rank": None, "rrf_score": 0.0},
            )
            entry[f"{retriever}_rank"] = rank
            entry[f"{retriever}_score"] = hit["score"]
            entry["rrf_score"] += 1 / (RRF_K + rank)
    ranked = sorted(pool.values(), key=lambda entry: entry["rrf_score"], reverse=True)
    ranked = ranked[:rerank_depth]
    model = reranker(settings.reranker_model, settings.embedding_cache_dir)
    scores = model.rerank(query, [embedding_text(entry) for entry in ranked])
    for entry, score in zip(ranked, scores, strict=True):
        entry["score"] = float(score)
    return sorted(ranked, key=lambda entry: entry["score"], reverse=True)[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index or search SEBI chunks in Qdrant."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index")
    search_command = commands.add_parser("search")
    search_command.add_argument("query")
    search_command.add_argument("-k", "--top-k", type=int, default=5)
    search_command.add_argument(
        "--mode", choices=["hybrid", "dense", "lexical"], default="hybrid"
    )
    args = parser.parse_args()

    settings = Settings()
    client = connect(settings)
    if args.command == "index":
        chunks = list(load_chunks(OUT_DIR))
        if not chunks:
            parser.exit(
                1, f"no processed documents in {OUT_DIR}; run inforce.ingest first\n"
            )
        print(
            f"indexed {index(client, settings, chunks)} chunks into {settings.qdrant_collection}"
        )
        return
    retrieve = {"hybrid": hybrid_search, "dense": search, "lexical": lexical_search}
    hits = retrieve[args.mode](client, settings, args.query, args.top_k)
    for rank, hit in enumerate(hits, start=1):
        ranks = f"  dense #{hit.get('dense_rank')} lexical #{hit.get('lexical_rank')}"
        print(
            f"\n{rank}. score={hit['score']:.3f}  {hit['doc_id']}  p.{hit['page_start']}-{hit['page_end']}"
            + (ranks if args.mode == "hybrid" else "")
        )
        print(
            f"   {hit['title']} | {hit['circular_number'] or hit['doc_type']} | {hit['source_url']}"
        )
        print(f"   section: {hit['section']}")
        print(f"   {hit['text'][:400]}")


if __name__ == "__main__":
    main()
