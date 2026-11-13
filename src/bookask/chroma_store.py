"""Chroma client setup and collection creation. No embedding/query logic here —
that's the part you're building yourself."""

import logging

import chromadb
from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)


def get_client(persist_dir: str) -> chromadb.ClientAPI:
    logger.debug("Opening Chroma persistent client at %s", persist_dir)
    return chromadb.PersistentClient(path=persist_dir)


def get_or_create_collection(client: chromadb.ClientAPI, name: str) -> Collection:
    return client.get_or_create_collection(name=name)
