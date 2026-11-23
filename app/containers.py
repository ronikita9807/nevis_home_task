"""Dependency injection container: the only place where ports meet implementations."""

from dependency_injector import containers, providers

from app.clients.infra.repository import PostgreSQLClientsUnitOfWork
from app.clients.service import ClientsService
from app.documents.infra.claude_summarizer import build_claude_summarizer
from app.documents.infra.repository import PostgreSQLDocumentsUnitOfWork
from app.documents.service import DocumentsService
from app.documents.summary import ExtractiveSummarizer, SummaryService
from app.search.infra.queries import PostgreSQLClientSearchQuery, PostgreSQLDocumentSearchQuery
from app.search.service import SearchService
from app.shared.db import Database
from app.shared.fastembed_embedder import FastEmbedEmbedder


class ApplicationContainer(containers.DeclarativeContainer):
    config = providers.Configuration(strict=True)

    # Singletons live for the whole process: the connection pool and the model weights.
    # Sessions are never singletons: each use case opens its own via a unit of work.
    database = providers.Singleton(
        Database,
        url=config.database_url,
        pool_size=config.db_pool_size,
        max_overflow=config.db_max_overflow,
        pool_timeout_seconds=config.db_pool_timeout_seconds,
        pool_recycle_seconds=config.db_pool_recycle_seconds,
    )
    embedder = providers.Singleton(
        FastEmbedEmbedder,
        model_name=config.embedding_model,
        cache_dir=config.embedding_cache_dir,
        threads=config.embedding_threads,
    )

    clients_uow = providers.Factory(PostgreSQLClientsUnitOfWork, database=database)
    clients_service = providers.Factory(
        ClientsService, uow_factory=clients_uow.provider, embedder=embedder
    )

    documents_uow = providers.Factory(PostgreSQLDocumentsUnitOfWork, database=database)
    documents_service = providers.Factory(
        DocumentsService,
        uow_factory=documents_uow.provider,
        embedder=embedder,
        clients=clients_service,
    )

    extractive_summarizer = providers.Singleton(ExtractiveSummarizer, embedder=embedder)
    # None when no ANTHROPIC_API_KEY is configured: summaries are then extractive only.
    llm_summarizer = providers.Singleton(
        build_claude_summarizer,
        api_key=config.anthropic_api_key,
        model=config.summary_model,
        timeout_seconds=config.summary_timeout_seconds,
    )
    summary_service = providers.Factory(
        SummaryService,
        uow_factory=documents_uow.provider,
        extractive=extractive_summarizer,
        llm=llm_summarizer,
    )

    client_search_query = providers.Factory(PostgreSQLClientSearchQuery, database=database)
    document_search_query = providers.Factory(PostgreSQLDocumentSearchQuery, database=database)
    search_service = providers.Factory(
        SearchService,
        clients=client_search_query,
        documents=document_search_query,
        embedder=embedder,
    )
