"""Dependency injection container: the only place where ports meet implementations."""

from dependency_injector import containers, providers

from app.clients.infra.repository import PostgreSQLClientsUnitOfWork
from app.clients.service import ClientsService
from app.documents.infra.embedder import FastEmbedEmbedder
from app.documents.infra.repository import PostgreSQLDocumentsUnitOfWork
from app.documents.service import DocumentsService
from app.search.infra.queries import PostgreSQLClientSearchQuery, PostgreSQLDocumentSearchQuery
from app.search.service import SearchService
from app.shared.db import Database


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
    clients_service = providers.Factory(ClientsService, uow_factory=clients_uow.provider)

    documents_uow = providers.Factory(PostgreSQLDocumentsUnitOfWork, database=database)
    documents_service = providers.Factory(
        DocumentsService,
        uow_factory=documents_uow.provider,
        embedder=embedder,
        clients=clients_service,
    )

    client_search_query = providers.Factory(PostgreSQLClientSearchQuery, database=database)
    document_search_query = providers.Factory(PostgreSQLDocumentSearchQuery, database=database)
    search_service = providers.Factory(
        SearchService,
        clients=client_search_query,
        documents=document_search_query,
        embedder=embedder,
    )
