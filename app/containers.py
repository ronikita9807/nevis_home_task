"""Dependency injection container: the only place where ports meet implementations."""

from dependency_injector import containers, providers

from app.clients.infra.repository import PostgreSQLClientsUnitOfWork
from app.clients.service import ClientsService
from app.shared.db import Database


class ApplicationContainer(containers.DeclarativeContainer):
    config = providers.Configuration(strict=True)

    # Singleton: the engine and its pool live for the whole process.
    # Sessions are never singletons: each use case opens its own via a unit of work.
    database = providers.Singleton(
        Database,
        url=config.database_url,
        pool_size=config.db_pool_size,
        max_overflow=config.db_max_overflow,
        pool_timeout_seconds=config.db_pool_timeout_seconds,
        pool_recycle_seconds=config.db_pool_recycle_seconds,
    )

    clients_uow = providers.Factory(PostgreSQLClientsUnitOfWork, database=database)
    clients_service = providers.Factory(ClientsService, uow_factory=clients_uow.provider)
