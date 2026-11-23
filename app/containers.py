"""Dependency injection container: the only place where ports meet implementations."""

from dependency_injector import containers, providers

from app.shared.db import Database


class ApplicationContainer(containers.DeclarativeContainer):
    config = providers.Configuration(strict=True)

    # Singleton: the engine and its pool live for the whole process.
    # Sessions are never singletons: each use case opens its own via `Database.session()`.
    database = providers.Singleton(
        Database,
        url=config.database_url,
        pool_size=config.db_pool_size,
        max_overflow=config.db_max_overflow,
        pool_timeout_seconds=config.db_pool_timeout_seconds,
        pool_recycle_seconds=config.db_pool_recycle_seconds,
    )
