"""Use cases of the documents module."""

import uuid
from collections.abc import Callable

from app.documents.chunking import split_into_chunks
from app.documents.commands import CreateDocumentCommand
from app.documents.domain import Document, DocumentChunk
from app.documents.dto import DocumentDTO
from app.documents.ports import ClientDirectory, DocumentsUnitOfWork
from app.shared.embeddings import Embedder


class DocumentsService:
    def __init__(
        self,
        uow_factory: Callable[[], DocumentsUnitOfWork],
        embedder: Embedder,
        clients: ClientDirectory,
    ) -> None:
        self._uow_factory = uow_factory
        self._embedder = embedder
        self._clients = clients

    async def create_document(
        self, advisor_id: uuid.UUID, client_id: uuid.UUID, command: CreateDocumentCommand
    ) -> DocumentDTO:
        # Cheap check first, so that an unknown client never costs model inference.
        await self._clients.ensure_client_exists(advisor_id, client_id)

        document = Document.create(
            client_id=client_id, advisor_id=advisor_id, title=command.title, content=command.content
        )
        texts = split_into_chunks(document.content)
        # The title gives every chunk its context ("Utility bill" + a line of kWh figures).
        vectors = await self._embedder.embed_documents([f"{document.title}\n{t}" for t in texts])
        chunks = [
            DocumentChunk(index=i, text=text, embedding=vector)
            for i, (text, vector) in enumerate(zip(texts, vectors, strict=True))
        ]

        # Inference is done before the transaction: no pooled connection is held meanwhile.
        # The client could have been removed since the check: the foreign key still guards it.
        async with self._uow_factory() as uow:
            await uow.documents.add(document, chunks, self._embedder.model_name)
            await uow.commit()
        return DocumentDTO.from_domain(document)

    async def get_document(self, advisor_id: uuid.UUID, document_id: uuid.UUID) -> DocumentDTO:
        async with self._uow_factory() as uow:
            document = await uow.documents.get(advisor_id, document_id)
        return DocumentDTO.from_domain(document)
